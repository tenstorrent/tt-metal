// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Proof that a multi-core sharded `DataflowBufferSpec::borrowed_from` add works on real hardware —
// the primitive the Metal 2.0 binary_ng factory is built on.
//
// Sharded borrowed_from is API-supported but exercised by no other test: every other borrowed/alias
// DFB test is interleaved + single-core, and the spec-time size check defers per-bank validation to
// attach time. This is the simplest case that exercises the sharded borrowed-DFB primitive:
//
//   - A small HEIGHT-sharded bf16 tensor, one 32x32 tile per core, across N cores.
//   - in0, in1, out are each backed by a borrowed_from sharded L1 tensor (no NoC traffic).
//   - reader publishes the resident shards; compute adds in0+in1 -> out; writer drains.
//   - These are the production DFB kernels the factory emits
//     (ttnn/.../binary_ng/device/kernels_dfb/{dataflow,compute}/...).
//
// If the per-bank attach check (dfb_total_bytes <= aligned_size_per_bank) rejects a sharded borrow,
// or the per-core shard address math is wrong, this test fails. Constant-valued tiles make the check
// tile-layout-invariant.

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/compute_hardware_config.hpp>
#include <tt-metalium/experimental/metal2_host_api/data_movement_hardware_config.hpp>
#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>
#include <tt-metalium/experimental/tensor/topology/tensor_topology.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/page_config.hpp>

#include "device_fixture.hpp"
#include "impl/dataflow_buffer/dataflow_buffer_impl.hpp"
#include "impl/program/program_impl.hpp"
#include "tt_metal/test_utils/packing.hpp"

namespace tt::tt_metal {
namespace {

using namespace experimental;
using tt::test_utils::pack_vector;
using tt::test_utils::unpack_vector;

// Production DFB kernels under test — the ones the Metal 2.0 binary_ng factory emits.
constexpr const char* READER_KERNEL =
    "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_dfb/dataflow/reader_sharded_no_bcast_dfb.cpp";
constexpr const char* WRITER_KERNEL =
    "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_dfb/dataflow/writer_sharded_no_bcast_dfb.cpp";
constexpr const char* COMPUTE_KERNEL =
    "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_dfb/compute/eltwise_binary_no_bcast_dfb.cpp";

// A height-sharded, tile-layout, bf16 L1 tensor: `num_cores` shards of `tiles_per_core` tiles each,
// laid out one shard per core along a 1xN core strip. Shape = {32 * num_cores * tiles_per_core, 32}.
TensorSpec make_height_sharded_bf16_spec(uint32_t num_cores, uint32_t tiles_per_core) {
    const uint32_t tile_h = tt::constants::TILE_HEIGHT;
    const uint32_t tile_w = tt::constants::TILE_WIDTH;
    const uint32_t shard_h = tiles_per_core * tile_h;
    const uint32_t total_h = num_cores * shard_h;

    auto shard_grid = num_cores_to_corerangeset(num_cores, CoreCoord{num_cores, 1}, /*row_wise=*/true);
    ShardSpec shard_spec{shard_grid, {shard_h, tile_w}, ShardOrientation::ROW_MAJOR};
    MemoryConfig memory_config{TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1, shard_spec};
    auto page_config = PageConfig(Layout::TILE);
    auto tensor_layout = TensorLayout(DataType::BFLOAT16, page_config, memory_config);
    return TensorSpec(Shape{total_h, tile_w}, tensor_layout);
}

// Fill a host bf16 vector (one element per logical element of the tensor) with a constant value.
// Constant tiles are identical in row-major and tile layout, so no host tilization is needed.
std::vector<uint32_t> packed_constant_bf16(uint32_t num_elements, float value) {
    std::vector<bfloat16> v(num_elements, bfloat16(value));
    return pack_vector<uint32_t, bfloat16>(v);
}

struct ShardedBorrowConfig {
    uint32_t num_cores = 4;
    uint32_t tiles_per_core = 1;
    float in0_value = 2.0f;
    float in1_value = 3.0f;
    bool fused_relu = false;
};

void run_sharded_borrowed_dfb_add(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const ShardedBorrowConfig& cfg) {
    IDevice* device = mesh_device->get_devices()[0];
    const uint32_t tile_bytes = tile_size(tt::DataFormat::Float16_b);  // 2048 for bf16 32x32
    const uint32_t elems_per_tile = tt::constants::TILE_HW;            // 1024

    // ---- Tensor specs (all three identical: height-sharded, one shard per core) ----
    const TensorSpec spec = make_height_sharded_bf16_spec(cfg.num_cores, cfg.tiles_per_core);
    const uint32_t total_tiles = cfg.num_cores * cfg.tiles_per_core;
    const uint32_t total_elems = total_tiles * elems_per_tile;

    // ---- ProgramSpec ----
    ProgramSpec pspec;
    pspec.name = "sharded_borrowed_dfb_add";

    const DFBSpecName IN0{"in0_dfb"};
    const DFBSpecName IN1{"in1_dfb"};
    const DFBSpecName OUT{"out_dfb"};
    const TensorParamName T_IN0{"in0_tensor"};
    const TensorParamName T_IN1{"in1_tensor"};
    const TensorParamName T_OUT{"out_tensor"};

    // DFBs borrow the per-core shards. entry_size = one tile; num_entries sized so the ring holds a
    // full shard (entry_size * num_entries must fit one shard == tiles_per_core tiles, the per-bank
    // check at attach time).
    auto make_dfb = [&](const DFBSpecName& name, const TensorParamName& borrowed) {
        return DataflowBufferSpec{
            .unique_id = name,
            .entry_size = tile_bytes,
            .num_entries = cfg.tiles_per_core,
            .data_format_metadata = tt::DataFormat::Float16_b,
            .borrowed_from = borrowed,
        };
    };
    DataflowBufferSpec in0_dfb = make_dfb(IN0, T_IN0);
    DataflowBufferSpec in1_dfb = make_dfb(IN1, T_IN1);
    DataflowBufferSpec out_dfb = make_dfb(OUT, T_OUT);

    // Compute defines for plain ADD (subset of the binary_ng factory's build_binary_defines()).
    KernelSpec::CompilerOptions::Defines defines;
    defines.emplace("ELTWISE_OP_TYPE", "EltwiseBinaryType::ELWADD");
    defines.emplace("ELTWISE_OP", "add_tiles");
    defines.emplace("ELTWISE_OP_INIT", "add_tiles_init");
    if (cfg.fused_relu) {
        defines.emplace("PACK_RELU", "1");
    }

    // ---- Reader (RISCV_1): publishes in0/in1 shards ----
    KernelSpec reader{
        .unique_id = KernelSpecName{"reader"},
        .source = std::filesystem::path(READER_KERNEL),
        .num_threads = 1,
        .dfb_bindings = {ProducerOf(IN0, "in0"), ProducerOf(IN1, "in1")},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles"}},
        .hw_config =
            DataMovementHardwareConfig{
                .gen1_config = DataMovementHardwareConfig::Gen1Config{.processor = DataMovementProcessor::RISCV_1}},
    };

    // ---- Writer (RISCV_0): drains out shard ----
    KernelSpec writer{
        .unique_id = KernelSpecName{"writer"},
        .source = std::filesystem::path(WRITER_KERNEL),
        .num_threads = 1,
        .dfb_bindings = {ConsumerOf(OUT, "out")},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles"}},
        .hw_config =
            DataMovementHardwareConfig{
                .gen1_config = DataMovementHardwareConfig::Gen1Config{.processor = DataMovementProcessor::RISCV_0}},
    };

    // ---- Compute: in0 + in1 -> out ----
    KernelSpec compute{
        .unique_id = KernelSpecName{"compute"},
        .source = std::filesystem::path(COMPUTE_KERNEL),
        .num_threads = 1,
        .compiler_options = {.defines = defines},
        .dfb_bindings = {ConsumerOf(IN0, "in0"), ConsumerOf(IN1, "in1"), ProducerOf(OUT, "out")},
        .compile_time_args = {{"num_tiles_per_cycle", cfg.tiles_per_core}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles"}},
        .hw_config =
            ComputeHardwareConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = false,  // bf16
            },
    };

    pspec.tensor_parameters = {
        {.unique_id = T_IN0, .spec = spec},
        {.unique_id = T_IN1, .spec = spec},
        {.unique_id = T_OUT, .spec = spec},
    };
    pspec.kernels = {reader, writer, compute};
    pspec.dataflow_buffers = {in0_dfb, in1_dfb, out_dfb};

    // Work unit covers the shard core strip (0,0)..(num_cores-1,0).
    std::set<CoreRange> ranges;
    for (uint32_t c = 0; c < cfg.num_cores; ++c) {
        ranges.insert(CoreRange(CoreCoord{c, 0}, CoreCoord{c, 0}));
    }
    NodeRangeSet target_nodes(ranges);
    pspec.work_units = {WorkUnitSpec{
        .name = "wu",
        .kernels = {KernelSpecName{"reader"}, KernelSpecName{"writer"}, KernelSpecName{"compute"}},
        .target_nodes = target_nodes}};

    Program program = MakeProgramFromSpec(*mesh_device, pspec);

    // ---- Allocate the three sharded L1 tensors and fill inputs ----
    MeshTensor in0_t = MeshTensor::allocate_on_device(*mesh_device, spec, TensorTopology{});
    MeshTensor in1_t = MeshTensor::allocate_on_device(*mesh_device, spec, TensorTopology{});
    MeshTensor out_t = MeshTensor::allocate_on_device(*mesh_device, spec, TensorTopology{});

    detail::WriteToBuffer(
        *in0_t.mesh_buffer().get_reference_buffer(), packed_constant_bf16(total_elems, cfg.in0_value));
    detail::WriteToBuffer(
        *in1_t.mesh_buffer().get_reference_buffer(), packed_constant_bf16(total_elems, cfg.in1_value));
    detail::WriteToBuffer(
        *out_t.mesh_buffer().get_reference_buffer(), packed_constant_bf16(total_elems, -99.0f));  // poison

    // ---- Per-core runtime args ----
    using NodeRuntimeArgs = ProgramRunArgs::KernelRunArgs::NodeRuntimeArgs;
    Group<NodeRuntimeArgs> reader_args, writer_args, compute_args;
    for (uint32_t c = 0; c < cfg.num_cores; ++c) {
        NodeCoord node{c, 0};
        reader_args.push_back({node, {{"num_tiles", cfg.tiles_per_core}}});
        writer_args.push_back({node, {{"num_tiles", cfg.tiles_per_core}}});
        compute_args.push_back({node, {{"num_tiles", cfg.tiles_per_core}}});
    }

    ProgramRunArgs params;
    params.kernel_run_args = {
        ProgramRunArgs::KernelRunArgs{.kernel = KernelSpecName{"reader"}, .runtime_arg_values = reader_args},
        ProgramRunArgs::KernelRunArgs{.kernel = KernelSpecName{"writer"}, .runtime_arg_values = writer_args},
        ProgramRunArgs::KernelRunArgs{.kernel = KernelSpecName{"compute"}, .runtime_arg_values = compute_args},
    };
    params.tensor_args.emplace(T_IN0, TensorArgument{in0_t});
    params.tensor_args.emplace(T_IN1, TensorArgument{in1_t});
    params.tensor_args.emplace(T_OUT, TensorArgument{out_t});
    SetProgramRunArgs(program, params);

    // ---- Launch ----
    detail::LaunchProgram(device, program, /*wait_until_cores_done=*/true);

    // Each borrowed DFB was attached to its core's shard. The three DFBs (in0/in1/out) each have a
    // uniform per-bank base equal to their backing tensor's (sharded) L1 address. Confirm the borrow
    // landed by matching each DFB's uniform_alloc_addr to one of the three tensor addresses.
    const std::set<uint32_t> tensor_addrs{
        static_cast<uint32_t>(in0_t.address()),
        static_cast<uint32_t>(in1_t.address()),
        static_cast<uint32_t>(out_t.address())};
    for (const auto& dfb : program.impl().dataflow_buffers()) {
        EXPECT_TRUE(tensor_addrs.count(dfb->uniform_alloc_addr()) == 1)
            << "borrowed DFB uniform_alloc_addr " << dfb->uniform_alloc_addr()
            << " does not match any sharded tensor L1 address";
    }

    // ---- Verify ----
    std::vector<uint32_t> packed_out;
    detail::ReadFromBuffer(*out_t.mesh_buffer().get_reference_buffer(), packed_out);
    auto out = unpack_vector<bfloat16, uint32_t>(packed_out);
    ASSERT_EQ(out.size(), total_elems);

    float expected = cfg.in0_value + cfg.in1_value;
    if (cfg.fused_relu) {
        expected = expected > 0.0f ? expected : 0.0f;
    }
    for (size_t i = 0; i < out.size(); ++i) {
        EXPECT_FLOAT_EQ(static_cast<float>(out[i]), expected)
            << "mismatch at element " << i << " (tile " << (i / elems_per_tile) << ")";
    }
}

}  // namespace

// =============================================================================
// Multi-core sharded borrowed-DFB add.
// =============================================================================

TEST_F(MeshDeviceFixture, ShardedBorrowedDFBAdd_4core_1tile) {
    run_sharded_borrowed_dfb_add(devices_.at(0), {.num_cores = 4, .tiles_per_core = 1});
}

TEST_F(MeshDeviceFixture, ShardedBorrowedDFBAdd_4core_2tile) {
    run_sharded_borrowed_dfb_add(devices_.at(0), {.num_cores = 4, .tiles_per_core = 2});
}

TEST_F(MeshDeviceFixture, ShardedBorrowedDFBAdd_8core_1tile) {
    run_sharded_borrowed_dfb_add(devices_.at(0), {.num_cores = 8, .tiles_per_core = 1});
}

TEST_F(MeshDeviceFixture, ShardedBorrowedDFBAdd_4core_fused_relu_positive) {
    run_sharded_borrowed_dfb_add(
        devices_.at(0),
        {.num_cores = 4, .tiles_per_core = 1, .in0_value = 2.0f, .in1_value = 3.0f, .fused_relu = true});
}

TEST_F(MeshDeviceFixture, ShardedBorrowedDFBAdd_4core_fused_relu_negative_clamps) {
    // in0 + in1 = -5 -> RELU clamps to 0.
    run_sharded_borrowed_dfb_add(
        devices_.at(0),
        {.num_cores = 4, .tiles_per_core = 1, .in0_value = -2.0f, .in1_value = -3.0f, .fused_relu = true});
}

}  // namespace tt::tt_metal
