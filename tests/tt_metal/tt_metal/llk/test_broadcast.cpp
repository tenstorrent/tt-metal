// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/reflection.hpp>
#include <chrono>
#include <fmt/base.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <bit>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/base_types.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include "llk_device_fixture.hpp"
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include "test_golden_impls.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tt_metal/test_utils/env_vars.hpp"
#include <umd/device/types/arch.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/float32.hpp"
#include "tt_metal/test_utils/packing.hpp"
#include "tt_metal/test_utils/stimulus.hpp"

namespace tt::tt_metal {
class IDevice;
}  // namespace tt::tt_metal

namespace tt::tt_metal {

using std::map;
using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

namespace unit_tests::compute::broadcast {

enum ApiConvention : std::uint8_t {
    DEFAULT = 0,
    SHORT_INIT = 1,  // call <op>_bcast_<dim>_init_short instead of init_bcast
    SHORT_CALL = 2,  // call <op>_tiles_bcast_<dim> instead of <op>_tiles_bcast
    SHORT_BOTH = 3   // both SHORT_INIT and SHORT_CALL
};

enum EltwiseOp : std::uint8_t { ADD = 0, SUB = 1, MUL = 2 };

enum BroadcastDim : std::uint8_t { ROW = 0, COL = 1, SCALAR = 2 };

enum TileShape : std::uint8_t { FULL_TILE = 0, TINY_TILE_16x32 = 1 };

const map<EltwiseOp, std::string> eltwise_op_to_type = {
    {EltwiseOp::ADD, "EltwiseBinaryType::ELWADD"},
    {EltwiseOp::SUB, "EltwiseBinaryType::ELWSUB"},
    {EltwiseOp::MUL, "EltwiseBinaryType::ELWMUL"}};

const map<EltwiseOp, std::string> eltwise_op_to_api_prefix = {
    {EltwiseOp::ADD, "add"}, {EltwiseOp::SUB, "sub"}, {EltwiseOp::MUL, "mul"}};

const map<BroadcastDim, std::string> broadcast_dim_to_type = {
    {BroadcastDim::ROW, "BroadcastType::ROW"},
    {BroadcastDim::COL, "BroadcastType::COL"},
    {BroadcastDim::SCALAR, "BroadcastType::SCALAR"},
};

const map<BroadcastDim, std::string> broadcast_dim_to_api_suffix = {
    {BroadcastDim::ROW, "rows"},
    {BroadcastDim::COL, "cols"},
    {BroadcastDim::SCALAR, "scalar"},
};

const map<TileShape, tt_metal::Tile> tile_shape_to_tile = {
    {TileShape::FULL_TILE, tt_metal::Tile({constants::TILE_HEIGHT, constants::TILE_WIDTH})},
    {TileShape::TINY_TILE_16x32, tt_metal::Tile({constants::TILE_HEIGHT / 2, constants::TILE_WIDTH})},
};

// Per-element bfloat16 tolerance shared by both runners below.
constexpr float k_broadcast_rtol = 0.0155;

// Quasar drives dataflow through the Gen2 config and takes its DFB sync explicitly; Gen1 arches pin
// the processor/NOC pair per direction. Identical for every runner in this file, hence the helpers.
experimental::DataMovementHardwareConfig make_reader_hw_config(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    if (mesh_device->arch() == tt::ARCH::QUASAR) {
        return experimental::DataMovementHardwareConfig{
            .gen2_specific = experimental::DataMovementHardwareConfig::DataMovement2XXConfig{
                .disable_dfb_implicit_sync_for_all = true}};
    }
    return experimental::DataMovementHardwareConfig{
        .gen1_specific = experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default}};
}

experimental::DataMovementHardwareConfig make_writer_hw_config(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    if (mesh_device->arch() == tt::ARCH::QUASAR) {
        return experimental::DataMovementHardwareConfig{
            .gen2_specific = experimental::DataMovementHardwareConfig::DataMovement2XXConfig{
                .disable_dfb_implicit_sync_for_all = true}};
    }
    return experimental::DataMovementHardwareConfig{
        .gen1_specific = experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default}};
}

experimental::ComputeHardwareConfig make_compute_hw_config(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, MathFidelity math_fidelity) {
    if (mesh_device->arch() == tt::ARCH::QUASAR) {
        return experimental::ComputeHardwareConfig{.fpu_math_fidelity = math_fidelity};
    }
    return experimental::ComputeHardwareConfig{.fpu_math_fidelity = math_fidelity};
}

struct BroadcastConfig {
    ApiConvention api_convention;
    EltwiseOp eltwise_op;
    BroadcastDim broadcast_dim;
    TileShape tile_shape = TileShape::FULL_TILE;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    std::uint32_t bcast_row_idx = 0;
};

void mask_src_b_for_broadcast(
    std::vector<bfloat16>& tile, const std::vector<std::uint32_t>& shape, BroadcastDim dim, std::uint32_t row_idx = 0) {
    int num_rows = shape.at(0);
    int num_cols = shape.at(1);

    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            if ((dim == BroadcastDim::ROW && i != row_idx) ||
                (dim == BroadcastDim::SCALAR && (i != row_idx || j != 0))) {
                tile[(i * num_cols) + j] = 0.0f;
            }
        }
    }
}

std::vector<bfloat16> gold_broadcast(
    std::vector<bfloat16>& src_a,
    std::vector<bfloat16>& src_b,
    const std::vector<std::uint32_t>& shape,
    EltwiseOp op,
    BroadcastDim dim,
    std::uint32_t row_idx = 0,
    MathFidelity math_fidelity = MathFidelity::HiFi4) {
    int num_rows = shape.at(0);
    int num_cols = shape.at(1);

    std::uint16_t srca_fid_mask = 0xFFFF;
    std::uint16_t srcb_fid_mask = 0xFFFF;

    std::vector<bfloat16> golden(num_cols * num_rows);

    switch (math_fidelity) {
        case MathFidelity::HiFi4:
        case MathFidelity::HiFi3: {
            break;
        }
        case MathFidelity::HiFi2: {
            srcb_fid_mask = 0xFFFE;
            break;
        }
        case MathFidelity::LoFi: {
            srca_fid_mask = 0xFFF8;
            srcb_fid_mask = 0xFFFE;
            break;
        }
        default: {
            TT_THROW("Unsupported MathFidelity={}", math_fidelity);
            break;
        }
    }

    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            bfloat16 broadcast_value{};
            switch (dim) {
                case BroadcastDim::ROW: {
                    broadcast_value = src_b[(row_idx * num_cols) + j];
                    break;
                }
                case BroadcastDim::COL: {
                    broadcast_value = src_b[i * num_cols];
                    break;
                }
                case BroadcastDim::SCALAR: {
                    broadcast_value = src_b[0];
                    break;
                }
                default: {
                    TT_THROW("Unsupported BroadcastDim={}", dim);
                    break;
                }
            }

            switch (op) {
                case EltwiseOp::ADD: {
                    golden[(i * num_cols) + j] =
                        static_cast<float>(src_a[(i * num_cols) + j]) + static_cast<float>(broadcast_value);
                    break;
                }
                case EltwiseOp::SUB: {
                    golden[(i * num_cols) + j] =
                        static_cast<float>(src_a[(i * num_cols) + j]) - static_cast<float>(broadcast_value);
                    break;
                }
                case EltwiseOp::MUL: {
                    golden[(i * num_cols) + j] =
                        static_cast<float>(std::bit_cast<bfloat16>(static_cast<std::uint16_t>(
                            std::bit_cast<std::uint16_t>(src_a[(i * num_cols) + j]) & srca_fid_mask))) *
                        static_cast<float>(std::bit_cast<bfloat16>(
                            static_cast<std::uint16_t>(std::bit_cast<std::uint16_t>(broadcast_value) & srcb_fid_mask)));
                    break;
                }
                default: {
                    TT_THROW("Unsupported EltwiseOp={}", op);
                    break;
                }
            }
        }
    }

    return golden;
}

constexpr std::uint32_t k_num_tiles_broadcast_test = 1;

auto CreateDramBufferForPageSize(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    std::uint32_t page_size_bytes,
    std::uint32_t num_pages) {
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = page_size_bytes, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = false};
    distributed::ReplicatedBufferConfig buffer_config{.size = page_size_bytes * num_pages};
    return distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
}

void run_single_core_broadcast(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const BroadcastConfig& test_config) {
    auto& cq = mesh_device->mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::MeshWorkload workload;

    const experimental::NodeCoord node{0, 0};

    tt_metal::Tile tile_dims = tile_shape_to_tile.at(test_config.tile_shape);
    std::uint32_t tile_width = tile_dims.get_tile_shape()[1];
    std::uint32_t tile_height = tile_dims.get_tile_shape()[0];
    if (test_config.tile_shape != TileShape::FULL_TILE) {
        log_info(tt::LogTest, "Tile shape is {{{}, {}}}", tile_height, tile_width);
    }

    std::uint32_t single_tile_size = tile_width * tile_height * sizeof(bfloat16);

    auto src_a_dram_buffer = CreateDramBufferForPageSize(mesh_device, single_tile_size, k_num_tiles_broadcast_test);
    std::uint32_t dram_buffer_src_a_addr = src_a_dram_buffer->address();

    auto src_b_dram_buffer = CreateDramBufferForPageSize(mesh_device, single_tile_size, k_num_tiles_broadcast_test);
    std::uint32_t dram_buffer_src_b_addr = src_b_dram_buffer->address();

    auto dst_dram_buffer = CreateDramBufferForPageSize(mesh_device, single_tile_size, k_num_tiles_broadcast_test);
    std::uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

    auto* device = mesh_device->get_devices().empty() ? nullptr : mesh_device->get_devices().front();
    TT_FATAL(device != nullptr, "mesh_device has no backing devices");
    const bool is_quasar = device->arch() == ARCH::QUASAR;

    std::map<std::string, std::string> defines = {
        {"BCAST_LLKOP", eltwise_op_to_type.at(test_config.eltwise_op)},
        {"BCAST_DIM", broadcast_dim_to_type.at(test_config.broadcast_dim)},
        {"BCAST_OP", eltwise_op_to_api_prefix.at(test_config.eltwise_op) + "_tiles_bcast"},
        {"BCAST_ROW_IDX", std::to_string(test_config.bcast_row_idx)}};

    // Add a helper define to indicate if this is a row broadcast
    if (test_config.broadcast_dim == BroadcastDim::ROW) {
        defines["BCAST_IS_ROW"] = "1";
    }

    log_info(
        tt::LogTest,
        "Testing BCAST_LLKOP={} BCAST_DIM={} ROW_IDX={}",
        defines["BCAST_LLKOP"],
        defines["BCAST_DIM"],
        test_config.bcast_row_idx);

    if (test_config.api_convention == ApiConvention::SHORT_INIT ||
        test_config.api_convention == ApiConvention::SHORT_BOTH) {
        defines["BCAST_OP_INIT"] = eltwise_op_to_api_prefix.at(test_config.eltwise_op) + "_bcast_" +
                                   broadcast_dim_to_api_suffix.at(test_config.broadcast_dim) + "_init_short";

        if ((test_config.eltwise_op == EltwiseOp::SUB || test_config.eltwise_op == EltwiseOp::MUL) &&
            test_config.broadcast_dim == BroadcastDim::SCALAR) {
            // FIXME sub_bcast_scalar_init_short and mul_bcast_scalar_init_short are instead called
            // sub_tiles_bcast_scalar_init_short and mul_tiles_bcast_scalar_init_short
            defines["BCAST_OP_INIT"] = eltwise_op_to_api_prefix.at(test_config.eltwise_op) + "_tiles_bcast_" +
                                       broadcast_dim_to_api_suffix.at(test_config.broadcast_dim) + "_init_short";
        }

        log_info(tt::LogTest, "Init function is {}", defines["BCAST_OP_INIT"]);
    } else {
        log_info(tt::LogTest, "Init function is init_bcast");
    }

    if (test_config.api_convention == ApiConvention::SHORT_CALL ||
        test_config.api_convention == ApiConvention::SHORT_BOTH) {
        defines["BCAST_SPECIFIC"] = "1";
        defines["BCAST_OP"] = defines["BCAST_OP"] + "_" + broadcast_dim_to_api_suffix.at(test_config.broadcast_dim);
    }

    log_info(tt::LogTest, "Compute function is {}", defines["BCAST_OP"]);

    experimental::KernelSpec::CompilerOptions::Defines defines_vec;
    for (auto& kv : defines) {
        defines_vec.emplace(kv.first, kv.second);
    }

    const experimental::DFBSpecName INP0_DFB{"inp0_dfb"};
    const experimental::DFBSpecName INP1_DFB{"inp1_dfb"};
    const experimental::DFBSpecName OUT_DFB{"out_dfb"};
    const experimental::KernelSpecName READER{"reader"};
    const experimental::KernelSpecName WRITER{"writer"};
    const experimental::KernelSpecName COMPUTE{"compute"};

    auto make_dfb = [&](const experimental::DFBSpecName& name) {
        return experimental::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = single_tile_size,
            .num_entries = k_num_tiles_broadcast_test,
            .data_format_metadata = tt::DataFormat::Float16_b,
            .tile_format_metadata = tile_dims,
        };
    };

    experimental::DataflowBufferSpec inp0_dfb_spec = make_dfb(INP0_DFB);
    experimental::DataflowBufferSpec inp1_dfb_spec = make_dfb(INP1_DFB);
    experimental::DataflowBufferSpec out_dfb_spec = make_dfb(OUT_DFB);

    experimental::DataMovementHardwareConfig reader_hw_config = make_reader_hw_config(mesh_device);
    experimental::KernelSpec reader_spec{
        .unique_id = READER,
        .source =

            "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings =
            {{
                 .dfb_spec_name = INP0_DFB,
                 .accessor_name = "in0",
                 .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = INP1_DFB,
                 .accessor_name = "in1",
                 .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             }},
        .runtime_arg_schema =
            {.runtime_arg_names = {"src0_addr", "src0_bank_id", "src1_addr", "src1_bank_id", "num_tiles"}},
        .hw_config = reader_hw_config,
    };

    experimental::DataMovementHardwareConfig writer_hw_config = make_writer_hw_config(mesh_device);
    experimental::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source =

            "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ConsumerOf(OUT_DFB, "in")},
        .runtime_arg_schema = {.runtime_arg_names = {"dst_addr", "bank_id", "num_tiles"}},
        .hw_config = writer_hw_config,
    };

    experimental::ComputeHardwareConfig compute_hw_config =
        make_compute_hw_config(mesh_device, test_config.math_fidelity);

    experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source =

            "tests/tt_metal/tt_metal/test_kernels/compute/broadcast_2_0.cpp",
        .num_threads = 1,
        .compiler_options = {.defines = defines_vec},
        .dfb_bindings =
            {{
                 .dfb_spec_name = INP0_DFB,
                 .accessor_name = "in0",
                 .endpoint_type = experimental::DFBEndpointType::CONSUMER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = INP1_DFB,
                 .accessor_name = "in1",
                 .endpoint_type = experimental::DFBEndpointType::CONSUMER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = OUT_DFB,
                 .accessor_name = "out",
                 .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             }},
        .hw_config = compute_hw_config,
    };

    experimental::WorkUnitSpec wu{
        .name = "main",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = node,
    };

    experimental::ProgramSpec spec{
        .name = "single_core_broadcast",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {inp0_dfb_spec, inp1_dfb_spec, out_dfb_spec},
        .work_units = {wu},
    };

    Program built_program = experimental::MakeProgramFromSpec(*mesh_device, spec);
    workload.add_program(device_range, std::move(built_program));
    auto& program_run = workload.get_programs().at(device_range);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node,
                {{"src0_addr", static_cast<std::uint32_t>(dram_buffer_src_a_addr)},
                 {"src0_bank_id", 0u},
                 {"src1_addr", static_cast<std::uint32_t>(dram_buffer_src_b_addr)},
                 {"src1_bank_id", 0u},
                 {"num_tiles", static_cast<std::uint32_t>(k_num_tiles_broadcast_test)}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node,
                {{"dst_addr", static_cast<std::uint32_t>(dram_buffer_dst_addr)},
                 {"bank_id", 0u},
                 {"num_tiles", static_cast<std::uint32_t>(k_num_tiles_broadcast_test)}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE},
    };
    experimental::SetProgramRunArgs(program_run, params);

    std::vector<bfloat16> input0 = generate_uniform_random_vector<bfloat16>(
        -1.0f, 1.0f, single_tile_size / sizeof(bfloat16), std::chrono::system_clock::now().time_since_epoch().count());

    std::vector<bfloat16> input1 = generate_uniform_random_vector<bfloat16>(
        -1.0f, 1.0f, single_tile_size / sizeof(bfloat16), std::chrono::system_clock::now().time_since_epoch().count());

    mask_src_b_for_broadcast(input1, {tile_height, tile_width}, test_config.broadcast_dim, test_config.bcast_row_idx);

    std::vector<bfloat16> golden = gold_broadcast(
        input0,
        input1,
        {tile_height, tile_width},
        test_config.eltwise_op,
        test_config.broadcast_dim,
        test_config.bcast_row_idx,
        test_config.math_fidelity);

    auto packed_input0 = pack_vector<std::uint32_t, bfloat16>(input0);
    auto packed_input1 = pack_vector<std::uint32_t, bfloat16>(input1);
    auto packed_golden = pack_vector<std::uint32_t, bfloat16>(golden);
    ::unit_tests::compute::GoldenConfig config = {
        .num_tiles_r_dim = 1,
        .num_tiles_c_dim = 1,
        .num_faces = tile_width / 16 * tile_height / 16,
        .tiny_tile = test_config.tile_shape != TileShape::FULL_TILE};
    auto tilized_input0 = ::unit_tests::compute::gold_standard_tilize(packed_input0, config);
    auto tilized_input1 = ::unit_tests::compute::gold_standard_tilize(packed_input1, config);

    distributed::WriteShard(cq, src_a_dram_buffer, tilized_input0, zero_coord);
    distributed::WriteShard(cq, src_b_dram_buffer, tilized_input1, zero_coord);

    distributed::EnqueueMeshWorkload(cq, workload, is_quasar);
    distributed::Finish(cq);

    std::vector<std::uint32_t> dest_buffer_data;
    distributed::ReadShard(cq, dest_buffer_data, dst_dram_buffer, zero_coord);
    auto dest_buffer_data_untilized = ::unit_tests::compute::gold_standard_untilize(dest_buffer_data, config);

    bool result = is_close_packed_vectors<bfloat16, std::uint32_t>(
        dest_buffer_data_untilized, packed_golden, [&](const bfloat16& a, const bfloat16& b) {
            return is_close(a, b, k_broadcast_rtol);
        });
    ASSERT_TRUE(result);
}

// --------------------------------------------------------------------------
// SDPA blocked bcast-col SUB with SrcB reuse (sub_tiles_bcast_cols_custom)
// --------------------------------------------------------------------------
// Distinct data flow from the stock broadcast path above, which is why it gets its own config and
// runner rather than another BroadcastConfig axis: each row of a block is ct_dim column tiles that
// all reuse ONE srcB tile, the op writes ct_dim dest slots itself, and srcB is never re-read from
// L1. Calling the stock one-tile broadcast ct_dim times would instead need ct_dim copies of srcB.
//
// A block is an rt_dim x ct_dim tile grid that fills one acquired dest section, so srcA is a
// (num_blocks * rt_dim) x ct_dim grid of tiles and srcB is rt_dim tiles (row r of every block reuses srcB tile r).
// rt_dim > 1 is the case that mirrors sub_exp_block_bcast_cols(): nonzero srcA / srcB tile indices
// and a nonzero dest base, all within one tile_regs_acquire().

struct SubBcastColCustomConfig {
    std::uint32_t ct_dim = 1;
    std::uint32_t rt_dim = 1;
    std::uint32_t num_blocks = 1;
    TileShape tile_shape = TileShape::FULL_TILE;
    MathFidelity math_fidelity = MathFidelity::LoFi;  // SUB is LoFi-only on Quasar (fidelity phases are MUL-only)
};

void run_sub_bcast_col_custom(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const SubBcastColCustomConfig& test_config) {
    auto& cq = mesh_device->mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::MeshWorkload workload;

    const experimental::NodeCoord node{0, 0};

    tt_metal::Tile tile_dims = tile_shape_to_tile.at(test_config.tile_shape);
    const std::uint32_t tile_width = tile_dims.get_tile_shape()[1];
    const std::uint32_t tile_height = tile_dims.get_tile_shape()[0];
    const std::uint32_t single_tile_size = tile_width * tile_height * sizeof(bfloat16);
    const std::uint32_t tiles_per_block = test_config.rt_dim * test_config.ct_dim;
    const std::uint32_t total_tiles = tiles_per_block * test_config.num_blocks;
    // Tile rows of the srcA / output grid, across all blocks. Tile columns are ct_dim.
    const std::uint32_t total_tile_rows = test_config.rt_dim * test_config.num_blocks;

    log_info(
        tt::LogTest,
        "Testing sub_bcast_cols_custom ct_dim={} rt_dim={} num_blocks={} tile={{{}, {}}}",
        test_config.ct_dim,
        test_config.rt_dim,
        test_config.num_blocks,
        tile_height,
        tile_width);

    // srcA and the output are total_tile_rows x ct_dim grids; srcB is rt_dim tiles, each reused by
    // its row in every block.
    auto src_a_dram_buffer = CreateDramBufferForPageSize(mesh_device, single_tile_size, total_tiles);
    std::uint32_t dram_buffer_src_a_addr = src_a_dram_buffer->address();

    auto src_b_dram_buffer = CreateDramBufferForPageSize(mesh_device, single_tile_size, test_config.rt_dim);
    std::uint32_t dram_buffer_src_b_addr = src_b_dram_buffer->address();

    auto dst_dram_buffer = CreateDramBufferForPageSize(mesh_device, single_tile_size, total_tiles);
    std::uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

    auto* device = mesh_device->get_devices().empty() ? nullptr : mesh_device->get_devices().front();
    TT_FATAL(device != nullptr, "mesh_device has no backing devices");
    const bool is_quasar = device->arch() == ARCH::QUASAR;

    experimental::KernelSpec::CompilerOptions::Defines defines_vec;
    defines_vec.emplace("CT_DIM", std::to_string(test_config.ct_dim));
    defines_vec.emplace("RT_DIM", std::to_string(test_config.rt_dim));
    defines_vec.emplace("NUM_BLOCKS", std::to_string(test_config.num_blocks));

    const experimental::DFBSpecName INP0_DFB{"inp0_dfb"};
    const experimental::DFBSpecName INP1_DFB{"inp1_dfb"};
    const experimental::DFBSpecName OUT_DFB{"out_dfb"};
    const experimental::KernelSpecName READER{"reader"};
    const experimental::KernelSpecName WRITER{"writer"};
    const experimental::KernelSpecName COMPUTE{"compute"};

    auto make_dfb = [&](const experimental::DFBSpecName& name, std::uint32_t num_entries) {
        return experimental::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = single_tile_size,
            .num_entries = num_entries,
            .data_format_metadata = tt::DataFormat::Float16_b,
            .tile_format_metadata = tile_dims,
        };
    };

    // in0/out hold one whole block: the compute kernel indexes every row of the block off the read
    // pointer inside a single dest acquire, so all rt_dim * ct_dim tiles must be resident. in1 holds
    // the rt_dim reused tiles.
    experimental::DataflowBufferSpec inp0_dfb_spec = make_dfb(INP0_DFB, tiles_per_block);
    experimental::DataflowBufferSpec inp1_dfb_spec = make_dfb(INP1_DFB, test_config.rt_dim);
    experimental::DataflowBufferSpec out_dfb_spec = make_dfb(OUT_DFB, tiles_per_block);

    experimental::DataMovementHardwareConfig reader_hw_config = make_reader_hw_config(mesh_device);
    experimental::KernelSpec reader_spec{
        .unique_id = READER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary_bcast_col_reuse.cpp",
        .num_threads = 1,
        .dfb_bindings =
            {{
                 .dfb_spec_name = INP0_DFB,
                 .accessor_name = "in0",
                 .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = INP1_DFB,
                 .accessor_name = "in1",
                 .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             }},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"src0_addr", "src0_bank_id", "src1_addr", "src1_bank_id", "num_tiles", "num_bcast_tiles"}},
        .hw_config = reader_hw_config,
    };

    experimental::DataMovementHardwareConfig writer_hw_config = make_writer_hw_config(mesh_device);
    experimental::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ConsumerOf(OUT_DFB, "in")},
        .runtime_arg_schema = {.runtime_arg_names = {"dst_addr", "bank_id", "num_tiles"}},
        .hw_config = writer_hw_config,
    };

    experimental::ComputeHardwareConfig compute_hw_config =
        make_compute_hw_config(mesh_device, test_config.math_fidelity);

    experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = "tests/tt_metal/tt_metal/test_kernels/compute/sub_bcast_col_custom.cpp",
        .num_threads = 1,
        .compiler_options = {.defines = defines_vec},
        .dfb_bindings =
            {{
                 .dfb_spec_name = INP0_DFB,
                 .accessor_name = "in0",
                 .endpoint_type = experimental::DFBEndpointType::CONSUMER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = INP1_DFB,
                 .accessor_name = "in1",
                 .endpoint_type = experimental::DFBEndpointType::CONSUMER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = OUT_DFB,
                 .accessor_name = "out",
                 .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             }},
        .hw_config = compute_hw_config,
    };

    experimental::WorkUnitSpec wu{
        .name = "main",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = node,
    };

    experimental::ProgramSpec spec{
        .name = "single_core_sub_bcast_col_custom",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {inp0_dfb_spec, inp1_dfb_spec, out_dfb_spec},
        .work_units = {wu},
    };

    Program built_program = experimental::MakeProgramFromSpec(*mesh_device, spec);
    workload.add_program(device_range, std::move(built_program));
    auto& program_run = workload.get_programs().at(device_range);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node,
                {{"src0_addr", static_cast<std::uint32_t>(dram_buffer_src_a_addr)},
                 {"src0_bank_id", 0u},
                 {"src1_addr", static_cast<std::uint32_t>(dram_buffer_src_b_addr)},
                 {"src1_bank_id", 0u},
                 // srcA tile count, then the srcB (bcast) tile count: one per row of a block.
                 {"num_tiles", static_cast<std::uint32_t>(total_tiles)},
                 {"num_bcast_tiles", static_cast<std::uint32_t>(test_config.rt_dim)}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node,
                {{"dst_addr", static_cast<std::uint32_t>(dram_buffer_dst_addr)},
                 {"bank_id", 0u},
                 {"num_tiles", static_cast<std::uint32_t>(total_tiles)}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE},
    };
    experimental::SetProgramRunArgs(program_run, params);

    const std::uint32_t tile_elems = tile_height * tile_width;
    // Row-major host geometry of the srcA / output tile grid, and of the srcB tile column.
    const std::uint32_t grid_width = tile_width * test_config.ct_dim;
    const std::uint32_t grid_height = tile_height * total_tile_rows;
    const std::uint32_t bcast_height = tile_height * test_config.rt_dim;

    std::vector<bfloat16> input0 = generate_uniform_random_vector<bfloat16>(
        -1.0f, 1.0f, tile_elems * total_tiles, std::chrono::system_clock::now().time_since_epoch().count());

    std::vector<bfloat16> input1 = generate_uniform_random_vector<bfloat16>(
        -1.0f, 1.0f, tile_elems * test_config.rt_dim, std::chrono::system_clock::now().time_since_epoch().count());

    // No mask_src_b_for_broadcast call: it is a no-op for COL (see its body): The FPU reads
    // column 0 of each srcB row and ignores the rest, which is what gold_broadcast models.
    // Tile srcB unmasked so host and device see identical bytes.

    // Replicate srcB across the grid so gold_broadcast's COL case (src_b[i * num_cols]) resolves to
    // column 0 of the right srcB row: the same tile subtracted from every srcA column tile of its
    // row, and (r % bcast_height) makes every block reuse the same rt_dim srcB tiles.
    std::vector<bfloat16> input1_grid(tile_elems * total_tiles);
    for (std::uint32_t r = 0; r < grid_height; r++) {
        for (std::uint32_t c = 0; c < grid_width; c++) {
            input1_grid[(r * grid_width) + c] = input1[((r % bcast_height) * tile_width) + (c % tile_width)];
        }
    }

    std::vector<bfloat16> golden = gold_broadcast(
        input0,
        input1_grid,
        {grid_height, grid_width},
        EltwiseOp::SUB,
        BroadcastDim::COL,
        0,
        test_config.math_fidelity);

    auto packed_input0 = pack_vector<std::uint32_t, bfloat16>(input0);
    auto packed_input1 = pack_vector<std::uint32_t, bfloat16>(input1);
    auto packed_golden = pack_vector<std::uint32_t, bfloat16>(golden);

    const int num_faces = static_cast<int>(tile_width / 16 * tile_height / 16);
    const bool tiny_tile = test_config.tile_shape != TileShape::FULL_TILE;
    // gold_standard_tilize/untilize walk a tile grid row-major, which is exactly the DRAM page order
    // the reader streams into in0: block b owns tile rows [b * rt_dim, (b + 1) * rt_dim).
    ::unit_tests::compute::GoldenConfig grid_config = {
        .num_tiles_r_dim = static_cast<int>(total_tile_rows),
        .num_tiles_c_dim = static_cast<int>(test_config.ct_dim),
        .num_faces = num_faces,
        .tiny_tile = tiny_tile};
    // srcB is a single column of rt_dim tiles.
    ::unit_tests::compute::GoldenConfig bcast_config = {
        .num_tiles_r_dim = static_cast<int>(test_config.rt_dim),
        .num_tiles_c_dim = 1,
        .num_faces = num_faces,
        .tiny_tile = tiny_tile};

    auto tilized_input0 = ::unit_tests::compute::gold_standard_tilize(packed_input0, grid_config);
    auto tilized_input1 = ::unit_tests::compute::gold_standard_tilize(packed_input1, bcast_config);

    distributed::WriteShard(cq, src_a_dram_buffer, tilized_input0, zero_coord);
    distributed::WriteShard(cq, src_b_dram_buffer, tilized_input1, zero_coord);

    distributed::EnqueueMeshWorkload(cq, workload, is_quasar);
    distributed::Finish(cq);

    std::vector<std::uint32_t> dest_buffer_data;
    distributed::ReadShard(cq, dest_buffer_data, dst_dram_buffer, zero_coord);
    auto dest_buffer_data_untilized = ::unit_tests::compute::gold_standard_untilize(dest_buffer_data, grid_config);

    bool result = is_close_packed_vectors<bfloat16, std::uint32_t>(
        dest_buffer_data_untilized, packed_golden, [&](const bfloat16& a, const bfloat16& b) {
            return is_close(a, b, k_broadcast_rtol);
        });
    ASSERT_TRUE(result);
}
}  // namespace unit_tests::compute::broadcast

class BroadcastParameterizedDeviceFixture
    : public LLKMeshDeviceFixture,
      public testing::WithParamInterface<unit_tests::compute::broadcast::BroadcastConfig> {};

TEST_P(BroadcastParameterizedDeviceFixture, TensixComputeSingleTileBroadcast) {
    if (this->arch_ == tt::ARCH::QUASAR) {
        GTEST_SKIP() << "Quasar uses TensixComputeBinaryBroadcastQuasarDfb";
    }
    unit_tests::compute::broadcast::BroadcastConfig test_config = GetParam();
    test_config.math_fidelity = MathFidelity::HiFi2;
    unit_tests::compute::broadcast::run_single_core_broadcast(this->devices_.at(0), test_config);
}

using namespace unit_tests::compute::broadcast;

INSTANTIATE_TEST_SUITE_P(
    ComputeSingleTileBroadcast,
    BroadcastParameterizedDeviceFixture,
    ::testing::Values(
        (BroadcastConfig){ApiConvention::DEFAULT, EltwiseOp::ADD, BroadcastDim::ROW},
        (BroadcastConfig){ApiConvention::DEFAULT, EltwiseOp::ADD, BroadcastDim::COL},
        (BroadcastConfig){ApiConvention::DEFAULT, EltwiseOp::ADD, BroadcastDim::SCALAR},
        (BroadcastConfig){ApiConvention::DEFAULT, EltwiseOp::SUB, BroadcastDim::ROW},
        (BroadcastConfig){ApiConvention::DEFAULT, EltwiseOp::SUB, BroadcastDim::COL},
        (BroadcastConfig){ApiConvention::DEFAULT, EltwiseOp::SUB, BroadcastDim::SCALAR},
        (BroadcastConfig){ApiConvention::DEFAULT, EltwiseOp::MUL, BroadcastDim::ROW},
        (BroadcastConfig){ApiConvention::DEFAULT, EltwiseOp::MUL, BroadcastDim::COL},
        (BroadcastConfig){ApiConvention::DEFAULT, EltwiseOp::MUL, BroadcastDim::SCALAR},
        (BroadcastConfig){ApiConvention::SHORT_INIT, EltwiseOp::ADD, BroadcastDim::ROW},
        (BroadcastConfig){ApiConvention::SHORT_INIT, EltwiseOp::ADD, BroadcastDim::COL},
        (BroadcastConfig){ApiConvention::SHORT_INIT, EltwiseOp::ADD, BroadcastDim::SCALAR},
        (BroadcastConfig){ApiConvention::SHORT_INIT, EltwiseOp::SUB, BroadcastDim::ROW},
        (BroadcastConfig){ApiConvention::SHORT_INIT, EltwiseOp::SUB, BroadcastDim::COL},
        (BroadcastConfig){ApiConvention::SHORT_INIT, EltwiseOp::SUB, BroadcastDim::SCALAR},
        (BroadcastConfig){ApiConvention::SHORT_INIT, EltwiseOp::MUL, BroadcastDim::ROW},
        (BroadcastConfig){ApiConvention::SHORT_INIT, EltwiseOp::MUL, BroadcastDim::COL},
        (BroadcastConfig){ApiConvention::SHORT_INIT, EltwiseOp::MUL, BroadcastDim::SCALAR},
        (BroadcastConfig){ApiConvention::SHORT_CALL, EltwiseOp::ADD, BroadcastDim::ROW},
        (BroadcastConfig){ApiConvention::SHORT_CALL, EltwiseOp::ADD, BroadcastDim::COL},
        (BroadcastConfig){ApiConvention::SHORT_CALL, EltwiseOp::ADD, BroadcastDim::SCALAR},
        (BroadcastConfig){ApiConvention::SHORT_CALL, EltwiseOp::SUB, BroadcastDim::ROW},
        (BroadcastConfig){ApiConvention::SHORT_CALL, EltwiseOp::SUB, BroadcastDim::COL},
        (BroadcastConfig){ApiConvention::SHORT_CALL, EltwiseOp::SUB, BroadcastDim::SCALAR},
        (BroadcastConfig){ApiConvention::SHORT_CALL, EltwiseOp::MUL, BroadcastDim::ROW},
        (BroadcastConfig){ApiConvention::SHORT_CALL, EltwiseOp::MUL, BroadcastDim::COL},
        (BroadcastConfig){ApiConvention::SHORT_CALL, EltwiseOp::MUL, BroadcastDim::SCALAR},
        (BroadcastConfig){ApiConvention::SHORT_BOTH, EltwiseOp::ADD, BroadcastDim::ROW},
        (BroadcastConfig){ApiConvention::SHORT_BOTH, EltwiseOp::ADD, BroadcastDim::COL},
        (BroadcastConfig){ApiConvention::SHORT_BOTH, EltwiseOp::ADD, BroadcastDim::SCALAR},
        (BroadcastConfig){ApiConvention::SHORT_BOTH, EltwiseOp::SUB, BroadcastDim::ROW},
        (BroadcastConfig){ApiConvention::SHORT_BOTH, EltwiseOp::SUB, BroadcastDim::COL},
        (BroadcastConfig){ApiConvention::SHORT_BOTH, EltwiseOp::SUB, BroadcastDim::SCALAR},
        (BroadcastConfig){ApiConvention::SHORT_BOTH, EltwiseOp::MUL, BroadcastDim::ROW},
        (BroadcastConfig){ApiConvention::SHORT_BOTH, EltwiseOp::MUL, BroadcastDim::COL},
        (BroadcastConfig){ApiConvention::SHORT_BOTH, EltwiseOp::MUL, BroadcastDim::SCALAR},
        (BroadcastConfig){ApiConvention::DEFAULT, EltwiseOp::ADD, BroadcastDim::COL, TileShape::TINY_TILE_16x32},
        (BroadcastConfig){ApiConvention::DEFAULT, EltwiseOp::SUB, BroadcastDim::COL, TileShape::TINY_TILE_16x32},
        (BroadcastConfig){ApiConvention::DEFAULT, EltwiseOp::MUL, BroadcastDim::COL, TileShape::TINY_TILE_16x32},
        (BroadcastConfig){ApiConvention::SHORT_INIT, EltwiseOp::ADD, BroadcastDim::COL, TileShape::TINY_TILE_16x32},
        (BroadcastConfig){ApiConvention::SHORT_INIT, EltwiseOp::SUB, BroadcastDim::COL, TileShape::TINY_TILE_16x32},
        (BroadcastConfig){ApiConvention::SHORT_INIT, EltwiseOp::MUL, BroadcastDim::COL, TileShape::TINY_TILE_16x32},
        (BroadcastConfig){ApiConvention::SHORT_BOTH, EltwiseOp::ADD, BroadcastDim::COL, TileShape::TINY_TILE_16x32},
        (BroadcastConfig){ApiConvention::SHORT_BOTH, EltwiseOp::SUB, BroadcastDim::COL, TileShape::TINY_TILE_16x32},
        (BroadcastConfig){ApiConvention::SHORT_BOTH, EltwiseOp::MUL, BroadcastDim::COL, TileShape::TINY_TILE_16x32},
        (BroadcastConfig){ApiConvention::DEFAULT,
                          EltwiseOp::ADD,
                          BroadcastDim::ROW,
                          TileShape::FULL_TILE,
                          MathFidelity::HiFi4,
                          15},  // Row 15 (middle)
        (BroadcastConfig){ApiConvention::DEFAULT,
                          EltwiseOp::SUB,
                          BroadcastDim::ROW,
                          TileShape::FULL_TILE,
                          MathFidelity::HiFi4,
                          15},  // Row 15 (middle)
        (BroadcastConfig){ApiConvention::DEFAULT,
                          EltwiseOp::ADD,
                          BroadcastDim::ROW,
                          TileShape::FULL_TILE,
                          MathFidelity::HiFi4,
                          31},  // Row 31 (last)
        (BroadcastConfig){ApiConvention::SHORT_CALL,
                          EltwiseOp::MUL,
                          BroadcastDim::ROW,
                          TileShape::FULL_TILE,
                          MathFidelity::HiFi4,
                          31},  // Row 31 with MUL
        (BroadcastConfig){ApiConvention::SHORT_BOTH,
                          EltwiseOp::ADD,
                          BroadcastDim::ROW,
                          TileShape::FULL_TILE,
                          MathFidelity::HiFi4,
                          20}));  // Row 20

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixComputeBinaryBroadcastQuasarDfb) {
    for (std::uint8_t op = std::uint8_t(EltwiseOp::ADD); op <= std::uint8_t(EltwiseOp::MUL); op++) {
        for (std::uint8_t dim = std::uint8_t(BroadcastDim::ROW); dim <= std::uint8_t(BroadcastDim::SCALAR); dim++) {
            for (std::uint8_t math_fid = std::uint8_t(MathFidelity::LoFi);
                 math_fid <= std::uint8_t(MathFidelity::HiFi4);
                 math_fid++) {
                // MathFidelity : {0, 2, 3, 4};
                if (math_fid == 1) {
                    continue;
                }
                if (!(EltwiseOp(op) == EltwiseOp::ADD && BroadcastDim(dim) == BroadcastDim::ROW &&
                      MathFidelity(math_fid) == MathFidelity::LoFi)) {
                    // TODO (#38092): Remove when we can run back to back tests on Quasar
                    continue;
                }
                unit_tests::compute::broadcast::BroadcastConfig cfg = {
                    .api_convention = ApiConvention::DEFAULT,
                    .eltwise_op = EltwiseOp(op),
                    .broadcast_dim = BroadcastDim(dim),
                    .tile_shape = TileShape::FULL_TILE,
                    .math_fidelity = MathFidelity(math_fid),
                    .bcast_row_idx = 0,
                };
                log_info(
                    tt::LogTest,
                    "Quasar binary broadcast DFB op={} dim={} math_fid={}",
                    eltwise_op_to_type.at(EltwiseOp(op)),
                    broadcast_dim_to_type.at(BroadcastDim(dim)),
                    math_fid);
                unit_tests::compute::broadcast::run_single_core_broadcast(this->devices_.at(0), cfg);
            }
        }
    }
}

// Blocked bcast-col SUB with SrcB reuse. All cases run back-to-back in one TEST_F.
// Cases are ordered so the first red one localizes the bug:
//   - ct_dim=1 carries no reuse at all, so a failure there means the srcB face traversal (the
//     +8/-8/+24 addr-mod walk) is wrong.
//   - ct_dim>1 adds the reuse itself: a failure there means the srcB hold is wrong i.e. the
//     per-tile SETRWC(CLR_A) is releasing srcB, or the L1 srcB tile pointer is advancing.
//   - num_blocks>1 additionally exercises dest-section switching and the per-block srcB re-unpack,
//     whose 1:1 pairing with the math thread's single CLR_B is what keeps blocks from deadlocking.
//   - rt_dim>1 is the only shape that exercises nonzero srcA/srcB tile indices and a nonzero dest
//     base, so a failure there (with the rt_dim=1 cases green) points at the API wrappers' index
//     arithmetic rather than the face walk.
//   - the tiny-tile (16x32) cases come last and mostly repeat a full-tile shape, so a red tiny-tile
//     case with its full-tile twin green isolates the shape-driven parts (face-row count, dest slot
//     stride, per-face unpack, buffer-descriptor z_dim).
TEST_F(QuasarMeshDeviceSingleCardFixture, ComputeSubBcastColCustom) {
    using unit_tests::compute::broadcast::SubBcastColCustomConfig;

    const std::vector<SubBcastColCustomConfig> cases = {
        // Full 32x32 tiles, one row per block: every call uses tile_index_a/b = 0 and dst_index = 0.
        // ct_dim=8 fills half-dest, 7 covers a non-power-of-two block width.
        {.ct_dim = 1, .num_blocks = 1},
        {.ct_dim = 4, .num_blocks = 1},
        {.ct_dim = 7, .num_blocks = 1},
        {.ct_dim = 8, .num_blocks = 2},
        // One multi-row block, mirroring sub_exp_block_bcast_cols(): row r reads srcA tile r * ct_dim
        // and srcB tile r and writes dest slots [r * ct_dim, (r + 1) * ct_dim), all inside one
        // acquire. rt_dim * ct_dim is capped by the same half-dest budget as ct_dim above.
        {.ct_dim = 4, .rt_dim = 2, .num_blocks = 1},
        // Tiny 16x32 tiles: one face-row, so the srcB walk runs its four-op face-row block once and
        // dest slots are 32 rows apart instead of 64. The full-tile shapes above are mirrored one for
        // one, because the tile shape changes the per-tile unpack (one UNPACR per face on Quasar), the
        // dest slot stride and the pack granularity all at once. The one shape with no full-tile twin
        // is ct_dim=16: a 32-row slot means twice as many tiles fit in a dest section, so this is the
        // only case that reaches the tiny-tile dest budget (and the only one where dst_index + ct_dim
        // lands exactly on the shape-derived max_dest_tiles bound).
        {.ct_dim = 1, .num_blocks = 1, .tile_shape = TileShape::TINY_TILE_16x32},
        {.ct_dim = 4, .num_blocks = 1, .tile_shape = TileShape::TINY_TILE_16x32},
        {.ct_dim = 7, .num_blocks = 1, .tile_shape = TileShape::TINY_TILE_16x32},
        {.ct_dim = 16, .num_blocks = 2, .tile_shape = TileShape::TINY_TILE_16x32},
        {.ct_dim = 4, .rt_dim = 2, .num_blocks = 1, .tile_shape = TileShape::TINY_TILE_16x32},
    };

    for (const auto& cfg : cases) {
        SCOPED_TRACE(
            "ct_dim=" + std::to_string(cfg.ct_dim) + " rt_dim=" + std::to_string(cfg.rt_dim) + " num_blocks=" +
            std::to_string(cfg.num_blocks) + " tiny_tile=" + std::to_string(cfg.tile_shape != TileShape::FULL_TILE));
        unit_tests::compute::broadcast::run_sub_bcast_col_custom(this->devices_.at(0), cfg);
        if (HasFatalFailure()) {
            // Later cases are not diagnostic once an earlier one is red.
            break;
        }
    }
}

}  // namespace tt::tt_metal
