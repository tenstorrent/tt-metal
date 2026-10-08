// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// C = A + B on one Quasar Neo cluster, written against the Metal 2.0 host API. The same shape as the
// Blackhole programming example eltwise_binary: one reader thread, one compute thread and one writer
// thread on a single node. A, B and C are bfloat16 TILE_LAYOUT tensors, DRAM interleaved, so DRAM page i
// is tile i. The DFBs use explicit reserve_back/push_back sync, so implicit sync is switched off.
// Fast dispatch only: tensors go through the mesh command queue.

#include "common/device_fixture.hpp"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>
#include <tt-metalium/tensor/tensor_apis.hpp>

using namespace tt;
using namespace tt::tt_metal;

namespace {

const experimental::DFBSpecName IN0_DFB{"in0_dfb"};
const experimental::DFBSpecName IN1_DFB{"in1_dfb"};
const experimental::DFBSpecName OUT_DFB{"out_dfb"};
const experimental::TensorParamName A_T{"a"};
const experimental::TensorParamName B_T{"b"};
const experimental::TensorParamName C_T{"c"};
const experimental::KernelSpecName READER{"reader"};
const experimental::KernelSpecName WRITER{"writer"};
const experimental::KernelSpecName COMPUTE{"compute"};

constexpr std::uint32_t kTileDim = 32;
constexpr std::uint32_t kTileBytes = kTileDim * kTileDim * sizeof(bfloat16);  // 2048
constexpr std::uint32_t kEntriesPerDfb = 2;  // double buffering, as the Blackhole example's CBs

// [1, 1, H, W] bfloat16 in TILE_LAYOUT, DRAM interleaved: one 2048-byte page per 32x32 tile, tiles in
// row-major tile order, so page i is tile i and every kernel walks pages 0 .. num_tiles-1.
TensorSpec make_tile_tensor_spec(std::uint32_t height, std::uint32_t width) {
    const auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    const auto tensor_layout = TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), memory_config);
    return TensorSpec(Shape{1, 1, height, width}, tensor_layout);
}

std::vector<bfloat16> random_bfloat16(size_t n, float lo, float hi, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<bfloat16> v(n);
    for (auto& x : v) {
        x = bfloat16(dist(rng));
    }
    return v;
}

experimental::ProgramSpec build_add_program_spec(
    std::uint32_t num_tiles,
    const MeshTensor& a,
    const MeshTensor& b,
    const MeshTensor& c,
    const experimental::NodeCoord& node) {
    auto make_dfb = [](const experimental::DFBSpecName& name) {
        return experimental::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = kTileBytes,
            .num_entries = kEntriesPerDfb,
            .data_format_metadata = tt::DataFormat::Float16_b,
        };
    };
    const experimental::DataMovementHardwareConfig dm_config{
        .config_2xx =
            experimental::DataMovementHardwareConfig::DataMovement2XXConfig{
                .disable_dfb_implicit_sync_for_all = true,
            },
    };

    experimental::KernelSpec reader_spec{
        .unique_id = READER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/eltwise_add_reader_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ProducerOf(IN0_DFB, "in0"), experimental::ProducerOf(IN1_DFB, "in1")},
        .tensor_bindings =
            {{.tensor_parameter_name = A_T, .accessor_name = "a"},
             {.tensor_parameter_name = B_T, .accessor_name = "b"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles"}},
        .hw_config = dm_config,
    };

    experimental::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/eltwise_add_writer_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ConsumerOf(OUT_DFB, "out")},
        .tensor_bindings = {{.tensor_parameter_name = C_T, .accessor_name = "c"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles"}},
        .hw_config = dm_config,
    };

    experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_add_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings =
            {experimental::ConsumerOf(IN0_DFB, "in0"),
             experimental::ConsumerOf(IN1_DFB, "in1"),
             experimental::ProducerOf(OUT_DFB, "out")},
        .compile_time_args = {{"num_tiles", num_tiles}},
        .hw_config = experimental::ComputeHardwareConfig{},
    };

    return experimental::ProgramSpec{
        .name = "eltwise_add_quasar",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {make_dfb(IN0_DFB), make_dfb(IN1_DFB), make_dfb(OUT_DFB)},
        .tensor_parameters =
            {{.unique_id = A_T, .spec = a.tensor_spec()},
             {.unique_id = B_T, .spec = b.tensor_spec()},
             {.unique_id = C_T, .spec = c.tensor_spec()}},
        .work_units = {{.name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = node}},
    };
}

// Element-wise A + B in float, rounded to bfloat16, as the FPU's bfloat16 result.
std::vector<bfloat16> golden_add(const std::vector<bfloat16>& a, const std::vector<bfloat16>& b) {
    std::vector<bfloat16> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        out[i] = bfloat16(static_cast<float>(a[i]) + static_cast<float>(b[i]));
    }
    return out;
}

void run_eltwise_add(distributed::MeshDevice& mesh_device, std::uint32_t height, std::uint32_t width) {
    const experimental::NodeCoord node{0, 0};
    const std::uint32_t num_tiles = (height / kTileDim) * (width / kTileDim);

    const TensorSpec tile_spec = make_tile_tensor_spec(height, width);
    MeshTensor a = MeshTensor::allocate_on_device(mesh_device, tile_spec);
    MeshTensor b = MeshTensor::allocate_on_device(mesh_device, tile_spec);
    MeshTensor c = MeshTensor::allocate_on_device(mesh_device, tile_spec);

    auto workload =
        experimental::MakeMeshWorkloadFromSpec(mesh_device, build_add_program_spec(num_tiles, a, b, c, node));

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(node, {{"num_tiles", num_tiles}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(node, {{"num_tiles", num_tiles}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE},
    };
    params.tensor_args = {
        {A_T, experimental::TensorArgument{a}},
        {B_T, experimental::TensorArgument{b}},
        {C_T, experimental::TensorArgument{c}},
    };
    experimental::SetProgramRunArgs(workload.get_programs().begin()->second, params);

    // Host data stays row-major.
    const size_t num_elems = static_cast<size_t>(height) * width;
    const std::vector<bfloat16> a_vec = random_bfloat16(num_elems, 0.0f, 1.0f, 0xA11);
    const std::vector<bfloat16> b_vec = random_bfloat16(num_elems, -0.5f, 0.5f, 0xB22);

    // from_vector tilizes the row-major host data; to_vector untilizes C back to row-major.
    auto& cq = mesh_device.mesh_command_queue();
    cq.enqueue_write_tensor(HostTensor::from_vector(a_vec, tile_spec), a);
    cq.enqueue_write_tensor(HostTensor::from_vector(b_vec, tile_spec), b);
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);
    const std::vector<bfloat16> c_vec = cq.enqueue_read_tensor(c).to_vector<bfloat16>();

    // The FPU adds in bfloat16, so allow about one bfloat16 ulp of rounding difference.
    ASSERT_EQ(c_vec.size(), num_elems);
    const std::vector<bfloat16> golden = golden_add(a_vec, b_vec);
    size_t mismatches = 0;
    for (size_t i = 0; i < num_elems; ++i) {
        const float got = static_cast<float>(c_vec[i]);
        const float want = static_cast<float>(golden[i]);
        if (std::fabs(got - want) > 0.02f + 0.01f * std::fabs(want)) {
            if (mismatches++ < 10) {
                ADD_FAILURE() << "C[" << i / width << ", " << i % width << "] = " << got << ", expected " << want;
            }
        }
    }
    EXPECT_EQ(mismatches, 0u) << "of " << num_elems << " elements";
}

}  // namespace

TEST_F(QuasarMeshDeviceSingleCardFixture, EltwiseAddSingleNeo) {
    if (this->IsSlowDispatch()) {
        GTEST_SKIP() << "Fast dispatch only";
    }
    // 256 x 256 bfloat16 = 8 x 8 = 64 tiles.
    run_eltwise_add(*devices_[0], /*height=*/256, /*width=*/256);
}
