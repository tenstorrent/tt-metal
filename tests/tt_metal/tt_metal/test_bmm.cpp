// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>
#include "test_gold_impls.hpp"
#include "impl/data_format/bfloat16_utils.hpp"

using std::vector;
using namespace tt;
using namespace tt::tt_metal;

namespace {

const experimental::DFBSpecName SRC0_DFB{"src0_dfb"};
const experimental::DFBSpecName SRC1_DFB{"src1_dfb"};
const experimental::DFBSpecName DST_DFB{"dst_dfb"};
const experimental::TensorParamName SRC0_T{"src0"};
const experimental::TensorParamName SRC1_T{"src1"};
const experimental::TensorParamName DST_T{"dst"};
const experimental::KernelSpecName READER{"reader"};
const experimental::KernelSpecName WRITER{"writer"};
const experimental::KernelSpecName COMPUTE{"compute"};

struct BmmParams {
    uint32_t Mt, Kt, Nt;
    uint32_t B_total;       // total batch count (buffer sizing + validation)
    uint32_t B_per_core;    // batch count per core (kernel runtime args)
    uint32_t num_threads = 2;
    uint32_t num_input_tiles = 4;
    uint32_t num_output_tiles = 4;
    uint32_t single_tile_size = 2 * 1024;
};

struct BmmTensors {
    MeshTensor src0;
    MeshTensor src1;
    MeshTensor dst;
};

// Flat 2D UINT32 page layout: one DRAM page per tile, tile_size bytes each. The element type
// is UINT32 only so DRAM exposes raw tile-paged storage at the buffer level; the kernels still
// operate on bfloat16 tiles via TensorAccessor / matmul LLKs.
TensorSpec make_flat_dram_tensor_spec(uint32_t tile_size, uint32_t num_tiles) {
    const uint32_t tile_size_words = tile_size / sizeof(uint32_t);
    auto page_config = PageConfig(Layout::ROW_MAJOR);
    auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    auto tensor_layout = TensorLayout(DataType::UINT32, page_config, memory_config);
    return TensorSpec(Shape{num_tiles, tile_size_words}, tensor_layout);
}

BmmTensors create_bmm_tensors(distributed::MeshDevice& mesh_device, const BmmParams& p) {
    const uint32_t num_tiles_A = p.Mt * p.Kt * p.B_total;
    const uint32_t num_tiles_B = p.Kt * p.Nt * p.B_total;
    const uint32_t num_tiles_C = p.Mt * p.Nt * p.B_total;
    return {
        MeshTensor::allocate_on_device(
            mesh_device, make_flat_dram_tensor_spec(p.single_tile_size, num_tiles_A), TensorTopology{}),
        MeshTensor::allocate_on_device(
            mesh_device, make_flat_dram_tensor_spec(p.single_tile_size, num_tiles_B), TensorTopology{}),
        MeshTensor::allocate_on_device(
            mesh_device, make_flat_dram_tensor_spec(p.single_tile_size, num_tiles_C), TensorTopology{}),
    };
}

template <typename TargetNodes>
experimental::ProgramSpec build_bmm_program_spec(
    const BmmParams& p, const BmmTensors& tensors, const TargetNodes& target_nodes, bool use_implicit_sync) {
    // Both kernels expose Gen1 + Gen2 data movement configs so the same spec runs on WH/BH and
    // Quasar; the runtime picks the right one per arch. The kernel sources use the unified
    // DataflowBuffer device API on both arches (CB-backed on Gen1, DFB-backed on Gen2).
    // On Quasar we also enable implicit sync on each DFB so the reader/writer kernels can drop
    // explicit reserve_back/wait_front/push_back/pop_front; on WH/BH implicit sync is unsupported
    // and must be disabled to match the explicit-sync kernel branch.
    const bool is_quasar = tensors.src0.device().arch() == ARCH::QUASAR;
    experimental::DataMovementHardwareConfig reader_config;
    experimental::DataMovementHardwareConfig writer_config;
    experimental::ComputeHardwareConfig compute_config;
    if (is_quasar) {
        reader_config = experimental::DataMovementGen2Config{.disable_dfb_implicit_sync_for_all = !use_implicit_sync};
        writer_config = experimental::DataMovementGen2Config{.disable_dfb_implicit_sync_for_all = !use_implicit_sync};
        compute_config = experimental::ComputeGen2Config{};
    } else {
        reader_config = experimental::DataMovementGen1Config{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default};
        writer_config = experimental::DataMovementGen1Config{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default};
        compute_config = experimental::ComputeGen1Config{};
    }

    experimental::DataflowBufferSpec src0_dfb_spec{
        .unique_id = SRC0_DFB,
        .entry_size = p.single_tile_size,
        .num_entries = p.num_input_tiles,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };
    experimental::DataflowBufferSpec src1_dfb_spec{
        .unique_id = SRC1_DFB,
        .entry_size = p.single_tile_size,
        .num_entries = p.num_input_tiles,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };
    experimental::DataflowBufferSpec dst_dfb_spec{
        .unique_id = DST_DFB,
        .entry_size = p.single_tile_size,
        .num_entries = p.num_output_tiles,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    experimental::KernelSpec reader_spec{
        .unique_id = READER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_bmm_8bank.cpp",
        .num_threads = p.num_threads,
        .dfb_bindings = {experimental::ProducerOf(SRC0_DFB, "src0"), experimental::ProducerOf(SRC1_DFB, "src1")},
        .tensor_bindings =
            {{.tensor_parameter_name = SRC0_T, .accessor_name = "src0"},
             {.tensor_parameter_name = SRC1_T, .accessor_name = "src1"}},
        // Only batch_start varies per node; everything else is identical across nodes and
        // lives in CRTAs for better dispatch efficiency.
        .runtime_arg_schema =
            {.runtime_arg_names = {"batch_start"},
             .common_runtime_arg_names = {"Mt", "Kt", "Nt", "MtKt", "KtNt", "batch", "do_bcast"}},
        .hw_config = reader_config,
    };

    experimental::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_bmm_8bank.cpp",
        .num_threads = p.num_threads,
        .dfb_bindings = {experimental::ConsumerOf(DST_DFB, "dst")},
        .tensor_bindings = {{.tensor_parameter_name = DST_T, .accessor_name = "dst"}},
        .runtime_arg_schema = {.runtime_arg_names = {"batch_start"}, .common_runtime_arg_names = {"Mt", "Nt", "batch"}},
        .hw_config = writer_config,
    };

    experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = "tests/tt_metal/tt_metal/test_kernels/compute/bmm.cpp",
        .num_threads = p.num_threads,
        .dfb_bindings =
            {experimental::ConsumerOf(SRC0_DFB, "src0"),
             experimental::AllConsumerOf(SRC1_DFB, "src1"),
             experimental::ProducerOf(DST_DFB, "dst")},
        .compile_time_args = {{"batch", p.B_per_core}, {"Mt", p.Mt}, {"Kt", p.Kt}, {"Nt", p.Nt}},
        .hw_config = compute_config,
    };

    return experimental::ProgramSpec{
        .name = "bmm",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {src0_dfb_spec, src1_dfb_spec, dst_dfb_spec},
        .tensor_parameters =
            {{.unique_id = SRC0_T, .spec = tensors.src0.tensor_spec()},
             {.unique_id = SRC1_T, .spec = tensors.src1.tensor_spec()},
             {.unique_id = DST_T, .spec = tensors.dst.tensor_spec()}},
        .work_units = {{.name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = target_nodes}},
    };
}

bool validate_bmm_result(
    const BmmParams& p,
    const std::vector<uint32_t>& src0_vec,
    const std::vector<uint32_t>& src1_vec,
    const std::vector<uint32_t>& result_vec,
    int* argfail) {
    auto comparison_function = [](float a, float b) {
        const float rtol = 0.05f;
        const float atol = 0.05f;
        float maxabs = fmaxf(fabsf(a), fabsf(b));
        float absdiff = fabsf(a - b);
        return (absdiff <= atol) || absdiff < rtol * maxabs;
    };
    vector<uint32_t> shapeA = {1, p.B_total, p.Mt * 32, p.Kt * 32};
    vector<uint32_t> shapeB = {1, p.B_total, p.Kt * 32, p.Nt * 32};
    vector<uint32_t> shapeC = {1, p.B_total, p.Mt * 32, p.Nt * 32};
    auto u16_src0 = u16_from_u32_vector(src0_vec);
    auto u16_src1 = u16_from_u32_vector(src1_vec);
    auto src0_linear =
        convert_layout<uint16_t>(u16_src0, shapeA, TensorLayoutType::TILED_NFACES, TensorLayoutType::LIN_ROW_MAJOR);
    auto src1_linear =
        convert_layout<uint16_t>(u16_src1, shapeB, TensorLayoutType::TILED_NFACES, TensorLayoutType::LIN_ROW_MAJOR);
    auto ref_bmm = gold_bmm(shapeA, src0_linear, shapeB, src1_linear);
    auto gold = u32_from_u16_vector(
        convert_layout<uint16_t>(ref_bmm, shapeC, TensorLayoutType::LIN_ROW_MAJOR, TensorLayoutType::TILED_NFACES));
    return packed_uint32_t_vector_comparison(result_vec, gold, comparison_function, argfail);
}

}  // namespace

TEST_F(MeshDeviceSingleCardFixture, Bmm) {
    auto& mesh_device = *devices_[0];
    IDevice* dev = mesh_device.get_devices()[0];

    BmmParams p;
    if (dev->arch() != ARCH::QUASAR) {
        p.Mt = 4; p.Kt = 2; p.Nt = 3;
        p.B_total = 2; p.B_per_core = 2;
        p.num_input_tiles = 2; p.num_output_tiles = 2;
        p.num_threads = 1;
    } else {
        p.Mt = 2; p.Kt = 2; p.Nt = 2;
        p.B_total = 1; p.B_per_core = 1;
        p.num_threads = 2;
    }

    auto tensors = create_bmm_tensors(mesh_device, p);
    const uint32_t bytesA = p.single_tile_size * p.Mt * p.Kt * p.B_total;
    const uint32_t bytesB = p.single_tile_size * p.Kt * p.Nt * p.B_total;

    const experimental::NodeCoord node{0, 0};
    const bool use_implicit_sync = (dev->arch() == ARCH::QUASAR);
    auto spec = build_bmm_program_spec(p, tensors, node, use_implicit_sync);
    auto program = experimental::MakeProgramFromSpec(mesh_device, spec);

    constexpr uint32_t do_bcast = 0;
    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values = {{node, {{"batch_start", 0u}}}},
            .common_runtime_arg_values =
                {{"Mt", p.Mt},
                 {"Kt", p.Kt},
                 {"Nt", p.Nt},
                 {"MtKt", p.Mt * p.Kt},
                 {"KtNt", p.Kt * p.Nt},
                 {"batch", p.B_per_core},
                 {"do_bcast", do_bcast}},
        },
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values = {{node, {{"batch_start", 0u}}}},
            .common_runtime_arg_values = {{"Mt", p.Mt}, {"Nt", p.Nt}, {"batch", p.B_per_core}},
        },
        experimental::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE},
    };
    params.tensor_args = {
        {SRC0_T, experimental::TensorArgument{tensors.src0}},
        {SRC1_T, experimental::TensorArgument{tensors.src1}},
        {DST_T, experimental::TensorArgument{tensors.dst}},
    };
    experimental::SetProgramRunArgs(program, params);

    auto src0_vec = create_random_vector_of_bfloat16(bytesA, 1.0f, 0x1234);
    auto src1_vec = create_random_vector_of_bfloat16(bytesB, 1.0f, 0x1234, -0.45f);
    // MeshTensor doesn't yet expose slow-dispatch read/write APIs, so route through the
    // underlying reference buffer to populate / read back DRAM.
    detail::WriteToBuffer(*tensors.src0.mesh_buffer().get_reference_buffer(), src0_vec);
    detail::WriteToBuffer(*tensors.src1.mesh_buffer().get_reference_buffer(), src1_vec);

    detail::LaunchProgram(dev, program, true);

    std::vector<uint32_t> result_vec;
    detail::ReadFromBuffer(*tensors.dst.mesh_buffer().get_reference_buffer(), result_vec);

    int argfail = -1;
    bool pass = validate_bmm_result(p, src0_vec, src1_vec, result_vec, &argfail);
    EXPECT_TRUE(pass) << "Failure position=" << argfail;
}

// This needs to be a separate test because we don't have a way of querying the correct compute grid size
// when running a multi-neo emu/sim build. Otherwise its the same test with batch split across nodes.
TEST_F(QuasarMeshDeviceSingleCardFixture, BmmMultinode) {
    auto& mesh_device = *devices_[0];
    if (mesh_device.compute_with_storage_grid_size().x < 2) {
        GTEST_SKIP() << "This test requires at least 2 worker nodes.";
    }

    IDevice* dev = mesh_device.get_devices()[0];

    BmmParams p;
    p.Mt = 2; p.Kt = 2; p.Nt = 2;
    p.B_total = 2;      // total batches across both cores
    p.B_per_core = 1;   // each core computes exactly one batch
    p.num_threads = 2;

    auto tensors = create_bmm_tensors(mesh_device, p);
    const uint32_t bytesA = p.single_tile_size * p.Mt * p.Kt * p.B_total;
    const uint32_t bytesB = p.single_tile_size * p.Kt * p.Nt * p.B_total;

    const experimental::NodeCoord node0{0, 0};
    const experimental::NodeCoord node1{1, 0};
    const experimental::NodeRange node_range{node0, node1};

    // QuasarMeshDeviceSingleCardFixture only opens Quasar devices, so implicit sync is always on.
    auto spec = build_bmm_program_spec(p, tensors, node_range, /*use_implicit_sync=*/true);
    auto program = experimental::MakeProgramFromSpec(mesh_device, spec);

    constexpr uint32_t do_bcast = 0;
    // node0 handles batch 0, node1 handles batch 1 (batch_start = node index)
    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values = {{node0, {{"batch_start", 0u}}}, {node1, {{"batch_start", 1u}}}},
            .common_runtime_arg_values =
                {{"Mt", p.Mt},
                 {"Kt", p.Kt},
                 {"Nt", p.Nt},
                 {"MtKt", p.Mt * p.Kt},
                 {"KtNt", p.Kt * p.Nt},
                 {"batch", p.B_per_core},
                 {"do_bcast", do_bcast}},
        },
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values = {{node0, {{"batch_start", 0u}}}, {node1, {{"batch_start", 1u}}}},
            .common_runtime_arg_values = {{"Mt", p.Mt}, {"Nt", p.Nt}, {"batch", p.B_per_core}},
        },
        experimental::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE},
    };
    params.tensor_args = {
        {SRC0_T, experimental::TensorArgument{tensors.src0}},
        {SRC1_T, experimental::TensorArgument{tensors.src1}},
        {DST_T, experimental::TensorArgument{tensors.dst}},
    };
    experimental::SetProgramRunArgs(program, params);

    auto src0_vec = create_random_vector_of_bfloat16(bytesA, 1.0f, 0x1234);
    auto src1_vec = create_random_vector_of_bfloat16(bytesB, 1.0f, 0x1234, -0.45f);
    // MeshTensor doesn't yet expose slow-dispatch read/write APIs, so route through the
    // underlying reference buffer to populate / read back DRAM.
    detail::WriteToBuffer(*tensors.src0.mesh_buffer().get_reference_buffer(), src0_vec);
    detail::WriteToBuffer(*tensors.src1.mesh_buffer().get_reference_buffer(), src1_vec);

    detail::LaunchProgram(dev, program, true);

    std::vector<uint32_t> result_vec;
    detail::ReadFromBuffer(*tensors.dst.mesh_buffer().get_reference_buffer(), result_vec);

    int argfail = -1;
    bool pass = validate_bmm_result(p, src0_vec, src1_vec, result_vec, &argfail);
    EXPECT_TRUE(pass) << "Failure position=" << argfail;
}
