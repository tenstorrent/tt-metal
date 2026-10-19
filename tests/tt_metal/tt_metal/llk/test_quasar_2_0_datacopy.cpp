// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Quasar 2.0 compute API datacopy: one input DFB -> compute datacopy (id-free 2.0 ops, explicit dest sync)
// -> one output DFB, read back and checked bit-exact against the input.

#include "llk_device_fixture.hpp"

#include <cstdint>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include "impl/data_format/bfloat16_utils.hpp"
#include "impl/program/program_impl.hpp"

using namespace tt;
using namespace tt::tt_metal;

TEST_F(LLKQuasarMeshDeviceSingleCardFixture, Quasar2_0Datacopy) {
    constexpr std::uint32_t NUM_TILES = 16;
    constexpr std::uint32_t BLOCK_TILES = 2;  // tiles per Dest section (copy_block / pack_block)
    constexpr std::uint32_t TILE_SIZE = 2 * 1024;  // Float16_b 32x32

    const std::shared_ptr<distributed::MeshDevice>& mesh_device = this->devices_.at(0);
    auto& cq = mesh_device->mesh_command_queue();
    const experimental::NodeCoord node{0, 0};

    // The direct reader/writer address one DRAM bank (bank 0): page_size = whole buffer keeps it in one bank.
    const distributed::DeviceLocalBufferConfig dram_config{
        .page_size = TILE_SIZE * NUM_TILES, .buffer_type = BufferType::DRAM};
    auto src_buffer = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = TILE_SIZE * NUM_TILES}, dram_config, mesh_device.get());
    auto dst_buffer = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = TILE_SIZE * NUM_TILES}, dram_config, mesh_device.get());

    const experimental::DFBSpecName IN_DFB{"in_dfb"};
    const experimental::DFBSpecName OUT_DFB{"out_dfb"};
    const experimental::KernelSpecName READER{"reader"};
    const experimental::KernelSpecName WRITER{"writer"};
    const experimental::KernelSpecName COMPUTE{"compute"};

    auto dfb_spec = [&](const experimental::DFBSpecName& name) {
        return experimental::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = TILE_SIZE,
            .num_entries = 2 * BLOCK_TILES,
            .data_format_metadata = tt::DataFormat::Float16_b,  // -> LLKOperandFrom<dfb::...> in the compute kernel
        };
    };

    experimental::KernelSpec reader_spec{
        .unique_id = READER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_reader_unary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ProducerOf(IN_DFB, "out")},
        .runtime_arg_schema = {.runtime_arg_names = {"src_addr", "src_bank_id", "num_tiles", "dram_page_stride"}},
        .hw_config = experimental::DataMovementHardwareConfig{},
    };
    experimental::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_writer_unary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ConsumerOf(OUT_DFB, "in")},
        .runtime_arg_schema = {.runtime_arg_names = {"dst_addr", "dst_bank_id", "num_tiles", "dram_page_stride"}},
        .hw_config = experimental::DataMovementHardwareConfig{},
    };
    experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = "tests/tt_metal/tt_metal/test_kernels/compute/datacopy_2_0_quasar.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ConsumerOf(IN_DFB, "in"), experimental::ProducerOf(OUT_DFB, "out")},
        .compile_time_args = {{"num_tiles", NUM_TILES}, {"block_tiles", BLOCK_TILES}},
        .hw_config = experimental::ComputeHardwareConfig{},
    };

    experimental::ProgramSpec spec{
        .name = "datacopy_2_0",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {dfb_spec(IN_DFB), dfb_spec(OUT_DFB)},
        .work_units = {experimental::WorkUnitSpec{
            .name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = node}},
    };
    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    std::vector<std::uint32_t> src_vec = create_random_vector_of_bfloat16(TILE_SIZE * NUM_TILES, 1.0f, 0x2020);
    distributed::EnqueueWriteMeshBuffer(cq, src_buffer, src_vec, /*blocking=*/true);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        {.kernel = READER,
         .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
             node,
             {{"src_addr", src_buffer->address()},
              {"src_bank_id", 0u},
              {"num_tiles", NUM_TILES},
              {"dram_page_stride", TILE_SIZE}})},
        {.kernel = WRITER,
         .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
             node,
             {{"dst_addr", dst_buffer->address()},
              {"dst_bank_id", 0u},
              {"num_tiles", NUM_TILES},
              {"dram_page_stride", TILE_SIZE}})},
    };
    experimental::SetProgramRunArgs(program, params);

    LaunchProgram(*mesh_device, std::move(program));

    std::vector<std::uint32_t> result_vec;
    distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_buffer, /*blocking=*/true);

    int argfail = -1;
    const bool pass =
        packed_uint32_t_vector_comparison(result_vec, src_vec, [](float a, float b) { return a == b; }, &argfail);
    EXPECT_TRUE(pass) << "Failure position=" << argfail;
}
