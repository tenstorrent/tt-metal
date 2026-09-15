// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Quasar BFD re-architecture POC: buffer descriptors are no longer programmed in
// hw_configure (coupled 1:1 with DFB ids); instead each op's llk_*_init allocates an
// id from its TRISC's partition (T0: [0,16), T2: [16,24), T3: [24,32)) via a
// bump-and-wrap counter and programs the table entry itself. This test copies three
// input DFBs to one output DFB the realistic way -- init once per operand, then a block
// loop of tiles -- and wraps the whole three-operand sequence in an outer loop of
// NUM_LOOPS. Each operand switch bump-allocates a fresh unpack BFD, so NUM_INPUTS*NUM_LOOPS
// (= 30) inits per run wrap the 16-entry unpack partition and reuse ids mid-run. The readers
// re-stream the same TILES_PER_INPUT tiles NUM_LOOPS times (the Quasar tile-counter model
// consumes a tile per unpack, so re-copying requires re-streaming). Every copy, including the
// post-wrap ones, is bit-exact-checked, so a wrapped id that programs the wrong input's
// descriptor is caught.

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
#include <tt-logger/tt-logger.hpp>

using namespace tt;
using namespace tt::tt_metal;

namespace {
constexpr std::uint32_t NUM_INPUTS = 3;
constexpr std::uint32_t TILES_PER_INPUT = 16;  // unique tiles per input, copied as one block
constexpr std::uint32_t NUM_LOOPS = 10;        // stream + copy the same tiles this many times
// NUM_CYCLES = one pass over all inputs (in0..in2, TILES_PER_INPUT each = 48). The compute wraps this in an
// outer loop of NUM_LOOPS, re-initing the unpack BFD once per operand (NUM_INPUTS*NUM_LOOPS = 30 inits > the
// 16-entry unpack partition, so it wraps). The readers re-stream the same TILES_PER_INPUT tiles NUM_LOOPS
// times: the same 16 tiles/input are copied 10x -> TILES_STREAMED_PER_INPUT (160) per input, TOTAL_TILES (480).
constexpr std::uint32_t NUM_CYCLES = NUM_INPUTS * TILES_PER_INPUT;
constexpr std::uint32_t TILES_STREAMED_PER_INPUT = TILES_PER_INPUT * NUM_LOOPS;
constexpr std::uint32_t TOTAL_TILES = NUM_CYCLES * NUM_LOOPS;
}  // namespace

TEST_F(LLKQuasarMeshDeviceSingleCardFixture, QuasarBfdDatacopy) {
    const std::shared_ptr<distributed::MeshDevice>& mesh_device = this->devices_.at(0);
    auto& cq = mesh_device->mesh_command_queue();
    const experimental::NodeCoord node{0, 0};

    const std::uint32_t single_tile_size = 2 * 1024;  // Float16_b 32x32 tile

    // The direct reader/writer kernels address a single DRAM bank (bank_id
    // 0). Use page_size = whole buffer so the allocator places each buffer in
    // one bank, and advance the DRAM pointer by the native tile size.
    distributed::ReplicatedBufferConfig src_global_config{.size = single_tile_size * TILES_STREAMED_PER_INPUT};
    distributed::DeviceLocalBufferConfig src_local_config{
        .page_size = single_tile_size * TILES_STREAMED_PER_INPUT, .buffer_type = BufferType::DRAM};
    auto src0_dram_buffer = distributed::MeshBuffer::create(src_global_config, src_local_config, mesh_device.get());
    auto src1_dram_buffer = distributed::MeshBuffer::create(src_global_config, src_local_config, mesh_device.get());
    auto src2_dram_buffer = distributed::MeshBuffer::create(src_global_config, src_local_config, mesh_device.get());

    auto dst_dram_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = single_tile_size * TOTAL_TILES},
        {.page_size = single_tile_size * TOTAL_TILES, .buffer_type = BufferType::DRAM},
        mesh_device.get());

    const experimental::DFBSpecName IN0_DFB{"in0_dfb"};
    const experimental::DFBSpecName IN1_DFB{"in1_dfb"};
    const experimental::DFBSpecName IN2_DFB{"in2_dfb"};
    const experimental::DFBSpecName OUT_DFB{"out_dfb"};
    const experimental::KernelSpecName READER0{"reader0"};
    const experimental::KernelSpecName READER1{"reader1"};
    const experimental::KernelSpecName READER2{"reader2"};
    const experimental::KernelSpecName WRITER{"writer"};
    const experimental::KernelSpecName COMPUTE{"compute"};

    experimental::DataflowBufferSpec in0_dfb_spec{
        .unique_id = IN0_DFB,
        .entry_size = single_tile_size,
        .num_entries = 2,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };
    experimental::DataflowBufferSpec in1_dfb_spec{
        .unique_id = IN1_DFB,
        .entry_size = single_tile_size,
        .num_entries = 2,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };
    experimental::DataflowBufferSpec in2_dfb_spec{
        .unique_id = IN2_DFB,
        .entry_size = single_tile_size,
        .num_entries = 2,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };
    experimental::DataflowBufferSpec out_dfb_spec{
        .unique_id = OUT_DFB,
        .entry_size = single_tile_size,
        .num_entries = 4,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    auto make_reader_spec = [](const experimental::DFBSpecName& dfb) {
        return experimental::KernelSpec{
            .unique_id = experimental::KernelSpecName{""},  // patched by caller
            .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_reader_unary_2_0.cpp",
            .num_threads = 1,
            .dfb_bindings = {experimental::ProducerOf(dfb, "out")},
            .runtime_arg_schema = {.runtime_arg_names = {"src_addr", "src_bank_id", "num_tiles", "dram_page_stride"}},
            .hw_config = experimental::DataMovementGen2Config{},
        };
    };
    experimental::KernelSpec reader0_spec = make_reader_spec(IN0_DFB);
    reader0_spec.unique_id = READER0;
    experimental::KernelSpec reader1_spec = make_reader_spec(IN1_DFB);
    reader1_spec.unique_id = READER1;
    experimental::KernelSpec reader2_spec = make_reader_spec(IN2_DFB);
    reader2_spec.unique_id = READER2;

    experimental::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_writer_unary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ConsumerOf(OUT_DFB, "in")},
        .runtime_arg_schema = {.runtime_arg_names = {"dst_addr", "dst_bank_id", "num_tiles", "dram_page_stride"}},
        .hw_config = experimental::DataMovementGen2Config{},
    };

    experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = "tests/tt_metal/tt_metal/test_kernels/compute/bfd_datacopy_quasar.cpp",
        .num_threads = 1,
        .dfb_bindings =
            {{
                 .dfb_spec_name = IN0_DFB,
                 .accessor_name = "in0",
                 .endpoint_type = experimental::DFBEndpointType::CONSUMER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = IN1_DFB,
                 .accessor_name = "in1",
                 .endpoint_type = experimental::DFBEndpointType::CONSUMER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = IN2_DFB,
                 .accessor_name = "in2",
                 .endpoint_type = experimental::DFBEndpointType::CONSUMER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = OUT_DFB,
                 .accessor_name = "out",
                 .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             }},
        .compile_time_args = {{"num_cycles", NUM_CYCLES}, {"num_loops", NUM_LOOPS}},
        .hw_config = experimental::ComputeGen2Config{},
    };

    experimental::WorkUnitSpec wu{
        .name = "main",
        .kernels = {READER0, READER1, READER2, WRITER, COMPUTE},
        .target_nodes = node,
    };

    experimental::ProgramSpec spec{
        .name = "bfd_datacopy",
        .kernels = {reader0_spec, reader1_spec, reader2_spec, writer_spec, compute_spec},
        .dataflow_buffers = {in0_dfb_spec, in1_dfb_spec, in2_dfb_spec, out_dfb_spec},
        .work_units = {wu},
    };

    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    // Stimulus: TILES_PER_INPUT distinct random tiles per input, repeated NUM_LOOPS times in DRAM so the
    // readers stream (and the compute copies) the SAME tiles over and over.
    auto make_streamed = [&](std::uint32_t seed) {
        std::vector<std::uint32_t> unique =
            create_random_vector_of_bfloat16(single_tile_size * TILES_PER_INPUT, 1.0f, seed);
        std::vector<std::uint32_t> streamed;
        streamed.reserve(unique.size() * NUM_LOOPS);
        for (std::uint32_t l = 0; l < NUM_LOOPS; ++l) {
            streamed.insert(streamed.end(), unique.begin(), unique.end());
        }
        return streamed;
    };
    std::vector<std::uint32_t> src0_vec = make_streamed(0xC0FFEE);
    std::vector<std::uint32_t> src1_vec = make_streamed(0xDEAD01);
    std::vector<std::uint32_t> src2_vec = make_streamed(0xBEEF02);
    distributed::EnqueueWriteMeshBuffer(cq, src0_dram_buffer, src0_vec, /*blocking=*/true);
    distributed::EnqueueWriteMeshBuffer(cq, src1_dram_buffer, src1_vec, /*blocking=*/true);
    distributed::EnqueueWriteMeshBuffer(cq, src2_dram_buffer, src2_vec, /*blocking=*/true);
    // Phase A dprint check: the compute kernel DPRINTs each input's tile0 word0 straight from L1;
    // these are the host-placed values it must match (compare against the kernel's word0=... lines).
    log_info(
        LogTest, "HOST tile0 word0: in0={:#010x} in1={:#010x} in2={:#010x}", src0_vec[0], src1_vec[0], src2_vec[0]);

    auto reader_run_args = [&](const std::shared_ptr<distributed::MeshBuffer>& buf) {
        return experimental::MakeRuntimeArgsForSingleNode(
            node,
            {{"src_addr", buf->address()},
             {"src_bank_id", 0u},
             {"num_tiles", TILES_STREAMED_PER_INPUT},
             {"dram_page_stride", single_tile_size}});
    };

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = READER0, .runtime_arg_values = reader_run_args(src0_dram_buffer)},
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = READER1, .runtime_arg_values = reader_run_args(src1_dram_buffer)},
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = READER2, .runtime_arg_values = reader_run_args(src2_dram_buffer)},
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node,
                {{"dst_addr", dst_dram_buffer->address()},
                 {"dst_bank_id", 0u},
                 {"num_tiles", TOTAL_TILES},
                 {"dram_page_stride", single_tile_size}}),
        },
    };
    experimental::SetProgramRunArgs(program, params);

    LaunchProgram(*mesh_device, std::move(program));

    std::vector<std::uint32_t> result_vec;
    distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_dram_buffer, /*blocking=*/true);

    // Output order = pack order: per outer loop, in0's whole block, then in1's, then in2's (blocked).
    // Within a loop, cycle i copies tile (i % TILES_PER_INPUT) of input (i / TILES_PER_INPUT), taken from
    // that loop's slice of the stream, i.e. stream tile (outer*TILES_PER_INPUT + i % TILES_PER_INPUT).
    const std::vector<std::uint32_t>* srcs[NUM_INPUTS] = {&src0_vec, &src1_vec, &src2_vec};
    const size_t words_per_tile = single_tile_size / sizeof(std::uint32_t);
    std::vector<std::uint32_t> golden;
    golden.reserve(words_per_tile * TOTAL_TILES);
    for (std::uint32_t outer = 0; outer < NUM_LOOPS; ++outer) {
        for (std::uint32_t i = 0; i < NUM_CYCLES; ++i) {
            const std::vector<std::uint32_t>& src = *srcs[i / TILES_PER_INPUT];
            const std::uint32_t tile_idx = outer * TILES_PER_INPUT + (i % TILES_PER_INPUT);
            golden.insert(
                golden.end(), src.begin() + tile_idx * words_per_tile, src.begin() + (tile_idx + 1) * words_per_tile);
        }
    }

    auto comparison_function = [](float a, float b) { return a == b; };  // datacopy is bit-exact
    int argfail = -1;
    bool pass = packed_uint32_t_vector_comparison(result_vec, golden, comparison_function, &argfail);
    EXPECT_TRUE(pass) << "Failure position=" << argfail;
}

// Phase B: DEST-register dprint validation. A deliberately SIMPLE datacopy -- one input, DEST_PRINT_TILES
// tiles, no BFD wrap -- that copies the tiles to DEST, reads each DEST tile via dprint_tensix_dest_reg
// (the Quasar RISC memory-mapped DEST read), then packs to L1. The compute kernel is phase-separated
// (copy all -> read all -> commit -> pack all). The output L1 is bit-exact-checked against the host
// input; with DPRINT enabled the DEST prints (0:0-0:N0TR1 "Tile ID"/rows) can be compared by eye
// against "HOST tile0 word0".
TEST_F(LLKQuasarMeshDeviceSingleCardFixture, QuasarBfdDatacopyDestPrint) {
    constexpr std::uint32_t DEST_PRINT_TILES = 1;

    const std::shared_ptr<distributed::MeshDevice>& mesh_device = this->devices_.at(0);
    auto& cq = mesh_device->mesh_command_queue();
    const experimental::NodeCoord node{0, 0};

    const std::uint32_t single_tile_size = 2 * 1024;  // Float16_b 32x32 tile

    distributed::ReplicatedBufferConfig src_global_config{.size = single_tile_size * DEST_PRINT_TILES};
    distributed::DeviceLocalBufferConfig src_local_config{
        .page_size = single_tile_size * DEST_PRINT_TILES, .buffer_type = BufferType::DRAM};
    auto src_dram_buffer = distributed::MeshBuffer::create(src_global_config, src_local_config, mesh_device.get());
    auto dst_dram_buffer = distributed::MeshBuffer::create(src_global_config, src_local_config, mesh_device.get());

    const experimental::DFBSpecName IN0_DFB{"dp_in0_dfb"};
    const experimental::DFBSpecName OUT_DFB{"dp_out_dfb"};
    const experimental::KernelSpecName READER0{"dp_reader0"};
    const experimental::KernelSpecName WRITER{"dp_writer"};
    const experimental::KernelSpecName COMPUTE{"dp_compute"};

    experimental::DataflowBufferSpec in0_dfb_spec{
        .unique_id = IN0_DFB,
        .entry_size = single_tile_size,
        .num_entries = 2,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };
    experimental::DataflowBufferSpec out_dfb_spec{
        .unique_id = OUT_DFB,
        .entry_size = single_tile_size,
        .num_entries = 2,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    experimental::KernelSpec reader0_spec{
        .unique_id = READER0,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_reader_unary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ProducerOf(IN0_DFB, "out")},
        .runtime_arg_schema = {.runtime_arg_names = {"src_addr", "src_bank_id", "num_tiles", "dram_page_stride"}},
        .hw_config = experimental::DataMovementGen2Config{},
    };
    experimental::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_writer_unary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ConsumerOf(OUT_DFB, "in")},
        .runtime_arg_schema = {.runtime_arg_names = {"dst_addr", "dst_bank_id", "num_tiles", "dram_page_stride"}},
        .hw_config = experimental::DataMovementGen2Config{},
    };
    experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = "tests/tt_metal/tt_metal/test_kernels/compute/datacopy_dest_print_quasar.cpp",
        .num_threads = 1,
        .dfb_bindings =
            {{
                 .dfb_spec_name = IN0_DFB,
                 .accessor_name = "in0",
                 .endpoint_type = experimental::DFBEndpointType::CONSUMER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = OUT_DFB,
                 .accessor_name = "out",
                 .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             }},
        .compile_time_args = {{"num_tiles", DEST_PRINT_TILES}},
        .hw_config = experimental::ComputeGen2Config{},
    };

    experimental::WorkUnitSpec wu{
        .name = "main",
        .kernels = {READER0, WRITER, COMPUTE},
        .target_nodes = node,
    };
    experimental::ProgramSpec spec{
        .name = "bfd_datacopy_dest_print",
        .kernels = {reader0_spec, writer_spec, compute_spec},
        .dataflow_buffers = {in0_dfb_spec, out_dfb_spec},
        .work_units = {wu},
    };
    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    std::vector<std::uint32_t> src_vec =
        create_random_vector_of_bfloat16(single_tile_size * DEST_PRINT_TILES, 1.0f, 0xC0FFEE);
    distributed::EnqueueWriteMeshBuffer(cq, src_dram_buffer, src_vec, /*blocking=*/true);
    log_info(LogTest, "HOST tile0 word0: in0={:#010x}", src_vec[0]);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = READER0,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node,
                {{"src_addr", src_dram_buffer->address()},
                 {"src_bank_id", 0u},
                 {"num_tiles", DEST_PRINT_TILES},
                 {"dram_page_stride", single_tile_size}})},
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node,
                {{"dst_addr", dst_dram_buffer->address()},
                 {"dst_bank_id", 0u},
                 {"num_tiles", DEST_PRINT_TILES},
                 {"dram_page_stride", single_tile_size}})},
    };
    experimental::SetProgramRunArgs(program, params);

    LaunchProgram(*mesh_device, std::move(program));

    std::vector<std::uint32_t> result_vec;
    distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_dram_buffer, /*blocking=*/true);

    auto comparison_function = [](float a, float b) { return a == b; };  // datacopy is bit-exact
    int argfail = -1;
    bool pass = packed_uint32_t_vector_comparison(result_vec, src_vec, comparison_function, &argfail);
    EXPECT_TRUE(pass) << "Failure position=" << argfail;
}

// Phase B (mid-pipeline): LIVE DEST-register read validation. Same one-input datacopy as
// QuasarBfdDatacopyDestPrint, but the compute kernel reads DEST from inside a running, tile-at-a-time
// pipeline -- BETWEEN copy_tile and pack_tile, with LIVE_TILES > 1 so the pipeline overlaps across
// iterations. This is the pattern that faults (TILE_COUNTERS) with an unbracketed live DEST read; it
// passes here because dprint_tensix_dest_reg now brackets the read with the ported dbg_halt/dbg_unhalt
// mailbox rendezvous, which quiesces unpack + pack. Positive proof the rendezvous enables mid-pipeline
// DEST inspection: validation is the bit-exact L1 check plus clean completion (no TILE_COUNTERS device
// fault) in the default functional env. NOTE: running this variant with DPRINT enabled (--debug)
// currently trips a separate, unrelated Quasar DPRINT-server host-side bug (bad wpos read); the
// device-side rendezvous is unaffected (this functional run confirms it), and the phase-separated
// QuasarBfdDatacopyDestPrint above still shows the actual DEST prints under --debug.
TEST_F(LLKQuasarMeshDeviceSingleCardFixture, QuasarBfdDatacopyDestPrintLive) {
    constexpr std::uint32_t LIVE_TILES = 4;

    const std::shared_ptr<distributed::MeshDevice>& mesh_device = this->devices_.at(0);
    auto& cq = mesh_device->mesh_command_queue();
    const experimental::NodeCoord node{0, 0};

    const std::uint32_t single_tile_size = 2 * 1024;  // Float16_b 32x32 tile

    distributed::ReplicatedBufferConfig src_global_config{.size = single_tile_size * LIVE_TILES};
    distributed::DeviceLocalBufferConfig src_local_config{
        .page_size = single_tile_size * LIVE_TILES, .buffer_type = BufferType::DRAM};
    auto src_dram_buffer = distributed::MeshBuffer::create(src_global_config, src_local_config, mesh_device.get());
    auto dst_dram_buffer = distributed::MeshBuffer::create(src_global_config, src_local_config, mesh_device.get());

    const experimental::DFBSpecName IN0_DFB{"dpl_in0_dfb"};
    const experimental::DFBSpecName OUT_DFB{"dpl_out_dfb"};
    const experimental::KernelSpecName READER0{"dpl_reader0"};
    const experimental::KernelSpecName WRITER{"dpl_writer"};
    const experimental::KernelSpecName COMPUTE{"dpl_compute"};

    experimental::DataflowBufferSpec in0_dfb_spec{
        .unique_id = IN0_DFB,
        .entry_size = single_tile_size,
        .num_entries = 2,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };
    experimental::DataflowBufferSpec out_dfb_spec{
        .unique_id = OUT_DFB,
        .entry_size = single_tile_size,
        .num_entries = 2,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    experimental::KernelSpec reader0_spec{
        .unique_id = READER0,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_reader_unary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ProducerOf(IN0_DFB, "out")},
        .runtime_arg_schema = {.runtime_arg_names = {"src_addr", "src_bank_id", "num_tiles", "dram_page_stride"}},
        .hw_config = experimental::DataMovementGen2Config{},
    };
    experimental::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_writer_unary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ConsumerOf(OUT_DFB, "in")},
        .runtime_arg_schema = {.runtime_arg_names = {"dst_addr", "dst_bank_id", "num_tiles", "dram_page_stride"}},
        .hw_config = experimental::DataMovementGen2Config{},
    };
    experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = "tests/tt_metal/tt_metal/test_kernels/compute/datacopy_dest_print_live_quasar.cpp",
        .num_threads = 1,
        .dfb_bindings =
            {{
                 .dfb_spec_name = IN0_DFB,
                 .accessor_name = "in0",
                 .endpoint_type = experimental::DFBEndpointType::CONSUMER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = OUT_DFB,
                 .accessor_name = "out",
                 .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             }},
        .compile_time_args = {{"num_tiles", LIVE_TILES}},
        .hw_config = experimental::ComputeGen2Config{},
    };

    experimental::WorkUnitSpec wu{
        .name = "main",
        .kernels = {READER0, WRITER, COMPUTE},
        .target_nodes = node,
    };
    experimental::ProgramSpec spec{
        .name = "bfd_datacopy_dest_print_live",
        .kernels = {reader0_spec, writer_spec, compute_spec},
        .dataflow_buffers = {in0_dfb_spec, out_dfb_spec},
        .work_units = {wu},
    };
    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    std::vector<std::uint32_t> src_vec =
        create_random_vector_of_bfloat16(single_tile_size * LIVE_TILES, 1.0f, 0xC0FFEE);
    distributed::EnqueueWriteMeshBuffer(cq, src_dram_buffer, src_vec, /*blocking=*/true);
    log_info(LogTest, "HOST tile0 word0: in0={:#010x}", src_vec[0]);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = READER0,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node,
                {{"src_addr", src_dram_buffer->address()},
                 {"src_bank_id", 0u},
                 {"num_tiles", LIVE_TILES},
                 {"dram_page_stride", single_tile_size}})},
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node,
                {{"dst_addr", dst_dram_buffer->address()},
                 {"dst_bank_id", 0u},
                 {"num_tiles", LIVE_TILES},
                 {"dram_page_stride", single_tile_size}})},
    };
    experimental::SetProgramRunArgs(program, params);

    LaunchProgram(*mesh_device, std::move(program));

    std::vector<std::uint32_t> result_vec;
    distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_dram_buffer, /*blocking=*/true);

    auto comparison_function = [](float a, float b) { return a == b; };  // datacopy is bit-exact
    int argfail = -1;
    bool pass = packed_uint32_t_vector_comparison(result_vec, src_vec, comparison_function, &argfail);
    EXPECT_TRUE(pass) << "Failure position=" << argfail;
}
