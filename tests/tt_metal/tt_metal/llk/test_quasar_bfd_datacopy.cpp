// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Quasar BFD re-architecture POC: buffer descriptors are no longer programmed in
// hw_configure (coupled 1:1 with DFB ids); instead each op's llk_*_init allocates an
// id from its TRISC's partition (T0: [0,16), T2: [16,24), T3: [24,32)) via a
// bump-and-wrap counter and programs the table entry itself. This test rotates a
// datacopy over three input DFBs with a full re-init every tile, churning BFD ids
// enough to wrap both the unpack partition (once) and the pack partition (twice).

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

using namespace tt;
using namespace tt::tt_metal;

namespace {
constexpr std::uint32_t NUM_INPUTS = 3;
constexpr std::uint32_t TILES_PER_INPUT = 6;
// 18 cycles: unpack allocs hit ids 0..15,0,1 (wraps once); pack allocs hit
// 16..23,16..23,16,17 (wraps twice). Every wrap overwrites an entry whose previous
// op has fully drained, so data must still land correctly.
constexpr std::uint32_t NUM_CYCLES = NUM_INPUTS * TILES_PER_INPUT;
}  // namespace

TEST_F(LLKQuasarMeshDeviceSingleCardFixture, QuasarBfdDatacopy) {
    const std::shared_ptr<distributed::MeshDevice>& mesh_device = this->devices_.at(0);
    IDevice* dev = mesh_device->get_devices()[0];
    const experimental::NodeCoord node{0, 0};

    const std::uint32_t single_tile_size = 2 * 1024;  // Float16_b 32x32 tile

    InterleavedBufferConfig src0_config{
        .device = dev,
        .size = single_tile_size * TILES_PER_INPUT,
        .page_size = single_tile_size,
        .buffer_type = BufferType::DRAM};
    auto src0_dram_buffer = CreateBuffer(src0_config);
    auto src1_dram_buffer = CreateBuffer(src0_config);
    auto src2_dram_buffer = CreateBuffer(src0_config);

    InterleavedBufferConfig dst_config{
        .device = dev,
        .size = single_tile_size * NUM_CYCLES,
        .page_size = single_tile_size,
        .buffer_type = BufferType::DRAM};
    auto dst_dram_buffer = CreateBuffer(dst_config);

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
        .compile_time_args = {{"num_cycles", NUM_CYCLES}},
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

    // Stimulus: distinct random tiles per input so a mis-routed copy is visible
    std::vector<std::uint32_t> src0_vec =
        create_random_vector_of_bfloat16(single_tile_size * TILES_PER_INPUT, 1.0f, 0xC0FFEE);
    std::vector<std::uint32_t> src1_vec =
        create_random_vector_of_bfloat16(single_tile_size * TILES_PER_INPUT, 1.0f, 0xDEAD01);
    std::vector<std::uint32_t> src2_vec =
        create_random_vector_of_bfloat16(single_tile_size * TILES_PER_INPUT, 1.0f, 0xBEEF02);
    tt::tt_metal::detail::WriteToBuffer(src0_dram_buffer, src0_vec);
    tt::tt_metal::detail::WriteToBuffer(src1_dram_buffer, src1_vec);
    tt::tt_metal::detail::WriteToBuffer(src2_dram_buffer, src2_vec);

    const std::uint32_t src_aligned_page_size = static_cast<std::uint32_t>(src0_dram_buffer->aligned_page_size());
    const std::uint32_t dst_aligned_page_size = static_cast<std::uint32_t>(dst_dram_buffer->aligned_page_size());

    auto reader_run_args = [&](const std::shared_ptr<Buffer>& buf) {
        return experimental::MakeRuntimeArgsForSingleNode(
            node,
            {{"src_addr", buf->address()},
             {"src_bank_id", 0u},
             {"num_tiles", TILES_PER_INPUT},
             {"dram_page_stride", src_aligned_page_size}});
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
                 {"num_tiles", NUM_CYCLES},
                 {"dram_page_stride", dst_aligned_page_size}}),
        },
    };
    experimental::SetProgramRunArgs(program, params);

    tt::tt_metal::detail::LaunchProgram(dev, program, true);

    std::vector<std::uint32_t> result_vec;
    tt::tt_metal::detail::ReadFromBuffer(dst_dram_buffer, result_vec);

    // Cycle i copies tile (i / 3) of input (i % 3)
    const std::vector<std::uint32_t>* srcs[NUM_INPUTS] = {&src0_vec, &src1_vec, &src2_vec};
    std::vector<std::uint32_t> golden;
    golden.reserve(src0_vec.size() * NUM_INPUTS);
    for (std::uint32_t i = 0; i < NUM_CYCLES; ++i) {
        const std::vector<std::uint32_t>& src = *srcs[i % NUM_INPUTS];
        const std::uint32_t tile_idx = i / NUM_INPUTS;
        const size_t words_per_tile = single_tile_size / sizeof(std::uint32_t);
        golden.insert(
            golden.end(), src.begin() + tile_idx * words_per_tile, src.begin() + (tile_idx + 1) * words_per_tile);
    }

    auto comparison_function = [](float a, float b) { return a == b; };  // datacopy is bit-exact
    int argfail = -1;
    bool pass = packed_uint32_t_vector_comparison(result_vec, golden, comparison_function, &argfail);
    EXPECT_TRUE(pass) << "Failure position=" << argfail;
}
