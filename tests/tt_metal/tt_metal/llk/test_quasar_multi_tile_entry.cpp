// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Exercises DFB entries that hold more than one tile, through both the unpacker and the
// packer, on Quasar.
//
// Every other Quasar compute test uses entry_size == one tile. At one tile per entry an
// entry index and a tile index are numerically identical, so the entry->tile conversion on
// each side of DEST is unobservable: a side that forgets to scale by tiles-per-entry still
// lands on the right tile. These cases set entry_size = tiles_per_entry * tile_size so the
// two indices diverge from the second entry onwards, and compare the copy bit-exactly.
//
// The DM kernels need no changes: direct_reader/writer_unary_2_0 already move one whole
// entry per iteration (ublock_size_bytes = dfb.get_entry_size()), so their "num_tiles"
// runtime arg is a count of ENTRIES here and the DRAM page stride is the entry size.

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

namespace {

constexpr std::uint32_t SINGLE_TILE_SIZE = 2 * 1024;  // Float16_b 32x32 tile

// Copy NUM_ENTRIES entries of TILES_PER_ENTRY tiles each, in -> out, and check bit-exactly.
void run_multi_tile_entry_datacopy(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    std::uint32_t tiles_per_entry,
    std::uint32_t num_entries) {
    auto& cq = mesh_device->mesh_command_queue();
    const experimental::NodeCoord node{0, 0};

    const std::uint32_t entry_size = tiles_per_entry * SINGLE_TILE_SIZE;
    const std::uint32_t total_bytes = entry_size * num_entries;

    // One DRAM bank per buffer (page_size == whole buffer), as in the other direct-reader tests.
    distributed::ReplicatedBufferConfig global_config{.size = total_bytes};
    distributed::DeviceLocalBufferConfig local_config{.page_size = total_bytes, .buffer_type = BufferType::DRAM};
    auto src_dram_buffer = distributed::MeshBuffer::create(global_config, local_config, mesh_device.get());
    auto dst_dram_buffer = distributed::MeshBuffer::create(global_config, local_config, mesh_device.get());

    const experimental::DFBSpecName IN_DFB{"in_dfb"};
    const experimental::DFBSpecName OUT_DFB{"out_dfb"};
    const experimental::KernelSpecName READER{"reader"};
    const experimental::KernelSpecName WRITER{"writer"};
    const experimental::KernelSpecName COMPUTE{"compute"};

    // The point of the test: entry_size is a whole multiple of the tile size, > 1 tile.
    experimental::DataflowBufferSpec in_dfb_spec{
        .unique_id = IN_DFB,
        .entry_size = entry_size,
        .num_entries = 2,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };
    experimental::DataflowBufferSpec out_dfb_spec{
        .unique_id = OUT_DFB,
        .entry_size = entry_size,
        .num_entries = 2,
        .data_format_metadata = tt::DataFormat::Float16_b,
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
        .source = "tests/tt_metal/tt_metal/test_kernels/compute/multi_tile_entry_datacopy_quasar.cpp",
        .num_threads = 1,
        .dfb_bindings =
            {{
                 .dfb_spec_name = IN_DFB,
                 .accessor_name = "in",
                 .endpoint_type = experimental::DFBEndpointType::CONSUMER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = OUT_DFB,
                 .accessor_name = "out",
                 .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             }},
        .compile_time_args = {{"num_entries", num_entries}, {"tiles_per_entry", tiles_per_entry}},
        .hw_config = experimental::ComputeHardwareConfig{},
    };

    experimental::WorkUnitSpec wu{
        .name = "main",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = node,
    };

    experimental::ProgramSpec spec{
        .name = "multi_tile_entry_datacopy",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {in_dfb_spec, out_dfb_spec},
        .work_units = {wu},
    };

    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    // Distinct data in every tile, so a tile landing at the wrong offset inside an entry --
    // or an entry's tiles landing in the previous entry -- fails the comparison.
    std::vector<std::uint32_t> src_vec = create_random_vector_of_bfloat16(total_bytes, 1.0f, 0x5EED42);
    distributed::EnqueueWriteMeshBuffer(cq, src_dram_buffer, src_vec, /*blocking=*/true);

    // The DM kernels move one ENTRY per loop iteration, so "num_tiles" is the entry count
    // and the DRAM stride is one entry.
    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node,
                {{"src_addr", src_dram_buffer->address()},
                 {"src_bank_id", 0u},
                 {"num_tiles", num_entries},
                 {"dram_page_stride", entry_size}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node,
                {{"dst_addr", dst_dram_buffer->address()},
                 {"dst_bank_id", 0u},
                 {"num_tiles", num_entries},
                 {"dram_page_stride", entry_size}}),
        },
    };
    experimental::SetProgramRunArgs(program, params);

    LaunchProgram(*mesh_device, std::move(program));

    std::vector<std::uint32_t> result_vec;
    distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_dram_buffer, /*blocking=*/true);

    // Pure datacopy: the output must equal the input, tile for tile.
    auto comparison_function = [](float a, float b) { return a == b; };
    int argfail = -1;
    bool pass = packed_uint32_t_vector_comparison(result_vec, src_vec, comparison_function, &argfail);
    EXPECT_TRUE(pass) << "tiles_per_entry=" << tiles_per_entry << " num_entries=" << num_entries
                      << " first mismatch at word " << argfail << " (tile "
                      << (argfail < 0 ? -1 : argfail / int(SINGLE_TILE_SIZE / sizeof(std::uint32_t))) << ")";
}

}  // namespace

// Two tiles per entry: the smallest case where an entry index and a tile index diverge.
TEST_F(LLKQuasarMeshDeviceSingleCardFixture, QuasarMultiTileEntryDatacopy2) {
    run_multi_tile_entry_datacopy(this->devices_.at(0), /*tiles_per_entry=*/2, /*num_entries=*/8);
}

// Four tiles per entry, still inside the SyncHalf 16-bit DEST section capacity (8 tiles).
TEST_F(LLKQuasarMeshDeviceSingleCardFixture, QuasarMultiTileEntryDatacopy4) {
    run_multi_tile_entry_datacopy(this->devices_.at(0), /*tiles_per_entry=*/4, /*num_entries=*/8);
}
