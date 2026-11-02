// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Quasar unpack-to-dest with tiny tiles: a 16x32 Float32 and a 1x32 Int32 DFB routed to a 32-bit DEST, several
// tiles per section placed with copy_tile at DEST indices 0..n-1 and packed back from the same indices. n goes up to
// the section capacity for the tile shape (DEST rows over the rows one tile spans: 16x32 takes 32 rows, 1x32 takes
// 16), so the last tile of a SyncHalf section sits at the top of its bank, where a 32x32-sized index would already
// be in the other bank. Three sections per run walk bank 0 -> 1 -> 0. Every datum is unique, so a tile or face
// that lands at the wrong DEST rows fails the bit-exact compare.

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <string>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-logger/tt-logger.hpp>

#include "llk_device_fixture.hpp"

namespace tt::tt_metal {

namespace unit_tests::compute::quasar_tiny_tile_to_dest {

// DEST rows one tile spans on the unpack-to-dest path: faces shorter than 8 rows take 8-row slots.
static std::uint32_t dest_tile_rows(std::uint32_t num_faces, std::uint32_t face_r_dim) {
    return num_faces * std::max<std::uint32_t>(face_r_dim, 8);
}

// Tiles one 32-bit DEST section holds for the shape: 256 rows in SyncHalf, 512 in SyncFull.
static std::uint32_t dest_section_tiles(std::uint32_t num_faces, std::uint32_t face_r_dim, bool dst_full_sync_en) {
    return (dst_full_sync_en ? 512u : 256u) / dest_tile_rows(num_faces, face_r_dim);
}

static void run_tiny_tile_copy_to_dest(
    distributed::MeshDevice& mesh_device,
    tt::DataFormat data_format,
    std::uint32_t face_r_dim,
    bool dst_full_sync_en,
    std::uint32_t section_ntiles,
    std::uint32_t num_sections) {
    constexpr std::uint32_t num_faces = 2;  // 1x2 face grid, Nx32 tiles
    const std::uint32_t num_tiles = section_ntiles * num_sections;
    const std::uint32_t tile_bytes = tt::datum_size(data_format) * num_faces * face_r_dim * tt::constants::FACE_WIDTH;
    const std::uint32_t words_per_tile = tile_bytes / sizeof(std::uint32_t);
    const std::uint32_t buffer_bytes = tile_bytes * num_tiles;

    auto& cq = mesh_device.mesh_command_queue();
    const experimental::NodeCoord node{0, 0};

    auto src_dram_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = buffer_bytes},
        {.page_size = buffer_bytes, .buffer_type = BufferType::DRAM},
        &mesh_device);
    auto dst_dram_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = buffer_bytes},
        {.page_size = buffer_bytes, .buffer_type = BufferType::DRAM},
        &mesh_device);

    const experimental::DFBSpecName INPUT_DFB{"input_dfb"};
    const experimental::DFBSpecName OUTPUT_DFB{"output_dfb"};
    const experimental::KernelSpecName READER{"reader"};
    const experimental::KernelSpecName WRITER{"writer"};
    const experimental::KernelSpecName COMPUTE{"compute"};

    const Tile tile = Tile::from_face_grid(1, num_faces, {face_r_dim, tt::constants::FACE_WIDTH});
    experimental::DataflowBufferSpec input_dfb_spec{
        .unique_id = INPUT_DFB,
        .entry_size = tile_bytes,
        .num_entries = 2 * section_ntiles,
        .data_format_metadata = data_format,
        .tile_format_metadata = tile,
    };
    experimental::DataflowBufferSpec output_dfb_spec{
        .unique_id = OUTPUT_DFB,
        .entry_size = tile_bytes,
        .num_entries = 2 * section_ntiles,
        .data_format_metadata = data_format,
        .tile_format_metadata = tile,
    };

    experimental::KernelSpec reader_spec{
        .unique_id = READER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_reader_unary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ProducerOf(INPUT_DFB, "out")},
        .runtime_arg_schema = {.runtime_arg_names = {"src_addr", "src_bank_id", "num_tiles", "dram_page_stride"}},
        .hw_config = experimental::DataMovementHardwareConfig{},
    };
    experimental::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_writer_unary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ConsumerOf(OUTPUT_DFB, "in")},
        .runtime_arg_schema = {.runtime_arg_names = {"dst_addr", "dst_bank_id", "num_tiles", "dram_page_stride"}},
        .hw_config = experimental::DataMovementHardwareConfig{},
    };

    // eltwise_sfpu_2_0.cpp in section mode: SECTION_NTILES copy_tile calls into DEST 0..n-1 per tile_regs_acquire,
    // packed back from the same indices; no SFPU op.
    experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_sfpu_2_0.cpp",
        .num_threads = 1,
        .compiler_options =
            {.defines = {{"SECTION_NTILES", std::to_string(section_ntiles)}, {"SECTION_USE_COPY_TILE", "1"}}},
        .dfb_bindings =
            {{
                 .dfb_spec_name = INPUT_DFB,
                 .accessor_name = "in",
                 .endpoint_type = experimental::DFBEndpointType::CONSUMER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = OUTPUT_DFB,
                 .accessor_name = "out",
                 .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             }},
        .compile_time_args = {{"per_core_block_cnt", num_sections}, {"per_core_block_size", 1u}},
        .hw_config =
            experimental::ComputeHardwareConfig{
                .enable_32_bit_dest = true,
                .double_buffer_dest = !dst_full_sync_en,
                .unpack_modes = {{INPUT_DFB, tt::tt_metal::UnpackMode::UnpackToDest}},
            },
    };

    experimental::WorkUnitSpec wu{
        .name = "main",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = node,
    };
    experimental::ProgramSpec spec{
        .name = "quasar_tiny_tile_to_dest",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {input_dfb_spec, output_dfb_spec},
        .work_units = {wu},
    };
    Program program = experimental::MakeProgramFromSpec(mesh_device, spec);

    // Unique per datum. Float32: small integers, exact. Int32: the upper halfword set too (bit 31 clear), so a DEST
    // path that dropped the high bits would fail the compare.
    std::vector<std::uint32_t> src_vec(buffer_bytes / sizeof(std::uint32_t));
    for (std::uint32_t i = 0; i < src_vec.size(); i++) {
        src_vec[i] = data_format == tt::DataFormat::Float32 ? std::bit_cast<std::uint32_t>(static_cast<float>(i))
                                                            : i * 0x00010001u;
    }
    distributed::EnqueueWriteMeshBuffer(cq, src_dram_buffer, src_vec, /*blocking=*/true);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node,
                {{"src_addr", static_cast<std::uint32_t>(src_dram_buffer->address())},
                 {"src_bank_id", 0u},
                 {"num_tiles", num_tiles},
                 {"dram_page_stride", tile_bytes}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node,
                {{"dst_addr", static_cast<std::uint32_t>(dst_dram_buffer->address())},
                 {"dst_bank_id", 0u},
                 {"num_tiles", num_tiles},
                 {"dram_page_stride", tile_bytes}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE},
    };
    experimental::SetProgramRunArgs(program, params);

    LaunchProgram(mesh_device, std::move(program));

    std::vector<std::uint32_t> result_vec;
    distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_dram_buffer, /*blocking=*/true);

    ASSERT_EQ(result_vec.size(), src_vec.size());
    const std::uint32_t words_per_face = face_r_dim * tt::constants::FACE_WIDTH;
    for (std::size_t i = 0; i < src_vec.size(); i++) {
        if (result_vec[i] != src_vec[i]) {
            const std::uint32_t tile = i / words_per_tile;
            const std::uint32_t face = (i % words_per_tile) / words_per_face;
            const std::uint32_t row = ((i % words_per_tile) % words_per_face) / tt::constants::FACE_WIDTH;
            FAIL() << "first mismatch at word " << i << ": tile " << tile << " (section " << tile / section_ntiles
                   << ", dst " << tile % section_ntiles << "), face " << face << ", row " << row << ": expected "
                   << src_vec[i] << " got " << result_vec[i];
        }
    }
}

static void run_tiny_tile_copy_to_dest_sweep(
    distributed::MeshDevice& mesh_device, tt::DataFormat data_format, std::uint32_t face_r_dim) {
    for (bool dst_full_sync_en : {false, true}) {
        const std::uint32_t capacity = dest_section_tiles(2, face_r_dim, dst_full_sync_en);
        for (std::uint32_t section_ntiles : {4u, capacity}) {
            log_info(
                tt::LogTest,
                "tiny-tile copy to DEST: {} {}x32 {} section_ntiles={} (capacity {})",
                data_format == tt::DataFormat::Float32 ? "Float32" : "Int32",
                face_r_dim,
                dst_full_sync_en ? "SyncFull" : "SyncHalf",
                section_ntiles,
                capacity);
            run_tiny_tile_copy_to_dest(
                mesh_device, data_format, face_r_dim, dst_full_sync_en, section_ntiles, /*num_sections=*/3);
        }
    }
}

}  // namespace unit_tests::compute::quasar_tiny_tile_to_dest

TEST_F(LLKQuasarMeshDeviceSingleCardFixture, QuasarCopyToDestTile16x32Float32) {
    unit_tests::compute::quasar_tiny_tile_to_dest::run_tiny_tile_copy_to_dest_sweep(
        this->device(), tt::DataFormat::Float32, /*face_r_dim=*/16);
}

TEST_F(LLKQuasarMeshDeviceSingleCardFixture, QuasarCopyToDestTile1x32Int32) {
    unit_tests::compute::quasar_tiny_tile_to_dest::run_tiny_tile_copy_to_dest_sweep(
        this->device(), tt::DataFormat::Int32, /*face_r_dim=*/1);
}

}  // namespace tt::tt_metal
