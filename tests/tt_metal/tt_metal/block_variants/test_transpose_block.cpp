// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

/**
 * Block Variant Tests: transpose
 *
 * Validates that block operations produce identical results to tile-by-tile processing.
 * Each test runs both a reference (tile-by-tile) and test (block) kernel with identical
 * input data and compares outputs using PCC (Pearson Correlation Coefficient).
 */

#include <gtest/gtest.h>
#include "common/command_queue_fixture.hpp"
#include "test_gold_impls.hpp"
#include "block_variants/block_variants_test_utils.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include "tt_metal/test_utils/stimulus.hpp"
#include <cstdint>
#include <vector>
#include <chrono>

using namespace tt;
using namespace tt::tt_metal;
using std::vector;

namespace {

vector<bfloat16> transpose_tile(const vector<bfloat16>& tile) {
    constexpr uint32_t tile_dim = 32;
    vector<bfloat16> transposed(tile.size());
    for (uint32_t r = 0; r < tile_dim; ++r) {
        for (uint32_t c = 0; c < tile_dim; ++c) {
            transposed[c * tile_dim + r] = tile[r * tile_dim + c];
        }
    }
    return transposed;
}

/**
 * Run transpose block test
 *
 * @param device Device to run on
 * @param Ht Block height in tiles
 * @param Wt Block width in tiles
 * @param num_blocks Number of blocks to process
 * @param data_format Data format for tiles
 */
void run_transpose_block_test(
    IDevice* device,
    uint32_t Ht,
    uint32_t Wt,
    uint32_t num_blocks = 10,
    tt::DataFormat data_format = tt::DataFormat::Float16_b) {
    // Validate block size
    ASSERT_LE(Ht * Wt, 16) << "Block size exceeds DEST capacity (max 16 tiles)";
    ASSERT_GT(Ht, 0) << "Block height must be > 0";
    ASSERT_GT(Wt, 0) << "Block width must be > 0";

    log_info(LogTest, "Testing transpose with Ht={}, Wt={}, blocks={}", Ht, Wt, num_blocks);

    // Create programs
    Program program_ref = CreateProgram();
    Program program_test = CreateProgram();

    CoreCoord core = {0, 0};
    uint32_t single_tile_size = 2 * 1024;
    uint32_t input_tiles = num_blocks * Ht * Wt;
    uint32_t output_tiles = num_blocks * Wt * Ht;  // transposed dimensions

    uint32_t input_buffer_size = single_tile_size * input_tiles;
    uint32_t output_buffer_size = single_tile_size * output_tiles;

    // Create DRAM buffers
    auto input_dram_buffer =
        CreateBuffer(InterleavedBufferConfig{device, input_buffer_size, single_tile_size, BufferType::DRAM});

    auto output_dram_buffer_ref =
        CreateBuffer(InterleavedBufferConfig{device, output_buffer_size, single_tile_size, BufferType::DRAM});

    auto output_dram_buffer_test =
        CreateBuffer(InterleavedBufferConfig{device, output_buffer_size, single_tile_size, BufferType::DRAM});

    // Setup circular buffers for reference program
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t out_cb_index = tt::CBIndex::c_2;

    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(Ht * Wt * single_tile_size, {{src0_cb_index, data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0_ref = CreateCircularBuffer(program_ref, core, cb_src0_config);
    auto cb_src0_test = CreateCircularBuffer(program_test, core, cb_src0_config);

    CircularBufferConfig cb_out_config = CircularBufferConfig(Wt * Ht * single_tile_size, {{out_cb_index, data_format}})
                                             .set_page_size(out_cb_index, single_tile_size);
    auto cb_out_ref = CreateCircularBuffer(program_ref, core, cb_out_config);
    auto cb_out_test = CreateCircularBuffer(program_test, core, cb_out_config);

    (void)cb_src0_ref;
    (void)cb_src0_test;
    (void)cb_out_ref;
    (void)cb_out_test;

    // Create reference kernel (tile-by-tile)
    auto reader_kernel_ref = CreateKernel(
        program_ref,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    auto writer_kernel_ref = CreateKernel(
        program_ref,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    vector<uint32_t> compute_args_ref = {Ht, Wt, num_blocks};
    auto compute_kernel_ref = CreateKernel(
        program_ref,
        "tests/tt_metal/tt_metal/test_kernels/compute/transpose_wh_tiles.cpp",
        core,
        ComputeConfig{.compile_args = compute_args_ref});
    (void)compute_kernel_ref;

    // Create test kernel (block operation)
    auto reader_kernel_test = CreateKernel(
        program_test,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    auto writer_kernel_test = CreateKernel(
        program_test,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    vector<uint32_t> compute_args_test = {Ht, Wt, num_blocks};
    auto compute_kernel_test = CreateKernel(
        program_test,
        "tests/tt_metal/tt_metal/test_kernels/compute/transpose_wh_block.cpp",
        core,
        ComputeConfig{.compile_args = compute_args_test});
    (void)compute_kernel_test;

    // Generate random input data
    uint32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
    uint32_t num_bfloat16 = input_buffer_size / sizeof(bfloat16);
    auto input_vec =
        tt::test_utils::generate_packed_uniform_random_vector<uint32_t, bfloat16>(-1.0f, 1.0f, num_bfloat16, seed);

    // Write input data to device
    detail::WriteToBuffer(input_dram_buffer, input_vec);

    // Set runtime args for reference program
    SetRuntimeArgs(program_ref, reader_kernel_ref, core, {input_dram_buffer->address(), 0, input_tiles});

    SetRuntimeArgs(program_ref, writer_kernel_ref, core, {output_dram_buffer_ref->address(), 0, output_tiles});

    // Set runtime args for test program
    SetRuntimeArgs(program_test, reader_kernel_test, core, {input_dram_buffer->address(), 0, input_tiles});

    SetRuntimeArgs(program_test, writer_kernel_test, core, {output_dram_buffer_test->address(), 0, output_tiles});

    // Execute programs
    detail::LaunchProgram(device, program_ref);
    detail::LaunchProgram(device, program_test);

    // Read results
    vector<uint32_t> result_vec_ref;
    detail::ReadFromBuffer(output_dram_buffer_ref, result_vec_ref);

    vector<uint32_t> result_vec_test;
    detail::ReadFromBuffer(output_dram_buffer_test, result_vec_test);

    // Convert to bfloat16 for comparison
    auto result_bfp16_ref = unpack_uint32_vec_into_bfloat16_vec(result_vec_ref);
    auto result_bfp16_test = unpack_uint32_vec_into_bfloat16_vec(result_vec_test);

    // Compare results
    ASSERT_EQ(result_bfp16_ref.size(), result_bfp16_test.size());

    // Calculate PCC
    float pcc = block_variants::compute_pcc(result_bfp16_ref, result_bfp16_test);
    log_info(LogTest, "PCC: {}", pcc);

    EXPECT_GE(pcc, 0.9999f) << "PCC between reference and test results is too low";

    // Generate golden reference for additional validation
    auto input_bfp16 = unpack_uint32_vec_into_bfloat16_vec(input_vec);

    // Create golden transpose result
    vector<bfloat16> golden_result;
    golden_result.reserve(result_bfp16_ref.size());

    const uint32_t tile_height = 32;
    const uint32_t tile_width = 32;

    for (uint32_t block = 0; block < num_blocks; block++) {
        vector<vector<bfloat16>> block_tiles;

        // Extract tiles for this block
        for (uint32_t h = 0; h < Ht; h++) {
            for (uint32_t w = 0; w < Wt; w++) {
                uint32_t tile_idx = block * Ht * Wt + h * Wt + w;
                uint32_t tile_start = tile_idx * tile_height * tile_width;
                vector<bfloat16> tile(
                    input_bfp16.begin() + tile_start, input_bfp16.begin() + tile_start + tile_height * tile_width);
                block_tiles.push_back(tile);
            }
        }

        // Transpose the block and add to golden result
        for (uint32_t w = 0; w < Wt; w++) {
            for (uint32_t h = 0; h < Ht; h++) {
                uint32_t src_tile_idx = h * Wt + w;
                auto transposed_tile = transpose_tile(block_tiles[src_tile_idx]);
                golden_result.insert(golden_result.end(), transposed_tile.begin(), transposed_tile.end());
            }
        }
    }

    // Compare with golden reference
    float pcc_golden = block_variants::compute_pcc(result_bfp16_test, golden_result);
    log_info(LogTest, "PCC vs Golden: {}", pcc_golden);

    EXPECT_GE(pcc_golden, 0.9999f) << "PCC between test result and golden reference is too low";
}

}  // namespace

// =============================================================================
// Test Cases
// =============================================================================

class BlockVariantsFixture : public tt::tt_metal::UnitMeshCQFixture {};

TEST_F(BlockVariantsFixture, TransposeBlock_1x1) {
    for (const auto& mesh_device : devices_) {
        run_transpose_block_test(mesh_device->get_devices().at(0), 1, 1);
    }
}

TEST_F(BlockVariantsFixture, TransposeBlock_1x2) {
    for (const auto& mesh_device : devices_) {
        run_transpose_block_test(mesh_device->get_devices().at(0), 1, 2);
    }
}

TEST_F(BlockVariantsFixture, TransposeBlock_1x4) {
    for (const auto& mesh_device : devices_) {
        run_transpose_block_test(mesh_device->get_devices().at(0), 1, 4);
    }
}

TEST_F(BlockVariantsFixture, TransposeBlock_1x8) {
    for (const auto& mesh_device : devices_) {
        run_transpose_block_test(mesh_device->get_devices().at(0), 1, 8);
    }
}

TEST_F(BlockVariantsFixture, TransposeBlock_1x16) {
    for (const auto& mesh_device : devices_) {
        run_transpose_block_test(mesh_device->get_devices().at(0), 1, 16);
    }
}

TEST_F(BlockVariantsFixture, TransposeBlock_2x1) {
    for (const auto& mesh_device : devices_) {
        run_transpose_block_test(mesh_device->get_devices().at(0), 2, 1);
    }
}

TEST_F(BlockVariantsFixture, TransposeBlock_2x2) {
    for (const auto& mesh_device : devices_) {
        run_transpose_block_test(mesh_device->get_devices().at(0), 2, 2);
    }
}

TEST_F(BlockVariantsFixture, TransposeBlock_2x4) {
    for (const auto& mesh_device : devices_) {
        run_transpose_block_test(mesh_device->get_devices().at(0), 2, 4);
    }
}

TEST_F(BlockVariantsFixture, TransposeBlock_2x8) {
    for (const auto& mesh_device : devices_) {
        run_transpose_block_test(mesh_device->get_devices().at(0), 2, 8);
    }
}

TEST_F(BlockVariantsFixture, TransposeBlock_4x1) {
    for (const auto& mesh_device : devices_) {
        run_transpose_block_test(mesh_device->get_devices().at(0), 4, 1);
    }
}

TEST_F(BlockVariantsFixture, TransposeBlock_4x2) {
    for (const auto& mesh_device : devices_) {
        run_transpose_block_test(mesh_device->get_devices().at(0), 4, 2);
    }
}

TEST_F(BlockVariantsFixture, TransposeBlock_4x4) {
    for (const auto& mesh_device : devices_) {
        run_transpose_block_test(mesh_device->get_devices().at(0), 4, 4);
    }
}

TEST_F(BlockVariantsFixture, TransposeBlock_8x1) {
    for (const auto& mesh_device : devices_) {
        run_transpose_block_test(mesh_device->get_devices().at(0), 8, 1);
    }
}

TEST_F(BlockVariantsFixture, TransposeBlock_8x2) {
    for (const auto& mesh_device : devices_) {
        run_transpose_block_test(mesh_device->get_devices().at(0), 8, 2);
    }
}

TEST_F(BlockVariantsFixture, TransposeBlock_16x1) {
    for (const auto& mesh_device : devices_) {
        run_transpose_block_test(mesh_device->get_devices().at(0), 16, 1);
    }
}

// =============================================================================
// Stress Tests
// =============================================================================

TEST_F(BlockVariantsFixture, TransposeBlock_Stress_ManyBlocks) {
    // Process many blocks to test stability
    for (const auto& mesh_device : devices_) {
        run_transpose_block_test(mesh_device->get_devices().at(0), 4, 4, 1000);
    }
}

TEST_F(BlockVariantsFixture, TransposeBlock_Stress_MaxCapacity) {
    // Use maximum DEST capacity
    for (const auto& mesh_device : devices_) {
        run_transpose_block_test(mesh_device->get_devices().at(0), 16, 1, 100);
    }
}
