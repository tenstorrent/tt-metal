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
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <cstdint>
#include <vector>
#include <random>

using namespace tt;
using namespace tt::tt_metal;
using std::vector;

namespace {

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
    Device* device,
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
    uint32_t src1_cb_index = tt::CBIndex::c_1;
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

    // Generate random input data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    vector<uint32_t> input_vec;
    input_vec.reserve(input_buffer_size / sizeof(uint32_t));

    for (uint32_t i = 0; i < input_tiles; i++) {
        auto tile_data = generate_uniform_random_vector<uint32_t>(-1.0f, 1.0f, single_tile_size / sizeof(uint32_t));
        input_vec.insert(input_vec.end(), tile_data.begin(), tile_data.end());
    }

    // Write input data to device
    WriteBuffer(device->command_queue(), input_dram_buffer, input_vec);

    // Set runtime args for reference program
    SetRuntimeArgs(
        program_ref,
        reader_kernel_ref,
        core,
        {input_dram_buffer->address(),
         (std::uint32_t)input_dram_buffer->noc_coordinates().x,
         (std::uint32_t)input_dram_buffer->noc_coordinates().y,
         input_tiles});

    SetRuntimeArgs(
        program_ref,
        writer_kernel_ref,
        core,
        {output_dram_buffer_ref->address(),
         (std::uint32_t)output_dram_buffer_ref->noc_coordinates().x,
         (std::uint32_t)output_dram_buffer_ref->noc_coordinates().y,
         output_tiles});

    // Set runtime args for test program
    SetRuntimeArgs(
        program_test,
        reader_kernel_test,
        core,
        {input_dram_buffer->address(),
         (std::uint32_t)input_dram_buffer->noc_coordinates().x,
         (std::uint32_t)input_dram_buffer->noc_coordinates().y,
         input_tiles});

    SetRuntimeArgs(
        program_test,
        writer_kernel_test,
        core,
        {output_dram_buffer_test->address(),
         (std::uint32_t)output_dram_buffer_test->noc_coordinates().x,
         (std::uint32_t)output_dram_buffer_test->noc_coordinates().y,
         output_tiles});

    // Execute programs
    EnqueueProgram(device->command_queue(), program_ref, false);
    Finish(device->command_queue());

    EnqueueProgram(device->command_queue(), program_test, false);
    Finish(device->command_queue());

    // Read results
    vector<uint32_t> result_vec_ref;
    ReadBuffer(device->command_queue(), output_dram_buffer_ref, result_vec_ref);

    vector<uint32_t> result_vec_test;
    ReadBuffer(device->command_queue(), output_dram_buffer_test, result_vec_test);

    // Convert to bfloat16 for comparison
    auto result_bfp16_ref = unpack_uint32_vec_into_bfloat16_vec(result_vec_ref);
    auto result_bfp16_test = unpack_uint32_vec_into_bfloat16_vec(result_vec_test);

    // Compare results
    ASSERT_EQ(result_bfp16_ref.size(), result_bfp16_test.size());

    // Calculate PCC
    float pcc = get_pcc(result_bfp16_ref, result_bfp16_test);
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
    float pcc_golden = get_pcc(result_bfp16_test, golden_result);
    log_info(LogTest, "PCC vs Golden: {}", pcc_golden);

    EXPECT_GE(pcc_golden, 0.9999f) << "PCC between test result and golden reference is too low";
}

}  // namespace

// =============================================================================
// Test Cases
// =============================================================================

class BlockVariantsFixture : public tt::tt_metal::CommandQueueFixture {};

TEST_F(BlockVariantsFixture, TransposeBlock_1x1) { run_transpose_block_test(this->device_.get(), 1, 1); }

TEST_F(BlockVariantsFixture, TransposeBlock_1x2) { run_transpose_block_test(this->device_.get(), 1, 2); }

TEST_F(BlockVariantsFixture, TransposeBlock_1x4) { run_transpose_block_test(this->device_.get(), 1, 4); }

TEST_F(BlockVariantsFixture, TransposeBlock_1x8) { run_transpose_block_test(this->device_.get(), 1, 8); }

TEST_F(BlockVariantsFixture, TransposeBlock_1x16) { run_transpose_block_test(this->device_.get(), 1, 16); }

TEST_F(BlockVariantsFixture, TransposeBlock_2x1) { run_transpose_block_test(this->device_.get(), 2, 1); }

TEST_F(BlockVariantsFixture, TransposeBlock_2x2) { run_transpose_block_test(this->device_.get(), 2, 2); }

TEST_F(BlockVariantsFixture, TransposeBlock_2x4) { run_transpose_block_test(this->device_.get(), 2, 4); }

TEST_F(BlockVariantsFixture, TransposeBlock_2x8) { run_transpose_block_test(this->device_.get(), 2, 8); }

TEST_F(BlockVariantsFixture, TransposeBlock_4x1) { run_transpose_block_test(this->device_.get(), 4, 1); }

TEST_F(BlockVariantsFixture, TransposeBlock_4x2) { run_transpose_block_test(this->device_.get(), 4, 2); }

TEST_F(BlockVariantsFixture, TransposeBlock_4x4) { run_transpose_block_test(this->device_.get(), 4, 4); }

TEST_F(BlockVariantsFixture, TransposeBlock_8x1) { run_transpose_block_test(this->device_.get(), 8, 1); }

TEST_F(BlockVariantsFixture, TransposeBlock_8x2) { run_transpose_block_test(this->device_.get(), 8, 2); }

TEST_F(BlockVariantsFixture, TransposeBlock_16x1) { run_transpose_block_test(this->device_.get(), 16, 1); }

// =============================================================================
// Stress Tests
// =============================================================================

TEST_F(BlockVariantsFixture, TransposeBlock_Stress_ManyBlocks) {
    // Process many blocks to test stability
    run_transpose_block_test(this->device_.get(), 4, 4, 1000);
}

TEST_F(BlockVariantsFixture, TransposeBlock_Stress_MaxCapacity) {
    // Use maximum DEST capacity
    run_transpose_block_test(this->device_.get(), 16, 1, 100);
}
