// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

/**
 * Block Variant Tests: reduce
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
#include <chrono>
#include <cstdint>
#include <array>
#include <map>
#include <string>
#include <vector>

using namespace tt;
using namespace tt::tt_metal;

namespace {

/**
 * Run reduce block test
 *
 * @param device Device to run on
 * @param Ht Block height in tiles
 * @param Wt Block width in tiles
 * @param num_blocks Number of blocks to process
 * @param data_format Data format for tiles
 */
void run_reduce_block_test(
    Device* device,
    uint32_t Ht,
    uint32_t Wt,
    uint32_t num_blocks = 10,
    tt::DataFormat data_format = tt::DataFormat::Float16_b) {
    // Validate block size
    ASSERT_LE(Ht * Wt, 16) << "Block size exceeds DEST capacity (max 16 tiles)";
    ASSERT_GT(Ht, 0) << "Block height must be > 0";
    ASSERT_GT(Wt, 0) << "Block width must be > 0";

    log_info(LogTest, "Testing reduce with Ht={}, Wt={}, blocks={}", Ht, Wt, num_blocks);

    // Create programs
    Program program_ref = CreateProgram();
    Program program_test = CreateProgram();

    CoreCoord core = {0, 0};

    uint32_t single_tile_size = 2 * 1024;
    uint32_t tiles_per_block = Ht * Wt;
    uint32_t total_tiles = num_blocks * tiles_per_block;
    uint32_t input_buffer_size = single_tile_size * total_tiles;
    uint32_t output_buffer_size = single_tile_size * num_blocks;  // One output tile per block

    tt::DataFormat cb_data_format = data_format;

    // Create input and output buffers
    std::shared_ptr<Buffer> src_dram_buffer =
        CreateBuffer(InterleavedBufferConfig{device, input_buffer_size, single_tile_size, BufferType::DRAM});

    std::shared_ptr<Buffer> dst_dram_buffer_ref =
        CreateBuffer(InterleavedBufferConfig{device, output_buffer_size, single_tile_size, BufferType::DRAM});

    std::shared_ptr<Buffer> dst_dram_buffer_test =
        CreateBuffer(InterleavedBufferConfig{device, output_buffer_size, single_tile_size, BufferType::DRAM});

    // Configure circular buffers for reference program
    uint32_t src_cb_index = 0;
    uint32_t dst_cb_index = 16;

    CircularBufferConfig cb_src_config =
        CircularBufferConfig(tiles_per_block * single_tile_size, {{src_cb_index, cb_data_format}})
            .set_page_size(src_cb_index, single_tile_size);
    auto cb_src_ref = CreateCircularBuffer(program_ref, core, cb_src_config);

    CircularBufferConfig cb_dst_config = CircularBufferConfig(single_tile_size, {{dst_cb_index, cb_data_format}})
                                             .set_page_size(dst_cb_index, single_tile_size);
    auto cb_dst_ref = CreateCircularBuffer(program_ref, core, cb_dst_config);

    // Configure circular buffers for test program
    auto cb_src_test = CreateCircularBuffer(program_test, core, cb_src_config);
    auto cb_dst_test = CreateCircularBuffer(program_test, core, cb_dst_config);

    // Create reference kernel (tile-by-tile)
    std::string ref_kernel_file = "tests/tt_metal/tt_metal/kernels/compute_reduce_tiles.cpp";
    std::vector<uint32_t> compute_args_ref = {Ht, Wt, num_blocks};
    auto compute_kernel_ref = CreateKernel(
        program_ref,
        ref_kernel_file,
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_args_ref,
            .defines = {}});

    // Create test kernel (block operation)
    std::string test_kernel_file = "tests/tt_metal/tt_metal/kernels/compute_reduce_block.cpp";
    std::vector<uint32_t> compute_args_test = {Ht, Wt, num_blocks};
    auto compute_kernel_test = CreateKernel(
        program_test,
        test_kernel_file,
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_args_test,
            .defines = {}});

    // Create data reader kernels
    std::vector<uint32_t> reader_args = {src_dram_buffer->address(), total_tiles, tiles_per_block};

    auto reader_kernel_ref = CreateKernel(
        program_ref,
        "tests/tt_metal/tt_metal/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = reader_args});

    auto reader_kernel_test = CreateKernel(
        program_test,
        "tests/tt_metal/tt_metal/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = reader_args});

    // Create data writer kernels
    std::vector<uint32_t> writer_args_ref = {dst_dram_buffer_ref->address(), num_blocks};

    std::vector<uint32_t> writer_args_test = {dst_dram_buffer_test->address(), num_blocks};

    auto writer_kernel_ref = CreateKernel(
        program_ref,
        "tests/tt_metal/tt_metal/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = writer_args_ref});

    auto writer_kernel_test = CreateKernel(
        program_test,
        "tests/tt_metal/tt_metal/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_args_test});

    // Generate input data
    std::vector<bfloat16> input_data;
    input_data.reserve(total_tiles * 512);  // 512 values per tile (32x32/2 for bfloat16)

    // Generate random data with some structure to ensure meaningful reduction
    std::srand(0);  // Fixed seed for reproducibility
    for (uint32_t i = 0; i < total_tiles * 512; i++) {
        float val = static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f;  // Range [-1, 1]
        input_data.push_back(bfloat16(val));
    }

    // Write input data to device
    EnqueueWriteBuffer(device->command_queue(), src_dram_buffer, input_data, false);

    // Run reference program
    EnqueueProgram(device->command_queue(), program_ref, false);
    Finish(device->command_queue());

    // Run test program
    EnqueueProgram(device->command_queue(), program_test, false);
    Finish(device->command_queue());

    // Read results
    std::vector<bfloat16> output_ref(num_blocks * 512);
    std::vector<bfloat16> output_test(num_blocks * 512);

    EnqueueReadBuffer(device->command_queue(), dst_dram_buffer_ref, output_ref, true);
    EnqueueReadBuffer(device->command_queue(), dst_dram_buffer_test, output_test, true);

    // Convert to float for comparison
    std::vector<float> output_ref_float, output_test_float;
    for (const auto& val : output_ref) {
        output_ref_float.push_back(val.to_float());
    }
    for (const auto& val : output_test) {
        output_test_float.push_back(val.to_float());
    }

    // Calculate PCC (Pearson Correlation Coefficient)
    auto pcc = tt::test_utils::get_pcc(output_ref_float, output_test_float);

    log_info(LogTest, "PCC: {}", pcc);
    EXPECT_GE(pcc, 0.9999) << "PCC too low: " << pcc;

    // Also validate that outputs are not all zeros (sanity check)
    bool has_non_zero_ref = false, has_non_zero_test = false;
    for (const auto& val : output_ref_float) {
        if (std::abs(val) > 1e-6f) {
            has_non_zero_ref = true;
            break;
        }
    }
    for (const auto& val : output_test_float) {
        if (std::abs(val) > 1e-6f) {
            has_non_zero_test = true;
            break;
        }
    }

    EXPECT_TRUE(has_non_zero_ref) << "Reference output is all zeros";
    EXPECT_TRUE(has_non_zero_test) << "Test output is all zeros";

    log_info(LogTest, "Reduce block test passed for {}x{} with {} blocks", Ht, Wt, num_blocks);
}

}  // namespace

// =============================================================================
// Test Cases
// =============================================================================

class BlockVariantsFixture : public tt::tt_metal::CommandQueueFixture {};

TEST_F(BlockVariantsFixture, ReduceBlock_1x1) { run_reduce_block_test(this->device_.get(), 1, 1); }

TEST_F(BlockVariantsFixture, ReduceBlock_1x2) { run_reduce_block_test(this->device_.get(), 1, 2); }

TEST_F(BlockVariantsFixture, ReduceBlock_1x4) { run_reduce_block_test(this->device_.get(), 1, 4); }

TEST_F(BlockVariantsFixture, ReduceBlock_1x8) { run_reduce_block_test(this->device_.get(), 1, 8); }

TEST_F(BlockVariantsFixture, ReduceBlock_1x16) { run_reduce_block_test(this->device_.get(), 1, 16); }

TEST_F(BlockVariantsFixture, ReduceBlock_2x1) { run_reduce_block_test(this->device_.get(), 2, 1); }

TEST_F(BlockVariantsFixture, ReduceBlock_2x2) { run_reduce_block_test(this->device_.get(), 2, 2); }

TEST_F(BlockVariantsFixture, ReduceBlock_2x4) { run_reduce_block_test(this->device_.get(), 2, 4); }

TEST_F(BlockVariantsFixture, ReduceBlock_2x8) { run_reduce_block_test(this->device_.get(), 2, 8); }

TEST_F(BlockVariantsFixture, ReduceBlock_4x1) { run_reduce_block_test(this->device_.get(), 4, 1); }

TEST_F(BlockVariantsFixture, ReduceBlock_4x2) { run_reduce_block_test(this->device_.get(), 4, 2); }

TEST_F(BlockVariantsFixture, ReduceBlock_4x4) { run_reduce_block_test(this->device_.get(), 4, 4); }

TEST_F(BlockVariantsFixture, ReduceBlock_8x1) { run_reduce_block_test(this->device_.get(), 8, 1); }

TEST_F(BlockVariantsFixture, ReduceBlock_8x2) { run_reduce_block_test(this->device_.get(), 8, 2); }

TEST_F(BlockVariantsFixture, ReduceBlock_16x1) { run_reduce_block_test(this->device_.get(), 16, 1); }

// =============================================================================
// Stress Tests
// =============================================================================

TEST_F(BlockVariantsFixture, ReduceBlock_Stress_ManyBlocks) {
    // Process many blocks to test stability
    run_reduce_block_test(this->device_.get(), 4, 4, 1000);
}

TEST_F(BlockVariantsFixture, ReduceBlock_Stress_MaxCapacity) {
    // Use maximum DEST capacity
    run_reduce_block_test(this->device_.get(), 16, 1, 100);
}
