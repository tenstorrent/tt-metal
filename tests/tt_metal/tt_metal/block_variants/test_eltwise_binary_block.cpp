// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

/**
 * Block Variant Tests: eltwise_binary
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
#include "tt_metal/common/bfloat16.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace {

/**
 * Run eltwise_binary block test
 *
 * @param device Device to run on
 * @param Ht Block height in tiles
 * @param Wt Block width in tiles
 * @param num_blocks Number of blocks to process
 * @param data_format Data format for tiles
 */
void run_eltwise_binary_block_test(
    Device* device,
    uint32_t Ht,
    uint32_t Wt,
    uint32_t num_blocks = 10,
    tt::DataFormat data_format = tt::DataFormat::Float16_b) {
    // Validate block size
    ASSERT_LE(Ht * Wt, 16) << "Block size exceeds DEST capacity (max 16 tiles)";
    ASSERT_GT(Ht, 0) << "Block height must be > 0";
    ASSERT_GT(Wt, 0) << "Block width must be > 0";

    log_info(LogTest, "Testing eltwise_binary with Ht={}, Wt={}, blocks={}", Ht, Wt, num_blocks);

    // Setup
    uint32_t single_tile_size = 2 * 1024;
    uint32_t total_tiles = num_blocks * Ht * Wt;
    uint32_t dram_buffer_size = single_tile_size * total_tiles;

    // Create programs
    Program program_ref = CreateProgram();
    Program program_test = CreateProgram();

    CoreCoord core = {0, 0};

    // Create buffers
    auto src0_dram_buffer_ref = CreateBuffer(InterleavedBufferConfig{
        .device = device, .size = dram_buffer_size, .page_size = single_tile_size, .buffer_type = BufferType::DRAM});

    auto src1_dram_buffer_ref = CreateBuffer(InterleavedBufferConfig{
        .device = device, .size = dram_buffer_size, .page_size = single_tile_size, .buffer_type = BufferType::DRAM});

    auto dst_dram_buffer_ref = CreateBuffer(InterleavedBufferConfig{
        .device = device, .size = dram_buffer_size, .page_size = single_tile_size, .buffer_type = BufferType::DRAM});

    auto src0_dram_buffer_test = CreateBuffer(InterleavedBufferConfig{
        .device = device, .size = dram_buffer_size, .page_size = single_tile_size, .buffer_type = BufferType::DRAM});

    auto src1_dram_buffer_test = CreateBuffer(InterleavedBufferConfig{
        .device = device, .size = dram_buffer_size, .page_size = single_tile_size, .buffer_type = BufferType::DRAM});

    auto dst_dram_buffer_test = CreateBuffer(InterleavedBufferConfig{
        .device = device, .size = dram_buffer_size, .page_size = single_tile_size, .buffer_type = BufferType::DRAM});

    // Create circular buffers for reference program
    CircularBufferConfig cb0_config = CircularBufferConfig(Ht * Wt * single_tile_size, {{CB::c_in0, data_format}})
                                          .set_page_size(CB::c_in0, single_tile_size);
    auto cb0_ref = CreateCircularBuffer(program_ref, core, cb0_config);

    CircularBufferConfig cb1_config = CircularBufferConfig(Ht * Wt * single_tile_size, {{CB::c_in1, data_format}})
                                          .set_page_size(CB::c_in1, single_tile_size);
    auto cb1_ref = CreateCircularBuffer(program_ref, core, cb1_config);

    CircularBufferConfig cb2_config = CircularBufferConfig(Ht * Wt * single_tile_size, {{CB::c_out0, data_format}})
                                          .set_page_size(CB::c_out0, single_tile_size);
    auto cb2_ref = CreateCircularBuffer(program_ref, core, cb2_config);

    // Create circular buffers for test program
    auto cb0_test = CreateCircularBuffer(program_test, core, cb0_config);
    auto cb1_test = CreateCircularBuffer(program_test, core, cb1_config);
    auto cb2_test = CreateCircularBuffer(program_test, core, cb2_config);

    // Create reference kernel (tile-by-tile)
    std::string ref_kernel_source = R"(
        #include "compute_kernel_api/eltwise_binary.h"
        #include "compute_kernel_api/tile_move_copy.h"

        namespace NAMESPACE {
        void MAIN {
            constexpr uint32_t Ht = get_compile_time_arg_val(0);
            constexpr uint32_t Wt = get_compile_time_arg_val(1);
            constexpr uint32_t num_blocks = get_compile_time_arg_val(2);

            binary_op_init_common(CB::c_in0, CB::c_in1);

            for (uint32_t block = 0; block < num_blocks; ++block) {
                cb_wait_front(CB::c_in0, Ht * Wt);
                cb_wait_front(CB::c_in1, Ht * Wt);
                cb_reserve_back(CB::c_out0, Ht * Wt);

                for (uint32_t h = 0; h < Ht; ++h) {
                    for (uint32_t w = 0; w < Wt; ++w) {
                        uint32_t tile_idx = h * Wt + w;
                        acquire_dst(tt::DstMode::Half);
                        add_tiles(CB::c_in0, CB::c_in1, tile_idx, tile_idx, tile_idx);
                        pack_tile(tile_idx, CB::c_out0);
                        release_dst(tt::DstMode::Half);
                    }
                }

                cb_push_back(CB::c_out0, Ht * Wt);
                cb_pop_front(CB::c_in0, Ht * Wt);
                cb_pop_front(CB::c_in1, Ht * Wt);
            }
        }
        }
    )";

    // Create test kernel (block operations)
    std::string test_kernel_source = R"(
        #include "compute_kernel_api/eltwise_binary.h"
        #include "compute_kernel_api/tile_move_copy.h"

        namespace NAMESPACE {
        void MAIN {
            constexpr uint32_t Ht = get_compile_time_arg_val(0);
            constexpr uint32_t Wt = get_compile_time_arg_val(1);
            constexpr uint32_t num_blocks = get_compile_time_arg_val(2);

            binary_op_init_common(CB::c_in0, CB::c_in1);

            for (uint32_t block = 0; block < num_blocks; ++block) {
                cb_wait_front(CB::c_in0, Ht * Wt);
                cb_wait_front(CB::c_in1, Ht * Wt);
                cb_reserve_back(CB::c_out0, Ht * Wt);

                acquire_dst(tt::DstMode::Half);
                add_tiles_block(CB::c_in0, CB::c_in1, 0, 0, 0, Ht, Wt);
                pack_tiles_block(0, CB::c_out0, Ht, Wt);
                release_dst(tt::DstMode::Half);

                cb_push_back(CB::c_out0, Ht * Wt);
                cb_pop_front(CB::c_in0, Ht * Wt);
                cb_pop_front(CB::c_in1, Ht * Wt);
            }
        }
        }
    )";

    // Write kernel files
    std::string ref_kernel_file = "eltwise_binary_ref_kernel.cpp";
    std::string test_kernel_file = "eltwise_binary_test_kernel.cpp";

    std::ofstream ref_file(ref_kernel_file);
    ref_file << ref_kernel_source;
    ref_file.close();

    std::ofstream test_file(test_kernel_file);
    test_file << test_kernel_source;
    test_file.close();

    // Create kernels
    auto ref_compute_kernel =
        CreateKernel(program_ref, ref_kernel_file, core, ComputeConfig{.compile_args = {Ht, Wt, num_blocks}});

    auto test_compute_kernel =
        CreateKernel(program_test, test_kernel_file, core, ComputeConfig{.compile_args = {Ht, Wt, num_blocks}});

    // Create reader/writer kernels for reference
    auto ref_unary_reader_kernel = CreateKernel(
        program_ref,
        "tt_metal/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = {(uint32_t)CB::c_in0}});

    auto ref_unary_reader_kernel2 = CreateKernel(
        program_ref,
        "tt_metal/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {(uint32_t)CB::c_in1}});

    auto ref_unary_writer_kernel = CreateKernel(
        program_ref,
        "tt_metal/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = {(uint32_t)CB::c_out0}});

    // Create reader/writer kernels for test
    auto test_unary_reader_kernel = CreateKernel(
        program_test,
        "tt_metal/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = {(uint32_t)CB::c_in0}});

    auto test_unary_reader_kernel2 = CreateKernel(
        program_test,
        "tt_metal/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {(uint32_t)CB::c_in1}});

    auto test_unary_writer_kernel = CreateKernel(
        program_test,
        "tt_metal/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = {(uint32_t)CB::c_out0}});

    // Generate random input data
    std::vector<bfloat16> src0_vec = create_random_vector_of_bfloat16(
        dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
    std::vector<bfloat16> src1_vec = create_random_vector_of_bfloat16(
        dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count() + 1);

    // Write input data to device
    EnqueueWriteBuffer(device->command_queue(), src0_dram_buffer_ref, src0_vec, false);
    EnqueueWriteBuffer(device->command_queue(), src1_dram_buffer_ref, src1_vec, false);
    EnqueueWriteBuffer(device->command_queue(), src0_dram_buffer_test, src0_vec, false);
    EnqueueWriteBuffer(device->command_queue(), src1_dram_buffer_test, src1_vec, false);

    // Set runtime args for reference program
    SetRuntimeArgs(program_ref, ref_unary_reader_kernel, core, {src0_dram_buffer_ref->address(), total_tiles, 0});

    SetRuntimeArgs(program_ref, ref_unary_reader_kernel2, core, {src1_dram_buffer_ref->address(), total_tiles, 0});

    SetRuntimeArgs(program_ref, ref_unary_writer_kernel, core, {dst_dram_buffer_ref->address(), total_tiles, 0});

    // Set runtime args for test program
    SetRuntimeArgs(program_test, test_unary_reader_kernel, core, {src0_dram_buffer_test->address(), total_tiles, 0});

    SetRuntimeArgs(program_test, test_unary_reader_kernel2, core, {src1_dram_buffer_test->address(), total_tiles, 0});

    SetRuntimeArgs(program_test, test_unary_writer_kernel, core, {dst_dram_buffer_test->address(), total_tiles, 0});

    // Execute programs
    EnqueueProgram(device->command_queue(), program_ref, false);
    EnqueueProgram(device->command_queue(), program_test, false);
    Finish(device->command_queue());

    // Read results
    std::vector<bfloat16> result_ref_vec;
    EnqueueReadBuffer(device->command_queue(), dst_dram_buffer_ref, result_ref_vec, true);

    std::vector<bfloat16> result_test_vec;
    EnqueueReadBuffer(device->command_queue(), dst_dram_buffer_test, result_test_vec, true);

    // Compare results
    auto [pcc_passed, pcc_value] = check_pcc(result_ref_vec, result_test_vec, 0.9999f);
    log_info(LogTest, "PCC value: {}", pcc_value);

    EXPECT_TRUE(pcc_passed) << "PCC comparison failed. PCC value: " << pcc_value << " (required: >= 0.9999)";

    // Additional validation against golden reference
    std::vector<bfloat16> golden_vec(total_tiles * 512);
    for (uint32_t i = 0; i < total_tiles * 512; ++i) {
        golden_vec[i] = src0_vec[i] + src1_vec[i];
    }

    auto [golden_pcc_passed, golden_pcc_value] = check_pcc(golden_vec, result_test_vec, 0.9999f);
    log_info(LogTest, "Golden PCC value: {}", golden_pcc_value);

    EXPECT_TRUE(golden_pcc_passed) << "Golden PCC comparison failed. PCC value: " << golden_pcc_value
                                   << " (required: >= 0.9999)";

    // Cleanup
    std::remove(ref_kernel_file.c_str());
    std::remove(test_kernel_file.c_str());
}

}  // namespace

// =============================================================================
// Test Cases
// =============================================================================

class BlockVariantsFixture : public tt::tt_metal::CommandQueueFixture {};

TEST_F(BlockVariantsFixture, EltwiseBinaryBlock_1x1) { run_eltwise_binary_block_test(this->device_.get(), 1, 1); }

TEST_F(BlockVariantsFixture, EltwiseBinaryBlock_1x2) { run_eltwise_binary_block_test(this->device_.get(), 1, 2); }

TEST_F(BlockVariantsFixture, EltwiseBinaryBlock_1x4) { run_eltwise_binary_block_test(this->device_.get(), 1, 4); }

TEST_F(BlockVariantsFixture, EltwiseBinaryBlock_1x8) { run_eltwise_binary_block_test(this->device_.get(), 1, 8); }

TEST_F(BlockVariantsFixture, EltwiseBinaryBlock_1x16) { run_eltwise_binary_block_test(this->device_.get(), 1, 16); }

TEST_F(BlockVariantsFixture, EltwiseBinaryBlock_2x1) { run_eltwise_binary_block_test(this->device_.get(), 2, 1); }

TEST_F(BlockVariantsFixture, EltwiseBinaryBlock_2x2) { run_eltwise_binary_block_test(this->device_.get(), 2, 2); }

TEST_F(BlockVariantsFixture, EltwiseBinaryBlock_2x4) { run_eltwise_binary_block_test(this->device_.get(), 2, 4); }

TEST_F(BlockVariantsFixture, EltwiseBinaryBlock_2x8) { run_eltwise_binary_block_test(this->device_.get(), 2, 8); }

TEST_F(BlockVariantsFixture, EltwiseBinaryBlock_4x1) { run_eltwise_binary_block_test(this->device_.get(), 4, 1); }

TEST_F(BlockVariantsFixture, EltwiseBinaryBlock_4x2) { run_eltwise_binary_block_test(this->device_.get(), 4, 2); }

TEST_F(BlockVariantsFixture, EltwiseBinaryBlock_4x4) { run_eltwise_binary_block_test(this->device_.get(), 4, 4); }

TEST_F(BlockVariantsFixture, EltwiseBinaryBlock_8x1) { run_eltwise_binary_block_test(this->device_.get(), 8, 1); }

TEST_F(BlockVariantsFixture, EltwiseBinaryBlock_8x2) { run_eltwise_binary_block_test(this->device_.get(), 8, 2); }

TEST_F(BlockVariantsFixture, EltwiseBinaryBlock_16x1) { run_eltwise_binary_block_test(this->device_.get(), 16, 1); }

// =============================================================================
// Stress Tests
// =============================================================================

TEST_F(BlockVariantsFixture, EltwiseBinaryBlock_Stress_ManyBlocks) {
    // Process many blocks to test stability
    run_eltwise_binary_block_test(this->device_.get(), 4, 4, 1000);
}

TEST_F(BlockVariantsFixture, EltwiseBinaryBlock_Stress_MaxCapacity) {
    // Use maximum DEST capacity
    run_eltwise_binary_block_test(this->device_.get(), 16, 1, 100);
}
