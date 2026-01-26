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
#include "block_variants/block_variants_test_utils.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
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
    IDevice* device,
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

    (void)cb_src_ref;
    (void)cb_dst_ref;
    (void)cb_src_test;
    (void)cb_dst_test;

    // Create reference kernel (tile-by-tile)
    std::string ref_kernel_file = "tests/tt_metal/tt_metal/block_variants/kernels/compute_reduce_tiles.cpp";
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
    (void)compute_kernel_ref;

    // Create test kernel (block operation)
    std::string test_kernel_file = "tests/tt_metal/tt_metal/block_variants/kernels/compute_reduce_block.cpp";
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
    (void)compute_kernel_test;

    // Create data reader kernels
    auto reader_kernel_ref = CreateKernel(
        program_ref,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    auto reader_kernel_test = CreateKernel(
        program_test,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    // Create data writer kernels
    auto writer_kernel_ref = CreateKernel(
        program_ref,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    auto writer_kernel_test = CreateKernel(
        program_test,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // Generate input data
    std::vector<bfloat16> input_data;
    input_data.reserve(total_tiles * 512);  // 512 values per tile (32x32/2 for bfloat16)

    // Generate random data with some structure to ensure meaningful reduction
    std::srand(0);  // Fixed seed for reproducibility
    const float rand_max = static_cast<float>(RAND_MAX);
    for (uint32_t i = 0; i < total_tiles * 512; i++) {
        float val = static_cast<float>(std::rand()) / rand_max * 2.0f - 1.0f;  // Range [-1, 1]
        input_data.push_back(bfloat16(val));
    }

    // Write input data to device
    detail::WriteToBuffer(src_dram_buffer, input_data);

    // Set runtime args
    SetRuntimeArgs(program_ref, reader_kernel_ref, core, {src_dram_buffer->address(), 0, total_tiles});
    SetRuntimeArgs(program_ref, writer_kernel_ref, core, {dst_dram_buffer_ref->address(), 0, num_blocks});

    SetRuntimeArgs(program_test, reader_kernel_test, core, {src_dram_buffer->address(), 0, total_tiles});
    SetRuntimeArgs(program_test, writer_kernel_test, core, {dst_dram_buffer_test->address(), 0, num_blocks});

    // Run reference program
    detail::LaunchProgram(device, program_ref);

    // Run test program
    detail::LaunchProgram(device, program_test);

    // Read results
    std::vector<bfloat16> output_ref(num_blocks * 512);
    std::vector<bfloat16> output_test(num_blocks * 512);

    detail::ReadFromBuffer(dst_dram_buffer_ref, output_ref);
    detail::ReadFromBuffer(dst_dram_buffer_test, output_test);

    // Convert to float for comparison
    std::vector<float> output_ref_float, output_test_float;
    for (const auto& val : output_ref) {
        output_ref_float.push_back(static_cast<float>(val));
    }
    for (const auto& val : output_test) {
        output_test_float.push_back(static_cast<float>(val));
    }

    // Calculate PCC (Pearson Correlation Coefficient)
    float pcc = block_variants::compute_pcc(output_ref_float, output_test_float);

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

class BlockVariantsFixture : public tt::tt_metal::UnitMeshCQFixture {};

TEST_F(BlockVariantsFixture, ReduceBlock_1x1) {
    for (const auto& mesh_device : devices_) {
        run_reduce_block_test(mesh_device->get_devices().at(0), 1, 1);
    }
}

TEST_F(BlockVariantsFixture, ReduceBlock_1x2) {
    for (const auto& mesh_device : devices_) {
        run_reduce_block_test(mesh_device->get_devices().at(0), 1, 2);
    }
}

TEST_F(BlockVariantsFixture, ReduceBlock_1x4) {
    for (const auto& mesh_device : devices_) {
        run_reduce_block_test(mesh_device->get_devices().at(0), 1, 4);
    }
}

TEST_F(BlockVariantsFixture, ReduceBlock_1x8) {
    for (const auto& mesh_device : devices_) {
        run_reduce_block_test(mesh_device->get_devices().at(0), 1, 8);
    }
}

TEST_F(BlockVariantsFixture, ReduceBlock_1x16) {
    for (const auto& mesh_device : devices_) {
        run_reduce_block_test(mesh_device->get_devices().at(0), 1, 16);
    }
}

TEST_F(BlockVariantsFixture, ReduceBlock_2x1) {
    for (const auto& mesh_device : devices_) {
        run_reduce_block_test(mesh_device->get_devices().at(0), 2, 1);
    }
}

TEST_F(BlockVariantsFixture, ReduceBlock_2x2) {
    for (const auto& mesh_device : devices_) {
        run_reduce_block_test(mesh_device->get_devices().at(0), 2, 2);
    }
}

TEST_F(BlockVariantsFixture, ReduceBlock_2x4) {
    for (const auto& mesh_device : devices_) {
        run_reduce_block_test(mesh_device->get_devices().at(0), 2, 4);
    }
}

TEST_F(BlockVariantsFixture, ReduceBlock_2x8) {
    for (const auto& mesh_device : devices_) {
        run_reduce_block_test(mesh_device->get_devices().at(0), 2, 8);
    }
}

TEST_F(BlockVariantsFixture, ReduceBlock_4x1) {
    for (const auto& mesh_device : devices_) {
        run_reduce_block_test(mesh_device->get_devices().at(0), 4, 1);
    }
}

TEST_F(BlockVariantsFixture, ReduceBlock_4x2) {
    for (const auto& mesh_device : devices_) {
        run_reduce_block_test(mesh_device->get_devices().at(0), 4, 2);
    }
}

TEST_F(BlockVariantsFixture, ReduceBlock_4x4) {
    for (const auto& mesh_device : devices_) {
        run_reduce_block_test(mesh_device->get_devices().at(0), 4, 4);
    }
}

TEST_F(BlockVariantsFixture, ReduceBlock_8x1) {
    for (const auto& mesh_device : devices_) {
        run_reduce_block_test(mesh_device->get_devices().at(0), 8, 1);
    }
}

TEST_F(BlockVariantsFixture, ReduceBlock_8x2) {
    for (const auto& mesh_device : devices_) {
        run_reduce_block_test(mesh_device->get_devices().at(0), 8, 2);
    }
}

TEST_F(BlockVariantsFixture, ReduceBlock_16x1) {
    for (const auto& mesh_device : devices_) {
        run_reduce_block_test(mesh_device->get_devices().at(0), 16, 1);
    }
}

// =============================================================================
// Stress Tests
// =============================================================================

TEST_F(BlockVariantsFixture, ReduceBlock_Stress_ManyBlocks) {
    // Process many blocks to test stability
    for (const auto& mesh_device : devices_) {
        run_reduce_block_test(mesh_device->get_devices().at(0), 4, 4, 1000);
    }
}

TEST_F(BlockVariantsFixture, ReduceBlock_Stress_MaxCapacity) {
    // Use maximum DEST capacity
    for (const auto& mesh_device : devices_) {
        run_reduce_block_test(mesh_device->get_devices().at(0), 16, 1, 100);
    }
}
