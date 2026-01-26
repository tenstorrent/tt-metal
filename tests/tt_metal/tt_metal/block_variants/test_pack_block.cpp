// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

/**
 * Block Variant Tests: pack
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
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace tt;
using namespace tt::tt_metal;
using std::map;
using std::string;
using std::vector;

namespace {

/**
 * Run pack block test
 *
 * @param device Device to run on
 * @param Ht Block height in tiles
 * @param Wt Block width in tiles
 * @param num_blocks Number of blocks to process
 * @param data_format Data format for tiles
 */
void run_pack_block_test(
    IDevice* device,
    uint32_t Ht,
    uint32_t Wt,
    uint32_t num_blocks = 10,
    tt::DataFormat data_format = tt::DataFormat::Float16_b) {
    // Validate block size
    ASSERT_LE(Ht * Wt, 16) << "Block size exceeds DEST capacity (max 16 tiles)";
    ASSERT_GT(Ht, 0) << "Block height must be > 0";
    ASSERT_GT(Wt, 0) << "Block width must be > 0";

    log_info(LogTest, "Testing pack with Ht={}, Wt={}, blocks={}", Ht, Wt, num_blocks);

    uint32_t single_tile_size = 2 * 1024;
    uint32_t block_size_tiles = Ht * Wt;
    uint32_t dram_buffer_size = single_tile_size * block_size_tiles * num_blocks;

    CoreCoord core = {0, 0};

    // Create programs
    Program program_ref = CreateProgram();
    Program program_test = CreateProgram();

    // Create buffers
    auto src_dram_buffer =
        CreateBuffer(InterleavedBufferConfig{device, dram_buffer_size, single_tile_size, BufferType::DRAM});

    auto dst_dram_buffer_ref =
        CreateBuffer(InterleavedBufferConfig{device, dram_buffer_size, single_tile_size, BufferType::DRAM});

    auto dst_dram_buffer_test =
        CreateBuffer(InterleavedBufferConfig{device, dram_buffer_size, single_tile_size, BufferType::DRAM});

    // Create circular buffers for reference kernel
    CircularBufferConfig cb_src0_config = CircularBufferConfig(block_size_tiles * single_tile_size, {{0, data_format}})
                                              .set_page_size(0, single_tile_size);
    auto cb_src0_ref = CreateCircularBuffer(program_ref, core, cb_src0_config);

    CircularBufferConfig cb_intermed0_config =
        CircularBufferConfig(block_size_tiles * single_tile_size, {{1, data_format}})
            .set_page_size(1, single_tile_size);
    auto cb_intermed0_ref = CreateCircularBuffer(program_ref, core, cb_intermed0_config);

    CircularBufferConfig cb_output_config =
        CircularBufferConfig(block_size_tiles * single_tile_size, {{2, data_format}})
            .set_page_size(2, single_tile_size);
    auto cb_output_ref = CreateCircularBuffer(program_ref, core, cb_output_config);

    // Create circular buffers for test kernel
    auto cb_src0_test = CreateCircularBuffer(program_test, core, cb_src0_config);
    auto cb_intermed0_test = CreateCircularBuffer(program_test, core, cb_intermed0_config);
    auto cb_output_test = CreateCircularBuffer(program_test, core, cb_output_config);

    (void)cb_src0_ref;
    (void)cb_intermed0_ref;
    (void)cb_output_ref;
    (void)cb_src0_test;
    (void)cb_intermed0_test;
    (void)cb_output_test;

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

    std::string compute_kernel_ref = "tests/tt_metal/tt_metal/test_kernels/compute/pack_tiles.cpp";
    std::map<string, string> defines_ref = {
        {"HT", std::to_string(Ht)}, {"WT", std::to_string(Wt)}, {"NUM_BLOCKS", std::to_string(num_blocks)}};

    auto compute_kernel_ref_id = CreateKernel(
        program_ref,
        compute_kernel_ref,
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4, .compile_args = {Ht, Wt, num_blocks}, .defines = defines_ref});

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

    std::string compute_kernel_test = "tests/tt_metal/tt_metal/test_kernels/compute/pack_block.cpp";
    std::map<string, string> defines_test = {
        {"HT", std::to_string(Ht)}, {"WT", std::to_string(Wt)}, {"NUM_BLOCKS", std::to_string(num_blocks)}};

    auto compute_kernel_test_id = CreateKernel(
        program_test,
        compute_kernel_test,
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4, .compile_args = {Ht, Wt, num_blocks}, .defines = defines_test});

    (void)compute_kernel_ref_id;
    (void)compute_kernel_test_id;

    // Generate random input data
    std::vector<uint32_t> src_vec;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-2.0f, 2.0f);

    for (uint32_t i = 0; i < dram_buffer_size / sizeof(uint32_t); ++i) {
        float val = dis(gen);
        src_vec.push_back(pack_two_bfloat16_into_uint32({bfloat16(val), bfloat16(val)}));
    }

    // Set runtime args for reference kernel
    SetRuntimeArgs(
        program_ref,
        reader_kernel_ref,
        core,
        {src_dram_buffer->address(), static_cast<uint32_t>(block_size_tiles * num_blocks), 0});

    SetRuntimeArgs(
        program_ref,
        writer_kernel_ref,
        core,
        {dst_dram_buffer_ref->address(), static_cast<uint32_t>(block_size_tiles * num_blocks), 0});

    // Set runtime args for test kernel
    SetRuntimeArgs(
        program_test,
        reader_kernel_test,
        core,
        {src_dram_buffer->address(), static_cast<uint32_t>(block_size_tiles * num_blocks), 0});

    SetRuntimeArgs(
        program_test,
        writer_kernel_test,
        core,
        {dst_dram_buffer_test->address(), static_cast<uint32_t>(block_size_tiles * num_blocks), 0});

    // Write input data to device
    detail::WriteToBuffer(src_dram_buffer, src_vec);

    // Execute reference program
    detail::LaunchProgram(device, program_ref);

    // Execute test program
    detail::LaunchProgram(device, program_test);

    // Read results
    std::vector<uint32_t> result_vec_ref;
    detail::ReadFromBuffer(dst_dram_buffer_ref, result_vec_ref);

    std::vector<uint32_t> result_vec_test;
    detail::ReadFromBuffer(dst_dram_buffer_test, result_vec_test);

    // Convert to float for comparison
    auto result_bfp16_ref = unpack_uint32_vec_into_bfloat16_vec(result_vec_ref);
    auto result_bfp16_test = unpack_uint32_vec_into_bfloat16_vec(result_vec_test);

    // Calculate PCC
    double pcc = block_variants::compute_pcc(result_bfp16_ref, result_bfp16_test);

    log_info(LogTest, "PCC between reference and test results: {}", pcc);

    // Validate PCC threshold
    EXPECT_GE(pcc, 0.9999) << "PCC too low: " << pcc << " (expected >= 0.9999)";

    // Additional validation: ensure results are identical for pack operation
    for (size_t i = 0; i < result_vec_ref.size(); ++i) {
        EXPECT_EQ(result_vec_ref[i], result_vec_test[i])
            << "Results differ at index " << i << " (pack should be bit-exact)";
    }
}

}  // namespace

// =============================================================================
// Test Cases
// =============================================================================

class BlockVariantsFixture : public tt::tt_metal::UnitMeshCQFixture {};

TEST_F(BlockVariantsFixture, PackBlock_1x1) {
    for (const auto& mesh_device : devices_) {
        run_pack_block_test(mesh_device->get_devices().at(0), 1, 1);
    }
}

TEST_F(BlockVariantsFixture, PackBlock_1x2) {
    for (const auto& mesh_device : devices_) {
        run_pack_block_test(mesh_device->get_devices().at(0), 1, 2);
    }
}

TEST_F(BlockVariantsFixture, PackBlock_1x4) {
    for (const auto& mesh_device : devices_) {
        run_pack_block_test(mesh_device->get_devices().at(0), 1, 4);
    }
}

TEST_F(BlockVariantsFixture, PackBlock_1x8) {
    for (const auto& mesh_device : devices_) {
        run_pack_block_test(mesh_device->get_devices().at(0), 1, 8);
    }
}

TEST_F(BlockVariantsFixture, PackBlock_1x16) {
    for (const auto& mesh_device : devices_) {
        run_pack_block_test(mesh_device->get_devices().at(0), 1, 16);
    }
}

TEST_F(BlockVariantsFixture, PackBlock_2x1) {
    for (const auto& mesh_device : devices_) {
        run_pack_block_test(mesh_device->get_devices().at(0), 2, 1);
    }
}

TEST_F(BlockVariantsFixture, PackBlock_2x2) {
    for (const auto& mesh_device : devices_) {
        run_pack_block_test(mesh_device->get_devices().at(0), 2, 2);
    }
}

TEST_F(BlockVariantsFixture, PackBlock_2x4) {
    for (const auto& mesh_device : devices_) {
        run_pack_block_test(mesh_device->get_devices().at(0), 2, 4);
    }
}

TEST_F(BlockVariantsFixture, PackBlock_2x8) {
    for (const auto& mesh_device : devices_) {
        run_pack_block_test(mesh_device->get_devices().at(0), 2, 8);
    }
}

TEST_F(BlockVariantsFixture, PackBlock_4x1) {
    for (const auto& mesh_device : devices_) {
        run_pack_block_test(mesh_device->get_devices().at(0), 4, 1);
    }
}

TEST_F(BlockVariantsFixture, PackBlock_4x2) {
    for (const auto& mesh_device : devices_) {
        run_pack_block_test(mesh_device->get_devices().at(0), 4, 2);
    }
}

TEST_F(BlockVariantsFixture, PackBlock_4x4) {
    for (const auto& mesh_device : devices_) {
        run_pack_block_test(mesh_device->get_devices().at(0), 4, 4);
    }
}

TEST_F(BlockVariantsFixture, PackBlock_8x1) {
    for (const auto& mesh_device : devices_) {
        run_pack_block_test(mesh_device->get_devices().at(0), 8, 1);
    }
}

TEST_F(BlockVariantsFixture, PackBlock_8x2) {
    for (const auto& mesh_device : devices_) {
        run_pack_block_test(mesh_device->get_devices().at(0), 8, 2);
    }
}

TEST_F(BlockVariantsFixture, PackBlock_16x1) {
    for (const auto& mesh_device : devices_) {
        run_pack_block_test(mesh_device->get_devices().at(0), 16, 1);
    }
}

// =============================================================================
// Stress Tests
// =============================================================================

TEST_F(BlockVariantsFixture, PackBlock_Stress_ManyBlocks) {
    // Process many blocks to test stability
    for (const auto& mesh_device : devices_) {
        run_pack_block_test(mesh_device->get_devices().at(0), 4, 4, 1000);
    }
}

TEST_F(BlockVariantsFixture, PackBlock_Stress_MaxCapacity) {
    // Use maximum DEST capacity
    for (const auto& mesh_device : devices_) {
        run_pack_block_test(mesh_device->get_devices().at(0), 16, 1, 100);
    }
}
