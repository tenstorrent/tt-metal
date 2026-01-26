// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

/**
 * Block Variant Tests: broadcast
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
 * Run broadcast block test
 *
 * @param device Device to run on
 * @param Ht Block height in tiles
 * @param Wt Block width in tiles
 * @param num_blocks Number of blocks to process
 * @param data_format Data format for tiles
 */
void run_broadcast_block_test(
    IDevice* device,
    uint32_t Ht,
    uint32_t Wt,
    uint32_t num_blocks = 10,
    tt::DataFormat data_format = tt::DataFormat::Float16_b) {
    // Validate block size
    ASSERT_LE(Ht * Wt, 16) << "Block size exceeds DEST capacity (max 16 tiles)";
    ASSERT_GT(Ht, 0) << "Block height must be > 0";
    ASSERT_GT(Wt, 0) << "Block width must be > 0";

    log_info(LogTest, "Testing broadcast with Ht={}, Wt={}, blocks={}", Ht, Wt, num_blocks);

    // Program configuration
    Program program_ref = CreateProgram();
    Program program_test = CreateProgram();
    CoreCoord core = {0, 0};

    // Buffer configuration
    uint32_t single_tile_size = 2 * 1024;  // bfloat16 tile size
    uint32_t total_tiles = Ht * Wt * num_blocks;
    uint32_t dram_buffer_size = single_tile_size * total_tiles;
    uint32_t scalar_buffer_size = single_tile_size;  // Single tile for scalar broadcast

    // Create DRAM buffers
    auto src0_dram_buffer =
        CreateBuffer(InterleavedBufferConfig{device, dram_buffer_size, single_tile_size, BufferType::DRAM});

    auto src1_dram_buffer =
        CreateBuffer(InterleavedBufferConfig{device, scalar_buffer_size, single_tile_size, BufferType::DRAM});

    auto dst_dram_buffer_ref =
        CreateBuffer(InterleavedBufferConfig{device, dram_buffer_size, single_tile_size, BufferType::DRAM});

    auto dst_dram_buffer_test =
        CreateBuffer(InterleavedBufferConfig{device, dram_buffer_size, single_tile_size, BufferType::DRAM});

    // Circular Buffer configuration
    uint32_t cb0_tiles = Ht * Wt;  // Input CB0
    uint32_t cb1_tiles = 1;        // Scalar CB1
    uint32_t cb2_tiles = Ht * Wt;  // Output CB2

    CircularBufferConfig cb_src0_config = CircularBufferConfig(cb0_tiles * single_tile_size, {{CB::c_in0, data_format}})
                                              .set_page_size(CB::c_in0, single_tile_size);
    CircularBufferConfig cb_src1_config = CircularBufferConfig(cb1_tiles * single_tile_size, {{CB::c_in1, data_format}})
                                              .set_page_size(CB::c_in1, single_tile_size);
    CircularBufferConfig cb_out_config = CircularBufferConfig(cb2_tiles * single_tile_size, {{CB::c_out0, data_format}})
                                             .set_page_size(CB::c_out0, single_tile_size);

    auto cb_src0_ref = CreateCircularBuffer(program_ref, core, cb_src0_config);
    auto cb_src1_ref = CreateCircularBuffer(program_ref, core, cb_src1_config);
    auto cb_out_ref = CreateCircularBuffer(program_ref, core, cb_out_config);

    auto cb_src0_test = CreateCircularBuffer(program_test, core, cb_src0_config);
    auto cb_src1_test = CreateCircularBuffer(program_test, core, cb_src1_config);
    auto cb_out_test = CreateCircularBuffer(program_test, core, cb_out_config);

    (void)cb_src0_ref;
    (void)cb_src1_ref;
    (void)cb_out_ref;
    (void)cb_src0_test;
    (void)cb_src1_test;
    (void)cb_out_test;

    // Reader kernels - identical for both programs
    std::map<string, string> reader_defines = {{"DRAM_UNRESERVED_BASE", "0"}};

    auto reader_kernel_ref = CreateKernel(
        program_ref,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = {total_tiles},
            .defines = reader_defines});

    auto reader_kernel_test = CreateKernel(
        program_test,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = {total_tiles},
            .defines = reader_defines});

    // Writer kernels - identical for both programs
    auto writer_kernel_ref = CreateKernel(
        program_ref,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = {total_tiles}});

    auto writer_kernel_test = CreateKernel(
        program_test,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = {total_tiles}});

    // Reference compute kernel (tile-by-tile)
    vector<uint32_t> compute_args_ref = {Ht, Wt, num_blocks};
    std::map<string, string> compute_defines_ref = {
        {"SFPU_OP_INIT_ACTIVATION", "0"},
        {"SFPU_OP_CHAIN_0", "mul_tiles"},
        {"ELWISE_BINARY_FIDELITY", "HiFi4"},
        {"ELWISE_BINARY", "mul_tiles"},
        {"EltwiseBinaryType", "EltwiseBinaryType::ELWMUL"}};

    auto compute_kernel_ref = CreateKernel(
        program_ref,
        "tests/tt_metal/tt_metal/test_kernels/compute/broadcast_tiles.cpp",
        core,
        ComputeConfig{.compile_args = compute_args_ref, .defines = compute_defines_ref});

    // Test compute kernel (block operation)
    vector<uint32_t> compute_args_test = {Ht, Wt, num_blocks};
    std::map<string, string> compute_defines_test = {
        {"SFPU_OP_INIT_ACTIVATION", "0"},
        {"SFPU_OP_CHAIN_0", "mul_tiles_bcast_block"},
        {"ELWISE_BINARY_FIDELITY", "HiFi4"},
        {"ELWISE_BINARY", "mul_tiles_bcast_block"},
        {"EltwiseBinaryType", "EltwiseBinaryType::ELWMUL"}};

    auto compute_kernel_test = CreateKernel(
        program_test,
        "tests/tt_metal/tt_metal/test_kernels/compute/broadcast_block.cpp",
        core,
        ComputeConfig{.compile_args = compute_args_test, .defines = compute_defines_test});

    (void)compute_kernel_ref;
    (void)compute_kernel_test;

    // Generate input data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-10.0f, 10.0f);

    // Input tensor data (multiple tiles)
    std::vector<bfloat16> src0_data(total_tiles * 512);  // 512 elements per tile
    for (uint32_t i = 0; i < src0_data.size(); i++) {
        src0_data[i] = bfloat16(dis(gen));
    }

    // Scalar data (single tile, broadcast)
    std::vector<bfloat16> src1_data(512);  // Single tile
    for (uint32_t i = 0; i < src1_data.size(); i++) {
        src1_data[i] = bfloat16(dis(gen));
    }

    // Write input data to device
    detail::WriteToBuffer(src0_dram_buffer, src0_data);
    detail::WriteToBuffer(src1_dram_buffer, src1_data);

    // Set runtime args for reference program
    SetRuntimeArgs(program_ref, reader_kernel_ref, core, {src0_dram_buffer->address(), src1_dram_buffer->address()});

    SetRuntimeArgs(program_ref, writer_kernel_ref, core, {dst_dram_buffer_ref->address()});

    // Set runtime args for test program
    SetRuntimeArgs(program_test, reader_kernel_test, core, {src0_dram_buffer->address(), src1_dram_buffer->address()});

    SetRuntimeArgs(program_test, writer_kernel_test, core, {dst_dram_buffer_test->address()});

    // Execute both programs
    detail::LaunchProgram(device, program_ref);
    detail::LaunchProgram(device, program_test);

    // Read results
    std::vector<bfloat16> result_ref(total_tiles * 512);
    std::vector<bfloat16> result_test(total_tiles * 512);

    detail::ReadFromBuffer(dst_dram_buffer_ref, result_ref);
    detail::ReadFromBuffer(dst_dram_buffer_test, result_test);

    // Compute PCC
    float pcc = block_variants::compute_pcc(result_ref, result_test);

    log_info(LogTest, "PCC: {}", pcc);
    EXPECT_GE(pcc, 0.9999f) << "PCC between reference and test results is too low";

    // Also validate against golden reference implementation
    std::vector<float> src0_float(src0_data.size());
    std::vector<float> src1_float(src1_data.size());
    std::vector<float> result_test_float(result_test.size());

    for (size_t i = 0; i < src0_data.size(); i++) {
        src0_float[i] = static_cast<float>(src0_data[i]);
    }
    for (size_t i = 0; i < src1_data.size(); i++) {
        src1_float[i] = static_cast<float>(src1_data[i]);
    }
    for (size_t i = 0; i < result_test.size(); i++) {
        result_test_float[i] = static_cast<float>(result_test[i]);
    }

    // Create golden reference by broadcasting scalar across all tiles
    std::vector<float> golden_ref(src0_float.size());
    for (uint32_t block = 0; block < num_blocks; block++) {
        for (uint32_t tile = 0; tile < Ht * Wt; tile++) {
            uint32_t tile_offset = (block * Ht * Wt + tile) * 512;
            for (uint32_t elem = 0; elem < 512; elem++) {
                golden_ref[tile_offset + elem] = src0_float[tile_offset + elem] * src1_float[elem];
            }
        }
    }

    float pcc_golden = block_variants::compute_pcc(result_test_float, golden_ref);
    log_info(LogTest, "PCC vs Golden: {}", pcc_golden);
    EXPECT_GE(pcc_golden, 0.9999f) << "PCC against golden reference is too low";
}

}  // namespace

// =============================================================================
// Test Cases
// =============================================================================

class BlockVariantsFixture : public tt::tt_metal::UnitMeshCQFixture {};

TEST_F(BlockVariantsFixture, BroadcastBlock_1x1) {
    for (const auto& mesh_device : devices_) {
        run_broadcast_block_test(mesh_device->get_devices().at(0), 1, 1);
    }
}

TEST_F(BlockVariantsFixture, BroadcastBlock_1x2) {
    for (const auto& mesh_device : devices_) {
        run_broadcast_block_test(mesh_device->get_devices().at(0), 1, 2);
    }
}

TEST_F(BlockVariantsFixture, BroadcastBlock_1x4) {
    for (const auto& mesh_device : devices_) {
        run_broadcast_block_test(mesh_device->get_devices().at(0), 1, 4);
    }
}

TEST_F(BlockVariantsFixture, BroadcastBlock_1x8) {
    for (const auto& mesh_device : devices_) {
        run_broadcast_block_test(mesh_device->get_devices().at(0), 1, 8);
    }
}

TEST_F(BlockVariantsFixture, BroadcastBlock_1x16) {
    for (const auto& mesh_device : devices_) {
        run_broadcast_block_test(mesh_device->get_devices().at(0), 1, 16);
    }
}

TEST_F(BlockVariantsFixture, BroadcastBlock_2x1) {
    for (const auto& mesh_device : devices_) {
        run_broadcast_block_test(mesh_device->get_devices().at(0), 2, 1);
    }
}

TEST_F(BlockVariantsFixture, BroadcastBlock_2x2) {
    for (const auto& mesh_device : devices_) {
        run_broadcast_block_test(mesh_device->get_devices().at(0), 2, 2);
    }
}

TEST_F(BlockVariantsFixture, BroadcastBlock_2x4) {
    for (const auto& mesh_device : devices_) {
        run_broadcast_block_test(mesh_device->get_devices().at(0), 2, 4);
    }
}

TEST_F(BlockVariantsFixture, BroadcastBlock_2x8) {
    for (const auto& mesh_device : devices_) {
        run_broadcast_block_test(mesh_device->get_devices().at(0), 2, 8);
    }
}

TEST_F(BlockVariantsFixture, BroadcastBlock_4x1) {
    for (const auto& mesh_device : devices_) {
        run_broadcast_block_test(mesh_device->get_devices().at(0), 4, 1);
    }
}

TEST_F(BlockVariantsFixture, BroadcastBlock_4x2) {
    for (const auto& mesh_device : devices_) {
        run_broadcast_block_test(mesh_device->get_devices().at(0), 4, 2);
    }
}

TEST_F(BlockVariantsFixture, BroadcastBlock_4x4) {
    for (const auto& mesh_device : devices_) {
        run_broadcast_block_test(mesh_device->get_devices().at(0), 4, 4);
    }
}

TEST_F(BlockVariantsFixture, BroadcastBlock_8x1) {
    for (const auto& mesh_device : devices_) {
        run_broadcast_block_test(mesh_device->get_devices().at(0), 8, 1);
    }
}

TEST_F(BlockVariantsFixture, BroadcastBlock_8x2) {
    for (const auto& mesh_device : devices_) {
        run_broadcast_block_test(mesh_device->get_devices().at(0), 8, 2);
    }
}

TEST_F(BlockVariantsFixture, BroadcastBlock_16x1) {
    for (const auto& mesh_device : devices_) {
        run_broadcast_block_test(mesh_device->get_devices().at(0), 16, 1);
    }
}

// =============================================================================
// Stress Tests
// =============================================================================

TEST_F(BlockVariantsFixture, BroadcastBlock_Stress_ManyBlocks) {
    // Process many blocks to test stability
    for (const auto& mesh_device : devices_) {
        run_broadcast_block_test(mesh_device->get_devices().at(0), 4, 4, 1000);
    }
}

TEST_F(BlockVariantsFixture, BroadcastBlock_Stress_MaxCapacity) {
    // Use maximum DEST capacity
    for (const auto& mesh_device : devices_) {
        run_broadcast_block_test(mesh_device->get_devices().at(0), 16, 1, 100);
    }
}
