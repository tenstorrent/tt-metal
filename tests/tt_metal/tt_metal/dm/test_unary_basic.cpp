// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "device_fixture.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"

namespace tt::tt_metal {

using std::vector;
using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::dm {

uint32_t runtime_host_id = 0;

// Test config
struct DmConfig {
    size_t total_num_tiles = 0;
    size_t num_tiles_per_ublock = 0;
    size_t tile_byte_size = 0;
    DataFormat l1_data_format = DataFormat::Invalid;
    CoreRangeSet cores = CoreRangeSet();
};

/// @brief Does Dram --> Reader --> CB --> Writer --> Dram.
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool run_dm(IDevice* device, const DmConfig& test_config) {
    // Program
    Program program = CreateProgram();

    // DRAM Buffers
    const size_t byte_size = test_config.total_num_tiles * test_config.tile_byte_size;
    // TODO: Test for sharded dram buffer as well
    InterleavedBufferConfig dram_config{
        .device = device, .size = byte_size, .page_size = byte_size, .buffer_type = BufferType::DRAM};

    auto input_dram_buffer = CreateBuffer(dram_config);
    uint32_t input_dram_byte_address = input_dram_buffer->address();
    auto output_dram_buffer = CreateBuffer(dram_config);
    uint32_t output_dram_byte_address = output_dram_buffer->address();

    // Input
    vector<uint32_t> packed_input = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -100.0f, 100.0f, byte_size / bfloat16::SIZEOF, std::chrono::system_clock::now().time_since_epoch().count());

    // Golden output
    vector<uint32_t> packed_golden = packed_input;

    uint8_t l1_cb_index = CBIndex::c_0;

    // Compile-time arguments for kernels
    vector<uint32_t> reader_compile_args = {
        (uint32_t)input_dram_byte_address,
        (uint32_t)0,
        (uint32_t)test_config.total_num_tiles,
        (uint32_t)test_config.num_tiles_per_ublock,
        (uint8_t)l1_cb_index,
    };

    vector<uint32_t> writer_compile_args = {
        (uint32_t)output_dram_byte_address,
        (uint32_t)0,
        (uint32_t)test_config.total_num_tiles,
        (uint32_t)test_config.num_tiles_per_ublock,
        (uint8_t)l1_cb_index,
    };

    // Create circular buffers
    CircularBufferConfig l1_cb_config = CircularBufferConfig(byte_size, {{l1_cb_index, test_config.l1_data_format}})
                                            .set_page_size(l1_cb_index, test_config.tile_byte_size);
    auto l1_cb = CreateCircularBuffer(program, test_config.cores, l1_cb_config);

    // Kernels
    auto reader_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/dm/reader_unary.cpp",
        test_config.cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_args});

    auto writer_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/dm/writer_unary.cpp",
        test_config.cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_args});

    // Assign unique id
    program.set_runtime_id(runtime_host_id++);

    // Launch program and record outputs
    vector<uint32_t> packed_output;
    detail::WriteToBuffer(input_dram_buffer, packed_input);
    detail::LaunchProgram(device, program);
    detail::ReadFromBuffer(output_dram_buffer, packed_output);

    // Print output and golden vectors
    log_info("Golden vector");
    print_vector<uint32_t>(packed_golden);
    log_info("Output vector");
    print_vector<uint32_t>(packed_output);

    // Return comparison
    return is_close_packed_vectors<bfloat16, uint32_t>(
        packed_output, packed_golden, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b); });
}
}  // namespace unit_tests::dm

/* ========== Test case for varying transaction numbers and sizes ========== */
TEST_F(DeviceFixture, TensixDataMovementDRAMInterleaved) {
    // TODO: Change total_num_tiles to test with different total data sizes
    // TODO: Change num_tiles_per_ublock to test with different packet sizes
    // TODO: Set tile byte size to minimum packet size (one flit) depending on ARCH
    // Parameters
    size_t num_of_transactions = 1;     // Number of transactions
    size_t transaction_size_tiles = 1;  // Transaction size
    size_t tile_byte_size = 16 * 16;    // Tile byte size

    // Cores
    CoreRange core_range({0, 0}, {0, 0});
    CoreRangeSet core_range_set({core_range});

    // Test config
    unit_tests::dm::DmConfig test_config = {
        .total_num_tiles = num_of_transactions,
        .num_tiles_per_ublock = transaction_size_tiles,
        .tile_byte_size = tile_byte_size,
        .l1_data_format = DataFormat::Float16_b,
        .cores = core_range_set};

    // Run
    for (unsigned int id = 0; id < num_devices_; id++) {
        for (unsigned int i = 0; i < 2; i++) {
            EXPECT_TRUE(run_dm(devices_.at(id), test_config));
        }
    }
}

// TODO: New test for different core locations
// TODO: Configure master cores here
// TODO: Use another core range set to configure slave cores, and add it to config

// TODO: New test for sharded DRAM buffer with
//      1. different transaction numbers and sizes
//      2. different core locations

}  // namespace tt::tt_metal
