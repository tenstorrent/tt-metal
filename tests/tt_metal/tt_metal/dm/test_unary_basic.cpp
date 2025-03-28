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
    size_t num_of_transactions = 0;
    size_t transaction_size_pages = 0;
    size_t page_size_bytes = 0;
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
    const size_t total_size_bytes =
        test_config.num_of_transactions * test_config.transaction_size_pages * test_config.page_size_bytes;
    // TODO: Test for sharded dram buffer as well
    InterleavedBufferConfig dram_config{
        .device = device, .size = total_size_bytes, .page_size = total_size_bytes, .buffer_type = BufferType::DRAM};

    auto input_dram_buffer = CreateBuffer(dram_config);
    uint32_t input_dram_byte_address = input_dram_buffer->address();
    auto output_dram_buffer = CreateBuffer(dram_config);
    uint32_t output_dram_byte_address = output_dram_buffer->address();

    // Input
    vector<uint32_t> packed_input = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -100.0f,
        100.0f,
        total_size_bytes / bfloat16::SIZEOF,
        std::chrono::system_clock::now().time_since_epoch().count());

    // Golden output
    vector<uint32_t> packed_golden = packed_input;

    uint8_t l1_cb_index = CBIndex::c_0;

    // Compile-time arguments for kernels
    vector<uint32_t> reader_compile_args = {
        (uint32_t)input_dram_byte_address,
        (uint32_t)0,
        (uint32_t)test_config.num_of_transactions,
        (uint32_t)test_config.transaction_size_pages,
        (uint32_t)test_config.page_size_bytes,
        (uint8_t)l1_cb_index,
    };

    vector<uint32_t> writer_compile_args = {
        (uint32_t)output_dram_byte_address,
        (uint32_t)0,
        (uint32_t)test_config.num_of_transactions,
        (uint32_t)test_config.transaction_size_pages,
        (uint32_t)test_config.page_size_bytes,
        (uint8_t)l1_cb_index,
    };

    // Create circular buffers
    CircularBufferConfig l1_cb_config =
        CircularBufferConfig(total_size_bytes, {{l1_cb_index, test_config.l1_data_format}})
            .set_page_size(l1_cb_index, test_config.page_size_bytes);
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
    log_info("Results for run id: {}", runtime_host_id);
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
TEST_F(DeviceFixture, TensixDataMovementDRAMInterleavedPacketSizes) {
    // Parameters
    size_t max_transactions = 2;            // Bound for testing different number of transactions
    size_t max_transaction_size_pages = 2;  // Bound for testing different transaction sizes
    size_t page_size_bytes = 32;            // Page size in bytes (=flit size): 32 bytes for WH, 64 for BH
    if (arch_ == tt::ARCH::BLACKHOLE) {
        page_size_bytes *= 2;
    }

    // Cores
    CoreRange core_range({0, 0}, {0, 0});
    CoreRangeSet core_range_set({core_range});

    for (size_t num_of_transactions = 1; num_of_transactions <= max_transactions; num_of_transactions *= 2) {
        for (size_t transaction_size_pages = 1; transaction_size_pages <= max_transaction_size_pages;
             transaction_size_pages *= 2) {
            // Test config
            unit_tests::dm::DmConfig test_config = {
                .num_of_transactions = num_of_transactions,
                .transaction_size_pages = transaction_size_pages,
                .page_size_bytes = page_size_bytes,
                .l1_data_format = DataFormat::Float16_b,
                .cores = core_range_set};

            // Run
            for (unsigned int id = 0; id < num_devices_; id++) {
                EXPECT_TRUE(run_dm(devices_.at(id), test_config));
            }
        }
    }
}

/* ========== Test case for varying core locations ========== */
TEST_F(DeviceFixture, TensixDataMovementDRAMInterleavedCoreLocations) {
    size_t num_of_transactions = 1;     // Bound for testing different number of transactions
    size_t transaction_size_pages = 1;  // Bound for testing different transaction sizes
    size_t page_size_bytes = 32;        // Page size in bytes (=flit size): 32 bytes for WH, 64 for BH
    if (arch_ == tt::ARCH::BLACKHOLE) {
        page_size_bytes *= 2;
    }

    for (unsigned int id = 0; id < num_devices_; id++) {
        // Cores
        auto grid_size = devices_.at(id)->compute_with_storage_grid_size();
        log_info("Grid size x: {}, y: {}", grid_size.x, grid_size.y);

        for (unsigned int x = 0; x < grid_size.x; x++) {
            for (unsigned int y = 0; y < grid_size.y; y++) {
                CoreRangeSet core_range_set(CoreRange({x, y}, {x, y}));

                // Test config
                unit_tests::dm::DmConfig test_config = {
                    .num_of_transactions = num_of_transactions,
                    .transaction_size_pages = transaction_size_pages,
                    .page_size_bytes = page_size_bytes,
                    .l1_data_format = DataFormat::Float16_b,
                    .cores = core_range_set};

                // Run
                EXPECT_TRUE(run_dm(devices_.at(id), test_config));
            }
        }
    }
}
// TODO: New test for sharded DRAM buffer with
//      1. different transaction numbers and sizes
//      2. different core locations
// TODO: New test for core-to-core transactions. May use master and slave core range sets, added in the test config.
// Might use a separate test file

}  // namespace tt::tt_metal
