// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "device_fixture.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "dm_common.hpp"

namespace tt::tt_metal {

using namespace std;
using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::dm::core_to_core {
// Test config, i.e. test parameters
struct OneFromOneConfig {
    uint32_t test_id = 0;
    CoreCoord master_core_coord = CoreCoord();
    CoreCoord subordinate_core_coord = CoreCoord();
    uint32_t num_of_transactions = 0;
    uint32_t transaction_size_pages = 0;
    uint32_t page_size_bytes = 0;
    DataFormat l1_data_format = DataFormat::Invalid;

    // TODO: Add the following parameters
    //  1. Virtual Channel
    //  2. Which NOC to use
};

/// @brief Does Requestor Core --> L1 Responder Core --> L1 Requestor Core
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool run_dm(IDevice* device, const OneFromOneConfig& test_config) {
    // Program
    Program program = CreateProgram();

    // Sharded L1 buffers
    const size_t total_size_bytes =
        test_config.num_of_transactions * test_config.transaction_size_pages * test_config.page_size_bytes;
    const size_t total_size_pages = test_config.num_of_transactions * test_config.transaction_size_pages;

    CoreRangeSet master_core_set({CoreRange(test_config.master_core_coord)});
    CoreRangeSet subordinate_core_set({CoreRange(test_config.subordinate_core_coord)});

    auto master_shard_parameters = ShardSpecBuffer(
        master_core_set,
        {1, total_size_bytes / 2},
        ShardOrientation::ROW_MAJOR,
        {1, test_config.page_size_bytes / 2},
        {1, total_size_pages});
    auto master_l1_buffer = CreateBuffer(ShardedBufferConfig{
        .device = device,
        .size = total_size_bytes,
        .page_size = test_config.page_size_bytes,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::WIDTH_SHARDED,
        .shard_parameters = std::move(master_shard_parameters),
    });
    uint32_t master_l1_byte_address = master_l1_buffer->address();

    auto subordinate_shard_parameters = ShardSpecBuffer(
        subordinate_core_set,
        {1, total_size_bytes / 2},
        ShardOrientation::ROW_MAJOR,
        {1, test_config.page_size_bytes / 2},
        {1, total_size_pages});
    auto subordinate_l1_buffer = CreateBuffer(ShardedBufferConfig{
        .device = device,
        .size = total_size_bytes,
        .page_size = test_config.page_size_bytes,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::WIDTH_SHARDED,
        .shard_parameters = std::move(subordinate_shard_parameters),
    });
    uint32_t subordinate_l1_byte_address = subordinate_l1_buffer->address();

    // Compile-time arguments for kernels
    vector<uint32_t> requestor_compile_args = {
        (uint32_t)subordinate_l1_byte_address,
        (uint32_t)master_l1_byte_address,
        (uint32_t)test_config.num_of_transactions,
        (uint32_t)test_config.transaction_size_pages,
        (uint32_t)test_config.page_size_bytes,
        (uint32_t)test_config.test_id};

    // Kernels
    auto requestor_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/data_movement/one_from_one/kernels/requestor.cpp",
        master_core_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = requestor_compile_args});

    // Runtime Arguments
    CoreCoord physical_subordinate_core = device->worker_core_from_logical_core(test_config.subordinate_core_coord);
    SetRuntimeArgs(
        program, requestor_kernel, master_core_set, {physical_subordinate_core.x, physical_subordinate_core.y});

    // Assign unique id
    log_info(tt::LogTest, "Running Test ID: {}, Run ID: {}", test_config.test_id, unit_tests::dm::runtime_host_id);
    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    // Input
    vector<uint32_t> packed_input = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -100.0f, 100.0f, total_size_bytes / bfloat16::SIZEOF, chrono::system_clock::now().time_since_epoch().count());

    // Golden output
    vector<uint32_t> packed_golden = packed_input;

    // Launch program and record outputs
    vector<uint32_t> packed_output;
    detail::WriteToBuffer(subordinate_l1_buffer, packed_input);
    MetalContext::instance().get_cluster().l1_barrier(device->id());
    detail::LaunchProgram(device, program);
    detail::ReadFromBuffer(master_l1_buffer, packed_output);

    // Results comparison
    bool pcc = is_close_packed_vectors<bfloat16, uint32_t>(
        packed_output, packed_golden, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b); });

    if (!pcc) {
        log_error(tt::LogTest, "PCC Check failed");
        log_info(tt::LogTest, "Golden vector");
        print_vector<uint32_t>(packed_golden);
        log_info(tt::LogTest, "Output vector");
        print_vector<uint32_t>(packed_output);
    }

    return pcc;
}
}  // namespace unit_tests::dm::core_to_core

/* ========== Test case for one from one data movement; Test id = 5 ========== */
TEST_F(DeviceFixture, TensixDataMovementOneFromOnePacketSizes) {
    // Parameters
    uint32_t max_transactions = 256;
    uint32_t max_transaction_size_pages = 64;
    uint32_t page_size_bytes = arch_ == tt::ARCH::BLACKHOLE ? 64 : 32;  // =Flit size: 32 bytes for WH, 64 for BH

    // Cores
    CoreCoord master_core_coord = {0, 0};
    CoreCoord subordinate_core_coord = {1, 1};

    for (uint32_t num_of_transactions = 1; num_of_transactions <= max_transactions; num_of_transactions *= 4) {
        for (uint32_t transaction_size_pages = 1; transaction_size_pages <= max_transaction_size_pages;
             transaction_size_pages *= 2) {
            if (num_of_transactions * transaction_size_pages * page_size_bytes >= 1024 * 1024) {
                continue;
            }

            // Test config
            unit_tests::dm::core_to_core::OneFromOneConfig test_config = {
                .test_id = 5,
                .master_core_coord = master_core_coord,
                .subordinate_core_coord = subordinate_core_coord,
                .num_of_transactions = num_of_transactions,
                .transaction_size_pages = transaction_size_pages,
                .page_size_bytes = page_size_bytes,
                .l1_data_format = DataFormat::Float16_b,
            };

            // Run
            for (unsigned int id = 0; id < num_devices_; id++) {
                EXPECT_TRUE(run_dm(devices_.at(id), test_config));
            }
        }
    }
}

/* ========== Test case for one from one data movement; Test id = 51 ========== */
TEST_F(DeviceFixture, TensixDataMovementOneFromOneDirectedIdeal) {
    uint32_t test_id = 51;  // Arbitrary test ID

    // Parameters
    /*
        L1 Capacity: 1.5 MB (I think, might be wrong)
        - Max transaction size
            = 4 * 32 pages
            = 128 pages * 32 (or 64) bytes/page
            = 4096 bytes for WH; 8192 bytes for BH
        - Max total transaction size
            = 128 transactions * 4096 bytes
            = 524,288 Bytes
            < 1.25 MB ~= L1 buffer capacity (.25 MB is allocated for the kernel code and other overheads)
    */
    uint32_t page_size_bytes, num_of_transactions;
    uint32_t transaction_size_pages = 4 * 32;
    if (arch_ == tt::ARCH::BLACKHOLE) {
        page_size_bytes = 64;  // (=flit size): 64 bytes for BH
        num_of_transactions = 64;
    } else {
        page_size_bytes = 32;  // (=flit size): 32 bytes for WH
        num_of_transactions = 128;
    }

    // Cores
    /*
        Any two cores that are next to each other on the torus
         - May be worth considering the performance of this test with different pairs of adjacent cores
    */
    CoreCoord master_core_coord = {0, 0};
    CoreCoord subordinate_core_coord = {0, 1};

    // Test Config
    unit_tests::dm::core_to_core::OneFromOneConfig test_config = {
        .test_id = test_id,
        .master_core_coord = master_core_coord,
        .subordinate_core_coord = subordinate_core_coord,
        .num_of_transactions = num_of_transactions,
        .transaction_size_pages = transaction_size_pages,
        .page_size_bytes = page_size_bytes,
        .l1_data_format = DataFormat::Float16_b,
    };

    // Run
    for (unsigned int id = 0; id < num_devices_; id++) {
        EXPECT_TRUE(run_dm(devices_.at(id), test_config));
    }
}

}  // namespace tt::tt_metal
