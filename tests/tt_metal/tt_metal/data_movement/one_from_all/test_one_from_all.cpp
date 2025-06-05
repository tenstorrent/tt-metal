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
struct OneFromAllConfig {
    uint32_t test_id = 0;
    CoreCoord master_core_coord = CoreCoord();
    CoreRangeSet subordinate_core_set;
    uint32_t num_of_transactions = 0;
    uint32_t transaction_size_pages = 0;
    uint32_t page_size_bytes = 0;
    DataFormat l1_data_format = DataFormat::Invalid;

    // TODO: Add the following parameters
    //  1. Virtual Channel
    //  2. Which NOC to use
};

/// @brief Does Gatherer Core --> L1 Responder Cores --> L1 Gatherer Core
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool run_dm(IDevice* device, const OneFromAllConfig& test_config) {
    // Program
    Program program = CreateProgram();

    // Sharded L1 buffers
    CoreRangeSet master_core_set({CoreRange(test_config.master_core_coord)});
    size_t total_subordinate_cores = test_config.subordinate_core_set.num_cores();

    const size_t total_size_bytes = test_config.num_of_transactions * test_config.transaction_size_pages *
                                    test_config.page_size_bytes * total_subordinate_cores;
    const size_t total_size_pages =
        test_config.num_of_transactions * test_config.transaction_size_pages * total_subordinate_cores;

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

    const size_t subordinate_size_bytes =
        test_config.num_of_transactions * test_config.transaction_size_pages * test_config.page_size_bytes;
    const size_t subordinate_size_pages = test_config.num_of_transactions * test_config.transaction_size_pages;

    // Compile-time arguments for kernels
    vector<uint32_t> gatherer_compile_args = {
        (uint32_t)master_l1_byte_address,
        (uint32_t)test_config.num_of_transactions,
        (uint32_t)test_config.transaction_size_pages,
        (uint32_t)test_config.page_size_bytes,
        (uint32_t)test_config.test_id,
        (uint32_t)total_subordinate_cores,
    };

    // Kernels
    auto gatherer_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/data_movement/one_from_all/kernels/gatherer.cpp",
        master_core_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = gatherer_compile_args});

    // Runtime Arguments
    vector<uint32_t> master_runtime_args;

    // Create buffers for each subordinate core and add to runtime args
    vector<shared_ptr<Buffer>> subordinate_l1_buffers;
    for (auto& core : corerange_to_cores(test_config.subordinate_core_set)) {
        auto subordinate_shard_parameters = ShardSpecBuffer(
            CoreRangeSet({CoreRange(core)}),
            {1, subordinate_size_bytes / 2},
            ShardOrientation::ROW_MAJOR,
            {1, test_config.page_size_bytes / 2},
            {1, subordinate_size_pages});
        ShardedBufferConfig subordinate_buffer_config{
            .device = device,
            .size = subordinate_size_bytes,
            .page_size = test_config.page_size_bytes,
            .buffer_type = BufferType::L1,
            .buffer_layout = TensorMemoryLayout::WIDTH_SHARDED,
            .shard_parameters = std::move(subordinate_shard_parameters),
        };
        auto subordinate_l1_buffer = CreateBuffer(subordinate_buffer_config);
        subordinate_l1_buffers.push_back(subordinate_l1_buffer);
        master_runtime_args.push_back((uint32_t)subordinate_l1_buffer->address());

        CoreCoord physical_core = device->worker_core_from_logical_core(core);
        master_runtime_args.push_back(physical_core.x);
        master_runtime_args.push_back(physical_core.y);
    }
    SetRuntimeArgs(program, gatherer_kernel, master_core_set, master_runtime_args);

    // Assign unique id
    log_info(tt::LogTest, "Running Test ID: {}, Run ID: {}", test_config.test_id, unit_tests::dm::runtime_host_id);
    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    // Input
    vector<uint32_t> packed_input = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -100.0f, 100.0f, total_size_bytes / bfloat16::SIZEOF, chrono::system_clock::now().time_since_epoch().count());

    // Golden output
    vector<uint32_t> packed_golden = packed_input;
    vector<uint32_t> packed_output;

    // Launch program and record outputs
    for (size_t i = 0; i < total_subordinate_cores; i++) {
        auto begin = packed_input.data() + i * subordinate_size_bytes / sizeof(uint32_t);
        auto end = packed_input.data() + (i + 1) * subordinate_size_bytes / sizeof(uint32_t);
        detail::WriteToBuffer(subordinate_l1_buffers[i], vector<uint32_t>(begin, end));
    }
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

/* ========== Test case for one from all data movement; Test id = 15 ========== */
TEST_F(DeviceFixture, TensixDataMovementOneFromAllPacketSizes) {
    // Parameters
    uint32_t max_transactions = 256;
    uint32_t max_transaction_size_pages = 64;
    uint32_t page_size_bytes = arch_ == tt::ARCH::BLACKHOLE ? 64 : 32;  // =Flit size: 32 bytes for WH, 64 for BH

    // Cores
    CoreCoord master_core_coord = {0, 0};
    CoreRangeSet subordinate_core_set = {CoreRange(CoreCoord(1, 1), CoreCoord(4, 4))};
    size_t total_subordinate_cores = subordinate_core_set.num_cores();

    uint32_t l1_size = 1024 * 1024;  // 1MB

    for (uint32_t num_of_transactions = 1; num_of_transactions <= max_transactions; num_of_transactions *= 4) {
        for (uint32_t transaction_size_pages = 1; transaction_size_pages <= max_transaction_size_pages;
             transaction_size_pages *= 2) {
            if (num_of_transactions * transaction_size_pages * page_size_bytes * total_subordinate_cores >=
                1024 * 1024) {
                continue;
            }

            // Test config
            unit_tests::dm::core_to_core::OneFromAllConfig test_config = {
                .test_id = 15,
                .master_core_coord = master_core_coord,
                .subordinate_core_set = subordinate_core_set,
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

/* ========== Test case for one from all data movement; Test id = 30 ========== */
TEST_F(DeviceFixture, TensixDataMovementOneFromAllDirectedIdeal) {
    // Parameters
    uint32_t num_of_transactions, page_size_bytes;
    uint32_t transaction_size_pages = 128;
    if (arch_ == tt::ARCH::BLACKHOLE) {
        page_size_bytes = 64;  // (=flit size): 64 bytes for BH
        num_of_transactions = 5;
    } else {
        page_size_bytes = 32;  // (=flit size): 32 bytes for WH
        num_of_transactions = 10;
    }

    // Cores
    CoreCoord master_core_coord = {0, 0};
    CoreRangeSet subordinate_core_set = {CoreRange(CoreCoord(1, 1), CoreCoord(4, 4))};
    size_t total_subordinate_cores = subordinate_core_set.num_cores();

    // Test config
    unit_tests::dm::core_to_core::OneFromAllConfig test_config = {
        .test_id = 30,
        .master_core_coord = master_core_coord,
        .subordinate_core_set = subordinate_core_set,
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
