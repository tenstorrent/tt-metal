// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "device_fixture.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"

namespace tt::tt_metal {

using namespace std;
using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::dm::core_to_core {

uint32_t runtime_host_id = 0;

// Test config, i.e. test parameters
struct OneToOneConfig {
    uint32_t test_id = 0;
    CoreCoord master_core_coord = CoreCoord();
    CoreCoord slave_core_coord = CoreCoord();
    uint32_t num_of_transactions = 0;
    uint32_t transaction_size_pages = 0;
    uint32_t page_size_bytes = 0;
    DataFormat l1_data_format = DataFormat::Invalid;

    // TODO: Add the following parameters
    //  1. Virtual Channel
    //  2. Which NOC to use
    //  3. Posted flag
};

/// @brief Does Dram --> Reader --> CB --> Writer --> Dram. // TODO: Change description
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool run_dm(IDevice* device, const OneToOneConfig& test_config) {
    // TODO: Steps to produce L1 to L1 tests
    //  1. Create sharded buffers for sender and receiver
    //  2. Create kernels for sender and receiver cores
    //  3. Input data into sender buffer (Do we need to wait for data to be written into L1?)
    //  4. Launch program (maybe two programs: one for sender, one for receiver)
    //  5. Read data from receiver buffer

    // Program
    Program program = CreateProgram();

    // Sharded L1 buffers
    const size_t total_size_bytes =
        test_config.num_of_transactions * test_config.transaction_size_pages * test_config.page_size_bytes;
    const size_t total_size_pages = test_config.num_of_transactions * test_config.transaction_size_pages;

    CoreRangeSet master_core_set({CoreRange(test_config.master_core_coord)});
    CoreRangeSet slave_core_set({CoreRange(test_config.slave_core_coord)});

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

    auto slave_shard_parameters = ShardSpecBuffer(
        slave_core_set,
        {1, total_size_bytes / 2},
        ShardOrientation::ROW_MAJOR,
        {1, test_config.page_size_bytes / 2},
        {1, total_size_pages});
    auto slave_l1_buffer = CreateBuffer(ShardedBufferConfig{
        .device = device,
        .size = total_size_bytes,
        .page_size = test_config.page_size_bytes,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::WIDTH_SHARDED,
        .shard_parameters = std::move(slave_shard_parameters),
    });
    uint32_t slave_l1_byte_address = slave_l1_buffer->address();

    // Compile-time arguments for kernels // TODO: Change compiletime args
    vector<uint32_t> sender_compile_args = {
        (uint32_t)master_l1_byte_address,
        (uint32_t)slave_l1_byte_address,
        (uint32_t)test_config.num_of_transactions,
        (uint32_t)test_config.transaction_size_pages,
        (uint32_t)test_config.page_size_bytes,
        (uint32_t)test_config.test_id};

    vector<uint32_t> receiver_compile_args = {
        (uint32_t)master_l1_byte_address,
        (uint32_t)slave_l1_byte_address,
        (uint32_t)test_config.num_of_transactions,
        (uint32_t)test_config.transaction_size_pages,
        (uint32_t)test_config.page_size_bytes,
        (uint32_t)test_config.test_id};

    // Kernels
    auto sender_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/dm/sender_core_to_core.cpp",
        master_core_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = sender_compile_args});

    auto receiver_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/dm/receiver_core_to_core.cpp",
        slave_core_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = receiver_compile_args});

    // Semaphores
    CoreRangeSet sem_core_set = slave_core_set.merge<CoreRangeSet>(master_core_set);
    const uint32_t sem_id = CreateSemaphore(program, sem_core_set, 0);
    CoreCoord physical_slave_core = device->worker_core_from_logical_core(test_config.slave_core_coord);

    // Runtime Arguments
    SetRuntimeArgs(program, sender_kernel, master_core_set, {sem_id, physical_slave_core.x, physical_slave_core.y});
    SetRuntimeArgs(program, receiver_kernel, slave_core_set, {sem_id});

    // Assign unique id
    log_info("Results for test id: {}", test_config.test_id);
    log_info("Results for run id: {}", runtime_host_id);
    program.set_runtime_id(runtime_host_id++);

    // Input
    vector<uint32_t> packed_input = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -100.0f, 100.0f, total_size_bytes / bfloat16::SIZEOF, chrono::system_clock::now().time_since_epoch().count());

    // Golden output
    vector<uint32_t> packed_golden = packed_input;

    // Launch program and record outputs
    vector<uint32_t> packed_output;
    detail::WriteToBuffer(master_l1_buffer, packed_input);
    MetalContext::instance().get_cluster().l1_barrier(device->id());
    detail::LaunchProgram(device, program);
    detail::ReadFromBuffer(slave_l1_buffer, packed_output);

    // Print output and golden vectors
    log_info("Golden vector");
    print_vector<uint32_t>(packed_golden);
    log_info("Output vector");
    print_vector<uint32_t>(packed_output);

    // Return comparison
    return is_close_packed_vectors<bfloat16, uint32_t>(
        packed_output, packed_golden, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b); });
}
}  // namespace unit_tests::dm::core_to_core

/* ========== Test case for one to one data movement; Test id = 3 ========== */
TEST_F(DeviceFixture, TensixDataMovementOneToOne) {
    // Parameters
    uint32_t num_of_transactions = 1;
    uint32_t transaction_size_pages = 1;
    uint32_t page_size_bytes = 32;  // =Flit size: 32 bytes for WH, 64 for BH
    if (arch_ == tt::ARCH::BLACKHOLE) {
        page_size_bytes *= 2;
    }

    // Cores
    CoreCoord master_core_coord = {0, 0};
    CoreCoord slave_core_coord = {1, 1};

    // Test config
    unit_tests::dm::core_to_core::OneToOneConfig test_config = {
        .test_id = 3,
        .master_core_coord = master_core_coord,
        .slave_core_coord = slave_core_coord,
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
