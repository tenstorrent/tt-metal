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

namespace unit_tests::dm::core_loopback {

constexpr uint32_t START_ID = 16;

// Test config, i.e. test parameters
struct LoopbackConfig {
    uint32_t test_id = 0;
    CoreCoord master_core_coord = CoreCoord();
    uint32_t num_of_transactions = 0;
    uint32_t transaction_size_pages = 0;
    uint32_t page_size_bytes = 0;
    DataFormat l1_data_format = DataFormat::Invalid;
    NOC noc_id = NOC::NOC_0;

    // TODO: Add the following parameters
    //  1. Virtual Channel (only useful for unicast)
    //  2. Posted flag (posted multicast has much better performance at larger grid sizes, than non-posted due to
    //  response packets) (60, 45, 23, vs 60, 60, 60 at posted)
};

/// @brief Does L1 Sender Core --> L1 Receiver Cores
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool run_dm(IDevice* device, const LoopbackConfig& test_config) {
    // Program
    Program program = CreateProgram();

    // Sharded L1 buffers
    const uint32_t transaction_size_bytes = test_config.transaction_size_pages * test_config.page_size_bytes;

    CoreRangeSet master_core_set({CoreRange(test_config.master_core_coord)});
    CoreRangeSet subordinate_core_set({CoreRange(test_config.master_core_coord)});

    auto master_shard_parameters = ShardSpecBuffer(
        master_core_set,
        {1, transaction_size_bytes / 2},
        ShardOrientation::ROW_MAJOR,
        {1, test_config.page_size_bytes / 2},
        {1, test_config.transaction_size_pages});
    auto master_l1_buffer = CreateBuffer(ShardedBufferConfig{
        .device = device,
        .size = transaction_size_bytes,
        .page_size = test_config.page_size_bytes,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::WIDTH_SHARDED,
        .shard_parameters = std::move(master_shard_parameters),
    });
    uint32_t master_l1_byte_address = master_l1_buffer->address();

    auto subordinate_shard_parameters = ShardSpecBuffer(
        subordinate_core_set,
        {1, transaction_size_bytes / 2},
        ShardOrientation::ROW_MAJOR,
        {1, test_config.page_size_bytes / 2},
        {1, test_config.transaction_size_pages});
    auto subordinate_l1_buffer = CreateBuffer(ShardedBufferConfig{
        .device = device,
        .size = transaction_size_bytes,
        .page_size = test_config.page_size_bytes,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::WIDTH_SHARDED,
        .shard_parameters = std::move(subordinate_shard_parameters),
    });
    uint32_t subordinate_l1_byte_address = subordinate_l1_buffer->address();

    // Compile-time arguments for kernels
    vector<uint32_t> sender_compile_args = {
        (uint32_t)master_l1_byte_address,
        (uint32_t)subordinate_l1_byte_address,
        (uint32_t)test_config.num_of_transactions,
        (uint32_t)test_config.transaction_size_pages,
        (uint32_t)test_config.page_size_bytes,
        (uint32_t)test_config.test_id};

    // Kernels
    auto sender_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/data_movement/loopback/kernels/sender.cpp",
        master_core_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = test_config.noc_id,
            .compile_args = sender_compile_args});

    // Semaphores
    CoreRangeSet sem_core_set = subordinate_core_set.merge<CoreRangeSet>(master_core_set);
    const uint32_t sem_id = CreateSemaphore(program, sem_core_set, 0);

    // Runtime Arguments
    CoreCoord worker = device->worker_core_from_logical_core(test_config.master_core_coord);
    std::vector<uint32_t> master_run_args = {sem_id, worker.x, worker.y};
    SetRuntimeArgs(program, sender_kernel, master_core_set, master_run_args);

    // Assign unique id
    log_info(tt::LogTest, "Running Test ID: {}, Run ID: {}", test_config.test_id, unit_tests::dm::runtime_host_id);
    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    // Input
    vector<uint32_t> packed_input = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -100.0f,
        100.0f,
        transaction_size_bytes / bfloat16::SIZEOF,
        chrono::system_clock::now().time_since_epoch().count());

    // Golden output
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    vector<uint32_t> packed_golden = packed_input;

    // Launch program and record outputs
    detail::WriteToBuffer(master_l1_buffer, packed_input);
    MetalContext::instance().get_cluster().l1_barrier(device->id());
    detail::LaunchProgram(device, program);

    vector<uint32_t> packed_output;
    detail::ReadFromBuffer(subordinate_l1_buffer, packed_output);
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
}  // namespace unit_tests::dm::core_loopback

/* ========== Test case for loopback data movement; ========== */
TEST_F(DeviceFixture, TensixDataMovementLoopbackPacketSizes) {
    // Parameters
    uint32_t max_transactions = 256;
    uint32_t max_transaction_size_pages =
        arch_ == tt::ARCH::BLACKHOLE ? 1024 : 2048;                     // Max total transaction size == 64 KB
    uint32_t page_size_bytes = arch_ == tt::ARCH::BLACKHOLE ? 64 : 32;  // =Flit size: 32 bytes for WH, 64 for BH
    CoreCoord master_core_coord = {0, 0};
    NOC noc_id = NOC::NOC_0;

    for (uint32_t num_of_transactions = 1; num_of_transactions <= max_transactions; num_of_transactions *= 4) {
        for (uint32_t transaction_size_pages = 1; transaction_size_pages <= max_transaction_size_pages;
             transaction_size_pages *= 2) {
            // Test config
            unit_tests::dm::core_loopback::LoopbackConfig test_config = {
                .test_id = unit_tests::dm::core_loopback::START_ID + 0,
                .master_core_coord = master_core_coord,
                .num_of_transactions = num_of_transactions,
                .transaction_size_pages = transaction_size_pages,
                .page_size_bytes = page_size_bytes,
                .l1_data_format = DataFormat::Float16_b,
                .noc_id = noc_id,
            };

            // Run
            for (unsigned int id = 0; id < num_devices_; id++) {
                EXPECT_TRUE(run_dm(devices_.at(id), test_config));
            }
        }
    }
}

}  // namespace tt::tt_metal
