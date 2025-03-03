// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <array>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/event.hpp>
#include <tt-metalium/sub_device.hpp>
#include "tt_metal/test_utils/stimulus.hpp"
#include "command_queue_fixture.hpp"
#include "sub_device_test_utils.hpp"
#include "dispatch_test_utils.hpp"

namespace tt::tt_metal {

TEST_F(CommandQueueSingleCardFixture, TensixTestSubDeviceSynchronization) {
    auto* device = devices_[0];
    uint32_t local_l1_size = 3200;
    SubDevice sub_device_1(std::array{CoreRangeSet(CoreRange({0, 0}, {2, 2}))});
    SubDevice sub_device_2(std::array{CoreRangeSet(std::vector{CoreRange({3, 3}, {3, 3}), CoreRange({4, 4}, {4, 4})})});
    CoreRangeSet sharded_cores_1 = CoreRange({0, 0}, {2, 2});

    auto sharded_cores_1_vec = corerange_to_cores(sharded_cores_1, std::nullopt, true);

    ShardSpecBuffer shard_spec_buffer_1 =
        ShardSpecBuffer(sharded_cores_1, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {sharded_cores_1.num_cores(), 1});
    uint32_t page_size_1 = 32;
    ShardedBufferConfig shard_config_1 = {
        nullptr,
        sharded_cores_1.num_cores() * page_size_1,
        page_size_1,
        BufferType::L1,
        TensorMemoryLayout::HEIGHT_SHARDED,
        shard_spec_buffer_1};
    auto input_1 =
        tt::test_utils::generate_uniform_random_vector<uint32_t>(0, 100, shard_config_1.size / sizeof(uint32_t));

    std::array sub_device_ids_to_block = {SubDeviceId{0}};

    auto sub_device_manager = device->create_sub_device_manager({sub_device_1, sub_device_2}, local_l1_size);

    shard_config_1.device = device;

    std::vector<CoreCoord> physical_cores_1;
    physical_cores_1.reserve(sharded_cores_1_vec.size());
    for (const auto& core : sharded_cores_1_vec) {
        physical_cores_1.push_back(device->worker_core_from_logical_core(core));
    }

    device->load_sub_device_manager(sub_device_manager);

    auto [program, syncer_core, global_semaphore] = create_single_sync_program(device, sub_device_2);
    EnqueueProgram(device->command_queue(), program, false);
    device->set_sub_device_stall_group(sub_device_ids_to_block);

    auto buffer_1 = CreateBuffer(shard_config_1, sub_device_ids_to_block[0]);

    // Test blocking synchronize doesn't stall
    Synchronize(device, 0);

    // Test blocking write buffer doesn't stall
    EnqueueWriteBuffer(device->command_queue(), buffer_1, input_1, true);

    // Test record event won't cause a stall
    auto event = std::make_shared<Event>();
    EnqueueRecordEvent(device->command_queue(), event);
    Synchronize(device, 0);

    // Test blocking read buffer doesn't stall
    std::vector<uint32_t> output_1;
    EnqueueReadBuffer(device->command_queue(), buffer_1, output_1, true);
    EXPECT_EQ(input_1, output_1);
    auto input_1_it = input_1.begin();
    for (const auto& physical_core : physical_cores_1) {
        auto readback = tt::llrt::read_hex_vec_from_core(device->id(), physical_core, buffer_1->address(), page_size_1);
        EXPECT_TRUE(std::equal(input_1_it, input_1_it + page_size_1 / sizeof(uint32_t), readback.begin()));
        input_1_it += page_size_1 / sizeof(uint32_t);
    }
    auto sem_addr = global_semaphore.address();
    auto physical_syncer_core = device->worker_core_from_logical_core(syncer_core);
    tt::llrt::write_hex_vec_to_core(device->id(), physical_syncer_core, std::vector<uint32_t>{1}, sem_addr);

    // Full synchronization
    device->reset_sub_device_stall_group();
    Synchronize(device);
}

TEST_F(CommandQueueSingleCardFixture, TensixTestSubDeviceBasicPrograms) {
    auto* device = devices_[0];
    SubDevice sub_device_1(std::array{CoreRangeSet(CoreRange({0, 0}, {2, 2}))});
    SubDevice sub_device_2(std::array{CoreRangeSet(std::vector{CoreRange({3, 3}, {3, 3}), CoreRange({4, 4}, {4, 4})})});
    uint32_t num_iters = 5;
    auto sub_device_manager = device->create_sub_device_manager({sub_device_1, sub_device_2}, 3200);
    device->load_sub_device_manager(sub_device_manager);

    auto [waiter_program, syncer_program, incrementer_program, global_sem] =
        create_basic_sync_program(device, sub_device_1, sub_device_2);

    for (uint32_t i = 0; i < num_iters; i++) {
        EnqueueProgram(device->command_queue(), waiter_program, false);
        device->set_sub_device_stall_group({SubDeviceId{0}});
        // Test blocking on one sub-device
        EnqueueProgram(device->command_queue(), syncer_program, true);
        EnqueueProgram(device->command_queue(), incrementer_program, false);
        device->reset_sub_device_stall_group();
    }
    Synchronize(device);
    detail::DumpDeviceProfileResults(device);
}

TEST_F(CommandQueueSingleCardFixture, TensixActiveEthTestSubDeviceBasicEthPrograms) {
    auto* device = devices_[0];
    SubDevice sub_device_1(std::array{CoreRangeSet(CoreRange({0, 0}, {2, 2}))});
    uint32_t num_iters = 5;
    if (!does_device_have_active_eth_cores(device)) {
        GTEST_SKIP() << "Skipping test because device " << device->id() << " does not have any active ethernet cores";
    }
    auto eth_core = *device->get_active_ethernet_cores(true).begin();
    SubDevice sub_device_2(std::array{
        CoreRangeSet(std::vector{CoreRange({3, 3}, {3, 3}), CoreRange({4, 4}, {4, 4})}),
        CoreRangeSet(CoreRange(eth_core, eth_core))});
    auto sub_device_manager = device->create_sub_device_manager({sub_device_1, sub_device_2}, 3200);
    device->load_sub_device_manager(sub_device_manager);

    auto [waiter_program, syncer_program, incrementer_program, global_sem] =
        create_basic_eth_sync_program(device, sub_device_1, sub_device_2);

    for (uint32_t i = 0; i < num_iters; i++) {
        EnqueueProgram(device->command_queue(), waiter_program, false);
        device->set_sub_device_stall_group({SubDeviceId{0}});
        // Test blocking on one sub-device
        EnqueueProgram(device->command_queue(), syncer_program, true);
        EnqueueProgram(device->command_queue(), incrementer_program, false);
        device->reset_sub_device_stall_group();
    }
    Synchronize(device);
    detail::DumpDeviceProfileResults(device);
}

}  // namespace tt::tt_metal
