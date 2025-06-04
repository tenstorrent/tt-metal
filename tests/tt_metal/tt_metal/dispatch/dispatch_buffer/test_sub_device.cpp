// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/sub_device.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include "command_queue_fixture.hpp"
#include "gtest/gtest.h"
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/host_api.hpp>
#include "llrt.hpp"
#include <tt_stl/span.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include "tt_metal/test_utils/stimulus.hpp"
#include "umd/device/types/xy_pair.h"

namespace tt::tt_metal {

TEST_F(CommandQueueSingleCardFixture, TensixTestSubDeviceAllocations) {
    uint32_t local_l1_size = 3200;
    SubDevice sub_device_1(std::array{CoreRangeSet(CoreRange({0, 0}, {2, 2}))});
    SubDevice sub_device_2(std::array{CoreRangeSet(std::vector{CoreRange({3, 3}, {3, 3}), CoreRange({4, 4}, {4, 4})})});
    CoreRangeSet sharded_cores_1 = CoreRange({0, 0}, {2, 2});
    CoreRangeSet sharded_cores_2 = CoreRange({4, 4}, {4, 4});

    auto device = devices_[0];
    auto sub_device_manager_1 = device->create_sub_device_manager({sub_device_1}, local_l1_size);
    auto sub_device_manager_2 = device->create_sub_device_manager({sub_device_1, sub_device_2}, local_l1_size);
    DeviceAddr l1_unreserved_base = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    DeviceAddr max_addr = l1_unreserved_base + local_l1_size;

    auto sharded_cores_1_vec = corerange_to_cores(sharded_cores_1, std::nullopt, true);
    auto sharded_cores_2_vec = corerange_to_cores(sharded_cores_2, std::nullopt, true);

    ShardSpecBuffer shard_spec_buffer_1 =
        ShardSpecBuffer(sharded_cores_1, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {sharded_cores_1.num_cores(), 1});
    uint32_t page_size_1 = 32;
    ShardedBufferConfig shard_config_1 = {
        device,
        sharded_cores_1.num_cores() * page_size_1,
        page_size_1,
        BufferType::L1,
        TensorMemoryLayout::HEIGHT_SHARDED,
        shard_spec_buffer_1};
    auto input_1 =
        tt::test_utils::generate_uniform_random_vector<uint32_t>(0, 100, shard_config_1.size / sizeof(uint32_t));

    ShardSpecBuffer shard_spec_buffer_2 =
        ShardSpecBuffer(sharded_cores_2, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {sharded_cores_2.num_cores(), 1});
    uint32_t page_size_2 = 64;
    ShardedBufferConfig shard_config_2 = {
        device,
        sharded_cores_2.num_cores() * page_size_2,
        page_size_2,
        BufferType::L1,
        TensorMemoryLayout::HEIGHT_SHARDED,
        shard_spec_buffer_2};
    auto input_2 =
        tt::test_utils::generate_uniform_random_vector<uint32_t>(0, 100, shard_config_2.size / sizeof(uint32_t));

    uint32_t page_size_3 = 1024;
    InterleavedBufferConfig interleaved_config = {
        device, page_size_3, page_size_3, BufferType::L1, TensorMemoryLayout::INTERLEAVED};
    auto input_3 =
        tt::test_utils::generate_uniform_random_vector<uint32_t>(0, 100, interleaved_config.size / sizeof(uint32_t));

    std::vector<CoreCoord> physical_cores_1;
    physical_cores_1.reserve(sharded_cores_1_vec.size());
    for (const auto& core : sharded_cores_1_vec) {
        physical_cores_1.push_back(device->worker_core_from_logical_core(core));
    }

    std::vector<CoreCoord> physical_cores_2;
    physical_cores_2.reserve(sharded_cores_2_vec.size());
    for (const auto& core : sharded_cores_2_vec) {
        physical_cores_2.push_back(device->worker_core_from_logical_core(core));
    }

    device->load_sub_device_manager(sub_device_manager_1);

    auto buffer_1 = CreateBuffer(shard_config_1, SubDeviceId{0});
    EXPECT_TRUE(buffer_1->address() <= max_addr - buffer_1->aligned_page_size());
    EnqueueWriteBuffer(device->command_queue(), buffer_1, input_1, false);
    std::vector<uint32_t> output_1;
    EnqueueReadBuffer(device->command_queue(), buffer_1, output_1, true);
    EXPECT_EQ(input_1, output_1);
    auto input_1_it = input_1.begin();
    for (const auto& physical_core : physical_cores_1) {
        auto readback = tt::llrt::read_hex_vec_from_core(device->id(), physical_core, buffer_1->address(), page_size_1);
        EXPECT_TRUE(std::equal(input_1_it, input_1_it + page_size_1 / sizeof(uint32_t), readback.begin()));
        input_1_it += page_size_1 / sizeof(uint32_t);
    }

    auto buffer_2 = CreateBuffer(interleaved_config);
    EXPECT_THROW(CreateBuffer(shard_config_1, SubDeviceId{1}), std::exception);
    EXPECT_THROW(device->clear_loaded_sub_device_manager(), std::exception);
    EXPECT_THROW(device->load_sub_device_manager(sub_device_manager_2), std::exception);
    DeallocateBuffer(*buffer_1);
    device->clear_loaded_sub_device_manager();
    device->load_sub_device_manager(sub_device_manager_2);

    auto buffer_3 = CreateBuffer(shard_config_2, SubDeviceId{1});
    EXPECT_TRUE(buffer_3->address() <= max_addr - buffer_3->aligned_page_size());
    EnqueueWriteBuffer(device->command_queue(), buffer_3, input_2, false);
    std::vector<uint32_t> output_2;
    EnqueueReadBuffer(device->command_queue(), buffer_3, output_2, true);
    EXPECT_EQ(input_2, output_2);
    auto input_2_it = input_2.begin();
    for (const auto& physical_core : physical_cores_2) {
        auto readback = tt::llrt::read_hex_vec_from_core(device->id(), physical_core, buffer_3->address(), page_size_2);
        EXPECT_TRUE(std::equal(input_2_it, input_2_it + page_size_2 / sizeof(uint32_t), readback.begin()));
        input_2_it += page_size_2 / sizeof(uint32_t);
    }

    auto buffer_4 = CreateBuffer(shard_config_1, SubDeviceId{0});
    EXPECT_TRUE(buffer_4->address() <= max_addr - buffer_4->aligned_page_size());
    EXPECT_THROW(CreateBuffer(interleaved_config, SubDeviceId{0}), std::exception);
}

TEST_F(CommandQueueSingleCardFixture, TensixTestSubDeviceBankIds) {
    uint32_t local_l1_size = 3200;
    SubDevice sub_device(std::array{
        CoreRangeSet(std::array{CoreRange({5, 4}, {5, 4}), CoreRange({3, 2}, {3, 3}), CoreRange({0, 0}, {2, 2})})});

    auto device = devices_[0];
    auto sub_device_manager = device->create_sub_device_manager({sub_device}, local_l1_size);
    device->load_sub_device_manager(sub_device_manager);

    auto cores_vec = corerange_to_cores(device->worker_cores(HalProgrammableCoreType::TENSIX, SubDeviceId{0}));
    for (const auto& core : cores_vec) {
        auto global_bank_id = device->allocator()->get_bank_ids_from_logical_core(BufferType::L1, core)[0];
        auto sub_device_bank_id =
            device->allocator(SubDeviceId{0})->get_bank_ids_from_logical_core(BufferType::L1, core)[0];
        EXPECT_EQ(global_bank_id, sub_device_bank_id);
    }
}

}  // namespace tt::tt_metal
