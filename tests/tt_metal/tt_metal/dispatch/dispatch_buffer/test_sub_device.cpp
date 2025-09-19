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
#include <umd/device/types/xy_pair.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_buffer.hpp>

namespace tt::tt_metal {

TEST_F(UnitMeshCQSingleCardFixture, TensixTestSubDeviceAllocations) {
    uint32_t local_l1_size = 3200;
    SubDevice sub_device_1(std::array{CoreRangeSet(CoreRange({0, 0}, {2, 2}))});
    SubDevice sub_device_2(std::array{CoreRangeSet(std::vector{CoreRange({3, 3}, {3, 3}), CoreRange({4, 4}, {4, 4})})});
    CoreRangeSet sharded_cores_1 = CoreRange({0, 0}, {2, 2});
    CoreRangeSet sharded_cores_2 = CoreRangeSet(std::vector{CoreRange({3, 3}, {3, 3}), CoreRange({4, 4}, {4, 4})});

    auto mesh_device = devices_[0];
    auto sub_device_manager_1 = mesh_device->create_sub_device_manager({sub_device_1}, local_l1_size);
    auto sub_device_manager_2 = mesh_device->create_sub_device_manager({sub_device_1, sub_device_2}, local_l1_size);
    DeviceAddr l1_unreserved_base = mesh_device->allocator()->get_base_allocator_addr(HalMemType::L1);
    DeviceAddr max_addr = l1_unreserved_base + local_l1_size;

    auto sharded_cores_1_vec = corerange_to_cores(sharded_cores_1, std::nullopt, true);
    auto sharded_cores_2_vec = corerange_to_cores(sharded_cores_2, std::nullopt, true);

    uint32_t page_size_1 = 32;
    distributed::ReplicatedBufferConfig replicated_config_1 = {
        sharded_cores_1.num_cores() * page_size_1,
    };

    ShardSpecBuffer shard_spec_buffer_1 =
        ShardSpecBuffer(sharded_cores_1, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {sharded_cores_1.num_cores(), 1});
    distributed::DeviceLocalBufferConfig local_config_1 = {
        .page_size = page_size_1,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(shard_spec_buffer_1, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = false,
        .sub_device_id = SubDeviceId{0}};
    auto input_1 = tt::test_utils::generate_uniform_random_vector<uint32_t>(
        0, 100, sharded_cores_1.num_cores() * page_size_1 / sizeof(uint32_t));

    ShardSpecBuffer shard_spec_buffer_2 =
        ShardSpecBuffer(sharded_cores_2, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {sharded_cores_2.num_cores(), 1});
    uint32_t page_size_2 = 64;
    distributed::ReplicatedBufferConfig replicated_config_2 = {
        sharded_cores_2.num_cores() * page_size_2,
    };
    distributed::DeviceLocalBufferConfig local_config_2 = {
        .page_size = page_size_2,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(shard_spec_buffer_2, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = false,
        .sub_device_id = SubDeviceId{1}};
    auto input_2 = tt::test_utils::generate_uniform_random_vector<uint32_t>(
        0, 100, sharded_cores_2.num_cores() * page_size_2 / sizeof(uint32_t));

    uint32_t page_size_3 = 1024;
    distributed::ReplicatedBufferConfig replicated_config_3 = {
        page_size_3,
    };
    distributed::DeviceLocalBufferConfig local_config_3 = {
        .page_size = page_size_3,
        .buffer_type = BufferType::L1,
        .sharding_args = std::nullopt,
        .bottom_up = false,
        .sub_device_id = std::nullopt};
    auto input_3 = tt::test_utils::generate_uniform_random_vector<uint32_t>(0, 100, page_size_3 / sizeof(uint32_t));

    std::vector<CoreCoord> physical_cores_1;
    physical_cores_1.reserve(sharded_cores_1_vec.size());
    for (const auto& core : sharded_cores_1_vec) {
        physical_cores_1.push_back(mesh_device->worker_core_from_logical_core(core));
    }

    std::vector<CoreCoord> physical_cores_2;
    physical_cores_2.reserve(sharded_cores_2_vec.size());
    for (const auto& core : sharded_cores_2_vec) {
        physical_cores_2.push_back(mesh_device->worker_core_from_logical_core(core));
    }

    mesh_device->load_sub_device_manager(sub_device_manager_1);

    auto buffer_1 = distributed::MeshBuffer::create(replicated_config_1, local_config_1, mesh_device.get());
    EXPECT_TRUE(buffer_1->address() <= max_addr - buffer_1->get_backing_buffer()->aligned_page_size());
    distributed::EnqueueWriteMeshBuffer(mesh_device->mesh_command_queue(), buffer_1, input_1, false);
    std::vector<uint32_t> output_1;
    distributed::ReadShard(mesh_device->mesh_command_queue(), output_1, buffer_1, zero_coord_, true);
    EXPECT_EQ(input_1, output_1);
    auto input_1_it = input_1.begin();
    for (const auto& physical_core : physical_cores_1) {
        auto readback = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
            mesh_device->get_devices()[0]->id(), physical_core, buffer_1->address(), page_size_1);
        EXPECT_TRUE(std::equal(input_1_it, input_1_it + page_size_1 / sizeof(uint32_t), readback.begin()));
        input_1_it += page_size_1 / sizeof(uint32_t);
    }

    auto buffer_2 = distributed::MeshBuffer::create(replicated_config_3, local_config_3, mesh_device.get());
    local_config_1.sub_device_id = SubDeviceId{1};
    EXPECT_THROW(
        distributed::MeshBuffer::create(replicated_config_1, local_config_1, mesh_device.get()), std::exception);
    EXPECT_THROW(mesh_device->clear_loaded_sub_device_manager(), std::exception);
    EXPECT_THROW(mesh_device->load_sub_device_manager(sub_device_manager_2), std::exception);
    buffer_1->deallocate();
    buffer_2->deallocate();
    mesh_device->clear_loaded_sub_device_manager();
    mesh_device->load_sub_device_manager(sub_device_manager_2);

    auto buffer_3 = distributed::MeshBuffer::create(replicated_config_2, local_config_2, mesh_device.get());
    EXPECT_TRUE(buffer_3->address() <= max_addr - buffer_3->get_backing_buffer()->aligned_page_size());
    distributed::EnqueueWriteMeshBuffer(mesh_device->mesh_command_queue(), buffer_3, input_2, false);
    std::vector<uint32_t> output_2;
    distributed::ReadShard(mesh_device->mesh_command_queue(), output_2, buffer_3, zero_coord_, true);
    EXPECT_EQ(input_2, output_2);
    auto input_2_it = input_2.begin();
    for (const auto& physical_core : physical_cores_2) {
        auto readback = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
            mesh_device->get_devices()[0]->id(), physical_core, buffer_3->address(), page_size_2);
        EXPECT_TRUE(std::equal(input_2_it, input_2_it + page_size_2 / sizeof(uint32_t), readback.begin()));
        input_2_it += page_size_2 / sizeof(uint32_t);
    }
    local_config_1.sub_device_id = SubDeviceId{0};
    auto buffer_4 = distributed::MeshBuffer::create(replicated_config_1, local_config_1, mesh_device.get());
    EXPECT_TRUE(buffer_4->address() <= max_addr - buffer_4->get_backing_buffer()->aligned_page_size());
    local_config_3.sub_device_id = SubDeviceId{0};
    EXPECT_THROW(
        distributed::MeshBuffer::create(replicated_config_3, local_config_3, mesh_device.get()), std::exception);
}

TEST_F(UnitMeshCQSingleCardFixture, TensixTestSubDeviceBankIds) {
    uint32_t local_l1_size = 3200;
    SubDevice sub_device(std::array{
        CoreRangeSet(std::array{CoreRange({5, 4}, {5, 4}), CoreRange({3, 2}, {3, 3}), CoreRange({0, 0}, {2, 2})})});

    auto mesh_device = devices_[0];
    auto sub_device_manager = mesh_device->create_sub_device_manager({sub_device}, local_l1_size);
    mesh_device->load_sub_device_manager(sub_device_manager);

    auto cores_vec = corerange_to_cores(mesh_device->worker_cores(HalProgrammableCoreType::TENSIX, SubDeviceId{0}));
    for (const auto& core : cores_vec) {
        auto global_bank_id = mesh_device->allocator()->get_bank_ids_from_logical_core(BufferType::L1, core)[0];
        auto sub_device_bank_id =
            mesh_device->allocator(SubDeviceId{0})->get_bank_ids_from_logical_core(BufferType::L1, core)[0];
        EXPECT_EQ(global_bank_id, sub_device_bank_id);
    }
}
}  // namespace tt::tt_metal
