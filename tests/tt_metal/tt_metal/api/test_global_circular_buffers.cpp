// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cstdint>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/global_circular_buffer.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <exception>
#include <map>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/kernel_types.hpp>
#include "mesh_dispatch_fixture.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

// Access to internal API: ProgramImpl::finalize_offsets
#include "impl/program/program_impl.hpp"
#include "impl/context/metal_context.hpp"

namespace tt::tt_metal {

TEST_F(MeshDispatchFixture, TensixCreateGlobalCircularBuffers) {
    CoreRangeSet cores(CoreRange({1, 1}, {1, 1}));
    CoreRangeSet cores2(CoreRange({1, 1}, {2, 2}));
    CoreRangeSet cores3(CoreRange({3, 3}, {3, 3}));
    auto mesh_device = devices_[0];

    {
        std::vector<std::pair<CoreCoord, CoreRangeSet>> sender_receiver_core_mapping = {{CoreCoord(0, 0), cores}};
        auto global_cb = tt::tt_metal::experimental::GlobalCircularBuffer(
            *mesh_device, sender_receiver_core_mapping, 3200, tt::tt_metal::BufferType::L1);
    }
    {
        std::vector<std::pair<CoreCoord, CoreRangeSet>> sender_receiver_core_mapping = {
            {CoreCoord(0, 0), cores}, {CoreCoord(1, 1), cores3}};
        // sender receiver cores overlap
        EXPECT_THROW(
            tt::tt_metal::experimental::GlobalCircularBuffer(
                *mesh_device, sender_receiver_core_mapping, 3200, tt::tt_metal::BufferType::L1),
            std::exception);
    }
    {
        std::vector<std::pair<CoreCoord, CoreRangeSet>> sender_receiver_core_mapping = {
            {CoreCoord(0, 0), cores}, {CoreCoord(0, 1), cores2}};
        // receiver cores overlap
        EXPECT_THROW(
            tt::tt_metal::experimental::GlobalCircularBuffer(
                *mesh_device, sender_receiver_core_mapping, 3200, tt::tt_metal::BufferType::L1),
            std::exception);
    }
}

TEST_F(MeshDispatchFixture, TensixProgramGlobalCircularBuffersAPI) {
    CoreCoord sender_core = CoreCoord(0, 0);
    CoreRangeSet sender_cores = CoreRangeSet(CoreRange(sender_core));
    CoreRangeSet receiver_cores(CoreRange({1, 1}, {2, 2}));
    CoreRangeSet dummy_receiver_cores(CoreRange({3, 3}, {3, 3}));
    uint32_t cb_page_size = 32;
    tt::DataFormat tile_format = tt::DataFormat::Float16_b;
    auto all_cores = sender_cores.merge(receiver_cores).merge(dummy_receiver_cores);

    auto mesh_device = devices_[0];

    std::vector<std::pair<CoreCoord, CoreRangeSet>> sender_receiver_core_mapping = {{sender_core, receiver_cores}};
    auto global_cb = tt::tt_metal::experimental::GlobalCircularBuffer(
        *mesh_device, sender_receiver_core_mapping, 3200, tt::tt_metal::BufferType::L1);
    std::vector<std::pair<CoreCoord, CoreRangeSet>> dummy_sender_receiver_core_mapping = {
        {CoreCoord(0, 0), dummy_receiver_cores}};
    auto dummy_global_cb = tt::tt_metal::experimental::GlobalCircularBuffer(
        *mesh_device, dummy_sender_receiver_core_mapping, 3200, tt::tt_metal::BufferType::L1);
    {
        distributed::MeshWorkload workload;
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        tt_metal::Program program = CreateProgram();

        tt::tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp",
            all_cores,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::RISCV_0_default});
        uint32_t remote_cb_index = 31;
        uint32_t local_cb_index = 0;
        tt::tt_metal::CircularBufferConfig global_cb_config = tt::tt_metal::CircularBufferConfig(cb_page_size);
        global_cb_config.remote_index(remote_cb_index).set_page_size(cb_page_size).set_data_format(tile_format);
        global_cb_config.index(local_cb_index).set_page_size(cb_page_size).set_data_format(tile_format);
        EXPECT_THROW(global_cb_config.remote_index(2), std::exception);
        EXPECT_THROW(
            tt::tt_metal::experimental::CreateCircularBuffer(
                program, CoreRangeSet(CoreRange({3, 3})), global_cb_config, global_cb),
            std::exception);
        auto remote_cb =
            tt::tt_metal::experimental::CreateCircularBuffer(program, receiver_cores, global_cb_config, global_cb);
        program.impl().compile(mesh_device.get());
        program.impl().finalize_offsets(mesh_device.get());
        tt::tt_metal::experimental::UpdateDynamicCircularBufferAddress(program, remote_cb, global_cb);
        EXPECT_THROW(UpdateDynamicCircularBufferAddress(program, remote_cb, dummy_global_cb), std::exception);
    }
    {
        distributed::MeshWorkload workload;
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        tt_metal::Program program = CreateProgram();

        tt::tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp",
            all_cores,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::RISCV_0_default});
        uint32_t remote_cb_index = 16;
        uint32_t local_cb_index = 17;
        tt::tt_metal::CircularBufferConfig global_cb_config = tt::tt_metal::CircularBufferConfig(cb_page_size);
        global_cb_config.remote_index(remote_cb_index).set_page_size(cb_page_size).set_data_format(tile_format);
        global_cb_config.index(local_cb_index).set_page_size(cb_page_size).set_data_format(tile_format);
        tt::tt_metal::experimental::CreateCircularBuffer(program, receiver_cores, global_cb_config, global_cb);
        workload.add_program(device_range, std::move(program));
        auto& program_ = workload.get_programs().at(device_range);
        EXPECT_THROW(program_.impl().finalize_offsets(mesh_device.get()), std::exception);
    }
}

TEST_F(MeshDispatchFixture, TensixGlobalCircularBufferFixedAddressReconstruction) {
    auto mesh_device = devices_[0];
    const CoreCoord sender{0, 0};
    const CoreRangeSet receivers{CoreRange{{1, 0}, {1, 0}}};
    const std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping{{sender, receivers}};
    const auto before = mesh_device->allocator()->get_statistics(BufferType::L1).total_allocated_bytes;
    auto original = experimental::GlobalCircularBuffer(*mesh_device, mapping, 65536);
    auto alias = original;
    const auto data_address = original.buffer_address();
    const auto config_address = original.config_address();
    distributed::Synchronize(*mesh_device, std::nullopt);
    auto* physical_device = mesh_device->get_devices().front();
    const auto alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
    // Eight header words, one NoC coordinate pair, and two aligned counters.
    const uint32_t config_bytes = ((10 * sizeof(uint32_t) + alignment - 1) / alignment + 2) * alignment;
    std::map<CoreCoord, std::vector<uint32_t>> expected_config;
    for (const auto core : {sender, CoreCoord{1, 0}}) {
        detail::ReadFromDeviceL1(
            physical_device, core, config_address, config_bytes, expected_config[core], CoreType::WORKER);
    }

    // Copies held by cached programs must not keep the allocation alive.
    original.deallocate();
    alias.deallocate();
    EXPECT_EQ(mesh_device->allocator()->get_statistics(BufferType::L1).total_allocated_bytes, before);
    std::vector<uint32_t> scratch(config_bytes / sizeof(uint32_t), 0xa5a5a5a5);
    for (const auto& [core, expected] : expected_config) {
        detail::WriteToDeviceL1(physical_device, core, config_address, scratch, CoreType::WORKER);
    }
    auto restored =
        experimental::GlobalCircularBuffer(*mesh_device, mapping, 65536, BufferType::L1, data_address, config_address);
    EXPECT_EQ(restored.buffer_address(), data_address);
    EXPECT_EQ(restored.config_address(), config_address);
    distributed::Synchronize(*mesh_device, std::nullopt);
    for (const auto& [core, expected] : expected_config) {
        std::vector<uint32_t> restored_config;
        detail::ReadFromDeviceL1(
            physical_device, core, config_address, config_bytes, restored_config, CoreType::WORKER);
        EXPECT_EQ(restored_config, expected);
    }
    restored.deallocate();
    EXPECT_EQ(mesh_device->allocator()->get_statistics(BufferType::L1).total_allocated_bytes, before);
}

TEST_F(MeshDispatchFixture, TensixGlobalCircularBufferFixedAddressCollisionRollsBack) {
    auto mesh_device = devices_[0];
    const CoreCoord sender{0, 0};
    const CoreRangeSet receivers{CoreRange{{1, 0}, {1, 0}}};
    const CoreRangeSet all_cores{CoreRange{{0, 0}, {1, 0}}};
    const std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping{{sender, receivers}};
    auto original = experimental::GlobalCircularBuffer(*mesh_device, mapping, 65536);
    const auto data_address = original.buffer_address();
    const auto config_address = original.config_address();
    distributed::Synchronize(*mesh_device, std::nullopt);
    original.deallocate();

    const distributed::DeviceLocalBufferConfig config{
        .page_size = 32,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(
            ShardSpecBuffer(all_cores, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {2, 1}),
            TensorMemoryLayout::HEIGHT_SHARDED),
    };
    auto* physical_device = mesh_device->get_devices().front();
    for (const auto blocked_address : {data_address, config_address}) {
        auto blocker = distributed::MeshBuffer::allocate_at_address(
            distributed::ReplicatedBufferConfig{.size = 64}, config, mesh_device.get(), blocked_address);
        const auto occupied = mesh_device->allocator()->get_statistics(BufferType::L1).total_allocated_bytes;
        // Existing explicit-address create remains a non-owning view.
        auto view = distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = 64}, config, mesh_device.get(), blocked_address);
        EXPECT_EQ(view->get_backing_buffer(), nullptr);
        view->deallocate();
        EXPECT_EQ(mesh_device->allocator()->get_statistics(BufferType::L1).total_allocated_bytes, occupied);
        std::vector<uint32_t> sentinel(8, 0xa5a5a5a5);
        detail::WriteToDeviceL1(physical_device, sender, blocked_address, sentinel, CoreType::WORKER);
        EXPECT_THROW(
            experimental::GlobalCircularBuffer(
                *mesh_device, mapping, 65536, BufferType::L1, data_address, config_address),
            std::runtime_error);
        EXPECT_EQ(mesh_device->allocator()->get_statistics(BufferType::L1).total_allocated_bytes, occupied);
        distributed::Synchronize(*mesh_device, std::nullopt);
        std::vector<uint32_t> after;
        detail::ReadFromDeviceL1(physical_device, sender, blocked_address, 32, after, CoreType::WORKER);
        EXPECT_EQ(after, sentinel);
    }
    EXPECT_THROW(
        experimental::GlobalCircularBuffer(*mesh_device, mapping, 65536, BufferType::L1, data_address),
        std::runtime_error);
    // A failed config allocation must have released the successfully allocated data.
    auto restored =
        experimental::GlobalCircularBuffer(*mesh_device, mapping, 65536, BufferType::L1, data_address, config_address);
    distributed::Synchronize(*mesh_device, std::nullopt);
}

}  // namespace tt::tt_metal
