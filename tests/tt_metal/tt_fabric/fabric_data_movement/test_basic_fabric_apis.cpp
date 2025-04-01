// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <tt-metalium/device_pool.hpp>
#include <tt-metalium/fabric_host_interface.h>

#include "fabric_fixture.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"
#include "test_common.hpp"

namespace tt::tt_fabric {
namespace fabric_router_tests {

// hack to let topology.cpp to know the binary is a unit test
// https://github.com/tenstorrent/tt-metal/issues/20000
// TODO: delete this once tt_fabric_api.h fully support low latency feature
extern "C" bool isFabricUnitTest();
bool isFabricUnitTest() { return true; }

TEST_F(Fabric2DPullFixture, TestAsyncWrite) {
    using tt::tt_metal::ShardedBufferConfig;
    using tt::tt_metal::ShardOrientation;
    using tt::tt_metal::ShardSpecBuffer;

    CoreCoord sender_logical_core = {0, 0};
    CoreRangeSet sender_logical_crs = {sender_logical_core};
    CoreCoord receiver_logical_core = {1, 0};
    CoreRangeSet receiver_logical_crs = {receiver_logical_core};
    std::pair<mesh_id_t, chip_id_t> start_mesh_chip_id;
    chip_id_t physical_start_device_id;
    std::pair<mesh_id_t, chip_id_t> end_mesh_chip_id;
    chip_id_t physical_end_device_id;

    auto control_plane = tt::Cluster::instance().get_control_plane();

    // Find a device with a neighbour in the East direction
    bool connection_found = false;
    for (auto* device : devices_) {
        start_mesh_chip_id = control_plane->get_mesh_chip_id_from_physical_chip_id(device->id());
        // Get neighbours within a mesh in the East direction
        auto neighbors = control_plane->get_intra_chip_neighbors(
            start_mesh_chip_id.first, start_mesh_chip_id.second, RoutingDirection::E);
        if (neighbors.size() > 0) {
            physical_start_device_id = device->id();
            end_mesh_chip_id = {start_mesh_chip_id.first, neighbors[0]};
            physical_end_device_id = control_plane->get_physical_chip_id_from_mesh_chip_id(end_mesh_chip_id);
            connection_found = true;
            break;
        }
    }
    if (!connection_found) {
        GTEST_SKIP() << "No path found between sender and receivers";
    }

    tt::log_info(tt::LogTest, "Async Write from {} to {}", start_mesh_chip_id.second, end_mesh_chip_id.second);
    // Get the optimal routers (no internal hops) on the start chip that will forward in the direction of the end chip
    auto routers = control_plane->get_routers_to_chip(
        start_mesh_chip_id.first, start_mesh_chip_id.second, end_mesh_chip_id.first, end_mesh_chip_id.second);

    auto* sender_device = DevicePool::instance().get_active_device(physical_start_device_id);
    auto* receiver_device = DevicePool::instance().get_active_device(physical_end_device_id);
    CoreCoord sender_virtual_core = sender_device->worker_core_from_logical_core(sender_logical_core);
    CoreCoord receiver_virtual_core = receiver_device->worker_core_from_logical_core(receiver_logical_core);

    uint32_t data_size = tt::constants::TILE_HW * sizeof(uint32_t);

    auto receiver_shard_parameters =
        ShardSpecBuffer(receiver_logical_crs, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
    ShardedBufferConfig receiver_shard_config = {
        .device = receiver_device,
        .size = data_size,
        .page_size = data_size,
        .buffer_type = tt_metal::BufferType::L1,
        .buffer_layout = tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = std::move(receiver_shard_parameters),
    };
    auto receiver_buffer = CreateBuffer(receiver_shard_config);
    // Reset buffer space for test validation
    std::vector<uint32_t> receiver_buffer_data(data_size / sizeof(uint32_t), 0);
    tt::tt_metal::detail::WriteToBuffer(receiver_buffer, receiver_buffer_data);

    // Packet header needs to be inlined with the data being sent, so this test just allocates buffer space for both
    // together on the sender
    uint32_t sender_packet_header_and_data_size = tt::tt_fabric::PACKET_HEADER_SIZE_BYTES + data_size;
    auto sender_shard_parameters =
        ShardSpecBuffer(sender_logical_crs, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
    ShardedBufferConfig sender_shard_config = {
        .device = sender_device,
        .size = sender_packet_header_and_data_size,
        .page_size = sender_packet_header_and_data_size,
        .buffer_type = tt_metal::BufferType::L1,
        .buffer_layout = tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = std::move(sender_shard_parameters),
    };
    auto sender_buffer = CreateBuffer(sender_shard_config);
    // Write the data to send to the buffer
    std::vector<uint32_t> sender_buffer_data(sender_packet_header_and_data_size / sizeof(uint32_t), 0);
    std::iota(sender_buffer_data.begin() + PACKET_HEADER_SIZE_BYTES / sizeof(uint32_t), sender_buffer_data.end(), 0);
    tt::tt_metal::detail::WriteToBuffer(sender_buffer, sender_buffer_data);

    // Extract the expected data to be read from the receiver
    std::copy(
        sender_buffer_data.begin() + tt::tt_fabric::PACKET_HEADER_SIZE_BYTES / sizeof(uint32_t),
        sender_buffer_data.end(),
        receiver_buffer_data.begin());

    // Wait for buffer data to be written to device
    tt::Cluster::instance().l1_barrier(physical_end_device_id);
    tt::Cluster::instance().l1_barrier(physical_start_device_id);

    auto receiver_noc_encoding =
        tt::tt_metal::hal_ref.noc_xy_encoding(receiver_virtual_core.x, receiver_virtual_core.y);

    // Create the sender program
    auto sender_program = tt_metal::CreateProgram();

    // Allocate space for the client interface
    uint32_t client_interface_cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig client_interface_cb_config =
        tt::tt_metal::CircularBufferConfig(
            tt::tt_fabric::CLIENT_INTERFACE_SIZE, {{client_interface_cb_index, DataFormat::UInt32}})
            .set_page_size(client_interface_cb_index, tt::tt_fabric::CLIENT_INTERFACE_SIZE);
    auto client_interface_cb =
        tt::tt_metal::CreateCircularBuffer(sender_program, sender_logical_core, client_interface_cb_config);

    std::vector<uint32_t> sender_compile_time_args = {client_interface_cb_index, 0, (uint32_t)fabric_mode::PULL};
    std::map<string, string> defines = {};
    defines["FVC_MODE_PULL"] = "";
    auto sender_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_async_write_sender.cpp",
        sender_logical_crs,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = sender_compile_time_args,
            .defines = defines});

    auto& sender_virtual_router_coord = routers[0].second;
    auto sender_router_noc_xy =
        tt_metal::hal_ref.noc_xy_encoding(sender_virtual_router_coord.x, sender_virtual_router_coord.y);

    std::vector<uint32_t> sender_runtime_args = {
        sender_buffer->address(),
        receiver_noc_encoding,
        receiver_buffer->address(),
        data_size,
        end_mesh_chip_id.first,
        end_mesh_chip_id.second,
        sender_router_noc_xy};
    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_core, sender_runtime_args);

    // Create the receiver program for validation
    auto receiver_program = tt_metal::CreateProgram();
    auto receiver_kernel = tt_metal::CreateKernel(
        receiver_program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_receiver.cpp",
        {receiver_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .defines = defines});

    std::vector<uint32_t> receiver_runtime_args = {
        receiver_buffer->address(),
        data_size,
    };
    tt_metal::SetRuntimeArgs(receiver_program, receiver_kernel, receiver_logical_core, receiver_runtime_args);

    // Launch sender and receiver programs and wait for them to finish
    this->RunProgramNonblocking(receiver_device, receiver_program);
    this->RunProgramNonblocking(sender_device, sender_program);
    this->WaitForSingleProgramDone(sender_device, sender_program);
    this->WaitForSingleProgramDone(receiver_device, receiver_program);

    // Validate the data received by the receiver
    std::vector<uint32_t> received_buffer_data;
    tt::tt_metal::detail::ReadFromBuffer(receiver_buffer, received_buffer_data);
    EXPECT_EQ(receiver_buffer_data, received_buffer_data);
}

TEST_F(Fabric2DPushFixture, TestAsyncWrite) {
    using tt::tt_metal::ShardedBufferConfig;
    using tt::tt_metal::ShardOrientation;
    using tt::tt_metal::ShardSpecBuffer;

    CoreCoord sender_logical_core = {0, 0};
    CoreRangeSet sender_logical_crs = {sender_logical_core};
    CoreCoord receiver_logical_core = {1, 0};
    CoreRangeSet receiver_logical_crs = {receiver_logical_core};
    std::pair<mesh_id_t, chip_id_t> start_mesh_chip_id;
    chip_id_t physical_start_device_id;
    std::pair<mesh_id_t, chip_id_t> end_mesh_chip_id;
    chip_id_t physical_end_device_id;

    auto control_plane = tt::Cluster::instance().get_control_plane();

    // Find a device with a neighbour in the East direction
    bool connection_found = false;
    for (auto* device : devices_) {
        start_mesh_chip_id = control_plane->get_mesh_chip_id_from_physical_chip_id(device->id());
        // Get neighbours within a mesh in the East direction
        auto neighbors = control_plane->get_intra_chip_neighbors(
            start_mesh_chip_id.first, start_mesh_chip_id.second, RoutingDirection::E);
        if (neighbors.size() > 0) {
            physical_start_device_id = device->id();
            end_mesh_chip_id = {start_mesh_chip_id.first, neighbors[0]};
            physical_end_device_id = control_plane->get_physical_chip_id_from_mesh_chip_id(end_mesh_chip_id);
            connection_found = true;
            break;
        }
    }
    if (!connection_found) {
        GTEST_SKIP() << "No path found between sender and receivers";
    }

    tt::log_info(tt::LogTest, "Async Write from {} to {}", start_mesh_chip_id.second, end_mesh_chip_id.second);
    // Get the optimal routers (no internal hops) on the start chip that will forward in the direction of the end chip
    auto routers = control_plane->get_routers_to_chip(
        start_mesh_chip_id.first, start_mesh_chip_id.second, end_mesh_chip_id.first, end_mesh_chip_id.second);

    auto* sender_device = DevicePool::instance().get_active_device(physical_start_device_id);
    auto* receiver_device = DevicePool::instance().get_active_device(physical_end_device_id);
    CoreCoord sender_virtual_core = sender_device->worker_core_from_logical_core(sender_logical_core);
    CoreCoord receiver_virtual_core = receiver_device->worker_core_from_logical_core(receiver_logical_core);

    uint32_t data_size = tt::constants::TILE_HW * sizeof(uint32_t);

    auto receiver_shard_parameters =
        ShardSpecBuffer(receiver_logical_crs, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
    ShardedBufferConfig receiver_shard_config = {
        .device = receiver_device,
        .size = data_size,
        .page_size = data_size,
        .buffer_type = tt_metal::BufferType::L1,
        .buffer_layout = tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = std::move(receiver_shard_parameters),
    };
    auto receiver_buffer = CreateBuffer(receiver_shard_config);
    // Reset buffer space for test validation
    std::vector<uint32_t> receiver_buffer_data(data_size / sizeof(uint32_t), 0);
    tt::tt_metal::detail::WriteToBuffer(receiver_buffer, receiver_buffer_data);

    // Packet header needs to be inlined with the data being sent, so this test just allocates buffer space for both
    // together on the sender
    uint32_t sender_packet_header_and_data_size = tt::tt_fabric::PACKET_HEADER_SIZE_BYTES + data_size;
    auto sender_shard_parameters =
        ShardSpecBuffer(sender_logical_crs, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
    ShardedBufferConfig sender_shard_config = {
        .device = sender_device,
        .size = sender_packet_header_and_data_size,
        .page_size = sender_packet_header_and_data_size,
        .buffer_type = tt_metal::BufferType::L1,
        .buffer_layout = tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = std::move(sender_shard_parameters),
    };
    auto sender_buffer = CreateBuffer(sender_shard_config);
    // Write the data to send to the buffer
    std::vector<uint32_t> sender_buffer_data(sender_packet_header_and_data_size / sizeof(uint32_t), 0);
    std::iota(sender_buffer_data.begin() + PACKET_HEADER_SIZE_BYTES / sizeof(uint32_t), sender_buffer_data.end(), 0);
    tt::tt_metal::detail::WriteToBuffer(sender_buffer, sender_buffer_data);

    // Extract the expected data to be read from the receiver
    std::copy(
        sender_buffer_data.begin() + tt::tt_fabric::PACKET_HEADER_SIZE_BYTES / sizeof(uint32_t),
        sender_buffer_data.end(),
        receiver_buffer_data.begin());

    // Wait for buffer data to be written to device
    tt::Cluster::instance().l1_barrier(physical_end_device_id);
    tt::Cluster::instance().l1_barrier(physical_start_device_id);

    auto receiver_noc_encoding =
        tt::tt_metal::hal_ref.noc_xy_encoding(receiver_virtual_core.x, receiver_virtual_core.y);

    // Create the sender program
    auto sender_program = tt_metal::CreateProgram();

    // Allocate space for the client interface
    uint32_t client_interface_cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig client_interface_cb_config =
        tt::tt_metal::CircularBufferConfig(
            tt::tt_fabric::CLIENT_INTERFACE_SIZE, {{client_interface_cb_index, DataFormat::UInt32}})
            .set_page_size(client_interface_cb_index, tt::tt_fabric::CLIENT_INTERFACE_SIZE);
    auto client_interface_cb =
        tt::tt_metal::CreateCircularBuffer(sender_program, sender_logical_core, client_interface_cb_config);

    std::vector<uint32_t> sender_compile_time_args = {client_interface_cb_index, 0, (uint32_t)fabric_mode::PUSH};
    std::map<string, string> defines = {};
    defines["DISABLE_LOW_LATENCY_ROUTING"] = "";
    auto sender_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_async_write_sender.cpp",
        sender_logical_crs,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = sender_compile_time_args,
            .defines = defines});

    auto& sender_virtual_router_coord = routers[0].second;
    auto sender_router_noc_xy =
        tt_metal::hal_ref.noc_xy_encoding(sender_virtual_router_coord.x, sender_virtual_router_coord.y);
    auto outbound_eth_channels = tt::Cluster::instance().get_fabric_ethernet_channels(physical_start_device_id);
    std::vector<uint32_t> sender_runtime_args = {
        sender_buffer->address(),
        receiver_noc_encoding,
        receiver_buffer->address(),
        data_size,
        end_mesh_chip_id.first,
        end_mesh_chip_id.second,
        sender_router_noc_xy,
        *outbound_eth_channels.begin()};
    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_core, sender_runtime_args);

    // Create the receiver program for validation
    auto receiver_program = tt_metal::CreateProgram();
    auto receiver_kernel = tt_metal::CreateKernel(
        receiver_program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_receiver.cpp",
        {receiver_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .defines = defines});

    std::vector<uint32_t> receiver_runtime_args = {
        receiver_buffer->address(),
        data_size,
    };
    tt_metal::SetRuntimeArgs(receiver_program, receiver_kernel, receiver_logical_core, receiver_runtime_args);

    // Launch sender and receiver programs and wait for them to finish
    this->RunProgramNonblocking(receiver_device, receiver_program);
    this->RunProgramNonblocking(sender_device, sender_program);
    this->WaitForSingleProgramDone(sender_device, sender_program);
    this->WaitForSingleProgramDone(receiver_device, receiver_program);

    // Validate the data received by the receiver
    std::vector<uint32_t> received_buffer_data;
    tt::tt_metal::detail::ReadFromBuffer(receiver_buffer, received_buffer_data);
    EXPECT_EQ(receiver_buffer_data, received_buffer_data);
}

TEST_F(Fabric2DPullFixture, TestAsyncRawWrite) {
    using tt::tt_metal::ShardedBufferConfig;
    using tt::tt_metal::ShardOrientation;
    using tt::tt_metal::ShardSpecBuffer;

    CoreCoord sender_logical_core = {0, 0};
    CoreRangeSet sender_logical_crs = {sender_logical_core};
    CoreCoord receiver_logical_core = {1, 0};
    CoreRangeSet receiver_logical_crs = {receiver_logical_core};
    std::pair<mesh_id_t, chip_id_t> start_mesh_chip_id;
    chip_id_t physical_start_device_id;
    std::pair<mesh_id_t, chip_id_t> end_mesh_chip_id;
    chip_id_t physical_end_device_id;

    auto control_plane = tt::Cluster::instance().get_control_plane();

    // Find a device with a neighbour in the East direction
    bool connection_found = false;
    for (auto* device : devices_) {
        start_mesh_chip_id = control_plane->get_mesh_chip_id_from_physical_chip_id(device->id());
        // Get neighbours within a mesh in the East direction
        auto neighbors = control_plane->get_intra_chip_neighbors(
            start_mesh_chip_id.first, start_mesh_chip_id.second, RoutingDirection::E);
        if (neighbors.size() > 0) {
            physical_start_device_id = device->id();
            end_mesh_chip_id = {start_mesh_chip_id.first, neighbors[0]};
            physical_end_device_id = control_plane->get_physical_chip_id_from_mesh_chip_id(end_mesh_chip_id);
            connection_found = true;
            break;
        }
    }
    if (!connection_found) {
        GTEST_SKIP() << "No path found between sender and receivers";
    }

    tt::log_info(tt::LogTest, "Raw Async Write from {} to {}", start_mesh_chip_id.second, end_mesh_chip_id.second);
    // Get the optimal routers (no internal hops) on the start chip that will forward in the direction of the end chip
    auto routers = control_plane->get_routers_to_chip(
        start_mesh_chip_id.first, start_mesh_chip_id.second, end_mesh_chip_id.first, end_mesh_chip_id.second);

    auto* sender_device = DevicePool::instance().get_active_device(physical_start_device_id);
    auto* receiver_device = DevicePool::instance().get_active_device(physical_end_device_id);
    CoreCoord sender_virtual_core = sender_device->worker_core_from_logical_core(sender_logical_core);
    CoreCoord receiver_virtual_core = receiver_device->worker_core_from_logical_core(receiver_logical_core);

    uint32_t data_size = tt::constants::TILE_HW * sizeof(uint32_t);

    auto receiver_shard_parameters =
        ShardSpecBuffer(receiver_logical_crs, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
    ShardedBufferConfig receiver_shard_config = {
        .device = receiver_device,
        .size = data_size,
        .page_size = data_size,
        .buffer_type = tt_metal::BufferType::L1,
        .buffer_layout = tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = std::move(receiver_shard_parameters),
    };
    auto receiver_buffer = CreateBuffer(receiver_shard_config);
    // Reset buffer space for test validation
    std::vector<uint32_t> receiver_buffer_data(data_size / sizeof(uint32_t), 0);
    tt::tt_metal::detail::WriteToBuffer(receiver_buffer, receiver_buffer_data);

    auto sender_shard_parameters =
        ShardSpecBuffer(sender_logical_crs, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
    ShardedBufferConfig sender_shard_config = {
        .device = sender_device,
        .size = data_size,
        .page_size = data_size,
        .buffer_type = tt_metal::BufferType::L1,
        .buffer_layout = tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = std::move(sender_shard_parameters),
    };
    auto sender_buffer = CreateBuffer(sender_shard_config);
    // Write the data to send to the buffer
    std::vector<uint32_t> sender_buffer_data(data_size / sizeof(uint32_t), 0);
    std::iota(sender_buffer_data.begin(), sender_buffer_data.end(), 0);
    tt::tt_metal::detail::WriteToBuffer(sender_buffer, sender_buffer_data);

    // Wait for buffer data to be written to device
    tt::Cluster::instance().l1_barrier(physical_end_device_id);
    tt::Cluster::instance().l1_barrier(physical_start_device_id);

    auto receiver_noc_encoding =
        tt::tt_metal::hal_ref.noc_xy_encoding(receiver_virtual_core.x, receiver_virtual_core.y);

    // Create the sender program
    auto sender_program = tt_metal::CreateProgram();

    // Allocate space for the client interface
    uint32_t client_interface_cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig client_interface_cb_config =
        tt::tt_metal::CircularBufferConfig(
            tt::tt_fabric::CLIENT_INTERFACE_SIZE, {{client_interface_cb_index, DataFormat::UInt32}})
            .set_page_size(client_interface_cb_index, tt::tt_fabric::CLIENT_INTERFACE_SIZE);
    auto client_interface_cb =
        tt::tt_metal::CreateCircularBuffer(sender_program, sender_logical_core, client_interface_cb_config);

    std::vector<uint32_t> sender_compile_time_args = {client_interface_cb_index, 1, (uint32_t)fabric_mode::PULL};
    std::map<string, string> defines = {};
    defines["FVC_MODE_PULL"] = "";
    auto sender_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_async_write_sender.cpp",
        sender_logical_crs,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = sender_compile_time_args,
            .defines = defines});

    auto& sender_virtual_router_coord = routers[0].second;
    auto sender_router_noc_xy =
        tt_metal::hal_ref.noc_xy_encoding(sender_virtual_router_coord.x, sender_virtual_router_coord.y);

    std::vector<uint32_t> sender_runtime_args = {
        sender_buffer->address(),
        receiver_noc_encoding,
        receiver_buffer->address(),
        data_size,
        end_mesh_chip_id.first,
        end_mesh_chip_id.second,
        sender_router_noc_xy};
    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_core, sender_runtime_args);

    // Create the receiver program for validation
    auto receiver_program = tt_metal::CreateProgram();
    auto receiver_kernel = tt_metal::CreateKernel(
        receiver_program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_receiver.cpp",
        {receiver_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .defines = defines});

    std::vector<uint32_t> receiver_runtime_args = {
        receiver_buffer->address(),
        data_size,
    };
    tt_metal::SetRuntimeArgs(receiver_program, receiver_kernel, receiver_logical_core, receiver_runtime_args);

    // Launch sender and receiver programs and wait for them to finish
    this->RunProgramNonblocking(receiver_device, receiver_program);
    this->RunProgramNonblocking(sender_device, sender_program);
    this->WaitForSingleProgramDone(sender_device, sender_program);
    this->WaitForSingleProgramDone(receiver_device, receiver_program);

    // Validate the data received by the receiver
    std::vector<uint32_t> received_buffer_data;
    tt::tt_metal::detail::ReadFromBuffer(receiver_buffer, received_buffer_data);
    EXPECT_EQ(sender_buffer_data, received_buffer_data);
}

TEST_F(Fabric2DPushFixture, TestAsyncRawWrite) {
    using tt::tt_metal::ShardedBufferConfig;
    using tt::tt_metal::ShardOrientation;
    using tt::tt_metal::ShardSpecBuffer;

    CoreCoord sender_logical_core = {0, 0};
    CoreRangeSet sender_logical_crs = {sender_logical_core};
    CoreCoord receiver_logical_core = {1, 0};
    CoreRangeSet receiver_logical_crs = {receiver_logical_core};
    std::pair<mesh_id_t, chip_id_t> start_mesh_chip_id;
    chip_id_t physical_start_device_id;
    std::pair<mesh_id_t, chip_id_t> end_mesh_chip_id;
    chip_id_t physical_end_device_id;

    auto control_plane = tt::Cluster::instance().get_control_plane();

    // Find a device with a neighbour in the East direction
    bool connection_found = false;
    for (auto* device : devices_) {
        start_mesh_chip_id = control_plane->get_mesh_chip_id_from_physical_chip_id(device->id());
        // Get neighbours within a mesh in the East direction
        auto neighbors = control_plane->get_intra_chip_neighbors(
            start_mesh_chip_id.first, start_mesh_chip_id.second, RoutingDirection::E);
        if (neighbors.size() > 0) {
            physical_start_device_id = device->id();
            end_mesh_chip_id = {start_mesh_chip_id.first, neighbors[0]};
            physical_end_device_id = control_plane->get_physical_chip_id_from_mesh_chip_id(end_mesh_chip_id);
            connection_found = true;
            break;
        }
    }
    if (!connection_found) {
        GTEST_SKIP() << "No path found between sender and receivers";
    }

    tt::log_info(tt::LogTest, "Raw Async Write from {} to {}", start_mesh_chip_id.second, end_mesh_chip_id.second);
    // Get the optimal routers (no internal hops) on the start chip that will forward in the direction of the end chip
    auto routers = control_plane->get_routers_to_chip(
        start_mesh_chip_id.first, start_mesh_chip_id.second, end_mesh_chip_id.first, end_mesh_chip_id.second);

    auto* sender_device = DevicePool::instance().get_active_device(physical_start_device_id);
    auto* receiver_device = DevicePool::instance().get_active_device(physical_end_device_id);
    CoreCoord sender_virtual_core = sender_device->worker_core_from_logical_core(sender_logical_core);
    CoreCoord receiver_virtual_core = receiver_device->worker_core_from_logical_core(receiver_logical_core);

    uint32_t data_size = tt::constants::TILE_HW * sizeof(uint32_t);

    auto receiver_shard_parameters =
        ShardSpecBuffer(receiver_logical_crs, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
    ShardedBufferConfig receiver_shard_config = {
        .device = receiver_device,
        .size = data_size,
        .page_size = data_size,
        .buffer_type = tt_metal::BufferType::L1,
        .buffer_layout = tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = std::move(receiver_shard_parameters),
    };
    auto receiver_buffer = CreateBuffer(receiver_shard_config);
    // Reset buffer space for test validation
    std::vector<uint32_t> receiver_buffer_data(data_size / sizeof(uint32_t), 0);
    tt::tt_metal::detail::WriteToBuffer(receiver_buffer, receiver_buffer_data);

    auto sender_shard_parameters =
        ShardSpecBuffer(sender_logical_crs, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
    ShardedBufferConfig sender_shard_config = {
        .device = sender_device,
        .size = data_size,
        .page_size = data_size,
        .buffer_type = tt_metal::BufferType::L1,
        .buffer_layout = tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = std::move(sender_shard_parameters),
    };
    auto sender_buffer = CreateBuffer(sender_shard_config);
    // Write the data to send to the buffer
    std::vector<uint32_t> sender_buffer_data(data_size / sizeof(uint32_t), 0);
    std::iota(sender_buffer_data.begin(), sender_buffer_data.end(), 0);
    tt::tt_metal::detail::WriteToBuffer(sender_buffer, sender_buffer_data);

    // Wait for buffer data to be written to device
    tt::Cluster::instance().l1_barrier(physical_end_device_id);
    tt::Cluster::instance().l1_barrier(physical_start_device_id);

    auto receiver_noc_encoding =
        tt::tt_metal::hal_ref.noc_xy_encoding(receiver_virtual_core.x, receiver_virtual_core.y);

    // Create the sender program
    auto sender_program = tt_metal::CreateProgram();

    // Allocate space for the client interface
    uint32_t client_interface_cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig client_interface_cb_config =
        tt::tt_metal::CircularBufferConfig(
            tt::tt_fabric::CLIENT_INTERFACE_SIZE, {{client_interface_cb_index, DataFormat::UInt32}})
            .set_page_size(client_interface_cb_index, tt::tt_fabric::CLIENT_INTERFACE_SIZE);
    auto client_interface_cb =
        tt::tt_metal::CreateCircularBuffer(sender_program, sender_logical_core, client_interface_cb_config);

    std::vector<uint32_t> sender_compile_time_args = {client_interface_cb_index, 1, (uint32_t)fabric_mode::PUSH};
    std::map<string, string> defines = {};
    defines["DISABLE_LOW_LATENCY_ROUTING"] = "";
    auto sender_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_async_write_sender.cpp",
        sender_logical_crs,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = sender_compile_time_args,
            .defines = defines});

    auto& sender_virtual_router_coord = routers[0].second;
    auto sender_router_noc_xy =
        tt_metal::hal_ref.noc_xy_encoding(sender_virtual_router_coord.x, sender_virtual_router_coord.y);
    auto outbound_eth_channels = tt::Cluster::instance().get_fabric_ethernet_channels(physical_start_device_id);
    std::vector<uint32_t> sender_runtime_args = {
        sender_buffer->address(),
        receiver_noc_encoding,
        receiver_buffer->address(),
        data_size,
        end_mesh_chip_id.first,
        end_mesh_chip_id.second,
        sender_router_noc_xy,
        *outbound_eth_channels.begin()};
    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_core, sender_runtime_args);

    // Create the receiver program for validation
    auto receiver_program = tt_metal::CreateProgram();
    auto receiver_kernel = tt_metal::CreateKernel(
        receiver_program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_receiver.cpp",
        {receiver_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .defines = defines});

    std::vector<uint32_t> receiver_runtime_args = {
        receiver_buffer->address(),
        data_size,
    };
    tt_metal::SetRuntimeArgs(receiver_program, receiver_kernel, receiver_logical_core, receiver_runtime_args);

    // Launch sender and receiver programs and wait for them to finish
    this->RunProgramNonblocking(receiver_device, receiver_program);
    this->RunProgramNonblocking(sender_device, sender_program);
    this->WaitForSingleProgramDone(sender_device, sender_program);
    this->WaitForSingleProgramDone(receiver_device, receiver_program);

    // Validate the data received by the receiver
    std::vector<uint32_t> received_buffer_data;
    tt::tt_metal::detail::ReadFromBuffer(receiver_buffer, received_buffer_data);
    EXPECT_EQ(sender_buffer_data, received_buffer_data);
}

TEST_F(Fabric2DPullFixture, TestAtomicInc) {
    using tt::tt_metal::ShardedBufferConfig;
    using tt::tt_metal::ShardOrientation;
    using tt::tt_metal::ShardSpecBuffer;

    CoreCoord sender_logical_core = {0, 0};
    CoreRangeSet sender_logical_crs = {sender_logical_core};
    CoreCoord receiver_logical_core = {1, 0};
    CoreRangeSet receiver_logical_crs = {receiver_logical_core};
    std::pair<mesh_id_t, chip_id_t> start_mesh_chip_id;
    chip_id_t physical_start_device_id;
    std::pair<mesh_id_t, chip_id_t> end_mesh_chip_id;
    chip_id_t physical_end_device_id;

    auto control_plane = tt::Cluster::instance().get_control_plane();

    // Find a device with a neighbour in the East direction
    bool connection_found = false;
    for (auto* device : devices_) {
        start_mesh_chip_id = control_plane->get_mesh_chip_id_from_physical_chip_id(device->id());
        // Get neighbours within a mesh in the East direction
        auto neighbors = control_plane->get_intra_chip_neighbors(
            start_mesh_chip_id.first, start_mesh_chip_id.second, RoutingDirection::E);
        if (neighbors.size() > 0) {
            physical_start_device_id = device->id();
            end_mesh_chip_id = {start_mesh_chip_id.first, neighbors[0]};
            physical_end_device_id = control_plane->get_physical_chip_id_from_mesh_chip_id(end_mesh_chip_id);
            connection_found = true;
            break;
        }
    }
    if (!connection_found) {
        GTEST_SKIP() << "No path found between sender and receivers";
    }

    // Get the optimal routers (no internal hops) on the start chip that will forward in the direction of the end chip
    auto routers = control_plane->get_routers_to_chip(
        start_mesh_chip_id.first, start_mesh_chip_id.second, end_mesh_chip_id.first, end_mesh_chip_id.second);

    auto* sender_device = DevicePool::instance().get_active_device(physical_start_device_id);
    auto* receiver_device = DevicePool::instance().get_active_device(physical_end_device_id);
    CoreCoord sender_virtual_core = sender_device->worker_core_from_logical_core(sender_logical_core);
    CoreCoord receiver_virtual_core = receiver_device->worker_core_from_logical_core(receiver_logical_core);

    uint32_t data_size = sizeof(uint32_t);

    auto receiver_shard_parameters =
        ShardSpecBuffer(receiver_logical_crs, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
    ShardedBufferConfig receiver_shard_config = {
        .device = receiver_device,
        .size = data_size,
        .page_size = data_size,
        .buffer_type = tt_metal::BufferType::L1,
        .buffer_layout = tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = std::move(receiver_shard_parameters),
    };
    auto receiver_buffer = CreateBuffer(receiver_shard_config);
    // Reset buffer space for test validation
    std::vector<uint32_t> receiver_buffer_data(data_size / sizeof(uint32_t), 0);
    tt::tt_metal::detail::WriteToBuffer(receiver_buffer, receiver_buffer_data);

    // Packet header needs to be inlined with the data being sent, so this test just allocates buffer space for both
    // together on the sender
    uint32_t sender_packet_header_and_data_size = tt::tt_fabric::PACKET_HEADER_SIZE_BYTES;
    auto sender_shard_parameters =
        ShardSpecBuffer(sender_logical_crs, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
    ShardedBufferConfig sender_shard_config = {
        .device = sender_device,
        .size = sender_packet_header_and_data_size,
        .page_size = sender_packet_header_and_data_size,
        .buffer_type = tt_metal::BufferType::L1,
        .buffer_layout = tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = std::move(sender_shard_parameters),
    };
    auto sender_buffer = CreateBuffer(sender_shard_config);
    // Write the data to send to the buffer
    std::vector<uint32_t> sender_buffer_data(sender_packet_header_and_data_size / sizeof(uint32_t), 0);
    tt::tt_metal::detail::WriteToBuffer(sender_buffer, sender_buffer_data);

    uint32_t atomic_inc = 5;
    uint32_t wrap_boundary = 31;

    // Extract the expected data to be read from the receiver
    receiver_buffer_data[0] = atomic_inc;

    // Wait for buffer data to be written to device
    tt::Cluster::instance().l1_barrier(physical_end_device_id);
    tt::Cluster::instance().l1_barrier(physical_start_device_id);

    auto receiver_noc_encoding =
        tt::tt_metal::hal_ref.noc_xy_encoding(receiver_virtual_core.x, receiver_virtual_core.y);

    // Create the sender program
    auto sender_program = tt_metal::CreateProgram();

    // Allocate space for the client interface
    uint32_t client_interface_cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig client_interface_cb_config =
        tt::tt_metal::CircularBufferConfig(
            tt::tt_fabric::CLIENT_INTERFACE_SIZE, {{client_interface_cb_index, DataFormat::UInt32}})
            .set_page_size(client_interface_cb_index, tt::tt_fabric::CLIENT_INTERFACE_SIZE);
    auto client_interface_cb =
        tt::tt_metal::CreateCircularBuffer(sender_program, sender_logical_core, client_interface_cb_config);

    std::map<string, string> defines = {};
    defines["FVC_MODE_PULL"] = "";
    std::vector<uint32_t> sender_compile_time_args = {client_interface_cb_index, fabric_mode::PULL};
    auto sender_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_atomic_inc_sender.cpp",
        sender_logical_crs,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = sender_compile_time_args,
            .defines = defines});

    auto& sender_virtual_router_coord = routers[0].second;
    auto sender_router_noc_xy =
        tt_metal::hal_ref.noc_xy_encoding(sender_virtual_router_coord.x, sender_virtual_router_coord.y);
    std::vector<uint32_t> sender_runtime_args = {
        sender_buffer->address(),
        receiver_noc_encoding,
        receiver_buffer->address(),
        atomic_inc,
        wrap_boundary,
        end_mesh_chip_id.first,
        end_mesh_chip_id.second,
        sender_router_noc_xy};
    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_core, sender_runtime_args);

    // Create the receiver program for validation
    auto receiver_program = tt_metal::CreateProgram();
    auto receiver_kernel = tt_metal::CreateKernel(
        receiver_program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_receiver.cpp",
        {receiver_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .defines = defines});

    std::vector<uint32_t> receiver_runtime_args = {
        receiver_buffer->address(),
        data_size,
    };
    tt_metal::SetRuntimeArgs(receiver_program, receiver_kernel, receiver_logical_core, receiver_runtime_args);

    // Launch sender and receiver programs and wait for them to finish
    this->RunProgramNonblocking(receiver_device, receiver_program);
    this->RunProgramNonblocking(sender_device, sender_program);
    this->WaitForSingleProgramDone(sender_device, sender_program);
    this->WaitForSingleProgramDone(receiver_device, receiver_program);

    // Validate the data received by the receiver
    std::vector<uint32_t> received_buffer_data;
    tt::tt_metal::detail::ReadFromBuffer(receiver_buffer, received_buffer_data);
    EXPECT_EQ(receiver_buffer_data, received_buffer_data);
}

TEST_F(Fabric2DPushFixture, TestAtomicInc) {
    using tt::tt_metal::ShardedBufferConfig;
    using tt::tt_metal::ShardOrientation;
    using tt::tt_metal::ShardSpecBuffer;

    CoreCoord sender_logical_core = {0, 0};
    CoreRangeSet sender_logical_crs = {sender_logical_core};
    CoreCoord receiver_logical_core = {1, 0};
    CoreRangeSet receiver_logical_crs = {receiver_logical_core};
    std::pair<mesh_id_t, chip_id_t> start_mesh_chip_id;
    chip_id_t physical_start_device_id;
    std::pair<mesh_id_t, chip_id_t> end_mesh_chip_id;
    chip_id_t physical_end_device_id;

    auto control_plane = tt::Cluster::instance().get_control_plane();

    // Find a device with a neighbour in the East direction
    bool connection_found = false;
    for (auto* device : devices_) {
        start_mesh_chip_id = control_plane->get_mesh_chip_id_from_physical_chip_id(device->id());
        // Get neighbours within a mesh in the East direction
        auto neighbors = control_plane->get_intra_chip_neighbors(
            start_mesh_chip_id.first, start_mesh_chip_id.second, RoutingDirection::E);
        if (neighbors.size() > 0) {
            physical_start_device_id = device->id();
            end_mesh_chip_id = {start_mesh_chip_id.first, neighbors[0]};
            physical_end_device_id = control_plane->get_physical_chip_id_from_mesh_chip_id(end_mesh_chip_id);
            connection_found = true;
            break;
        }
    }
    if (!connection_found) {
        GTEST_SKIP() << "No path found between sender and receivers";
    }

    // Get the optimal routers (no internal hops) on the start chip that will forward in the direction of the end chip
    auto routers = control_plane->get_routers_to_chip(
        start_mesh_chip_id.first, start_mesh_chip_id.second, end_mesh_chip_id.first, end_mesh_chip_id.second);

    auto* sender_device = DevicePool::instance().get_active_device(physical_start_device_id);
    auto* receiver_device = DevicePool::instance().get_active_device(physical_end_device_id);
    CoreCoord sender_virtual_core = sender_device->worker_core_from_logical_core(sender_logical_core);
    CoreCoord receiver_virtual_core = receiver_device->worker_core_from_logical_core(receiver_logical_core);

    uint32_t data_size = sizeof(uint32_t);

    auto receiver_shard_parameters =
        ShardSpecBuffer(receiver_logical_crs, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
    ShardedBufferConfig receiver_shard_config = {
        .device = receiver_device,
        .size = data_size,
        .page_size = data_size,
        .buffer_type = tt_metal::BufferType::L1,
        .buffer_layout = tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = std::move(receiver_shard_parameters),
    };
    auto receiver_buffer = CreateBuffer(receiver_shard_config);
    // Reset buffer space for test validation
    std::vector<uint32_t> receiver_buffer_data(data_size / sizeof(uint32_t), 0);
    tt::tt_metal::detail::WriteToBuffer(receiver_buffer, receiver_buffer_data);

    // Packet header needs to be inlined with the data being sent, so this test just allocates buffer space for both
    // together on the sender
    uint32_t sender_packet_header_and_data_size = tt::tt_fabric::PACKET_HEADER_SIZE_BYTES;
    auto sender_shard_parameters =
        ShardSpecBuffer(sender_logical_crs, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
    ShardedBufferConfig sender_shard_config = {
        .device = sender_device,
        .size = sender_packet_header_and_data_size,
        .page_size = sender_packet_header_and_data_size,
        .buffer_type = tt_metal::BufferType::L1,
        .buffer_layout = tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = std::move(sender_shard_parameters),
    };
    auto sender_buffer = CreateBuffer(sender_shard_config);
    // Write the data to send to the buffer
    std::vector<uint32_t> sender_buffer_data(sender_packet_header_and_data_size / sizeof(uint32_t), 0);
    tt::tt_metal::detail::WriteToBuffer(sender_buffer, sender_buffer_data);

    uint32_t atomic_inc = 5;
    uint32_t wrap_boundary = 31;

    // Extract the expected data to be read from the receiver
    receiver_buffer_data[0] = atomic_inc;

    // Wait for buffer data to be written to device
    tt::Cluster::instance().l1_barrier(physical_end_device_id);
    tt::Cluster::instance().l1_barrier(physical_start_device_id);

    auto receiver_noc_encoding =
        tt::tt_metal::hal_ref.noc_xy_encoding(receiver_virtual_core.x, receiver_virtual_core.y);

    // Create the sender program
    auto sender_program = tt_metal::CreateProgram();

    // Allocate space for the client interface
    uint32_t client_interface_cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig client_interface_cb_config =
        tt::tt_metal::CircularBufferConfig(
            tt::tt_fabric::CLIENT_INTERFACE_SIZE, {{client_interface_cb_index, DataFormat::UInt32}})
            .set_page_size(client_interface_cb_index, tt::tt_fabric::CLIENT_INTERFACE_SIZE);
    auto client_interface_cb =
        tt::tt_metal::CreateCircularBuffer(sender_program, sender_logical_core, client_interface_cb_config);

    std::map<string, string> defines = {};
    defines["DISABLE_LOW_LATENCY_ROUTING"] = "";
    std::vector<uint32_t> sender_compile_time_args = {client_interface_cb_index, fabric_mode::PUSH};
    auto sender_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_atomic_inc_sender.cpp",
        sender_logical_crs,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = sender_compile_time_args,
            .defines = defines});

    auto& sender_virtual_router_coord = routers[0].second;
    auto sender_router_noc_xy =
        tt_metal::hal_ref.noc_xy_encoding(sender_virtual_router_coord.x, sender_virtual_router_coord.y);
    auto outbound_eth_channels = tt::Cluster::instance().get_fabric_ethernet_channels(physical_start_device_id);
    std::vector<uint32_t> sender_runtime_args = {
        sender_buffer->address(),
        receiver_noc_encoding,
        receiver_buffer->address(),
        atomic_inc,
        wrap_boundary,
        end_mesh_chip_id.first,
        end_mesh_chip_id.second,
        sender_router_noc_xy,
        *outbound_eth_channels.begin()};
    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_core, sender_runtime_args);

    // Create the receiver program for validation
    auto receiver_program = tt_metal::CreateProgram();
    auto receiver_kernel = tt_metal::CreateKernel(
        receiver_program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_receiver.cpp",
        {receiver_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .defines = defines});

    std::vector<uint32_t> receiver_runtime_args = {
        receiver_buffer->address(),
        data_size,
    };
    tt_metal::SetRuntimeArgs(receiver_program, receiver_kernel, receiver_logical_core, receiver_runtime_args);

    // Launch sender and receiver programs and wait for them to finish
    this->RunProgramNonblocking(receiver_device, receiver_program);
    this->RunProgramNonblocking(sender_device, sender_program);
    this->WaitForSingleProgramDone(sender_device, sender_program);
    this->WaitForSingleProgramDone(receiver_device, receiver_program);

    // Validate the data received by the receiver
    std::vector<uint32_t> received_buffer_data;
    tt::tt_metal::detail::ReadFromBuffer(receiver_buffer, received_buffer_data);
    EXPECT_EQ(receiver_buffer_data, received_buffer_data);
}

TEST_F(Fabric2DPullFixture, TestAsyncWriteAtomicInc) {
    using tt::tt_metal::ShardedBufferConfig;
    using tt::tt_metal::ShardOrientation;
    using tt::tt_metal::ShardSpecBuffer;

    CoreCoord sender_logical_core = {0, 0};
    CoreRangeSet sender_logical_crs = {sender_logical_core};
    CoreCoord receiver_logical_core = {1, 0};
    CoreRangeSet receiver_logical_crs = {receiver_logical_core};
    std::pair<mesh_id_t, chip_id_t> start_mesh_chip_id;
    chip_id_t physical_start_device_id;
    std::pair<mesh_id_t, chip_id_t> end_mesh_chip_id;
    chip_id_t physical_end_device_id;

    auto control_plane = tt::Cluster::instance().get_control_plane();

    // Find a device with a neighbour in the East direction
    bool connection_found = false;
    for (auto* device : devices_) {
        start_mesh_chip_id = control_plane->get_mesh_chip_id_from_physical_chip_id(device->id());
        // Get neighbours within a mesh in the East direction
        auto neighbors = control_plane->get_intra_chip_neighbors(
            start_mesh_chip_id.first, start_mesh_chip_id.second, RoutingDirection::E);
        if (neighbors.size() > 0) {
            physical_start_device_id = device->id();
            end_mesh_chip_id = {start_mesh_chip_id.first, neighbors[0]};
            physical_end_device_id = control_plane->get_physical_chip_id_from_mesh_chip_id(end_mesh_chip_id);
            connection_found = true;
            break;
        }
    }
    if (!connection_found) {
        GTEST_SKIP() << "No path found between sender and receivers";
    }

    // Get the optimal routers (no internal hops) on the start chip that will forward in the direction of the end chip
    auto routers = control_plane->get_routers_to_chip(
        start_mesh_chip_id.first, start_mesh_chip_id.second, end_mesh_chip_id.first, end_mesh_chip_id.second);

    auto* sender_device = DevicePool::instance().get_active_device(physical_start_device_id);
    auto* receiver_device = DevicePool::instance().get_active_device(physical_end_device_id);
    CoreCoord sender_virtual_core = sender_device->worker_core_from_logical_core(sender_logical_core);
    CoreCoord receiver_virtual_core = receiver_device->worker_core_from_logical_core(receiver_logical_core);

    uint32_t data_size = tt::constants::TILE_HW * sizeof(uint32_t);
    uint32_t atomic_inc_size = sizeof(uint32_t);

    auto receiver_shard_parameters =
        ShardSpecBuffer(receiver_logical_crs, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
    ShardedBufferConfig receiver_shard_config = {
        .device = receiver_device,
        .size = data_size,
        .page_size = data_size,
        .buffer_type = tt_metal::BufferType::L1,
        .buffer_layout = tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = receiver_shard_parameters,
    };
    auto receiver_buffer = CreateBuffer(receiver_shard_config);
    ShardedBufferConfig receiver_atomic_shard_config = {
        .device = receiver_device,
        .size = atomic_inc_size,
        .page_size = atomic_inc_size,
        .buffer_type = tt_metal::BufferType::L1,
        .buffer_layout = tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = receiver_shard_parameters,
    };
    auto receiver_atomic_buffer = CreateBuffer(receiver_atomic_shard_config);
    // Reset buffer space for test validation
    std::vector<uint32_t> receiver_buffer_data(atomic_inc_size / sizeof(uint32_t), 0);
    tt::tt_metal::detail::WriteToBuffer(receiver_atomic_buffer, receiver_buffer_data);
    receiver_buffer_data.resize(data_size / sizeof(uint32_t), 0);
    tt::tt_metal::detail::WriteToBuffer(receiver_buffer, receiver_buffer_data);

    // Packet header needs to be inlined with the data being sent, so this test just allocates buffer space for both
    // together on the sender
    uint32_t sender_packet_header_and_data_size = tt::tt_fabric::PACKET_HEADER_SIZE_BYTES + data_size;
    auto sender_shard_parameters =
        ShardSpecBuffer(sender_logical_crs, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
    ShardedBufferConfig sender_shard_config = {
        .device = sender_device,
        .size = sender_packet_header_and_data_size,
        .page_size = sender_packet_header_and_data_size,
        .buffer_type = tt_metal::BufferType::L1,
        .buffer_layout = tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = std::move(sender_shard_parameters),
    };
    auto sender_buffer = CreateBuffer(sender_shard_config);
    // Write the data to send to the buffer
    std::vector<uint32_t> sender_buffer_data(sender_packet_header_and_data_size / sizeof(uint32_t), 0);
    std::iota(sender_buffer_data.begin() + PACKET_HEADER_SIZE_BYTES / sizeof(uint32_t), sender_buffer_data.end(), 0);
    tt::tt_metal::detail::WriteToBuffer(sender_buffer, sender_buffer_data);

    uint32_t atomic_inc = 5;

    // Extract the expected data to be read from the receiver
    std::copy(
        sender_buffer_data.begin() + tt::tt_fabric::PACKET_HEADER_SIZE_BYTES / sizeof(uint32_t),
        sender_buffer_data.end(),
        receiver_buffer_data.begin());

    // Wait for buffer data to be written to device
    tt::Cluster::instance().l1_barrier(physical_end_device_id);
    tt::Cluster::instance().l1_barrier(physical_start_device_id);

    auto receiver_noc_encoding =
        tt::tt_metal::hal_ref.noc_xy_encoding(receiver_virtual_core.x, receiver_virtual_core.y);

    // Create the sender program
    auto sender_program = tt_metal::CreateProgram();

    // Allocate space for the client interface
    uint32_t client_interface_cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig client_interface_cb_config =
        tt::tt_metal::CircularBufferConfig(
            tt::tt_fabric::CLIENT_INTERFACE_SIZE, {{client_interface_cb_index, DataFormat::UInt32}})
            .set_page_size(client_interface_cb_index, tt::tt_fabric::CLIENT_INTERFACE_SIZE);
    auto client_interface_cb =
        tt::tt_metal::CreateCircularBuffer(sender_program, sender_logical_core, client_interface_cb_config);

    std::map<string, string> defines = {};
    defines["FVC_MODE_PULL"] = "";
    std::vector<uint32_t> sender_compile_time_args = {client_interface_cb_index, 0, fabric_mode::PULL};
    auto sender_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_async_write_atomic_inc_sender.cpp",
        sender_logical_crs,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = sender_compile_time_args,
            .defines = defines});

    auto& sender_virtual_router_coord = routers[0].second;
    auto sender_router_noc_xy =
        tt_metal::hal_ref.noc_xy_encoding(sender_virtual_router_coord.x, sender_virtual_router_coord.y);

    std::vector<uint32_t> sender_runtime_args = {
        sender_buffer->address(),
        receiver_noc_encoding,
        receiver_buffer->address(),
        receiver_atomic_buffer->address(),
        data_size,
        atomic_inc,
        end_mesh_chip_id.first,
        end_mesh_chip_id.second,
        sender_router_noc_xy};
    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_core, sender_runtime_args);

    // Create the receiver program for validation
    auto receiver_program = tt_metal::CreateProgram();
    auto receiver_kernel = tt_metal::CreateKernel(
        receiver_program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_receiver.cpp",
        {receiver_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .defines = defines});

    std::vector<uint32_t> receiver_runtime_args = {
        receiver_buffer->address(),
        data_size,
    };
    tt_metal::SetRuntimeArgs(receiver_program, receiver_kernel, receiver_logical_core, receiver_runtime_args);

    // Launch sender and receiver programs and wait for them to finish
    this->RunProgramNonblocking(receiver_device, receiver_program);
    this->RunProgramNonblocking(sender_device, sender_program);
    this->WaitForSingleProgramDone(sender_device, sender_program);
    this->WaitForSingleProgramDone(receiver_device, receiver_program);

    // Validate the data received by the receiver
    std::vector<uint32_t> received_buffer_data;
    tt::tt_metal::detail::ReadFromBuffer(receiver_buffer, received_buffer_data);
    EXPECT_EQ(receiver_buffer_data, received_buffer_data);
    received_buffer_data.clear();
    tt::tt_metal::detail::ReadFromBuffer(receiver_atomic_buffer, received_buffer_data);
    EXPECT_EQ(atomic_inc, received_buffer_data[0]);
}

TEST_F(Fabric2DPushFixture, TestAsyncWriteAtomicInc) {
    using tt::tt_metal::ShardedBufferConfig;
    using tt::tt_metal::ShardOrientation;
    using tt::tt_metal::ShardSpecBuffer;

    CoreCoord sender_logical_core = {0, 0};
    CoreRangeSet sender_logical_crs = {sender_logical_core};
    CoreCoord receiver_logical_core = {1, 0};
    CoreRangeSet receiver_logical_crs = {receiver_logical_core};
    std::pair<mesh_id_t, chip_id_t> start_mesh_chip_id;
    chip_id_t physical_start_device_id;
    std::pair<mesh_id_t, chip_id_t> end_mesh_chip_id;
    chip_id_t physical_end_device_id;

    auto control_plane = tt::Cluster::instance().get_control_plane();

    // Find a device with a neighbour in the East direction
    bool connection_found = false;
    for (auto* device : devices_) {
        start_mesh_chip_id = control_plane->get_mesh_chip_id_from_physical_chip_id(device->id());
        // Get neighbours within a mesh in the East direction
        auto neighbors = control_plane->get_intra_chip_neighbors(
            start_mesh_chip_id.first, start_mesh_chip_id.second, RoutingDirection::E);
        if (neighbors.size() > 0) {
            physical_start_device_id = device->id();
            end_mesh_chip_id = {start_mesh_chip_id.first, neighbors[0]};
            physical_end_device_id = control_plane->get_physical_chip_id_from_mesh_chip_id(end_mesh_chip_id);
            connection_found = true;
            break;
        }
    }
    if (!connection_found) {
        GTEST_SKIP() << "No path found between sender and receivers";
    }

    // Get the optimal routers (no internal hops) on the start chip that will forward in the direction of the end chip
    auto routers = control_plane->get_routers_to_chip(
        start_mesh_chip_id.first, start_mesh_chip_id.second, end_mesh_chip_id.first, end_mesh_chip_id.second);

    auto* sender_device = DevicePool::instance().get_active_device(physical_start_device_id);
    auto* receiver_device = DevicePool::instance().get_active_device(physical_end_device_id);
    CoreCoord sender_virtual_core = sender_device->worker_core_from_logical_core(sender_logical_core);
    CoreCoord receiver_virtual_core = receiver_device->worker_core_from_logical_core(receiver_logical_core);

    uint32_t data_size = tt::constants::TILE_HW * sizeof(uint32_t);
    uint32_t atomic_inc_size = sizeof(uint32_t);

    auto receiver_shard_parameters =
        ShardSpecBuffer(receiver_logical_crs, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
    ShardedBufferConfig receiver_shard_config = {
        .device = receiver_device,
        .size = data_size,
        .page_size = data_size,
        .buffer_type = tt_metal::BufferType::L1,
        .buffer_layout = tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = receiver_shard_parameters,
    };
    auto receiver_buffer = CreateBuffer(receiver_shard_config);
    ShardedBufferConfig receiver_atomic_shard_config = {
        .device = receiver_device,
        .size = atomic_inc_size,
        .page_size = atomic_inc_size,
        .buffer_type = tt_metal::BufferType::L1,
        .buffer_layout = tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = receiver_shard_parameters,
    };
    auto receiver_atomic_buffer = CreateBuffer(receiver_atomic_shard_config);
    // Reset buffer space for test validation
    std::vector<uint32_t> receiver_buffer_data(atomic_inc_size / sizeof(uint32_t), 0);
    tt::tt_metal::detail::WriteToBuffer(receiver_atomic_buffer, receiver_buffer_data);
    receiver_buffer_data.resize(data_size / sizeof(uint32_t), 0);
    tt::tt_metal::detail::WriteToBuffer(receiver_buffer, receiver_buffer_data);

    // Packet header needs to be inlined with the data being sent, so this test just allocates buffer space for both
    // together on the sender
    uint32_t sender_packet_header_and_data_size = tt::tt_fabric::PACKET_HEADER_SIZE_BYTES + data_size;
    auto sender_shard_parameters =
        ShardSpecBuffer(sender_logical_crs, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
    ShardedBufferConfig sender_shard_config = {
        .device = sender_device,
        .size = sender_packet_header_and_data_size,
        .page_size = sender_packet_header_and_data_size,
        .buffer_type = tt_metal::BufferType::L1,
        .buffer_layout = tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = std::move(sender_shard_parameters),
    };
    auto sender_buffer = CreateBuffer(sender_shard_config);
    // Write the data to send to the buffer
    std::vector<uint32_t> sender_buffer_data(sender_packet_header_and_data_size / sizeof(uint32_t), 0);
    std::iota(sender_buffer_data.begin() + PACKET_HEADER_SIZE_BYTES / sizeof(uint32_t), sender_buffer_data.end(), 0);
    tt::tt_metal::detail::WriteToBuffer(sender_buffer, sender_buffer_data);

    uint32_t atomic_inc = 5;

    // Extract the expected data to be read from the receiver
    std::copy(
        sender_buffer_data.begin() + tt::tt_fabric::PACKET_HEADER_SIZE_BYTES / sizeof(uint32_t),
        sender_buffer_data.end(),
        receiver_buffer_data.begin());

    // Wait for buffer data to be written to device
    tt::Cluster::instance().l1_barrier(physical_end_device_id);
    tt::Cluster::instance().l1_barrier(physical_start_device_id);

    auto receiver_noc_encoding =
        tt::tt_metal::hal_ref.noc_xy_encoding(receiver_virtual_core.x, receiver_virtual_core.y);

    // Create the sender program
    auto sender_program = tt_metal::CreateProgram();

    // Allocate space for the client interface
    uint32_t client_interface_cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig client_interface_cb_config =
        tt::tt_metal::CircularBufferConfig(
            tt::tt_fabric::CLIENT_INTERFACE_SIZE, {{client_interface_cb_index, DataFormat::UInt32}})
            .set_page_size(client_interface_cb_index, tt::tt_fabric::CLIENT_INTERFACE_SIZE);
    auto client_interface_cb =
        tt::tt_metal::CreateCircularBuffer(sender_program, sender_logical_core, client_interface_cb_config);

    std::map<string, string> defines = {};
    defines["DISABLE_LOW_LATENCY_ROUTING"] = "";
    std::vector<uint32_t> sender_compile_time_args = {client_interface_cb_index, 0, fabric_mode::PUSH};
    auto sender_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_async_write_atomic_inc_sender.cpp",
        sender_logical_crs,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = sender_compile_time_args,
            .defines = defines});

    auto& sender_virtual_router_coord = routers[0].second;
    auto sender_router_noc_xy =
        tt_metal::hal_ref.noc_xy_encoding(sender_virtual_router_coord.x, sender_virtual_router_coord.y);
    auto outbound_eth_channels = tt::Cluster::instance().get_fabric_ethernet_channels(physical_start_device_id);
    std::vector<uint32_t> sender_runtime_args = {
        sender_buffer->address(),
        receiver_noc_encoding,
        receiver_buffer->address(),
        receiver_atomic_buffer->address(),
        data_size,
        atomic_inc,
        end_mesh_chip_id.first,
        end_mesh_chip_id.second,
        sender_router_noc_xy,
        *outbound_eth_channels.begin()};
    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_core, sender_runtime_args);

    // Create the receiver program for validation
    auto receiver_program = tt_metal::CreateProgram();
    auto receiver_kernel = tt_metal::CreateKernel(
        receiver_program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_receiver.cpp",
        {receiver_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .defines = defines});

    std::vector<uint32_t> receiver_runtime_args = {
        receiver_buffer->address(),
        data_size,
    };
    tt_metal::SetRuntimeArgs(receiver_program, receiver_kernel, receiver_logical_core, receiver_runtime_args);

    // Launch sender and receiver programs and wait for them to finish
    this->RunProgramNonblocking(receiver_device, receiver_program);
    this->RunProgramNonblocking(sender_device, sender_program);
    this->WaitForSingleProgramDone(sender_device, sender_program);
    this->WaitForSingleProgramDone(receiver_device, receiver_program);

    // Validate the data received by the receiver
    std::vector<uint32_t> received_buffer_data;
    tt::tt_metal::detail::ReadFromBuffer(receiver_buffer, received_buffer_data);
    EXPECT_EQ(receiver_buffer_data, received_buffer_data);
    received_buffer_data.clear();
    tt::tt_metal::detail::ReadFromBuffer(receiver_atomic_buffer, received_buffer_data);
    EXPECT_EQ(atomic_inc, received_buffer_data[0]);
}

TEST_F(Fabric2DPullFixture, TestAsyncRawWriteAtomicInc) {
    using tt::tt_metal::ShardedBufferConfig;
    using tt::tt_metal::ShardOrientation;
    using tt::tt_metal::ShardSpecBuffer;

    CoreCoord sender_logical_core = {0, 0};
    CoreRangeSet sender_logical_crs = {sender_logical_core};
    CoreCoord receiver_logical_core = {1, 0};
    CoreRangeSet receiver_logical_crs = {receiver_logical_core};
    std::pair<mesh_id_t, chip_id_t> start_mesh_chip_id;
    chip_id_t physical_start_device_id;
    std::pair<mesh_id_t, chip_id_t> end_mesh_chip_id;
    chip_id_t physical_end_device_id;

    auto control_plane = tt::Cluster::instance().get_control_plane();

    // Find a device with a neighbour in the East direction
    bool connection_found = false;
    for (auto* device : devices_) {
        start_mesh_chip_id = control_plane->get_mesh_chip_id_from_physical_chip_id(device->id());
        // Get neighbours within a mesh in the East direction
        auto neighbors = control_plane->get_intra_chip_neighbors(
            start_mesh_chip_id.first, start_mesh_chip_id.second, RoutingDirection::E);
        if (neighbors.size() > 0) {
            physical_start_device_id = device->id();
            end_mesh_chip_id = {start_mesh_chip_id.first, neighbors[0]};
            physical_end_device_id = control_plane->get_physical_chip_id_from_mesh_chip_id(end_mesh_chip_id);
            connection_found = true;
            break;
        }
    }
    if (!connection_found) {
        GTEST_SKIP() << "No path found between sender and receivers";
    }

    // Get the optimal routers (no internal hops) on the start chip that will forward in the direction of the end chip
    auto routers = control_plane->get_routers_to_chip(
        start_mesh_chip_id.first, start_mesh_chip_id.second, end_mesh_chip_id.first, end_mesh_chip_id.second);

    auto* sender_device = DevicePool::instance().get_active_device(physical_start_device_id);
    auto* receiver_device = DevicePool::instance().get_active_device(physical_end_device_id);
    CoreCoord sender_virtual_core = sender_device->worker_core_from_logical_core(sender_logical_core);
    CoreCoord receiver_virtual_core = receiver_device->worker_core_from_logical_core(receiver_logical_core);

    uint32_t data_size = tt::constants::TILE_HW * sizeof(uint32_t);
    uint32_t atomic_inc_size = sizeof(uint32_t);

    auto receiver_shard_parameters =
        ShardSpecBuffer(receiver_logical_crs, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
    ShardedBufferConfig receiver_shard_config = {
        .device = receiver_device,
        .size = data_size,
        .page_size = data_size,
        .buffer_type = tt_metal::BufferType::L1,
        .buffer_layout = tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = receiver_shard_parameters,
    };
    auto receiver_buffer = CreateBuffer(receiver_shard_config);
    ShardedBufferConfig receiver_atomic_shard_config = {
        .device = receiver_device,
        .size = atomic_inc_size,
        .page_size = atomic_inc_size,
        .buffer_type = tt_metal::BufferType::L1,
        .buffer_layout = tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = receiver_shard_parameters,
    };
    auto receiver_atomic_buffer = CreateBuffer(receiver_atomic_shard_config);
    // Reset buffer space for test validation
    std::vector<uint32_t> receiver_buffer_data(atomic_inc_size / sizeof(uint32_t), 0);
    tt::tt_metal::detail::WriteToBuffer(receiver_atomic_buffer, receiver_buffer_data);
    receiver_buffer_data.resize(data_size / sizeof(uint32_t), 0);
    tt::tt_metal::detail::WriteToBuffer(receiver_buffer, receiver_buffer_data);

    auto sender_shard_parameters =
        ShardSpecBuffer(sender_logical_crs, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
    ShardedBufferConfig sender_shard_config = {
        .device = sender_device,
        .size = data_size,
        .page_size = data_size,
        .buffer_type = tt_metal::BufferType::L1,
        .buffer_layout = tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = std::move(sender_shard_parameters),
    };
    auto sender_buffer = CreateBuffer(sender_shard_config);
    // Write the data to send to the buffer
    std::vector<uint32_t> sender_buffer_data(data_size / sizeof(uint32_t), 0);
    std::iota(sender_buffer_data.begin(), sender_buffer_data.end(), 0);
    tt::tt_metal::detail::WriteToBuffer(sender_buffer, sender_buffer_data);

    uint32_t atomic_inc = 5;

    // Wait for buffer data to be written to device
    tt::Cluster::instance().l1_barrier(physical_end_device_id);
    tt::Cluster::instance().l1_barrier(physical_start_device_id);

    auto receiver_noc_encoding =
        tt::tt_metal::hal_ref.noc_xy_encoding(receiver_virtual_core.x, receiver_virtual_core.y);

    // Create the sender program
    auto sender_program = tt_metal::CreateProgram();

    // Allocate space for the client interface
    uint32_t client_interface_cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig client_interface_cb_config =
        tt::tt_metal::CircularBufferConfig(
            tt::tt_fabric::CLIENT_INTERFACE_SIZE, {{client_interface_cb_index, DataFormat::UInt32}})
            .set_page_size(client_interface_cb_index, tt::tt_fabric::CLIENT_INTERFACE_SIZE);
    auto client_interface_cb =
        tt::tt_metal::CreateCircularBuffer(sender_program, sender_logical_core, client_interface_cb_config);

    std::map<string, string> defines = {};
    defines["FVC_MODE_PULL"] = "";
    std::vector<uint32_t> sender_compile_time_args = {client_interface_cb_index, 1, fabric_mode::PULL};
    auto sender_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_async_write_atomic_inc_sender.cpp",
        sender_logical_crs,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = sender_compile_time_args,
            .defines = defines});

    auto& sender_virtual_router_coord = routers[0].second;
    auto sender_router_noc_xy =
        tt_metal::hal_ref.noc_xy_encoding(sender_virtual_router_coord.x, sender_virtual_router_coord.y);

    std::vector<uint32_t> sender_runtime_args = {
        sender_buffer->address(),
        receiver_noc_encoding,
        receiver_buffer->address(),
        receiver_atomic_buffer->address(),
        data_size,
        atomic_inc,
        end_mesh_chip_id.first,
        end_mesh_chip_id.second,
        sender_router_noc_xy};
    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_core, sender_runtime_args);

    // Create the receiver program for validation
    auto receiver_program = tt_metal::CreateProgram();
    auto receiver_kernel = tt_metal::CreateKernel(
        receiver_program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_receiver.cpp",
        {receiver_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .defines = defines});

    std::vector<uint32_t> receiver_runtime_args = {
        receiver_buffer->address(),
        data_size,
    };
    tt_metal::SetRuntimeArgs(receiver_program, receiver_kernel, receiver_logical_core, receiver_runtime_args);

    // Launch sender and receiver programs and wait for them to finish
    this->RunProgramNonblocking(receiver_device, receiver_program);
    this->RunProgramNonblocking(sender_device, sender_program);
    this->WaitForSingleProgramDone(sender_device, sender_program);
    this->WaitForSingleProgramDone(receiver_device, receiver_program);

    // Validate the data received by the receiver
    std::vector<uint32_t> received_buffer_data;
    tt::tt_metal::detail::ReadFromBuffer(receiver_buffer, received_buffer_data);
    EXPECT_EQ(sender_buffer_data, received_buffer_data);
    received_buffer_data.clear();
    tt::tt_metal::detail::ReadFromBuffer(receiver_atomic_buffer, received_buffer_data);
    EXPECT_EQ(atomic_inc, received_buffer_data[0]);
}

TEST_F(Fabric2DPushFixture, TestAsyncRawWriteAtomicInc) {
    using tt::tt_metal::ShardedBufferConfig;
    using tt::tt_metal::ShardOrientation;
    using tt::tt_metal::ShardSpecBuffer;

    CoreCoord sender_logical_core = {0, 0};
    CoreRangeSet sender_logical_crs = {sender_logical_core};
    CoreCoord receiver_logical_core = {1, 0};
    CoreRangeSet receiver_logical_crs = {receiver_logical_core};
    std::pair<mesh_id_t, chip_id_t> start_mesh_chip_id;
    chip_id_t physical_start_device_id;
    std::pair<mesh_id_t, chip_id_t> end_mesh_chip_id;
    chip_id_t physical_end_device_id;

    auto control_plane = tt::Cluster::instance().get_control_plane();

    // Find a device with a neighbour in the East direction
    bool connection_found = false;
    for (auto* device : devices_) {
        start_mesh_chip_id = control_plane->get_mesh_chip_id_from_physical_chip_id(device->id());
        // Get neighbours within a mesh in the East direction
        auto neighbors = control_plane->get_intra_chip_neighbors(
            start_mesh_chip_id.first, start_mesh_chip_id.second, RoutingDirection::E);
        if (neighbors.size() > 0) {
            physical_start_device_id = device->id();
            end_mesh_chip_id = {start_mesh_chip_id.first, neighbors[0]};
            physical_end_device_id = control_plane->get_physical_chip_id_from_mesh_chip_id(end_mesh_chip_id);
            connection_found = true;
            break;
        }
    }
    if (!connection_found) {
        GTEST_SKIP() << "No path found between sender and receivers";
    }

    // Get the optimal routers (no internal hops) on the start chip that will forward in the direction of the end chip
    auto routers = control_plane->get_routers_to_chip(
        start_mesh_chip_id.first, start_mesh_chip_id.second, end_mesh_chip_id.first, end_mesh_chip_id.second);

    auto* sender_device = DevicePool::instance().get_active_device(physical_start_device_id);
    auto* receiver_device = DevicePool::instance().get_active_device(physical_end_device_id);
    CoreCoord sender_virtual_core = sender_device->worker_core_from_logical_core(sender_logical_core);
    CoreCoord receiver_virtual_core = receiver_device->worker_core_from_logical_core(receiver_logical_core);

    uint32_t data_size = tt::constants::TILE_HW * sizeof(uint32_t);
    uint32_t atomic_inc_size = sizeof(uint32_t);

    auto receiver_shard_parameters =
        ShardSpecBuffer(receiver_logical_crs, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
    ShardedBufferConfig receiver_shard_config = {
        .device = receiver_device,
        .size = data_size,
        .page_size = data_size,
        .buffer_type = tt_metal::BufferType::L1,
        .buffer_layout = tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = receiver_shard_parameters,
    };
    auto receiver_buffer = CreateBuffer(receiver_shard_config);
    ShardedBufferConfig receiver_atomic_shard_config = {
        .device = receiver_device,
        .size = atomic_inc_size,
        .page_size = atomic_inc_size,
        .buffer_type = tt_metal::BufferType::L1,
        .buffer_layout = tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = receiver_shard_parameters,
    };
    auto receiver_atomic_buffer = CreateBuffer(receiver_atomic_shard_config);
    // Reset buffer space for test validation
    std::vector<uint32_t> receiver_buffer_data(atomic_inc_size / sizeof(uint32_t), 0);
    tt::tt_metal::detail::WriteToBuffer(receiver_atomic_buffer, receiver_buffer_data);
    receiver_buffer_data.resize(data_size / sizeof(uint32_t), 0);
    tt::tt_metal::detail::WriteToBuffer(receiver_buffer, receiver_buffer_data);

    auto sender_shard_parameters =
        ShardSpecBuffer(sender_logical_crs, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
    ShardedBufferConfig sender_shard_config = {
        .device = sender_device,
        .size = data_size,
        .page_size = data_size,
        .buffer_type = tt_metal::BufferType::L1,
        .buffer_layout = tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = std::move(sender_shard_parameters),
    };
    auto sender_buffer = CreateBuffer(sender_shard_config);
    // Write the data to send to the buffer
    std::vector<uint32_t> sender_buffer_data(data_size / sizeof(uint32_t), 0);
    std::iota(sender_buffer_data.begin(), sender_buffer_data.end(), 0);
    tt::tt_metal::detail::WriteToBuffer(sender_buffer, sender_buffer_data);

    uint32_t atomic_inc = 5;

    // Wait for buffer data to be written to device
    tt::Cluster::instance().l1_barrier(physical_end_device_id);
    tt::Cluster::instance().l1_barrier(physical_start_device_id);

    auto receiver_noc_encoding =
        tt::tt_metal::hal_ref.noc_xy_encoding(receiver_virtual_core.x, receiver_virtual_core.y);

    // Create the sender program
    auto sender_program = tt_metal::CreateProgram();

    // Allocate space for the client interface
    uint32_t client_interface_cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig client_interface_cb_config =
        tt::tt_metal::CircularBufferConfig(
            tt::tt_fabric::CLIENT_INTERFACE_SIZE, {{client_interface_cb_index, DataFormat::UInt32}})
            .set_page_size(client_interface_cb_index, tt::tt_fabric::CLIENT_INTERFACE_SIZE);
    auto client_interface_cb =
        tt::tt_metal::CreateCircularBuffer(sender_program, sender_logical_core, client_interface_cb_config);

    std::map<string, string> defines = {};
    defines["DISABLE_LOW_LATENCY_ROUTING"] = "";
    std::vector<uint32_t> sender_compile_time_args = {client_interface_cb_index, 1, fabric_mode::PUSH};
    auto sender_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_async_write_atomic_inc_sender.cpp",
        sender_logical_crs,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = sender_compile_time_args,
            .defines = defines});

    auto& sender_virtual_router_coord = routers[0].second;
    auto sender_router_noc_xy =
        tt_metal::hal_ref.noc_xy_encoding(sender_virtual_router_coord.x, sender_virtual_router_coord.y);
    auto outbound_eth_channels = tt::Cluster::instance().get_fabric_ethernet_channels(physical_start_device_id);
    std::vector<uint32_t> sender_runtime_args = {
        sender_buffer->address(),
        receiver_noc_encoding,
        receiver_buffer->address(),
        receiver_atomic_buffer->address(),
        data_size,
        atomic_inc,
        end_mesh_chip_id.first,
        end_mesh_chip_id.second,
        sender_router_noc_xy,
        *outbound_eth_channels.begin()};
    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_core, sender_runtime_args);

    // Create the receiver program for validation
    auto receiver_program = tt_metal::CreateProgram();
    auto receiver_kernel = tt_metal::CreateKernel(
        receiver_program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_receiver.cpp",
        {receiver_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .defines = defines});

    std::vector<uint32_t> receiver_runtime_args = {
        receiver_buffer->address(),
        data_size,
    };
    tt_metal::SetRuntimeArgs(receiver_program, receiver_kernel, receiver_logical_core, receiver_runtime_args);

    // Launch sender and receiver programs and wait for them to finish
    this->RunProgramNonblocking(receiver_device, receiver_program);
    this->RunProgramNonblocking(sender_device, sender_program);
    this->WaitForSingleProgramDone(sender_device, sender_program);
    this->WaitForSingleProgramDone(receiver_device, receiver_program);

    // Validate the data received by the receiver
    std::vector<uint32_t> received_buffer_data;
    tt::tt_metal::detail::ReadFromBuffer(receiver_buffer, received_buffer_data);
    EXPECT_EQ(sender_buffer_data, received_buffer_data);
    received_buffer_data.clear();
    tt::tt_metal::detail::ReadFromBuffer(receiver_atomic_buffer, received_buffer_data);
    EXPECT_EQ(atomic_inc, received_buffer_data[0]);
}

TEST_F(Fabric2DPullFixture, TestAsyncWriteMulticast) {
    using tt::tt_metal::ShardedBufferConfig;
    using tt::tt_metal::ShardOrientation;
    using tt::tt_metal::ShardSpecBuffer;

    CoreCoord sender_logical_core = {0, 0};
    CoreRangeSet sender_logical_crs = {sender_logical_core};
    CoreCoord receiver_logical_core = {1, 0};
    CoreRangeSet receiver_logical_crs = {receiver_logical_core};
    std::pair<mesh_id_t, chip_id_t> start_mesh_chip_id;
    chip_id_t physical_start_device_id;
    std::unordered_map<RoutingDirection, std::vector<std::pair<mesh_id_t, chip_id_t>>> end_mesh_chip_ids_by_dir;
    std::unordered_map<RoutingDirection, std::vector<chip_id_t>> physical_end_device_ids_by_dir;
    std::unordered_map<RoutingDirection, uint32_t> mcast_hops;
    auto routing_direction = RoutingDirection::E;
    mcast_hops[routing_direction] = 1;

    auto control_plane = tt::Cluster::instance().get_control_plane();

    // Find a device with enough neighbours in the specified direction
    bool connection_found = false;
    for (auto* device : devices_) {
        start_mesh_chip_id = control_plane->get_mesh_chip_id_from_physical_chip_id(device->id());
        std::unordered_map<RoutingDirection, std::vector<std::pair<mesh_id_t, chip_id_t>>>
            temp_end_mesh_chip_ids_by_dir;
        std::unordered_map<RoutingDirection, std::vector<chip_id_t>> temp_physical_end_device_ids_by_dir;
        connection_found = true;
        for (auto [routing_direction, num_hops] : mcast_hops) {
            bool direction_found = true;
            auto& temp_end_mesh_chip_ids = temp_end_mesh_chip_ids_by_dir[routing_direction];
            auto& temp_physical_end_device_ids = temp_physical_end_device_ids_by_dir[routing_direction];
            uint32_t curr_mesh_id = start_mesh_chip_id.first;
            uint32_t curr_chip_id = start_mesh_chip_id.second;
            for (uint32_t i = 0; i < num_hops; i++) {
                auto neighbors = control_plane->get_intra_chip_neighbors(curr_mesh_id, curr_chip_id, routing_direction);
                if (neighbors.size() > 0) {
                    temp_end_mesh_chip_ids.emplace_back(curr_mesh_id, neighbors[0]);
                    temp_physical_end_device_ids.push_back(
                        control_plane->get_physical_chip_id_from_mesh_chip_id(temp_end_mesh_chip_ids.back()));
                    curr_mesh_id = temp_end_mesh_chip_ids.back().first;
                    curr_chip_id = temp_end_mesh_chip_ids.back().second;
                } else {
                    direction_found = false;
                    break;
                }
            }
            if (!direction_found) {
                connection_found = false;
                break;
            }
        }
        if (connection_found) {
            physical_start_device_id = device->id();
            end_mesh_chip_ids_by_dir = std::move(temp_end_mesh_chip_ids_by_dir);
            physical_end_device_ids_by_dir = std::move(temp_physical_end_device_ids_by_dir);
            break;
        }
    }

    if (!connection_found) {
        GTEST_SKIP() << "No path found between sender and receivers";
    }

    auto* sender_device = DevicePool::instance().get_active_device(physical_start_device_id);
    CoreCoord sender_virtual_core = sender_device->worker_core_from_logical_core(sender_logical_core);
    // Virtual coordinate space. All devices have the same logical to virtual mapping
    CoreCoord receiver_virtual_core = sender_device->worker_core_from_logical_core(receiver_logical_core);

    uint32_t data_size = tt::constants::TILE_HW * sizeof(uint32_t);

    auto receiver_shard_parameters =
        ShardSpecBuffer(receiver_logical_crs, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});

    // Reset buffer space for test validation
    std::vector<uint32_t> receiver_buffer_data(data_size / sizeof(uint32_t), 0);

    std::map<string, string> defines = {};
    defines["FVC_MODE_PULL"] = "";
    std::vector<tt_metal::Program> receiver_programs;
    std::vector<std::shared_ptr<tt_metal::Buffer>> receiver_buffers;
    for (auto& [routing_direction, physical_end_device_ids] : physical_end_device_ids_by_dir) {
        for (auto physical_end_device_id : physical_end_device_ids) {
            auto* receiver_device = DevicePool::instance().get_active_device(physical_end_device_id);
            ShardedBufferConfig receiver_shard_config = {
                .device = receiver_device,
                .size = data_size,
                .page_size = data_size,
                .buffer_type = tt_metal::BufferType::L1,
                .buffer_layout = tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
                .shard_parameters = receiver_shard_parameters,
            };
            auto receiver_buffer = CreateBuffer(receiver_shard_config);
            tt::tt_metal::detail::WriteToBuffer(receiver_buffer, receiver_buffer_data);
            tt::Cluster::instance().l1_barrier(physical_end_device_id);
            // Create the receiver program for validation
            auto receiver_program = tt_metal::CreateProgram();
            auto receiver_kernel = tt_metal::CreateKernel(
                receiver_program,
                "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_receiver.cpp",
                {receiver_logical_core},
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt_metal::NOC::RISCV_0_default,
                    .defines = defines});

            std::vector<uint32_t> receiver_runtime_args = {
                receiver_buffer->address(),
                data_size,
            };
            tt_metal::SetRuntimeArgs(receiver_program, receiver_kernel, receiver_logical_core, receiver_runtime_args);

            this->RunProgramNonblocking(receiver_device, receiver_program);
            receiver_programs.push_back(std::move(receiver_program));
            receiver_buffers.push_back(std::move(receiver_buffer));
        }
    }
    // Assume all receiver buffers are at the same address
    uint32_t receiver_buffer_addr = receiver_buffers[0]->address();
    for (const auto& receiver_buffer : receiver_buffers) {
        if (receiver_buffer_addr != receiver_buffer->address()) {
            GTEST_SKIP() << "Receiver buffers are not at the same address";
        }
    }

    // Packet header needs to be inlined with the data being sent, so this test just allocates buffer space for both
    // together on the sender
    uint32_t sender_packet_header_and_data_size = tt::tt_fabric::PACKET_HEADER_SIZE_BYTES + data_size;
    auto sender_shard_parameters =
        ShardSpecBuffer(sender_logical_crs, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
    ShardedBufferConfig sender_shard_config = {
        .device = sender_device,
        .size = sender_packet_header_and_data_size,
        .page_size = sender_packet_header_and_data_size,
        .buffer_type = tt_metal::BufferType::L1,
        .buffer_layout = tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = std::move(sender_shard_parameters),
    };
    auto sender_buffer = CreateBuffer(sender_shard_config);
    // Write the data to send to the buffer
    std::vector<uint32_t> sender_buffer_data(sender_packet_header_and_data_size / sizeof(uint32_t), 0);
    std::iota(sender_buffer_data.begin() + PACKET_HEADER_SIZE_BYTES / sizeof(uint32_t), sender_buffer_data.end(), 0);
    tt::tt_metal::detail::WriteToBuffer(sender_buffer, sender_buffer_data);

    // Extract the expected data to be read from the receiver
    std::copy(
        sender_buffer_data.begin() + tt::tt_fabric::PACKET_HEADER_SIZE_BYTES / sizeof(uint32_t),
        sender_buffer_data.end(),
        receiver_buffer_data.begin());

    // Wait for buffer data to be written to device
    tt::Cluster::instance().l1_barrier(physical_start_device_id);

    auto receiver_noc_encoding =
        tt::tt_metal::hal_ref.noc_xy_encoding(receiver_virtual_core.x, receiver_virtual_core.y);

    // Create the sender program
    auto sender_program = tt_metal::CreateProgram();

    // Allocate space for the client interface
    uint32_t client_interface_cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig client_interface_cb_config =
        tt::tt_metal::CircularBufferConfig(
            mcast_hops.size() * tt::tt_fabric::CLIENT_INTERFACE_SIZE, {{client_interface_cb_index, DataFormat::UInt32}})
            .set_page_size(client_interface_cb_index, tt::tt_fabric::CLIENT_INTERFACE_SIZE);
    auto client_interface_cb =
        tt::tt_metal::CreateCircularBuffer(sender_program, sender_logical_core, client_interface_cb_config);

    std::vector<uint32_t> sender_compile_time_args = {client_interface_cb_index, 0, fabric_mode::PULL};
    auto sender_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_async_write_multicast_sender.cpp",
        sender_logical_crs,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = sender_compile_time_args,
            .defines = defines});

    std::unordered_map<RoutingDirection, uint32_t> sender_router_noc_xys;
    for (auto& [routing_direction, end_mesh_chip_ids] : end_mesh_chip_ids_by_dir) {
        auto routers = control_plane->get_routers_to_chip(
            start_mesh_chip_id.first,
            start_mesh_chip_id.second,
            end_mesh_chip_ids[0].first,
            end_mesh_chip_ids[0].second);
        auto& sender_virtual_router_coord = routers[0].second;
        sender_router_noc_xys.try_emplace(
            routing_direction,
            tt_metal::hal_ref.noc_xy_encoding(sender_virtual_router_coord.x, sender_virtual_router_coord.y));
    }

    std::vector<uint32_t> sender_runtime_args = {
        sender_buffer->address(),
        receiver_noc_encoding,
        receiver_buffer_addr,
        data_size,
        end_mesh_chip_ids_by_dir[routing_direction][0].first,
        end_mesh_chip_ids_by_dir[routing_direction][0].second,
        mcast_hops[routing_direction],
        sender_router_noc_xys[routing_direction]};
    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_core, sender_runtime_args);

    // Launch sender and receiver programs and wait for them to finish
    this->RunProgramNonblocking(sender_device, sender_program);
    this->WaitForSingleProgramDone(sender_device, sender_program);
    for (auto [routing_direction, physical_end_device_ids] : physical_end_device_ids_by_dir) {
        for (uint32_t i = 0; i < physical_end_device_ids.size(); i++) {
            auto* receiver_device = DevicePool::instance().get_active_device(physical_end_device_ids[i]);
            this->WaitForSingleProgramDone(receiver_device, receiver_programs[i]);
        }
    }

    // Validate the data received by the receiver
    for (auto [routing_direction, physical_end_device_ids] : physical_end_device_ids_by_dir) {
        for (uint32_t i = 0; i < physical_end_device_ids.size(); i++) {
            std::vector<uint32_t> received_buffer_data;
            tt::tt_metal::detail::ReadFromBuffer(receiver_buffers[i], received_buffer_data);
            EXPECT_EQ(receiver_buffer_data, received_buffer_data);
        }
    }
}

TEST_F(Fabric2DPullFixture, TestAsyncRawWriteMulticast) {
    using tt::tt_metal::ShardedBufferConfig;
    using tt::tt_metal::ShardOrientation;
    using tt::tt_metal::ShardSpecBuffer;

    CoreCoord sender_logical_core = {0, 0};
    CoreRangeSet sender_logical_crs = {sender_logical_core};
    CoreCoord receiver_logical_core = {1, 0};
    CoreRangeSet receiver_logical_crs = {receiver_logical_core};
    std::pair<mesh_id_t, chip_id_t> start_mesh_chip_id;
    chip_id_t physical_start_device_id;
    std::unordered_map<RoutingDirection, std::vector<std::pair<mesh_id_t, chip_id_t>>> end_mesh_chip_ids_by_dir;
    std::unordered_map<RoutingDirection, std::vector<chip_id_t>> physical_end_device_ids_by_dir;
    std::unordered_map<RoutingDirection, uint32_t> mcast_hops;
    auto routing_direction = RoutingDirection::E;
    mcast_hops[routing_direction] = 1;

    auto control_plane = tt::Cluster::instance().get_control_plane();

    // Find a device with enough neighbours in the specified direction
    bool connection_found = false;
    for (auto* device : devices_) {
        start_mesh_chip_id = control_plane->get_mesh_chip_id_from_physical_chip_id(device->id());
        std::unordered_map<RoutingDirection, std::vector<std::pair<mesh_id_t, chip_id_t>>>
            temp_end_mesh_chip_ids_by_dir;
        std::unordered_map<RoutingDirection, std::vector<chip_id_t>> temp_physical_end_device_ids_by_dir;
        connection_found = true;
        for (auto [routing_direction, num_hops] : mcast_hops) {
            bool direction_found = true;
            auto& temp_end_mesh_chip_ids = temp_end_mesh_chip_ids_by_dir[routing_direction];
            auto& temp_physical_end_device_ids = temp_physical_end_device_ids_by_dir[routing_direction];
            uint32_t curr_mesh_id = start_mesh_chip_id.first;
            uint32_t curr_chip_id = start_mesh_chip_id.second;
            for (uint32_t i = 0; i < num_hops; i++) {
                auto neighbors = control_plane->get_intra_chip_neighbors(curr_mesh_id, curr_chip_id, routing_direction);
                if (neighbors.size() > 0) {
                    temp_end_mesh_chip_ids.emplace_back(curr_mesh_id, neighbors[0]);
                    temp_physical_end_device_ids.push_back(
                        control_plane->get_physical_chip_id_from_mesh_chip_id(temp_end_mesh_chip_ids.back()));
                    curr_mesh_id = temp_end_mesh_chip_ids.back().first;
                    curr_chip_id = temp_end_mesh_chip_ids.back().second;
                } else {
                    direction_found = false;
                    break;
                }
            }
            if (!direction_found) {
                connection_found = false;
                break;
            }
        }
        if (connection_found) {
            physical_start_device_id = device->id();
            end_mesh_chip_ids_by_dir = std::move(temp_end_mesh_chip_ids_by_dir);
            physical_end_device_ids_by_dir = std::move(temp_physical_end_device_ids_by_dir);
            break;
        }
    }

    if (!connection_found) {
        GTEST_SKIP() << "No path found between sender and receivers";
    }

    tt::log_info(
        tt::LogTest, "Async Raw Write Mcast from {} to {}", start_mesh_chip_id.second, end_mesh_chip_ids_by_dir);

    auto* sender_device = DevicePool::instance().get_active_device(physical_start_device_id);
    CoreCoord sender_virtual_core = sender_device->worker_core_from_logical_core(sender_logical_core);
    // Virtual coordinate space. All devices have the same logical to virtual mapping
    CoreCoord receiver_virtual_core = sender_device->worker_core_from_logical_core(receiver_logical_core);

    uint32_t data_size = tt::constants::TILE_HW * sizeof(uint32_t);

    auto receiver_shard_parameters =
        ShardSpecBuffer(receiver_logical_crs, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});

    // Reset buffer space for test validation
    std::vector<uint32_t> receiver_buffer_data(data_size / sizeof(uint32_t), 0);

    std::map<string, string> defines = {};
    defines["FVC_MODE_PULL"] = "";
    std::vector<tt_metal::Program> receiver_programs;
    std::vector<std::shared_ptr<tt_metal::Buffer>> receiver_buffers;
    for (auto& [routing_direction, physical_end_device_ids] : physical_end_device_ids_by_dir) {
        for (auto physical_end_device_id : physical_end_device_ids) {
            auto* receiver_device = DevicePool::instance().get_active_device(physical_end_device_id);
            ShardedBufferConfig receiver_shard_config = {
                .device = receiver_device,
                .size = data_size,
                .page_size = data_size,
                .buffer_type = tt_metal::BufferType::L1,
                .buffer_layout = tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
                .shard_parameters = receiver_shard_parameters,
            };
            auto receiver_buffer = CreateBuffer(receiver_shard_config);
            tt::tt_metal::detail::WriteToBuffer(receiver_buffer, receiver_buffer_data);
            tt::Cluster::instance().l1_barrier(physical_end_device_id);
            // Create the receiver program for validation
            auto receiver_program = tt_metal::CreateProgram();
            auto receiver_kernel = tt_metal::CreateKernel(
                receiver_program,
                "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_receiver.cpp",
                {receiver_logical_core},
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt_metal::NOC::RISCV_0_default,
                    .defines = defines});

            std::vector<uint32_t> receiver_runtime_args = {
                receiver_buffer->address(),
                data_size,
            };
            tt_metal::SetRuntimeArgs(receiver_program, receiver_kernel, receiver_logical_core, receiver_runtime_args);

            this->RunProgramNonblocking(receiver_device, receiver_program);
            receiver_programs.push_back(std::move(receiver_program));
            receiver_buffers.push_back(std::move(receiver_buffer));
        }
    }
    // Assume all receiver buffers are at the same address
    uint32_t receiver_buffer_addr = receiver_buffers[0]->address();
    for (const auto& receiver_buffer : receiver_buffers) {
        if (receiver_buffer_addr != receiver_buffer->address()) {
            GTEST_SKIP() << "Receiver buffers are not at the same address";
        }
    }

    auto sender_shard_parameters =
        ShardSpecBuffer(sender_logical_crs, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
    ShardedBufferConfig sender_shard_config = {
        .device = sender_device,
        .size = data_size,
        .page_size = data_size,
        .buffer_type = tt_metal::BufferType::L1,
        .buffer_layout = tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = std::move(sender_shard_parameters),
    };
    auto sender_buffer = CreateBuffer(sender_shard_config);
    // Write the data to send to the buffer
    std::vector<uint32_t> sender_buffer_data(data_size / sizeof(uint32_t), 0);
    std::iota(sender_buffer_data.begin(), sender_buffer_data.end(), 0);
    tt::tt_metal::detail::WriteToBuffer(sender_buffer, sender_buffer_data);

    // Wait for buffer data to be written to device
    tt::Cluster::instance().l1_barrier(physical_start_device_id);

    auto receiver_noc_encoding =
        tt::tt_metal::hal_ref.noc_xy_encoding(receiver_virtual_core.x, receiver_virtual_core.y);

    // Create the sender program
    auto sender_program = tt_metal::CreateProgram();

    // Allocate space for the client interface
    uint32_t client_interface_cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig client_interface_cb_config =
        tt::tt_metal::CircularBufferConfig(
            mcast_hops.size() * tt::tt_fabric::CLIENT_INTERFACE_SIZE, {{client_interface_cb_index, DataFormat::UInt32}})
            .set_page_size(client_interface_cb_index, tt::tt_fabric::CLIENT_INTERFACE_SIZE);
    auto client_interface_cb =
        tt::tt_metal::CreateCircularBuffer(sender_program, sender_logical_core, client_interface_cb_config);

    std::vector<uint32_t> sender_compile_time_args = {client_interface_cb_index, 1, fabric_mode::PULL};
    auto sender_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_async_write_multicast_sender.cpp",
        sender_logical_crs,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = sender_compile_time_args,
            .defines = defines});

    std::unordered_map<RoutingDirection, uint32_t> sender_router_noc_xys;
    for (auto& [routing_direction, end_mesh_chip_ids] : end_mesh_chip_ids_by_dir) {
        auto routers = control_plane->get_routers_to_chip(
            start_mesh_chip_id.first,
            start_mesh_chip_id.second,
            end_mesh_chip_ids[0].first,
            end_mesh_chip_ids[0].second);
        auto& sender_virtual_router_coord = routers[0].second;
        sender_router_noc_xys.try_emplace(
            routing_direction,
            tt_metal::hal_ref.noc_xy_encoding(sender_virtual_router_coord.x, sender_virtual_router_coord.y));
    }

    std::vector<uint32_t> sender_runtime_args = {
        sender_buffer->address(),
        receiver_noc_encoding,
        receiver_buffer_addr,
        data_size,
        end_mesh_chip_ids_by_dir[routing_direction][0].first,
        end_mesh_chip_ids_by_dir[routing_direction][0].second,
        mcast_hops[routing_direction],
        sender_router_noc_xys[routing_direction]};
    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_core, sender_runtime_args);

    // Launch sender and receiver programs and wait for them to finish
    this->RunProgramNonblocking(sender_device, sender_program);
    this->WaitForSingleProgramDone(sender_device, sender_program);
    for (auto [routing_direction, physical_end_device_ids] : physical_end_device_ids_by_dir) {
        for (uint32_t i = 0; i < physical_end_device_ids.size(); i++) {
            auto* receiver_device = DevicePool::instance().get_active_device(physical_end_device_ids[i]);
            this->WaitForSingleProgramDone(receiver_device, receiver_programs[i]);
        }
    }

    // Validate the data received by the receiver
    for (auto [routing_direction, physical_end_device_ids] : physical_end_device_ids_by_dir) {
        for (uint32_t i = 0; i < physical_end_device_ids.size(); i++) {
            std::vector<uint32_t> received_buffer_data;
            tt::tt_metal::detail::ReadFromBuffer(receiver_buffers[i], received_buffer_data);
            EXPECT_EQ(sender_buffer_data, received_buffer_data);
        }
    }
}

TEST_F(Fabric2DPullFixture, TestAsyncWriteMulticastMultidirectional) {
    using tt::tt_metal::ShardedBufferConfig;
    using tt::tt_metal::ShardOrientation;
    using tt::tt_metal::ShardSpecBuffer;

    CoreCoord sender_logical_core = {0, 0};
    CoreRangeSet sender_logical_crs = {sender_logical_core};
    CoreCoord receiver_logical_core = {1, 0};
    CoreRangeSet receiver_logical_crs = {receiver_logical_core};
    std::pair<mesh_id_t, chip_id_t> start_mesh_chip_id;
    chip_id_t physical_start_device_id;
    std::unordered_map<RoutingDirection, std::vector<std::pair<mesh_id_t, chip_id_t>>> end_mesh_chip_ids_by_dir;
    std::unordered_map<RoutingDirection, std::vector<chip_id_t>> physical_end_device_ids_by_dir;
    RoutingDirection routing_direction = RoutingDirection::E;
    std::unordered_map<RoutingDirection, uint32_t> mcast_hops;
    mcast_hops[RoutingDirection::E] = 1;
    mcast_hops[RoutingDirection::W] = 2;

    auto control_plane = tt::Cluster::instance().get_control_plane();

    // Find a device with enough neighbours in the specified direction
    bool connection_found = false;
    for (auto* device : devices_) {
        start_mesh_chip_id = control_plane->get_mesh_chip_id_from_physical_chip_id(device->id());
        std::unordered_map<RoutingDirection, std::vector<std::pair<mesh_id_t, chip_id_t>>>
            temp_end_mesh_chip_ids_by_dir;
        std::unordered_map<RoutingDirection, std::vector<chip_id_t>> temp_physical_end_device_ids_by_dir;
        connection_found = true;
        for (auto [routing_direction, num_hops] : mcast_hops) {
            bool direction_found = true;
            auto& temp_end_mesh_chip_ids = temp_end_mesh_chip_ids_by_dir[routing_direction];
            auto& temp_physical_end_device_ids = temp_physical_end_device_ids_by_dir[routing_direction];
            uint32_t curr_mesh_id = start_mesh_chip_id.first;
            uint32_t curr_chip_id = start_mesh_chip_id.second;
            for (uint32_t i = 0; i < num_hops; i++) {
                auto neighbors = control_plane->get_intra_chip_neighbors(curr_mesh_id, curr_chip_id, routing_direction);
                if (neighbors.size() > 0) {
                    temp_end_mesh_chip_ids.emplace_back(curr_mesh_id, neighbors[0]);
                    temp_physical_end_device_ids.push_back(
                        control_plane->get_physical_chip_id_from_mesh_chip_id(temp_end_mesh_chip_ids.back()));
                    curr_mesh_id = temp_end_mesh_chip_ids.back().first;
                    curr_chip_id = temp_end_mesh_chip_ids.back().second;
                } else {
                    direction_found = false;
                    break;
                }
            }
            if (!direction_found) {
                connection_found = false;
                break;
            }
        }
        if (connection_found) {
            physical_start_device_id = device->id();
            end_mesh_chip_ids_by_dir = std::move(temp_end_mesh_chip_ids_by_dir);
            physical_end_device_ids_by_dir = std::move(temp_physical_end_device_ids_by_dir);
            break;
        }
    }

    if (!connection_found) {
        GTEST_SKIP() << "No path found between sender and receivers";
    }
    tt::log_info(
        tt::LogTest,
        "Async Write Mcast Multidirection from {} to {}",
        start_mesh_chip_id.second,
        end_mesh_chip_ids_by_dir);

    auto* sender_device = DevicePool::instance().get_active_device(physical_start_device_id);
    CoreCoord sender_virtual_core = sender_device->worker_core_from_logical_core(sender_logical_core);
    // Virtual coordinate space. All devices have the same logical to virtual mapping
    CoreCoord receiver_virtual_core = sender_device->worker_core_from_logical_core(receiver_logical_core);

    uint32_t data_size = tt::constants::TILE_HW * sizeof(uint32_t);

    auto receiver_shard_parameters =
        ShardSpecBuffer(receiver_logical_crs, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});

    // Reset buffer space for test validation
    std::vector<uint32_t> receiver_buffer_data(data_size / sizeof(uint32_t), 0);

    std::map<string, string> defines = {};
    defines["FVC_MODE_PULL"] = "";
    std::vector<tt_metal::Program> receiver_programs;
    std::vector<std::shared_ptr<tt_metal::Buffer>> receiver_buffers;
    for (auto& [routing_direction, physical_end_device_ids] : physical_end_device_ids_by_dir) {
        for (auto physical_end_device_id : physical_end_device_ids) {
            auto* receiver_device = DevicePool::instance().get_active_device(physical_end_device_id);
            ShardedBufferConfig receiver_shard_config = {
                .device = receiver_device,
                .size = data_size,
                .page_size = data_size,
                .buffer_type = tt_metal::BufferType::L1,
                .buffer_layout = tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
                .shard_parameters = receiver_shard_parameters,
            };
            auto receiver_buffer = CreateBuffer(receiver_shard_config);
            tt::tt_metal::detail::WriteToBuffer(receiver_buffer, receiver_buffer_data);
            tt::Cluster::instance().l1_barrier(physical_end_device_id);
            // Create the receiver program for validation
            auto receiver_program = tt_metal::CreateProgram();
            auto receiver_kernel = tt_metal::CreateKernel(
                receiver_program,
                "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_receiver.cpp",
                {receiver_logical_core},
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt_metal::NOC::RISCV_0_default,
                    .defines = defines});

            std::vector<uint32_t> receiver_runtime_args = {
                receiver_buffer->address(),
                data_size,
            };
            tt_metal::SetRuntimeArgs(receiver_program, receiver_kernel, receiver_logical_core, receiver_runtime_args);

            this->RunProgramNonblocking(receiver_device, receiver_program);
            receiver_programs.push_back(std::move(receiver_program));
            receiver_buffers.push_back(std::move(receiver_buffer));
        }
    }
    // Assume all receiver buffers are at the same address
    uint32_t receiver_buffer_addr = receiver_buffers[0]->address();
    for (const auto& receiver_buffer : receiver_buffers) {
        if (receiver_buffer_addr != receiver_buffer->address()) {
            GTEST_SKIP() << "Receiver buffers are not at the same address";
        }
    }

    // Packet header needs to be inlined with the data being sent, so this test just allocates buffer space for both
    // together on the sender
    uint32_t sender_packet_header_and_data_size = tt::tt_fabric::PACKET_HEADER_SIZE_BYTES + data_size;
    auto sender_shard_parameters =
        ShardSpecBuffer(sender_logical_crs, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
    ShardedBufferConfig sender_shard_config = {
        .device = sender_device,
        .size = sender_packet_header_and_data_size,
        .page_size = sender_packet_header_and_data_size,
        .buffer_type = tt_metal::BufferType::L1,
        .buffer_layout = tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = std::move(sender_shard_parameters),
    };
    auto sender_buffer = CreateBuffer(sender_shard_config);
    // Write the data to send to the buffer
    std::vector<uint32_t> sender_buffer_data(sender_packet_header_and_data_size / sizeof(uint32_t), 0);
    std::iota(sender_buffer_data.begin() + PACKET_HEADER_SIZE_BYTES / sizeof(uint32_t), sender_buffer_data.end(), 0);
    tt::tt_metal::detail::WriteToBuffer(sender_buffer, sender_buffer_data);

    // Extract the expected data to be read from the receiver
    std::copy(
        sender_buffer_data.begin() + tt::tt_fabric::PACKET_HEADER_SIZE_BYTES / sizeof(uint32_t),
        sender_buffer_data.end(),
        receiver_buffer_data.begin());

    // Wait for buffer data to be written to device
    tt::Cluster::instance().l1_barrier(physical_start_device_id);

    auto receiver_noc_encoding =
        tt::tt_metal::hal_ref.noc_xy_encoding(receiver_virtual_core.x, receiver_virtual_core.y);

    // Create the sender program
    auto sender_program = tt_metal::CreateProgram();

    // Allocate space for the client interface
    uint32_t client_interface_cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig client_interface_cb_config =
        tt::tt_metal::CircularBufferConfig(
            mcast_hops.size() * tt::tt_fabric::CLIENT_INTERFACE_SIZE, {{client_interface_cb_index, DataFormat::UInt32}})
            .set_page_size(client_interface_cb_index, tt::tt_fabric::CLIENT_INTERFACE_SIZE);
    auto client_interface_cb =
        tt::tt_metal::CreateCircularBuffer(sender_program, sender_logical_core, client_interface_cb_config);

    std::vector<uint32_t> sender_compile_time_args = {client_interface_cb_index, 0, fabric_mode::PULL};
    auto sender_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/"
        "fabric_async_write_multicast_multidirectional_sender.cpp",
        sender_logical_crs,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = sender_compile_time_args,
            .defines = defines});

    std::unordered_map<RoutingDirection, uint32_t> sender_router_noc_xys;
    for (auto& [routing_direction, end_mesh_chip_ids] : end_mesh_chip_ids_by_dir) {
        auto routers = control_plane->get_routers_to_chip(
            start_mesh_chip_id.first,
            start_mesh_chip_id.second,
            end_mesh_chip_ids[0].first,
            end_mesh_chip_ids[0].second);
        auto& sender_virtual_router_coord = routers[0].second;
        sender_router_noc_xys.try_emplace(
            routing_direction,
            tt_metal::hal_ref.noc_xy_encoding(sender_virtual_router_coord.x, sender_virtual_router_coord.y));
    }

    std::vector<uint32_t> sender_runtime_args = {
        sender_buffer->address(),
        receiver_noc_encoding,
        receiver_buffer_addr,
        data_size,
        end_mesh_chip_ids_by_dir[RoutingDirection::E][0].first,
        end_mesh_chip_ids_by_dir[RoutingDirection::E][0].second,
        mcast_hops[RoutingDirection::E],
        sender_router_noc_xys[RoutingDirection::E],
        end_mesh_chip_ids_by_dir[RoutingDirection::W][0].first,
        end_mesh_chip_ids_by_dir[RoutingDirection::W][0].second,
        mcast_hops[RoutingDirection::W],
        sender_router_noc_xys[RoutingDirection::W]};
    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_core, sender_runtime_args);

    // Launch sender and receiver programs and wait for them to finish
    this->RunProgramNonblocking(sender_device, sender_program);
    this->WaitForSingleProgramDone(sender_device, sender_program);
    for (auto [routing_direction, physical_end_device_ids] : physical_end_device_ids_by_dir) {
        for (uint32_t i = 0; i < physical_end_device_ids.size(); i++) {
            auto* receiver_device = DevicePool::instance().get_active_device(physical_end_device_ids[i]);
            this->WaitForSingleProgramDone(receiver_device, receiver_programs[i]);
        }
    }

    // Validate the data received by the receiver
    for (auto [routing_direction, physical_end_device_ids] : physical_end_device_ids_by_dir) {
        for (uint32_t i = 0; i < physical_end_device_ids.size(); i++) {
            std::vector<uint32_t> received_buffer_data;
            tt::tt_metal::detail::ReadFromBuffer(receiver_buffers[i], received_buffer_data);
            EXPECT_EQ(receiver_buffer_data, received_buffer_data);
        }
    }
}

TEST_F(Fabric2DPullFixture, TestAsyncRawWriteMulticastMultidirectional) {
    using tt::tt_metal::ShardedBufferConfig;
    using tt::tt_metal::ShardOrientation;
    using tt::tt_metal::ShardSpecBuffer;

    CoreCoord sender_logical_core = {0, 0};
    CoreRangeSet sender_logical_crs = {sender_logical_core};
    CoreCoord receiver_logical_core = {1, 0};
    CoreRangeSet receiver_logical_crs = {receiver_logical_core};
    std::pair<mesh_id_t, chip_id_t> start_mesh_chip_id;
    chip_id_t physical_start_device_id;
    std::unordered_map<RoutingDirection, std::vector<std::pair<mesh_id_t, chip_id_t>>> end_mesh_chip_ids_by_dir;
    std::unordered_map<RoutingDirection, std::vector<chip_id_t>> physical_end_device_ids_by_dir;
    RoutingDirection routing_direction = RoutingDirection::E;
    std::unordered_map<RoutingDirection, uint32_t> mcast_hops;
    mcast_hops[RoutingDirection::E] = 1;
    mcast_hops[RoutingDirection::W] = 2;

    auto control_plane = tt::Cluster::instance().get_control_plane();

    // Find a device with enough neighbours in the specified direction
    bool connection_found = false;
    for (auto* device : devices_) {
        start_mesh_chip_id = control_plane->get_mesh_chip_id_from_physical_chip_id(device->id());
        std::unordered_map<RoutingDirection, std::vector<std::pair<mesh_id_t, chip_id_t>>>
            temp_end_mesh_chip_ids_by_dir;
        std::unordered_map<RoutingDirection, std::vector<chip_id_t>> temp_physical_end_device_ids_by_dir;
        connection_found = true;
        for (auto [routing_direction, num_hops] : mcast_hops) {
            bool direction_found = true;
            auto& temp_end_mesh_chip_ids = temp_end_mesh_chip_ids_by_dir[routing_direction];
            auto& temp_physical_end_device_ids = temp_physical_end_device_ids_by_dir[routing_direction];
            uint32_t curr_mesh_id = start_mesh_chip_id.first;
            uint32_t curr_chip_id = start_mesh_chip_id.second;
            for (uint32_t i = 0; i < num_hops; i++) {
                auto neighbors = control_plane->get_intra_chip_neighbors(curr_mesh_id, curr_chip_id, routing_direction);
                if (neighbors.size() > 0) {
                    temp_end_mesh_chip_ids.emplace_back(curr_mesh_id, neighbors[0]);
                    temp_physical_end_device_ids.push_back(
                        control_plane->get_physical_chip_id_from_mesh_chip_id(temp_end_mesh_chip_ids.back()));
                    curr_mesh_id = temp_end_mesh_chip_ids.back().first;
                    curr_chip_id = temp_end_mesh_chip_ids.back().second;
                } else {
                    direction_found = false;
                    break;
                }
            }
            if (!direction_found) {
                connection_found = false;
                break;
            }
        }
        if (connection_found) {
            physical_start_device_id = device->id();
            end_mesh_chip_ids_by_dir = std::move(temp_end_mesh_chip_ids_by_dir);
            physical_end_device_ids_by_dir = std::move(temp_physical_end_device_ids_by_dir);
            break;
        }
    }

    if (!connection_found) {
        GTEST_SKIP() << "No path found between sender and receivers";
    }
    tt::log_info(
        tt::LogTest,
        "Async Raw Write Mcast Multidirection from {} to {}",
        start_mesh_chip_id.second,
        end_mesh_chip_ids_by_dir);

    auto* sender_device = DevicePool::instance().get_active_device(physical_start_device_id);
    CoreCoord sender_virtual_core = sender_device->worker_core_from_logical_core(sender_logical_core);
    // Virtual coordinate space. All devices have the same logical to virtual mapping
    CoreCoord receiver_virtual_core = sender_device->worker_core_from_logical_core(receiver_logical_core);

    uint32_t data_size = tt::constants::TILE_HW * sizeof(uint32_t);

    auto receiver_shard_parameters =
        ShardSpecBuffer(receiver_logical_crs, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});

    // Reset buffer space for test validation
    std::vector<uint32_t> receiver_buffer_data(data_size / sizeof(uint32_t), 0);

    std::map<string, string> defines = {};
    defines["FVC_MODE_PULL"] = "";
    std::vector<tt_metal::Program> receiver_programs;
    std::vector<std::shared_ptr<tt_metal::Buffer>> receiver_buffers;
    for (auto& [routing_direction, physical_end_device_ids] : physical_end_device_ids_by_dir) {
        for (auto physical_end_device_id : physical_end_device_ids) {
            auto* receiver_device = DevicePool::instance().get_active_device(physical_end_device_id);
            ShardedBufferConfig receiver_shard_config = {
                .device = receiver_device,
                .size = data_size,
                .page_size = data_size,
                .buffer_type = tt_metal::BufferType::L1,
                .buffer_layout = tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
                .shard_parameters = receiver_shard_parameters,
            };
            auto receiver_buffer = CreateBuffer(receiver_shard_config);
            tt::tt_metal::detail::WriteToBuffer(receiver_buffer, receiver_buffer_data);
            tt::Cluster::instance().l1_barrier(physical_end_device_id);
            // Create the receiver program for validation
            auto receiver_program = tt_metal::CreateProgram();
            auto receiver_kernel = tt_metal::CreateKernel(
                receiver_program,
                "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_receiver.cpp",
                {receiver_logical_core},
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt_metal::NOC::RISCV_0_default,
                    .defines = defines});

            std::vector<uint32_t> receiver_runtime_args = {
                receiver_buffer->address(),
                data_size,
            };
            tt_metal::SetRuntimeArgs(receiver_program, receiver_kernel, receiver_logical_core, receiver_runtime_args);

            this->RunProgramNonblocking(receiver_device, receiver_program);
            receiver_programs.push_back(std::move(receiver_program));
            receiver_buffers.push_back(std::move(receiver_buffer));
        }
    }
    // Assume all receiver buffers are at the same address
    uint32_t receiver_buffer_addr = receiver_buffers[0]->address();
    for (const auto& receiver_buffer : receiver_buffers) {
        if (receiver_buffer_addr != receiver_buffer->address()) {
            GTEST_SKIP() << "Receiver buffers are not at the same address";
        }
    }

    auto sender_shard_parameters =
        ShardSpecBuffer(sender_logical_crs, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
    ShardedBufferConfig sender_shard_config = {
        .device = sender_device,
        .size = data_size,
        .page_size = data_size,
        .buffer_type = tt_metal::BufferType::L1,
        .buffer_layout = tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = std::move(sender_shard_parameters),
    };
    auto sender_buffer = CreateBuffer(sender_shard_config);
    // Write the data to send to the buffer
    std::vector<uint32_t> sender_buffer_data(data_size / sizeof(uint32_t), 0);
    std::iota(sender_buffer_data.begin(), sender_buffer_data.end(), 0);
    tt::tt_metal::detail::WriteToBuffer(sender_buffer, sender_buffer_data);

    // Wait for buffer data to be written to device
    tt::Cluster::instance().l1_barrier(physical_start_device_id);

    auto receiver_noc_encoding =
        tt::tt_metal::hal_ref.noc_xy_encoding(receiver_virtual_core.x, receiver_virtual_core.y);

    // Create the sender program
    auto sender_program = tt_metal::CreateProgram();

    // Allocate space for the client interface
    uint32_t client_interface_cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig client_interface_cb_config =
        tt::tt_metal::CircularBufferConfig(
            mcast_hops.size() * tt::tt_fabric::CLIENT_INTERFACE_SIZE, {{client_interface_cb_index, DataFormat::UInt32}})
            .set_page_size(client_interface_cb_index, tt::tt_fabric::CLIENT_INTERFACE_SIZE);
    auto client_interface_cb =
        tt::tt_metal::CreateCircularBuffer(sender_program, sender_logical_core, client_interface_cb_config);

    std::vector<uint32_t> sender_compile_time_args = {client_interface_cb_index, 1, fabric_mode::PULL};
    auto sender_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/"
        "fabric_async_write_multicast_multidirectional_sender.cpp",
        sender_logical_crs,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = sender_compile_time_args,
            .defines = defines});

    std::unordered_map<RoutingDirection, uint32_t> sender_router_noc_xys;
    for (auto& [routing_direction, end_mesh_chip_ids] : end_mesh_chip_ids_by_dir) {
        auto routers = control_plane->get_routers_to_chip(
            start_mesh_chip_id.first,
            start_mesh_chip_id.second,
            end_mesh_chip_ids[0].first,
            end_mesh_chip_ids[0].second);
        auto& sender_virtual_router_coord = routers[0].second;
        sender_router_noc_xys.try_emplace(
            routing_direction,
            tt_metal::hal_ref.noc_xy_encoding(sender_virtual_router_coord.x, sender_virtual_router_coord.y));
    }

    std::vector<uint32_t> sender_runtime_args = {
        sender_buffer->address(),
        receiver_noc_encoding,
        receiver_buffer_addr,
        data_size,
        end_mesh_chip_ids_by_dir[RoutingDirection::E][0].first,
        end_mesh_chip_ids_by_dir[RoutingDirection::E][0].second,
        mcast_hops[RoutingDirection::E],
        sender_router_noc_xys[RoutingDirection::E],
        end_mesh_chip_ids_by_dir[RoutingDirection::W][0].first,
        end_mesh_chip_ids_by_dir[RoutingDirection::W][0].second,
        mcast_hops[RoutingDirection::W],
        sender_router_noc_xys[RoutingDirection::W]};
    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_core, sender_runtime_args);

    // Launch sender and receiver programs and wait for them to finish
    this->RunProgramNonblocking(sender_device, sender_program);
    this->WaitForSingleProgramDone(sender_device, sender_program);
    for (auto [routing_direction, physical_end_device_ids] : physical_end_device_ids_by_dir) {
        for (uint32_t i = 0; i < physical_end_device_ids.size(); i++) {
            auto* receiver_device = DevicePool::instance().get_active_device(physical_end_device_ids[i]);
            this->WaitForSingleProgramDone(receiver_device, receiver_programs[i]);
        }
    }

    // Validate the data received by the receiver
    for (auto [routing_direction, physical_end_device_ids] : physical_end_device_ids_by_dir) {
        for (uint32_t i = 0; i < physical_end_device_ids.size(); i++) {
            std::vector<uint32_t> received_buffer_data;
            tt::tt_metal::detail::ReadFromBuffer(receiver_buffers[i], received_buffer_data);
            EXPECT_EQ(sender_buffer_data, received_buffer_data);
        }
    }
}

}  // namespace fabric_router_tests
}  // namespace tt::tt_fabric
