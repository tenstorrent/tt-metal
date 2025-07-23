// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <stdint.h>
#include <tt-metalium/device_pool.hpp>
#include <tt-metalium/fabric_host_interface.h>
#include <algorithm>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include "fabric_fixture.hpp"
#include "t3k_mesh_descriptor_chip_mappings.hpp"
#include "utils.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include "hostdevcommon/kernel_structs.h"
#include <tt-metalium/kernel_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/mesh_graph.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include "impl/context/metal_context.hpp"
#include "test_common.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "umd/device/types/xy_pair.h"
#include <tt-metalium/utils.hpp>

namespace tt::tt_fabric {
namespace fabric_router_tests {

// hack to let topology.cpp to know the binary is a unit test
// https://github.com/tenstorrent/tt-metal/issues/20000
// TODO: delete this once tt_fabric_api.h fully support low latency feature
extern "C" bool isFabricUnitTest();
bool isFabricUnitTest() { return true; }

using tt::tt_metal::ShardedBufferConfig;
using tt::tt_metal::ShardOrientation;
using tt::tt_metal::ShardSpecBuffer;

void ValidateBuffer(const std::vector<uint32_t>& expected_data, std::shared_ptr<tt_metal::Buffer>& buffer) {
    std::vector<uint32_t> actual_data;
    tt::tt_metal::detail::ReadFromBuffer(buffer, actual_data);
    EXPECT_EQ(expected_data, actual_data);
}

void ValidateBuffer(const uint32_t& expected_data, std::shared_ptr<tt_metal::Buffer>& buffer) {
    std::vector<uint32_t> actual_data;
    tt::tt_metal::detail::ReadFromBuffer(buffer, actual_data);
    EXPECT_EQ(expected_data, actual_data[0]);
}

void CreateSenderKernel(
    tt::tt_metal::Program& sender_program,
    const std::string& sender_kernel_name,
    std::vector<uint32_t>&& sender_compile_time_args,
    const CoreCoord& sender_logical_core,
    const std::map<std::string, std::string>& defines,
    std::vector<uint32_t>&& sender_runtime_args) {
    // Allocate space for the client interface
    uint32_t client_interface_cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig client_interface_cb_config =
        tt::tt_metal::CircularBufferConfig(
            tt::tt_fabric::CLIENT_INTERFACE_SIZE, {{client_interface_cb_index, DataFormat::UInt32}})
            .set_page_size(client_interface_cb_index, tt::tt_fabric::CLIENT_INTERFACE_SIZE);
    auto client_interface_cb =
        tt::tt_metal::CreateCircularBuffer(sender_program, sender_logical_core, client_interface_cb_config);

    sender_compile_time_args.insert(sender_compile_time_args.begin(), client_interface_cb_index);
    auto sender_kernel = tt_metal::CreateKernel(
        sender_program,
        sender_kernel_name,
        {sender_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = std::move(sender_compile_time_args),
            .defines = defines});

    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_core, std::move(sender_runtime_args));
}

void CreateReceiverKernel(
    tt::tt_metal::Program& receiver_program,
    const CoreCoord& receiver_logical_core,
    const std::map<std::string, std::string>& defines,
    const uint32_t address,
    const uint32_t data_size) {
    auto receiver_kernel = tt_metal::CreateKernel(
        receiver_program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_receiver.cpp",
        {receiver_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .defines = defines});

    std::vector<uint32_t> receiver_runtime_args = {
        address,
        data_size,
    };
    tt_metal::SetRuntimeArgs(receiver_program, receiver_kernel, receiver_logical_core, receiver_runtime_args);
}

std::shared_ptr<tt_metal::Buffer> PrepareBuffer(
    tt::tt_metal::IDevice* device, uint32_t size, CoreRangeSet& logical_crs, const std::vector<uint32_t>& fill_data) {
    auto shard_parameters = ShardSpecBuffer(logical_crs, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
    ShardedBufferConfig shard_config = {
        .device = device,
        .size = size,
        .page_size = size,
        .buffer_type = tt_metal::BufferType::L1,
        .buffer_layout = tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = std::move(shard_parameters),
    };
    auto buffer = CreateBuffer(shard_config);
    tt::tt_metal::detail::WriteToBuffer(buffer, fill_data);
    return buffer;
}

void RunAsyncWriteTest(
    BaseFabricFixture* fixture, fabric_mode mode, bool is_raw_write, RoutingDirection direction = RoutingDirection::E) {
    CoreCoord sender_logical_core = {0, 0};
    CoreRangeSet sender_logical_crs = {sender_logical_core};
    CoreCoord receiver_logical_core = {1, 0};
    CoreRangeSet receiver_logical_crs = {receiver_logical_core};
    FabricNodeId start_fabric_node_id(MeshId{0}, 0);
    chip_id_t physical_start_device_id;
    FabricNodeId end_fabric_node_id(MeshId{0}, 0);
    chip_id_t physical_end_device_id;

    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    // Find a device with a neighbour in the specified direction
    if (!find_device_with_neighbor_in_direction(
            fixture,
            start_fabric_node_id,
            end_fabric_node_id,
            physical_start_device_id,
            physical_end_device_id,
            direction)) {
        GTEST_SKIP() << "No path found between sender and receivers";
    }

    std::string test_type = is_raw_write ? "Raw Async Write" : "Async Write";
    log_info(tt::LogTest, "{} from {} to {}", test_type, start_fabric_node_id.chip_id, end_fabric_node_id.chip_id);

    // Get the optimal channels (no internal hops) on the start chip that will forward in the direction of the end chip
    auto router_chans = control_plane.get_forwarding_eth_chans_to_chip(start_fabric_node_id, end_fabric_node_id);

    auto* sender_device = DevicePool::instance().get_active_device(physical_start_device_id);
    auto* receiver_device = DevicePool::instance().get_active_device(physical_end_device_id);
    CoreCoord sender_virtual_core = sender_device->worker_core_from_logical_core(sender_logical_core);
    CoreCoord receiver_virtual_core = receiver_device->worker_core_from_logical_core(receiver_logical_core);

    uint32_t data_size = tt::constants::TILE_HW * sizeof(uint32_t);
    std::vector<uint32_t> receiver_buffer_data(data_size / sizeof(uint32_t), 0);
    auto receiver_buffer = PrepareBuffer(receiver_device, data_size, receiver_logical_crs, receiver_buffer_data);

    // Prepare sender buffer based on whether it's raw write or not
    std::vector<uint32_t> sender_buffer_data;
    std::shared_ptr<tt_metal::Buffer> sender_buffer;

    if (is_raw_write) {
        // For raw write, we don't need packet header
        sender_buffer_data.resize(data_size / sizeof(uint32_t), 0);
        std::iota(sender_buffer_data.begin(), sender_buffer_data.end(), 0);
        sender_buffer = PrepareBuffer(sender_device, data_size, sender_logical_crs, sender_buffer_data);
    } else {
        // Packet header needs to be inlined with the data being sent
        uint32_t sender_packet_header_and_data_size = tt::tt_fabric::PACKET_HEADER_SIZE_BYTES + data_size;
        sender_buffer_data.resize(sender_packet_header_and_data_size / sizeof(uint32_t), 0);
        std::iota(
            sender_buffer_data.begin() + PACKET_HEADER_SIZE_BYTES / sizeof(uint32_t), sender_buffer_data.end(), 0);
        sender_buffer =
            PrepareBuffer(sender_device, sender_packet_header_and_data_size, sender_logical_crs, sender_buffer_data);

        // Extract the expected data to be read from the receiver
        std::copy(
            sender_buffer_data.begin() + tt::tt_fabric::PACKET_HEADER_SIZE_BYTES / sizeof(uint32_t),
            sender_buffer_data.end(),
            receiver_buffer_data.begin());
    }

    // Wait for buffer data to be written to device
    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(physical_end_device_id);
    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(physical_start_device_id);

    auto receiver_noc_encoding =
        tt::tt_metal::MetalContext::instance().hal().noc_xy_encoding(receiver_virtual_core.x, receiver_virtual_core.y);

    // Create the sender program
    std::vector<uint32_t> sender_compile_time_args = {
        (uint32_t)mode, (uint32_t)test_mode::TEST_ASYNC_WRITE, (uint32_t)is_raw_write};
    auto outbound_eth_channels = control_plane.get_active_fabric_eth_channels(start_fabric_node_id);
    auto router_virtual_core = tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_eth_core_from_channel(
        physical_start_device_id, *router_chans.begin());
    std::vector<uint32_t> sender_runtime_args = {
        sender_buffer->address(),
        receiver_noc_encoding,
        receiver_buffer->address(),
        data_size,
        *end_fabric_node_id.mesh_id,
        end_fabric_node_id.chip_id,
        tt_metal::MetalContext::instance().hal().noc_xy_encoding(router_virtual_core.x, router_virtual_core.y),
        outbound_eth_channels.begin()->first};
    std::map<std::string, std::string> defines = {};
    if (mode == fabric_mode::PULL) {
        defines["FVC_MODE_PULL"] = "";
    }
    defines["DISABLE_LOW_LATENCY_ROUTING"] = "";
    auto sender_program = tt_metal::CreateProgram();
    CreateSenderKernel(
        sender_program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_async_write_sender.cpp",
        std::move(sender_compile_time_args),
        sender_logical_core,
        defines,
        std::move(sender_runtime_args));

    // Create the receiver program for validation
    auto receiver_program = tt_metal::CreateProgram();
    CreateReceiverKernel(receiver_program, receiver_logical_core, defines, receiver_buffer->address(), data_size);

    // Launch sender and receiver programs and wait for them to finish
    fixture->RunProgramNonblocking(receiver_device, receiver_program);
    fixture->RunProgramNonblocking(sender_device, sender_program);
    fixture->WaitForSingleProgramDone(sender_device, sender_program);
    fixture->WaitForSingleProgramDone(receiver_device, receiver_program);

    // Validate the data received by the receiver
    if (is_raw_write) {
        ValidateBuffer(sender_buffer_data, receiver_buffer);
    } else {
        ValidateBuffer(receiver_buffer_data, receiver_buffer);
    }
}

void RunAtomicIncTest(BaseFabricFixture* fixture, fabric_mode mode) {
    CoreCoord sender_logical_core = {0, 0};
    CoreRangeSet sender_logical_crs = {sender_logical_core};
    CoreCoord receiver_logical_core = {1, 0};
    CoreRangeSet receiver_logical_crs = {receiver_logical_core};
    FabricNodeId start_fabric_node_id(MeshId{0}, 0);
    chip_id_t physical_start_device_id;
    FabricNodeId end_fabric_node_id(MeshId{0}, 0);
    chip_id_t physical_end_device_id;

    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    // Find a device with a neighbour in the East direction
    if (!find_device_with_neighbor_in_direction(
            fixture,
            start_fabric_node_id,
            end_fabric_node_id,
            physical_start_device_id,
            physical_end_device_id,
            RoutingDirection::E)) {
        GTEST_SKIP() << "No path found between sender and receivers";
    }

    // Get the optimal channels (no internal hops) on the start chip that will forward in the direction of the end chip
    auto router_chans = control_plane.get_forwarding_eth_chans_to_chip(start_fabric_node_id, end_fabric_node_id);

    auto* sender_device = DevicePool::instance().get_active_device(physical_start_device_id);
    auto* receiver_device = DevicePool::instance().get_active_device(physical_end_device_id);
    CoreCoord sender_virtual_core = sender_device->worker_core_from_logical_core(sender_logical_core);
    CoreCoord receiver_virtual_core = receiver_device->worker_core_from_logical_core(receiver_logical_core);

    uint32_t data_size = sizeof(uint32_t);

    std::vector<uint32_t> receiver_buffer_data(data_size / sizeof(uint32_t), 0);
    auto receiver_buffer = PrepareBuffer(receiver_device, data_size, receiver_logical_crs, receiver_buffer_data);

    // Packet header needs to be inlined with the data being sent, so this test just allocates buffer space for both
    // together on the sender
    uint32_t sender_packet_header_and_data_size = tt::tt_fabric::PACKET_HEADER_SIZE_BYTES;
    std::vector<uint32_t> sender_buffer_data(sender_packet_header_and_data_size / sizeof(uint32_t), 0);
    auto sender_buffer =
        PrepareBuffer(sender_device, sender_packet_header_and_data_size, sender_logical_crs, sender_buffer_data);

    uint32_t atomic_inc = 5;
    uint32_t wrap_boundary = 31;

    // Extract the expected data to be read from the receiver
    receiver_buffer_data[0] = atomic_inc;

    // Wait for buffer data to be written to device
    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(physical_end_device_id);
    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(physical_start_device_id);

    auto receiver_noc_encoding =
        tt::tt_metal::MetalContext::instance().hal().noc_xy_encoding(receiver_virtual_core.x, receiver_virtual_core.y);

    // Create the sender program
    auto sender_program = tt_metal::CreateProgram();
    std::map<std::string, std::string> defines = {};
    if (mode == fabric_mode::PULL) {
        defines["FVC_MODE_PULL"] = "";
    }
    defines["DISABLE_LOW_LATENCY_ROUTING"] = "";
    std::vector<uint32_t> sender_compile_time_args = {(uint32_t)mode, (uint32_t)TEST_ATOMIC_INC, 0};
    auto outbound_eth_channels = control_plane.get_active_fabric_eth_channels(start_fabric_node_id);
    auto router_virtual_core = tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_eth_core_from_channel(
        physical_start_device_id, *router_chans.begin());
    std::vector<uint32_t> sender_runtime_args = {
        sender_buffer->address(),
        receiver_noc_encoding,
        receiver_buffer->address(),
        atomic_inc,
        wrap_boundary,
        *end_fabric_node_id.mesh_id,
        end_fabric_node_id.chip_id,
        tt_metal::MetalContext::instance().hal().noc_xy_encoding(router_virtual_core.x, router_virtual_core.y),
        outbound_eth_channels.begin()->first};

    CreateSenderKernel(
        sender_program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_atomic_inc_sender.cpp",
        std::move(sender_compile_time_args),
        sender_logical_core,
        defines,
        std::move(sender_runtime_args));

    // Create the receiver program for validation
    auto receiver_program = tt_metal::CreateProgram();
    CreateReceiverKernel(receiver_program, receiver_logical_core, defines, receiver_buffer->address(), data_size);

    // Launch sender and receiver programs and wait for them to finish
    fixture->RunProgramNonblocking(receiver_device, receiver_program);
    fixture->RunProgramNonblocking(sender_device, sender_program);
    fixture->WaitForSingleProgramDone(sender_device, sender_program);
    fixture->WaitForSingleProgramDone(receiver_device, receiver_program);

    // Validate the data received by the receiver
    ValidateBuffer(receiver_buffer_data, receiver_buffer);
}

void RunAsyncWriteAtomicIncTest(BaseFabricFixture* fixture, fabric_mode mode, bool is_raw_write) {
    CoreCoord sender_logical_core = {0, 0};
    CoreRangeSet sender_logical_crs = {sender_logical_core};
    CoreCoord receiver_logical_core = {1, 0};
    CoreRangeSet receiver_logical_crs = {receiver_logical_core};
    FabricNodeId start_fabric_node_id(MeshId{0}, 0);
    chip_id_t physical_start_device_id;
    FabricNodeId end_fabric_node_id(MeshId{0}, 0);
    chip_id_t physical_end_device_id;

    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    // Find a device with a neighbour in the East direction
    if (!find_device_with_neighbor_in_direction(
            fixture,
            start_fabric_node_id,
            end_fabric_node_id,
            physical_start_device_id,
            physical_end_device_id,
            RoutingDirection::E)) {
        GTEST_SKIP() << "No path found between sender and receivers";
    }

    // Get the optimal channels (no internal hops) on the start chip that will forward in the direction of the end chip
    auto router_chans = control_plane.get_forwarding_eth_chans_to_chip(start_fabric_node_id, end_fabric_node_id);

    auto* sender_device = DevicePool::instance().get_active_device(physical_start_device_id);
    auto* receiver_device = DevicePool::instance().get_active_device(physical_end_device_id);
    CoreCoord sender_virtual_core = sender_device->worker_core_from_logical_core(sender_logical_core);
    CoreCoord receiver_virtual_core = receiver_device->worker_core_from_logical_core(receiver_logical_core);

    uint32_t data_size = tt::constants::TILE_HW * sizeof(uint32_t);
    uint32_t atomic_inc_size = sizeof(uint32_t);

    std::vector<uint32_t> receiver_buffer_data(atomic_inc_size / sizeof(uint32_t), 0);
    auto receiver_atomic_buffer =
        PrepareBuffer(receiver_device, atomic_inc_size, receiver_logical_crs, receiver_buffer_data);
    receiver_buffer_data.resize(data_size / sizeof(uint32_t), 0);
    auto receiver_buffer = PrepareBuffer(receiver_device, data_size, receiver_logical_crs, receiver_buffer_data);

    // Prepare sender buffer based on whether it's raw write or not
    std::vector<uint32_t> sender_buffer_data;
    std::shared_ptr<tt_metal::Buffer> sender_buffer;

    if (is_raw_write) {
        // For raw write, we don't need packet header
        sender_buffer_data.resize(data_size / sizeof(uint32_t), 0);
        std::iota(sender_buffer_data.begin(), sender_buffer_data.end(), 0);
        sender_buffer = PrepareBuffer(sender_device, data_size, sender_logical_crs, sender_buffer_data);
    } else {
        // Packet header needs to be inlined with the data being sent
        uint32_t sender_packet_header_and_data_size = tt::tt_fabric::PACKET_HEADER_SIZE_BYTES + data_size;
        sender_buffer_data.resize(sender_packet_header_and_data_size / sizeof(uint32_t), 0);
        std::iota(
            sender_buffer_data.begin() + PACKET_HEADER_SIZE_BYTES / sizeof(uint32_t), sender_buffer_data.end(), 0);
        sender_buffer =
            PrepareBuffer(sender_device, sender_packet_header_and_data_size, sender_logical_crs, sender_buffer_data);

        // Extract the expected data to be read from the receiver
        std::copy(
            sender_buffer_data.begin() + tt::tt_fabric::PACKET_HEADER_SIZE_BYTES / sizeof(uint32_t),
            sender_buffer_data.end(),
            receiver_buffer_data.begin());
    }

    uint32_t atomic_inc = 5;

    // Wait for buffer data to be written to device
    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(physical_end_device_id);
    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(physical_start_device_id);

    auto receiver_noc_encoding =
        tt::tt_metal::MetalContext::instance().hal().noc_xy_encoding(receiver_virtual_core.x, receiver_virtual_core.y);

    // Create the sender program
    auto sender_program = tt_metal::CreateProgram();
    std::map<std::string, std::string> defines = {};
    if (mode == fabric_mode::PULL) {
        defines["FVC_MODE_PULL"] = "";
    }
    defines["DISABLE_LOW_LATENCY_ROUTING"] = "";
    std::vector<uint32_t> sender_compile_time_args = {
        (uint32_t)mode, (uint32_t)TEST_ASYNC_WRITE_ATOMIC_INC, (uint32_t)is_raw_write};
    auto outbound_eth_channels = control_plane.get_active_fabric_eth_channels(start_fabric_node_id);
    auto router_virtual_core = tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_eth_core_from_channel(
        physical_start_device_id, *router_chans.begin());
    std::vector<uint32_t> sender_runtime_args = {
        sender_buffer->address(),
        receiver_noc_encoding,
        receiver_buffer->address(),
        receiver_atomic_buffer->address(),
        data_size,
        atomic_inc,
        *end_fabric_node_id.mesh_id,
        end_fabric_node_id.chip_id,
        tt_metal::MetalContext::instance().hal().noc_xy_encoding(router_virtual_core.x, router_virtual_core.y),
        outbound_eth_channels.begin()->first};

    CreateSenderKernel(
        sender_program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_async_write_atomic_inc_sender.cpp",
        std::move(sender_compile_time_args),
        sender_logical_core,
        defines,
        std::move(sender_runtime_args));

    // Create the receiver program for validation
    auto receiver_program = tt_metal::CreateProgram();
    CreateReceiverKernel(receiver_program, receiver_logical_core, defines, receiver_buffer->address(), data_size);

    // Launch sender and receiver programs and wait for them to finish
    fixture->RunProgramNonblocking(receiver_device, receiver_program);
    fixture->RunProgramNonblocking(sender_device, sender_program);
    fixture->WaitForSingleProgramDone(sender_device, sender_program);
    fixture->WaitForSingleProgramDone(receiver_device, receiver_program);

    // Validate the data received by the receiver
    if (is_raw_write) {
        ValidateBuffer(sender_buffer_data, receiver_buffer);
    } else {
        ValidateBuffer(receiver_buffer_data, receiver_buffer);
    }
    ValidateBuffer(atomic_inc, receiver_atomic_buffer);
}

void RunAsyncWriteMulticastTest(
    BaseFabricFixture* fixture, fabric_mode mode, bool is_raw_write, bool multidirectional = false) {
    CoreCoord sender_logical_core = {0, 0};
    CoreRangeSet sender_logical_crs = {sender_logical_core};
    CoreCoord receiver_logical_core = {1, 0};
    CoreRangeSet receiver_logical_crs = {receiver_logical_core};
    FabricNodeId start_fabric_node_id(MeshId{0}, 0);
    chip_id_t physical_start_device_id;
    std::unordered_map<RoutingDirection, std::vector<FabricNodeId>> end_fabric_node_ids_by_dir;
    std::unordered_map<RoutingDirection, std::vector<chip_id_t>> physical_end_device_ids_by_dir;

    // Configure directions and hops based on test type
    std::unordered_map<RoutingDirection, uint32_t> mcast_hops;
    if (multidirectional) {
        mcast_hops[RoutingDirection::E] = 1;
        mcast_hops[RoutingDirection::W] = 2;
    } else {
        mcast_hops[RoutingDirection::E] = 1;
    }

    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    // Find a device with enough neighbours in the specified directions
    if (!find_device_with_neighbor_in_multi_direction(
            fixture,
            start_fabric_node_id,
            end_fabric_node_ids_by_dir,
            physical_start_device_id,
            physical_end_device_ids_by_dir,
            mcast_hops)) {
        GTEST_SKIP() << "No path found between sender and receivers";
    }

    // Log test configuration
    std::string test_type = is_raw_write ? "Raw" : "";
    std::string direction_type = multidirectional ? "Multidirectional" : "";
    log_info(
        tt::LogTest,
        "Async {} Write Mcast {} from {} to {}",
        test_type,
        direction_type,
        start_fabric_node_id.chip_id,
        end_fabric_node_ids_by_dir);

    auto* sender_device = DevicePool::instance().get_active_device(physical_start_device_id);
    CoreCoord sender_virtual_core = sender_device->worker_core_from_logical_core(sender_logical_core);
    // Virtual coordinate space. All devices have the same logical to virtual mapping
    CoreCoord receiver_virtual_core = sender_device->worker_core_from_logical_core(receiver_logical_core);

    uint32_t data_size = tt::constants::TILE_HW * sizeof(uint32_t);

    // Reset buffer space for test validation
    std::vector<uint32_t> receiver_buffer_data(data_size / sizeof(uint32_t), 0);

    // Create receiver programs and buffers
    std::map<std::string, std::string> defines = {};
    if (mode == fabric_mode::PULL) {
        defines["FVC_MODE_PULL"] = "";
    }
    defines["DISABLE_LOW_LATENCY_ROUTING"] = "";
    std::vector<tt_metal::Program> receiver_programs;
    std::vector<std::shared_ptr<tt_metal::Buffer>> receiver_buffers;

    for (auto& [routing_direction, physical_end_device_ids] : physical_end_device_ids_by_dir) {
        for (auto physical_end_device_id : physical_end_device_ids) {
            auto* receiver_device = DevicePool::instance().get_active_device(physical_end_device_id);
            auto receiver_buffer =
                PrepareBuffer(receiver_device, data_size, receiver_logical_crs, receiver_buffer_data);
            tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(physical_end_device_id);
            // Create the receiver program for validation
            auto receiver_program = tt_metal::CreateProgram();
            CreateReceiverKernel(
                receiver_program, receiver_logical_core, defines, receiver_buffer->address(), data_size);
            fixture->RunProgramNonblocking(receiver_device, receiver_program);
            receiver_programs.push_back(std::move(receiver_program));
            receiver_buffers.push_back(std::move(receiver_buffer));
        }
    }

    // Ensure all receiver buffers are at the same address
    uint32_t receiver_buffer_addr = receiver_buffers[0]->address();
    for (const auto& receiver_buffer : receiver_buffers) {
        if (receiver_buffer_addr != receiver_buffer->address()) {
            GTEST_SKIP() << "Receiver buffers are not at the same address";
        }
    }

    // Prepare sender buffer based on whether it's raw write or not
    std::vector<uint32_t> sender_buffer_data;
    std::shared_ptr<tt_metal::Buffer> sender_buffer;

    if (is_raw_write) {
        // For raw write, we don't need packet header
        sender_buffer_data.resize(data_size / sizeof(uint32_t), 0);
        std::iota(sender_buffer_data.begin(), sender_buffer_data.end(), 0);
        sender_buffer = PrepareBuffer(sender_device, data_size, sender_logical_crs, sender_buffer_data);
    } else {
        // Packet header needs to be inlined with the data being sent
        uint32_t sender_packet_header_and_data_size = tt::tt_fabric::PACKET_HEADER_SIZE_BYTES + data_size;
        sender_buffer_data.resize(sender_packet_header_and_data_size / sizeof(uint32_t), 0);
        std::iota(
            sender_buffer_data.begin() + PACKET_HEADER_SIZE_BYTES / sizeof(uint32_t), sender_buffer_data.end(), 0);
        sender_buffer =
            PrepareBuffer(sender_device, sender_packet_header_and_data_size, sender_logical_crs, sender_buffer_data);

        // Extract the expected data to be read from the receiver
        std::copy(
            sender_buffer_data.begin() + tt::tt_fabric::PACKET_HEADER_SIZE_BYTES / sizeof(uint32_t),
            sender_buffer_data.end(),
            receiver_buffer_data.begin());
    }

    // Wait for buffer data to be written to device
    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(physical_start_device_id);

    auto receiver_noc_encoding =
        tt::tt_metal::MetalContext::instance().hal().noc_xy_encoding(receiver_virtual_core.x, receiver_virtual_core.y);

    // Create the sender program
    auto sender_program = tt_metal::CreateProgram();
    std::vector<uint32_t> sender_compile_time_args = {(uint32_t)is_raw_write, (uint32_t)mode};

    // Get router encodings for each direction
    std::unordered_map<RoutingDirection, uint32_t> sender_router_noc_xys;
    for (auto& [routing_direction, end_fabric_node_ids] : end_fabric_node_ids_by_dir) {
        auto router_chans =
            control_plane.get_forwarding_eth_chans_to_chip(start_fabric_node_id, end_fabric_node_ids[0]);
        const auto& sender_virtual_router_coord =
            tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_eth_core_from_channel(
                physical_start_device_id, *router_chans.begin());
        sender_router_noc_xys.try_emplace(
            routing_direction,
            tt_metal::MetalContext::instance().hal().noc_xy_encoding(
                sender_virtual_router_coord.x, sender_virtual_router_coord.y));
    }

    // Prepare runtime args based on whether it's multidirectional or not
    auto outbound_eth_channels = control_plane.get_active_fabric_eth_channels(start_fabric_node_id);
    std::vector<uint32_t> sender_runtime_args;

    if (multidirectional) {
        sender_runtime_args = {
            sender_buffer->address(),
            receiver_noc_encoding,
            receiver_buffer_addr,
            data_size,
            *end_fabric_node_ids_by_dir[RoutingDirection::E][0].mesh_id,
            end_fabric_node_ids_by_dir[RoutingDirection::E][0].chip_id,
            mcast_hops[RoutingDirection::E],
            sender_router_noc_xys[RoutingDirection::E],
            *end_fabric_node_ids_by_dir[RoutingDirection::W][0].mesh_id,
            end_fabric_node_ids_by_dir[RoutingDirection::W][0].chip_id,
            mcast_hops[RoutingDirection::W],
            sender_router_noc_xys[RoutingDirection::W],
            outbound_eth_channels.begin()->first};
    } else {
        auto routing_direction = RoutingDirection::E;
        sender_runtime_args = {
            sender_buffer->address(),
            receiver_noc_encoding,
            receiver_buffer_addr,
            data_size,
            *end_fabric_node_ids_by_dir[routing_direction][0].mesh_id,
            end_fabric_node_ids_by_dir[routing_direction][0].chip_id,
            mcast_hops[routing_direction],
            sender_router_noc_xys[routing_direction],
            outbound_eth_channels.begin()->first};
    }

    // Choose the appropriate kernel based on whether it's multidirectional or not
    std::string kernel_path;
    if (multidirectional) {
        kernel_path =
            "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/"
            "fabric_async_write_multicast_multidirectional_sender.cpp";
    } else {
        kernel_path =
            "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/"
            "fabric_async_write_multicast_sender.cpp";
    }

    CreateSenderKernel(
        sender_program,
        kernel_path,
        std::move(sender_compile_time_args),
        sender_logical_core,
        defines,
        std::move(sender_runtime_args));

    // Launch sender and receiver programs and wait for them to finish
    fixture->RunProgramNonblocking(sender_device, sender_program);
    fixture->WaitForSingleProgramDone(sender_device, sender_program);

    for (auto [routing_direction, physical_end_device_ids] : physical_end_device_ids_by_dir) {
        for (uint32_t i = 0; i < physical_end_device_ids.size(); i++) {
            auto* receiver_device = DevicePool::instance().get_active_device(physical_end_device_ids[i]);
            fixture->WaitForSingleProgramDone(receiver_device, receiver_programs[i]);
        }
    }

    // Validate the data received by the receiver
    for (auto [routing_direction, physical_end_device_ids] : physical_end_device_ids_by_dir) {
        for (uint32_t i = 0; i < physical_end_device_ids.size(); i++) {
            if (is_raw_write) {
                ValidateBuffer(sender_buffer_data, receiver_buffers[i]);
            } else {
                ValidateBuffer(receiver_buffer_data, receiver_buffers[i]);
            }
        }
    }
}

void RunGetNextHopRouterDirectionTest(BaseFabricFixture* fixture, bool is_multi_mesh = false) {
    CoreCoord logical_core = {0, 0};

    auto devices = DevicePool::instance().get_all_active_devices();
    const size_t NUM_DEVICES = devices.size();
    bool invalid_test_scenario = !is_multi_mesh && NUM_DEVICES < 2;
    if (invalid_test_scenario) {
        GTEST_SKIP() << "Test requires at least 2 devices, found " << NUM_DEVICES;
    }

    std::vector<tt::tt_metal::Program> programs(NUM_DEVICES);
    std::vector<std::shared_ptr<tt_metal::Buffer>> result_buffers(NUM_DEVICES);
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    for (size_t src_idx = 0; src_idx < NUM_DEVICES; src_idx++) {
        auto* src_device = devices[src_idx];
        auto src_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(src_device->id());
        uint32_t src_fabric_chip_id = src_fabric_node_id.chip_id;

        uint32_t result_size = NUM_DEVICES * sizeof(uint32_t);
        std::vector<uint32_t> result_buffer_data(NUM_DEVICES, 0);
        CoreRangeSet core_range = {logical_core};
        result_buffers[src_idx] = PrepareBuffer(src_device, result_size, core_range, result_buffer_data);
        programs[src_idx] = tt::tt_metal::CreateProgram();

        uint32_t result_addr = result_buffers[src_idx]->address();
        std::vector<uint32_t> runtime_args = {
            *src_fabric_node_id.mesh_id,         // src_mesh_id
            src_fabric_chip_id,                  // src_chip_id
            result_addr,                         // result_addr
            static_cast<uint32_t>(NUM_DEVICES),  // num_devices
        };

        // Add mesh_id and chip_id pairs for all destinations
        for (size_t dst_idx = 0; dst_idx < NUM_DEVICES; dst_idx++) {
            auto dst_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(devices[dst_idx]->id());
            runtime_args.push_back(*dst_fabric_node_id.mesh_id);  // dst_mesh_id
            runtime_args.push_back(dst_fabric_node_id.chip_id);   // dst_chip_id
        }

        auto kernel = tt_metal::CreateKernel(
            programs[src_idx],
            "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/test_get_next_hop_router_direction.cpp",
            {logical_core},
            tt_metal::DataMovementConfig{});

        tt_metal::SetRuntimeArgs(programs[src_idx], kernel, logical_core, runtime_args);
    }

    for (size_t src_idx = 0; src_idx < NUM_DEVICES; src_idx++) {
        fixture->RunProgramNonblocking(devices[src_idx], programs[src_idx]);
    }
    for (size_t src_idx = 0; src_idx < NUM_DEVICES; src_idx++) {
        fixture->WaitForSingleProgramDone(devices[src_idx], programs[src_idx]);
    }

    for (size_t src_idx = 0; src_idx < NUM_DEVICES; src_idx++) {
        auto* src_device = devices[src_idx];
        auto src_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(src_device->id());
        uint32_t src_fabric_chip_id = src_fabric_node_id.chip_id;

        std::vector<uint32_t> result_data;
        tt::tt_metal::detail::ReadFromBuffer(result_buffers[src_idx], result_data);
        for (size_t dst_idx = 0; dst_idx < NUM_DEVICES; dst_idx++) {
            auto dst_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(devices[dst_idx]->id());
            uint32_t actual_direction = result_data[dst_idx];
            if (src_fabric_node_id == dst_fabric_node_id) {
                // Self-routing should return INVALID_DIRECTION
                EXPECT_EQ(actual_direction, (uint32_t)eth_chan_magic_values::INVALID_DIRECTION);
            } else {
                auto expected_direction = control_plane.get_forwarding_direction(src_fabric_node_id, dst_fabric_node_id)
                                              .value_or(RoutingDirection::NONE);

                if (expected_direction != RoutingDirection::NONE) {
                    // Route exists - should return valid direction
                    auto expected_eth_direction = control_plane.routing_direction_to_eth_direction(expected_direction);
                    EXPECT_EQ(actual_direction, expected_eth_direction);
                } else {
                    // No route exists - should return INVALID_DIRECTION
                    EXPECT_EQ(actual_direction, (uint32_t)eth_chan_magic_values::INVALID_DIRECTION);
                }
            }
        }
    }
}

TEST_F(Fabric2DFixture, DISABLED_TestAsyncWrite) { RunAsyncWriteTest(this, fabric_mode::PUSH, false); }

TEST_F(Fabric2DFixture, TestUnicastRaw) {
    for (uint32_t i = 0; i < 10; i++) {
        RunTestUnicastRaw(this);
    }
}

TEST_F(Fabric2DFixture, TestUnicastConnAPI) { RunTestUnicastConnAPI(this, 1); }

TEST_F(Fabric2DFixture, TestUnicastConnAPIDRAM) { RunTestUnicastConnAPI(this, 1, RoutingDirection::E, true); }

TEST_F(Fabric2DFixture, TestUnicastConnAPIRandom) {
    for (uint32_t i = 0; i < 10; i++) {
        RunTestUnicastConnAPIRandom(this);
    }
}

TEST_F(Fabric2DFixture, TestMCastConnAPI_1W1E) {
    RunTestMCastConnAPI(this, RoutingDirection::W, 1, RoutingDirection::E, 1);
}

TEST_F(Fabric2DFixture, TestMCastConnAPI_1W2E) {
    RunTestMCastConnAPI(this, RoutingDirection::W, 1, RoutingDirection::E, 2);
}

TEST_F(Fabric2DFixture, TestMCastConnAPI_2W1E) {
    RunTestMCastConnAPI(this, RoutingDirection::W, 2, RoutingDirection::E, 1);
}

TEST_F(Fabric2DFixture, TestMCastConnAPI_2W2E) {
    RunTestMCastConnAPI(this, RoutingDirection::W, 2, RoutingDirection::E, 2);
}

TEST_F(Fabric2DFixture, TestMCastConnAPI_3W3E) {
    RunTestMCastConnAPI(this, RoutingDirection::W, 3, RoutingDirection::E, 3);
}

TEST_F(Fabric2DFixture, TestMCastConnAPI_4W3E) {
    RunTestMCastConnAPI(this, RoutingDirection::W, 4, RoutingDirection::E, 3);
}

TEST_F(Fabric2DFixture, TestMCastConnAPI_3W4E) {
    RunTestMCastConnAPI(this, RoutingDirection::W, 3, RoutingDirection::E, 4);
}

TEST_F(Fabric2DFixture, TestMCastConnAPI_1N2S) {
    RunTestMCastConnAPI(this, RoutingDirection::N, 1, RoutingDirection::S, 2);
}

TEST_F(Fabric2DFixture, TestMCastConnAPI_2N1S) {
    RunTestMCastConnAPI(this, RoutingDirection::N, 2, RoutingDirection::S, 1);
}

TEST_F(Fabric2DFixture, Test2DMCastConnAPI_1S1E1W) { RunTest2DMCastConnAPI(this, RoutingDirection::S, 1, 1, 1); }

TEST_F(Fabric2DFixture, Test2DMCastConnAPI_1S2E1W) { RunTest2DMCastConnAPI(this, RoutingDirection::S, 1, 2, 1); }

TEST_F(Fabric2DFixture, Test2DMCastConnAPI_1S1E2W) { RunTest2DMCastConnAPI(this, RoutingDirection::S, 1, 1, 2); }

TEST_F(Fabric2DFixture, Test2DMCastConnAPI_2S1E1W) { RunTest2DMCastConnAPI(this, RoutingDirection::S, 2, 1, 1); }

TEST_F(Fabric2DFixture, Test2DMCastConnAPI_2S2E1W) { RunTest2DMCastConnAPI(this, RoutingDirection::S, 2, 2, 1); }

TEST_F(Fabric2DFixture, Test2DMCastConnAPI_2S1E2W) { RunTest2DMCastConnAPI(this, RoutingDirection::S, 2, 1, 2); }

TEST_F(NightlyFabric2DFixture, Test2DMCastConnAPI_2S1E6W) { RunTest2DMCastConnAPI(this, RoutingDirection::S, 2, 1, 6); }

TEST_F(NightlyFabric2DFixture, Test2DMCastConnAPI_2S6E1W) { RunTest2DMCastConnAPI(this, RoutingDirection::S, 2, 6, 1); }

TEST_F(NightlyFabric2DFixture, Test2DMCastConnAPI_3S1E6W) { RunTest2DMCastConnAPI(this, RoutingDirection::S, 3, 1, 6); }

TEST_F(NightlyFabric2DFixture, Test2DMCastConnAPI_3S6E1W) { RunTest2DMCastConnAPI(this, RoutingDirection::S, 3, 6, 1); }

TEST_F(Fabric2DFixture, Test2DMCastConnAPI_1N1E1W) { RunTest2DMCastConnAPI(this, RoutingDirection::N, 1, 1, 1); }

TEST_F(Fabric2DFixture, Test2DMCastConnAPI_1N2E1W) { RunTest2DMCastConnAPI(this, RoutingDirection::N, 1, 2, 1); }

TEST_F(Fabric2DFixture, Test2DMCastConnAPI_1N1E2W) { RunTest2DMCastConnAPI(this, RoutingDirection::N, 1, 1, 2); }

TEST_F(Fabric2DFixture, Test2DMCastConnAPI_2N1E1W) { RunTest2DMCastConnAPI(this, RoutingDirection::N, 2, 1, 1); }

TEST_F(Fabric2DFixture, Test2DMCastConnAPI_2N2E1W) { RunTest2DMCastConnAPI(this, RoutingDirection::N, 2, 2, 1); }

TEST_F(Fabric2DFixture, Test2DMCastConnAPI_2N1E2W) { RunTest2DMCastConnAPI(this, RoutingDirection::N, 2, 1, 2); }

TEST_F(NightlyFabric2DFixture, Test2DMCastConnAPI_2N1E6W) { RunTest2DMCastConnAPI(this, RoutingDirection::N, 2, 1, 6); }

TEST_F(NightlyFabric2DFixture, Test2DMCastConnAPI_2N6E1W) { RunTest2DMCastConnAPI(this, RoutingDirection::N, 2, 6, 1); }

TEST_F(NightlyFabric2DFixture, Test2DMCastConnAPI_3N1E6W) { RunTest2DMCastConnAPI(this, RoutingDirection::N, 3, 1, 6); }

TEST_F(NightlyFabric2DFixture, Test2DMCastConnAPI_3N6E1W) { RunTest2DMCastConnAPI(this, RoutingDirection::N, 3, 6, 1); }

TEST_F(Fabric2DFixture, DISABLED_TestAtomicInc) { RunAtomicIncTest(this, fabric_mode::PUSH); }

TEST_F(Fabric2DFixture, DISABLED_TestAsyncWriteAtomicInc) {
    RunAsyncWriteAtomicIncTest(this, fabric_mode::PUSH, false);
}

TEST_F(Fabric2DFixture, DISABLED_TestAsyncRawWriteAtomicInc) {
    RunAsyncWriteAtomicIncTest(this, fabric_mode::PUSH, true);
}

// 1D Routing Validation Test
TEST_F(Fabric1DFixture, TestGetNextHopRouterDirection1D) { RunGetNextHopRouterDirectionTest(this, false); }

// 2D Dynamic Routing Unicast Tests
TEST_F(Fabric2DDynamicFixture, TestUnicastRaw) {
    for (uint32_t i = 0; i < 10; i++) {
        RunTestUnicastRaw(this);
    }
}

// 2D Dynamic Routing Unicast Tests
TEST_P(T3kCustomMeshGraphFabric2DDynamicFixture, TestUnicastRaw) {
    auto [mesh_graph_desc_path, mesh_graph_eth_coords] = GetParam();
    CustomMeshGraphFabric2DDynamicFixture::SetUp(
        mesh_graph_desc_path, get_physical_chip_mapping_from_eth_coords_mapping(mesh_graph_eth_coords));
    for (uint32_t i = 0; i < 10; i++) {
        RunTestUnicastRaw(this);
    }
}

TEST_F(Fabric2DDynamicFixture, TestGetNextHopRouterDirection1MeshAllToAll) {
    if (tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() == tt::ClusterType::TG) {
        GTEST_SKIP() << "Test not applicable for TG cluster type";
    }
    RunGetNextHopRouterDirectionTest(this, false);
}

// Multi-Mesh Test - Using parameterized test with connected mesh descriptor
TEST_P(T3kCustomMeshGraphFabric2DDynamicFixture, TestGetNextHopRouterDirectionMultiMesh) {
    auto [mesh_graph_desc_path, mesh_graph_eth_coords] = GetParam();
    CustomMeshGraphFabric2DDynamicFixture::SetUp(
        mesh_graph_desc_path, get_physical_chip_mapping_from_eth_coords_mapping(mesh_graph_eth_coords));
    RunGetNextHopRouterDirectionTest(this, true);
}

TEST_P(T3kCustomMeshGraphFabric2DDynamicFixture, TestGetNextHopRouterDirectionDisjointMultiMesh) {
    auto [mesh_graph_desc_path, mesh_graph_eth_coords] = GetParam();
    CustomMeshGraphFabric2DDynamicFixture::SetUp(
        mesh_graph_desc_path, get_physical_chip_mapping_from_eth_coords_mapping(mesh_graph_eth_coords));
    RunGetNextHopRouterDirectionTest(this, true);
}

INSTANTIATE_TEST_SUITE_P(
    T3kDisjointMeshGraphFabric2DDynamicTests,
    T3kCustomMeshGraphFabric2DDynamicFixture,
    ::testing::ValuesIn(t3k_disjoint_mesh_descriptor_chip_mappings));

INSTANTIATE_TEST_SUITE_P(
    T3kCustomMeshGraphFabric2DDynamicTests,
    T3kCustomMeshGraphFabric2DDynamicFixture,
    ::testing::ValuesIn(t3k_mesh_descriptor_chip_mappings));

TEST_F(Fabric2DDynamicFixture, TestUnicastConnAPI) { RunTestUnicastConnAPI(this, 1); }

TEST_F(Fabric2DDynamicFixture, TestUnicastConnAPIDRAM) { RunTestUnicastConnAPI(this, 1, RoutingDirection::E, true); }

// 2D Dynamic Routing Unidirectional mcast tests (no turns)
TEST_F(Fabric2DDynamicFixture, TestLineMcastE2Hops) {
    auto routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 2};
    RunTestLineMcast(this, {routing_info});
}

TEST_F(Fabric2DDynamicFixture, TestLineMcastE3Hops) {
    auto routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 3};
    RunTestLineMcast(this, {routing_info});
}

TEST_F(NightlyFabric2DDynamicFixture, TestLineMcastE7Hops) {
    auto routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 7};
    RunTestLineMcast(this, {routing_info});
}

TEST_F(Fabric2DDynamicFixture, TestLineMcastW2Hops) {
    auto routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 2};
    RunTestLineMcast(this, {routing_info});
}

TEST_F(Fabric2DDynamicFixture, TestLineMcastW3Hops) {
    auto routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 3};
    RunTestLineMcast(this, {routing_info});
}

TEST_F(NightlyFabric2DDynamicFixture, TestLineMcastW7Hops) {
    auto routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 7};
    RunTestLineMcast(this, {routing_info});
}

// 2D Dynamic Routing Unidirectional mcast tests (with turns)
TEST_F(Fabric2DDynamicFixture, TestLineMcastN1HopE3Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 3};
    auto n_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::N, .num_mcast_hops = 1};
    RunTestLineMcast(this, {e_routing_info, n_routing_info});
}

TEST_F(NightlyFabric2DDynamicFixture, TestLineMcastN1HopE7Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 7};
    auto n_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::N, .num_mcast_hops = 1};
    RunTestLineMcast(this, {e_routing_info, n_routing_info});
}

TEST_F(Fabric2DDynamicFixture, TestLineMcastN2HopE3Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 3};
    auto n_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::N, .num_mcast_hops = 2};
    RunTestLineMcast(this, {e_routing_info, n_routing_info});
}

TEST_F(NightlyFabric2DDynamicFixture, TestLineMcastN2HopE7Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 7};
    auto n_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::N, .num_mcast_hops = 2};
    RunTestLineMcast(this, {e_routing_info, n_routing_info});
}

TEST_F(Fabric2DDynamicFixture, TestLineMcastS1HopE3Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 3};
    auto s_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::S, .num_mcast_hops = 1};
    RunTestLineMcast(this, {e_routing_info, s_routing_info});
}

TEST_F(NightlyFabric2DDynamicFixture, TestLineMcastS1HopE7Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 7};
    auto s_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::S, .num_mcast_hops = 1};
    RunTestLineMcast(this, {e_routing_info, s_routing_info});
}

TEST_F(Fabric2DDynamicFixture, TestLineMcastS2HopE3Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 3};
    auto s_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::S, .num_mcast_hops = 2};
    RunTestLineMcast(this, {e_routing_info, s_routing_info});
}

TEST_F(NightlyFabric2DDynamicFixture, TestLineMcastS2HopE7Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 7};
    auto s_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::S, .num_mcast_hops = 2};
    RunTestLineMcast(this, {e_routing_info, s_routing_info});
}

TEST_F(Fabric2DDynamicFixture, TestLineMcastN1HopW3Hops) {
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 3};
    auto n_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::N, .num_mcast_hops = 1};
    RunTestLineMcast(this, {w_routing_info, n_routing_info});
}

TEST_F(NightlyFabric2DDynamicFixture, TestLineMcastN1HopW7Hops) {
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 7};
    auto n_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::N, .num_mcast_hops = 1};
    RunTestLineMcast(this, {w_routing_info, n_routing_info});
}

TEST_F(Fabric2DDynamicFixture, TestLineMcastS1HopW3Hops) {
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 3};
    auto s_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::S, .num_mcast_hops = 1};
    RunTestLineMcast(this, {w_routing_info, s_routing_info});
}

TEST_F(NightlyFabric2DDynamicFixture, TestLineMcastS1HopW7Hops) {
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 7};
    auto s_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::S, .num_mcast_hops = 1};
    RunTestLineMcast(this, {w_routing_info, s_routing_info});
}

TEST_F(Fabric2DDynamicFixture, TestLineMcastS1HopE1HopW1Hop) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 1};
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 1};
    auto s_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::S, .num_mcast_hops = 1};
    RunTestLineMcast(this, {e_routing_info, w_routing_info, s_routing_info});
}

TEST_F(Fabric2DDynamicFixture, TestLineMcastN1HopE1HopW1Hop) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 1};
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 1};
    auto n_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::N, .num_mcast_hops = 1};
    RunTestLineMcast(this, {e_routing_info, w_routing_info, n_routing_info});
}

TEST_F(Fabric2DDynamicFixture, TestLineMcastS1HopE2HopsW1Hop) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 2};
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 1};
    auto s_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::S, .num_mcast_hops = 1};
    RunTestLineMcast(this, {e_routing_info, w_routing_info, s_routing_info});
}

TEST_F(Fabric2DDynamicFixture, TestLineMcastN1HopE2HopsW1Hop) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 2};
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 1};
    auto n_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::N, .num_mcast_hops = 1};
    RunTestLineMcast(this, {e_routing_info, w_routing_info, n_routing_info});
}

TEST_F(Fabric2DDynamicFixture, TestLineMcastS1HopE1HopW2Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 1};
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 2};
    auto s_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::S, .num_mcast_hops = 1};
    RunTestLineMcast(this, {e_routing_info, w_routing_info, s_routing_info});
}

TEST_F(Fabric2DDynamicFixture, TestLineMcastN1HopE1HopW2Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 1};
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 2};
    auto n_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::N, .num_mcast_hops = 1};
    RunTestLineMcast(this, {e_routing_info, w_routing_info, n_routing_info});
}

TEST_F(Fabric2DDynamicFixture, TestLineMcastS2HopsE4HopsW3Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 4};
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 3};
    auto s_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::S, .num_mcast_hops = 2};
    RunTestLineMcast(this, {e_routing_info, w_routing_info, s_routing_info});
}

TEST_F(NightlyFabric2DDynamicFixture, TestLineMcastS2HopsE3HopsW4Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 3};
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 4};
    auto s_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::S, .num_mcast_hops = 2};
    RunTestLineMcast(this, {e_routing_info, w_routing_info, s_routing_info});
}

TEST_F(NightlyFabric2DDynamicFixture, TestLineMcastS3HopsE4HopsW3Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 4};
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 3};
    auto s_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::S, .num_mcast_hops = 3};
    RunTestLineMcast(this, {e_routing_info, w_routing_info, s_routing_info});
}

TEST_F(NightlyFabric2DDynamicFixture, TestLineMcastS3HopsE3HopsW4Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 3};
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 4};
    auto s_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::S, .num_mcast_hops = 3};
    RunTestLineMcast(this, {e_routing_info, w_routing_info, s_routing_info});
}

TEST_F(NightlyFabric2DDynamicFixture, TestLineMcastN2HopsE4HopsW3Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 4};
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 3};
    auto n_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::N, .num_mcast_hops = 2};
    RunTestLineMcast(this, {e_routing_info, w_routing_info, n_routing_info});
}

TEST_F(NightlyFabric2DDynamicFixture, TestLineMcastN2HopsE3HopsW4Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 3};
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 4};
    auto n_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::N, .num_mcast_hops = 2};
    RunTestLineMcast(this, {e_routing_info, w_routing_info, n_routing_info});
}

TEST_F(NightlyFabric2DDynamicFixture, TestLineMcastN3HopsE4HopsW3Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 4};
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 3};
    auto n_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::N, .num_mcast_hops = 3};
    RunTestLineMcast(this, {e_routing_info, w_routing_info, n_routing_info});
}

TEST_F(NightlyFabric2DDynamicFixture, TestLineMcastN3HopsE3HopsW4Hops) {
    auto e_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 3};
    auto w_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 4};
    auto n_routing_info = McastRoutingInfo{.mcast_dir = RoutingDirection::N, .num_mcast_hops = 3};
    RunTestLineMcast(this, {e_routing_info, w_routing_info, n_routing_info});
}

}  // namespace fabric_router_tests
}  // namespace tt::tt_fabric
