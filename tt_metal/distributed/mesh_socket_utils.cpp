// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/mesh_socket_utils.hpp"
#include <tt-metalium/system_mesh.hpp>
#include "impl/context/metal_context.hpp"

#include "tt_metal/hw/inc/socket.h"

namespace tt::tt_metal::distributed {

std::shared_ptr<MeshBuffer> create_sender_socket_config_buffer(
    std::shared_ptr<MeshDevice> sender, const socket_config_t& config) {
    const auto& socket_connections = config.socket_connection_config;
    const auto& socket_mem_config = config.socket_mem_config;
    auto l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
    uint32_t sender_config_buffer_size = sizeof(sender_socket_md);
    std::set<CoreRange> all_sender_cores_set = {};
    for (const auto& connection : socket_connections) {
        all_sender_cores_set.insert(connection.sender_core.second);
    }
    auto all_sender_cores = CoreRangeSet(all_sender_cores_set);
    auto num_sender_cores = all_sender_cores_set.size();
    auto total_sender_config_buffer_size = num_sender_cores * sender_config_buffer_size;
    auto sender_config_shard_params =
        ShardSpecBuffer(all_sender_cores, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {num_sender_cores, 1});
    DeviceLocalBufferConfig sender_config_buffer_specs = {
        .page_size = sender_config_buffer_size,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = sender_config_shard_params,
        .bottom_up = std::nullopt,
        .sub_device_id = socket_mem_config.sender_sub_device,
    };

    // If sub device specified - we should assert that the socket is entirely contained in the sub device
    // i.e. all sender cores are in the sub device
    if (socket_mem_config.sender_sub_device.has_value()) {
        auto sender_sub_device_id = socket_mem_config.sender_sub_device.value();
        auto sender_sub_device_cores = sender->worker_cores(HalProgrammableCoreType::TENSIX, sender_sub_device_id);
        auto sender_sub_device_cores_set = CoreRangeSet(sender_sub_device_cores);
        TT_FATAL(
            sender_sub_device_cores_set.contains(all_sender_cores),
            "Socket sender cores must be contained in the specified sub device");
    }

    MeshBufferConfig sender_config_mesh_buffer_specs = ReplicatedBufferConfig{
        .size = total_sender_config_buffer_size,
    };
    return MeshBuffer::create(sender_config_mesh_buffer_specs, sender_config_buffer_specs, sender.get());
}

std::shared_ptr<MeshBuffer> create_receiver_socket_config_buffer(
    std::shared_ptr<MeshDevice> receiver, const socket_config_t& config) {
    const auto& socket_connections = config.socket_connection_config;
    const auto& socket_mem_config = config.socket_mem_config;
    auto l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
    uint32_t recv_config_buffer_size = sizeof(receiver_socket_md);
    std::set<CoreRange> all_recv_cores_set = {};
    for (const auto& connection : socket_connections) {
        all_recv_cores_set.insert(connection.receiver_core.second);
    }
    auto all_recv_cores = CoreRangeSet(all_recv_cores_set);
    auto num_recv_cores = all_recv_cores_set.size();
    auto total_recv_config_buffer_size = num_recv_cores * recv_config_buffer_size;

    auto recv_config_shard_params =
        ShardSpecBuffer(all_recv_cores, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {num_recv_cores, 1});

    DeviceLocalBufferConfig recv_config_buffer_specs = {
        .page_size = recv_config_buffer_size,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = recv_config_shard_params,
        .bottom_up = std::nullopt,
        .sub_device_id = socket_mem_config.receiver_sub_device,
    };

    if (socket_mem_config.receiver_sub_device.has_value()) {
        auto recv_sub_device_id = socket_mem_config.receiver_sub_device.value();
        auto recv_sub_device_cores = receiver->worker_cores(HalProgrammableCoreType::TENSIX, recv_sub_device_id);
        auto recv_sub_device_cores_set = CoreRangeSet(recv_sub_device_cores);
        TT_FATAL(
            recv_sub_device_cores_set.contains(all_recv_cores),
            "Socket sender cores must be contained in the specified sub device");
    }

    MeshBufferConfig recv_config_mesh_buffer_specs = ReplicatedBufferConfig{
        .size = total_recv_config_buffer_size,
    };

    return MeshBuffer::create(recv_config_mesh_buffer_specs, recv_config_buffer_specs, receiver.get());
}

std::shared_ptr<MeshBuffer> create_socket_data_buffer(
    std::shared_ptr<MeshDevice> receiver, const socket_config_t& config) {
    const auto& socket_mem_config = config.socket_mem_config;

    uint32_t num_data_cores = 0;
    CoreRangeSet shard_grid;
    if (socket_mem_config.socket_storage_type == BufferType::DRAM) {
        // Allocate DRAM Sharded Buffer
        shard_grid =
            CoreRange(CoreCoord(0, 0), CoreCoord(receiver->dram_grid_size().x - 1, receiver->dram_grid_size().y - 1));
        num_data_cores = receiver->dram_grid_size().x * receiver->dram_grid_size().y;
    } else {
        // Allocate Sharded buffer on worker cores
        auto receiver_sub_device_id = socket_mem_config.receiver_sub_device.value_or(SubDeviceId{0});
        shard_grid = receiver->worker_cores(HalProgrammableCoreType::TENSIX, receiver_sub_device_id);
        num_data_cores = receiver->num_worker_cores(HalProgrammableCoreType::TENSIX, receiver_sub_device_id);
    }
    // Allocate a shard of size fifo_size on each data core. User decides how these data-cores must be used
    const auto total_data_buffer_size = num_data_cores * socket_mem_config.fifo_size;
    DeviceLocalBufferConfig socket_data_buffer_specs = {
        .page_size = socket_mem_config.fifo_size,
        .buffer_type = socket_mem_config.socket_storage_type,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters =
            ShardSpecBuffer(shard_grid, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {num_data_cores, 1}),
        .bottom_up = std::nullopt,
        .sub_device_id = socket_mem_config.socket_storage_type == BufferType::DRAM
                             ? std::nullopt
                             : socket_mem_config.receiver_sub_device,
    };
    MeshBufferConfig socket_data_mesh_buffer_specs = ReplicatedBufferConfig{
        .size = total_data_buffer_size,
    };

    return MeshBuffer::create(socket_data_mesh_buffer_specs, socket_data_buffer_specs, receiver.get());
}

std::unordered_map<MeshCoordinate, std::vector<socket_connection_t>> group_socket_connections_by_sender(
    const socket_config_t& config) {
    std::unordered_map<MeshCoordinate, std::vector<socket_connection_t>> grouped_connections;
    for (const auto& connection : config.socket_connection_config) {
        grouped_connections[connection.sender_core.first].push_back(connection);
    }
    return grouped_connections;
}

std::unordered_map<MeshCoordinate, std::vector<socket_connection_t>> group_socket_connections_by_receiver(
    const socket_config_t& config) {
    std::unordered_map<MeshCoordinate, std::vector<socket_connection_t>> grouped_connections;
    for (const auto& connection : config.socket_connection_config) {
        grouped_connections[connection.receiver_core.first].push_back(connection);
    }
    return grouped_connections;
}

uint32_t get_sender_receiver_chip_fabric_encoding(
    MeshDevice* sender_device,
    MeshDevice* recv_device,
    const MeshCoordinate& sender_coord,
    const MeshCoordinate& recv_coord,
    FabricConfig fabric_config,
    bool get_sender_encoding) {
    const auto sender_physical_device_id = sender_device->get_device(sender_coord)->id();
    const auto recv_physical_device_id = recv_device->get_device(recv_coord)->id();
    if (fabric_config == FabricConfig::FABRIC_1D or fabric_config == FabricConfig::FABRIC_1D_RING) {
        // 1D Fabric requires passing in the number of hops between the sender and receiver
        auto sender_global_coord = SystemMesh::instance().get_global_device_coordinate(sender_physical_device_id);
        auto recv_global_coord = SystemMesh::instance().get_global_device_coordinate(recv_physical_device_id);

        if (fabric_config == FabricConfig::FABRIC_1D) {
            TT_FATAL(
                sender_global_coord[0] == recv_global_coord[0] || sender_global_coord[1] == recv_global_coord[1],
                "Sender and receiver chips must be in the same row or column when using 1D Line Fabric");
        }
        return std::abs(static_cast<int>(sender_global_coord[0]) - static_cast<int>(recv_global_coord[0])) +
               std::abs(static_cast<int>(sender_global_coord[1]) - static_cast<int>(recv_global_coord[1]));
    }
    // 2D fabric requires the physical chip id
    return get_sender_encoding ? sender_physical_device_id : recv_physical_device_id;
}

uint32_t get_physical_mesh_id(MeshDevice* mesh_device, const MeshCoordinate& coord) {
    auto physical_device_id = mesh_device->get_device(coord)->id();
    auto global_coord = SystemMesh::instance().get_global_device_coordinate(physical_device_id);
    return SystemMesh::instance().get_physical_mesh_id(global_coord);
}

void populate_sender_socket_config_buffer(
    std::shared_ptr<MeshBuffer> sender_config_buffer,
    std::shared_ptr<MeshBuffer> recv_config_buffer,
    std::shared_ptr<MeshBuffer> socket_data_buffer,
    const socket_config_t& config) {
    auto socket_connections_per_sender = group_socket_connections_by_sender(config);
    auto recv_bytes_sent_addr = recv_config_buffer->address();
    const auto& sender_core_to_core_id =
        sender_config_buffer->get_backing_buffer()->get_buffer_page_mapping()->core_to_core_id_;
    auto sender_mesh_device = sender_config_buffer->device();
    auto recv_mesh_device = recv_config_buffer->device();

    FabricConfig fabric_config = tt::tt_metal::MetalContext::instance().get_cluster().get_fabric_config();

    for (const auto& [device_coord, outgoing_connections] : socket_connections_per_sender) {
        std::vector<sender_socket_md> sender_config_buffer_data(
            sender_config_buffer->size() / sizeof(sender_socket_md), sender_socket_md());
        for (const auto& [sender_core, recv_core] : outgoing_connections) {
            auto downstream_chip_id = get_sender_receiver_chip_fabric_encoding(
                sender_mesh_device, recv_mesh_device, sender_core.first, recv_core.first, fabric_config, false);
            auto downstream_mesh_id = get_physical_mesh_id(recv_mesh_device, recv_core.first);
            TT_FATAL(
                sender_core.first == device_coord,
                "Internal Error: Sender cores in socket connection are incorrectly grouped.");
            auto recv_virtual_core = sender_mesh_device->worker_core_from_logical_core(recv_core.second);
            // Lookup which shard corresponding to the sender core. Socket config data for this sender will be
            // accordingly inserted on host
            uint32_t sender_idx = sender_core_to_core_id.at(sender_core.second);
            auto& curr_sender_md = sender_config_buffer_data[sender_idx];
            curr_sender_md.write_ptr = socket_data_buffer->address();
            curr_sender_md.downstream_mesh_id = downstream_mesh_id;
            curr_sender_md.downstream_chip_id = downstream_chip_id;
            curr_sender_md.downstream_noc_y = recv_virtual_core.y;
            curr_sender_md.downstream_noc_x = recv_virtual_core.x;
            curr_sender_md.downstream_bytes_sent_addr = recv_bytes_sent_addr;
            curr_sender_md.downstream_fifo_addr = socket_data_buffer->address();
            curr_sender_md.downstream_fifo_total_size = config.socket_mem_config.fifo_size;
            curr_sender_md.is_sender = 1;
        }
        distributed::WriteShard(
            sender_mesh_device->mesh_command_queue(0),
            sender_config_buffer,
            sender_config_buffer_data,
            device_coord,
            true);
    }
}

void populate_receiver_socket_config_buffer(
    std::shared_ptr<MeshBuffer> recv_config_buffer,
    std::shared_ptr<MeshBuffer> sender_config_buffer,
    std::shared_ptr<MeshBuffer> socket_data_buffer,
    const socket_config_t& config) {
    auto socket_connections_per_receiver = group_socket_connections_by_receiver(config);
    auto sender_bytes_acked_addr = sender_config_buffer->address();
    std::vector<receiver_socket_md> recv_config_buffer_data(
        recv_config_buffer->size() / sizeof(receiver_socket_md), receiver_socket_md());
    const auto& recv_core_to_core_id =
        recv_config_buffer->get_backing_buffer()->get_buffer_page_mapping()->core_to_core_id_;
    auto recv_mesh_device = recv_config_buffer->device();
    auto sender_mesh_device = sender_config_buffer->device();

    FabricConfig fabric_config = tt::tt_metal::MetalContext::instance().get_cluster().get_fabric_config();

    for (const auto& [device_coord, incoming_connections] : socket_connections_per_receiver) {
        for (const auto& [sender_core, recv_core] : incoming_connections) {
            auto upstream_chip_id = get_sender_receiver_chip_fabric_encoding(
                sender_mesh_device, recv_mesh_device, sender_core.first, recv_core.first, fabric_config, true);
            auto upstream_mesh_id = get_physical_mesh_id(sender_mesh_device, sender_core.first);

            TT_FATAL(
                recv_core.first == device_coord,
                "Internal Error: Receiver cores in socket connection are incorrectly grouped.");
            auto sender_virtual_core = recv_mesh_device->worker_core_from_logical_core(sender_core.second);
            // Lookup which shard corresponding to the receiver core. Socket config data for this receiver will be
            // accordingly inserted on host
            uint32_t recv_idx = recv_core_to_core_id.at(recv_core.second);
            auto& curr_recv_md = recv_config_buffer_data[recv_idx];
            curr_recv_md.read_ptr = socket_data_buffer->address();
            curr_recv_md.fifo_addr = socket_data_buffer->address();
            curr_recv_md.fifo_total_size = config.socket_mem_config.fifo_size;
            curr_recv_md.upstream_mesh_id = upstream_mesh_id;
            curr_recv_md.upstream_chip_id = upstream_chip_id;
            curr_recv_md.upstream_noc_y = sender_virtual_core.y;
            curr_recv_md.upstream_noc_x = sender_virtual_core.x;
            curr_recv_md.upstream_bytes_acked_addr = sender_bytes_acked_addr;
            curr_recv_md.is_sender = 0;
        }
        distributed::WriteShard(
            recv_mesh_device->mesh_command_queue(0), recv_config_buffer, recv_config_buffer_data, device_coord, true);
    }
}

}  // namespace tt::tt_metal::distributed
