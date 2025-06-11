// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/mesh_socket_utils.hpp"
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/system_mesh.hpp>
#include "impl/context/metal_context.hpp"

#include "tt_metal/hw/inc/socket.h"

namespace tt::tt_metal::distributed {

namespace {

std::unordered_map<MeshCoordinate, std::vector<SocketConnection>> group_socket_connections(
    const SocketConfig& config, SocketEndpoint socket_endpoint) {
    bool is_sender = socket_endpoint == SocketEndpoint::SENDER;
    std::unordered_map<MeshCoordinate, std::vector<SocketConnection>> grouped_connections;
    for (const auto& connection : config.socket_connection_config) {
        grouped_connections[is_sender ? connection.sender_core.device_coord : connection.receiver_core.device_coord]
            .push_back(connection);
    }
    return grouped_connections;
}

void validate_fabric_config_for_sockets(
    FabricConfig fabric_config, chip_id_t sender_physical_device_id, chip_id_t recv_physical_device_id) {
    if (sender_physical_device_id != recv_physical_device_id) {
        TT_FATAL(fabric_config != FabricConfig::DISABLED, "Can only create multi-device sockets with fabric enabled.");
    }

    static const std::unordered_set<FabricConfig> supported_fabrics = {
        FabricConfig::FABRIC_1D,
        FabricConfig::FABRIC_1D_RING,
        FabricConfig::FABRIC_2D_DYNAMIC,
        FabricConfig::DISABLED  // Fabric can be disabled as long as socket endpoints are on the same physical device
    };

    bool fabric_config_supported = supported_fabrics.count(fabric_config) > 0;
    TT_FATAL(fabric_config_supported, "Unsupported Fabric Config for Sockets specified {}", fabric_config);
}

uint32_t get_sender_receiver_chip_fabric_encoding(
    MeshDevice* sender_device,
    MeshDevice* recv_device,
    const MeshCoordinate& sender_coord,
    const MeshCoordinate& recv_coord,
    FabricConfig fabric_config,
    SocketEndpoint socket_endpoint) {
    const auto sender_physical_device_id = sender_device->get_device(sender_coord)->id();
    const auto recv_physical_device_id = recv_device->get_device(recv_coord)->id();
    bool is_sender = socket_endpoint == SocketEndpoint::SENDER;

    validate_fabric_config_for_sockets(fabric_config, sender_physical_device_id, recv_physical_device_id);

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
    } else {
        // 2D/Mesh Fabric requires looking up "logical" encodings from the control plane
        auto& control_plane= tt::tt_metal::MetalContext::instance().get_control_plane();
        if (is_sender) {
            return control_plane.get_fabric_node_id_from_physical_chip_id(recv_physical_device_id).chip_id;
        } else {
            return control_plane.get_fabric_node_id_from_physical_chip_id(sender_physical_device_id).chip_id;
        }
    }
}

}  // namespace

std::shared_ptr<MeshBuffer> create_socket_config_buffer(
    const std::shared_ptr<MeshDevice>& device, const SocketConfig& config, SocketEndpoint socket_endpoint) {
    const auto& socket_connections = config.socket_connection_config;
    const auto& socket_mem_config = config.socket_mem_config;
    auto l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
    bool is_sender = socket_endpoint == SocketEndpoint::SENDER;

    uint32_t config_buffer_size = is_sender ? sizeof(sender_socket_md) : sizeof(receiver_socket_md);
    std::set<CoreRange> all_cores_set;
    std::unordered_map<MeshCoordinate, std::set<CoreRange>> socket_cores_per_device;
    for (const auto& connection : socket_connections) {
        const auto& socket_device =
            is_sender ? connection.sender_core.device_coord : connection.receiver_core.device_coord;
        const auto& socket_core = is_sender ? connection.sender_core.core_coord : connection.receiver_core.core_coord;
        TT_FATAL(
            socket_cores_per_device[socket_device].insert(socket_core).second,
            "Cannot reuse sender or receiver cores in a single socket.");
        all_cores_set.insert(socket_core);
    }

    auto all_cores = CoreRangeSet(all_cores_set);
    auto num_cores = all_cores_set.size();
    auto total_config_buffer_size = num_cores * config_buffer_size;

    auto shard_params = ShardSpecBuffer(all_cores, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {num_cores, 1});

    DeviceLocalBufferConfig buffer_specs = {
        .page_size = config_buffer_size,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = shard_params,
        .bottom_up = std::nullopt,
        .sub_device_id = is_sender ? socket_mem_config.sender_sub_device : socket_mem_config.receiver_sub_device,
    };

    if (buffer_specs.sub_device_id.has_value()) {
        auto sub_device_id = buffer_specs.sub_device_id.value();
        auto device_cores = device->worker_cores(HalProgrammableCoreType::TENSIX, sub_device_id);
        auto device_cores_set = CoreRangeSet(device_cores);
        TT_FATAL(device_cores_set.contains(all_cores), "Socket cores must be contained in the specified sub device");
    }

    MeshBufferConfig mesh_buffer_specs = ReplicatedBufferConfig{
        .size = total_config_buffer_size,
    };

    return MeshBuffer::create(mesh_buffer_specs, buffer_specs, device.get());
}

std::shared_ptr<MeshBuffer> create_socket_data_buffer(
    const std::shared_ptr<MeshDevice>& receiver, const SocketConfig& config) {
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

void write_socket_configs(
    const std::shared_ptr<MeshBuffer>& config_buffer,
    const std::shared_ptr<MeshBuffer>& peer_config_buffer,
    const std::shared_ptr<MeshBuffer>& socket_data_buffer,
    const SocketConfig& config,
    SocketEndpoint socket_endpoint) {
    auto mesh_device = config_buffer->device();
    auto peer_device = peer_config_buffer->device();
    auto& core_to_core_id = config_buffer->get_backing_buffer()->get_buffer_page_mapping()->core_to_core_id;
    bool is_sender = socket_endpoint == SocketEndpoint::SENDER;

    auto grouped_connections = group_socket_connections(config, socket_endpoint);
    auto peer_addr = peer_config_buffer->address();

    FabricConfig fabric_config = tt::tt_metal::MetalContext::instance().get_fabric_config();

    if (is_sender) {
        std::vector<sender_socket_md> config_data(config_buffer->size() / sizeof(sender_socket_md), sender_socket_md());

        for (const auto& [device_coord, connections] : grouped_connections) {
            for (const auto& [sender_core, recv_core] : connections) {
                TT_FATAL(sender_core.device_coord == device_coord, "Internal Error: Sender cores incorrectly grouped.");
                auto downstream_chip_id = get_sender_receiver_chip_fabric_encoding(
                    mesh_device,
                    peer_device,
                    sender_core.device_coord,
                    recv_core.device_coord,
                    fabric_config,
                    SocketEndpoint::SENDER);
                auto downstream_mesh_id = get_physical_mesh_id(peer_device, recv_core.device_coord);
                auto recv_virtual_core = mesh_device->worker_core_from_logical_core(recv_core.core_coord);

                uint32_t idx = core_to_core_id.at(sender_core.core_coord);
                auto& md = config_data[idx];
                md.write_ptr = socket_data_buffer->address();
                md.downstream_fifo_addr = socket_data_buffer->address();
                md.downstream_fifo_total_size = config.socket_mem_config.fifo_size;
                md.downstream_mesh_id = downstream_mesh_id;
                md.downstream_chip_id = downstream_chip_id;
                md.downstream_noc_y = recv_virtual_core.y;
                md.downstream_noc_x = recv_virtual_core.x;
                md.downstream_bytes_sent_addr = peer_addr;
                md.is_sender = is_sender;
            }
            distributed::WriteShard(mesh_device->mesh_command_queue(0), config_buffer, config_data, device_coord, true);
        }
    } else {
        std::vector<receiver_socket_md> config_data(
            config_buffer->size() / sizeof(receiver_socket_md), receiver_socket_md());

        for (const auto& [device_coord, connections] : grouped_connections) {
            for (const auto& [sender_core, recv_core] : connections) {
                TT_FATAL(recv_core.device_coord == device_coord, "Internal Error: Receiver cores incorrectly grouped.");
                auto upstream_chip_id = get_sender_receiver_chip_fabric_encoding(
                    peer_device,
                    mesh_device,
                    sender_core.device_coord,
                    recv_core.device_coord,
                    fabric_config,
                    SocketEndpoint::RECEIVER);
                auto upstream_mesh_id = get_physical_mesh_id(peer_device, sender_core.device_coord);
                auto sender_virtual_core = mesh_device->worker_core_from_logical_core(sender_core.core_coord);

                uint32_t idx = core_to_core_id.at(recv_core.core_coord);
                auto& md = config_data[idx];
                md.read_ptr = socket_data_buffer->address();
                md.fifo_addr = socket_data_buffer->address();
                md.fifo_total_size = config.socket_mem_config.fifo_size;
                md.upstream_mesh_id = upstream_mesh_id;
                md.upstream_chip_id = upstream_chip_id;
                md.upstream_noc_y = sender_virtual_core.y;
                md.upstream_noc_x = sender_virtual_core.x;
                md.upstream_bytes_acked_addr = peer_addr;
                md.is_sender = is_sender;
            }
            distributed::WriteShard(mesh_device->mesh_command_queue(0), config_buffer, config_data, device_coord, true);
        }
    }
}

uint32_t get_physical_mesh_id(MeshDevice* mesh_device, const MeshCoordinate& coord) {
    auto physical_device_id = mesh_device->get_device(coord)->id();
    auto global_coord = SystemMesh::instance().get_global_device_coordinate(physical_device_id);
    return SystemMesh::instance().get_physical_mesh_id(global_coord);
}

}  // namespace tt::tt_metal::distributed
