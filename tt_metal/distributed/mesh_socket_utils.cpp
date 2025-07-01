// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/mesh_socket_utils.hpp"
#include "tt_metal/distributed/mesh_socket_serialization.hpp"
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/system_mesh.hpp>
#include "impl/context/metal_context.hpp"

#include "tt_metal/hw/inc/socket.h"

using namespace tt::tt_metal::distributed::multihost;

namespace tt::tt_metal::distributed {

namespace {

std::unordered_map<MeshCoordinate, std::vector<std::pair<uint32_t, SocketConnection>>> group_socket_connections(
    const SocketConfig& config, SocketEndpoint socket_endpoint) {
    bool is_sender = socket_endpoint == SocketEndpoint::SENDER;
    std::unordered_map<MeshCoordinate, std::vector<std::pair<uint32_t, SocketConnection>>> grouped_connections;
    uint32_t connection_index = 0;
    for (const auto& connection : config.socket_connection_config) {
        grouped_connections[is_sender ? connection.sender_core.device_coord : connection.receiver_core.device_coord]
            .push_back({connection_index, connection});
        connection_index++;
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
    uint32_t sender_device_id, uint32_t recv_device_id, FabricConfig fabric_config, SocketEndpoint socket_endpoint) {
    bool is_sender = socket_endpoint == SocketEndpoint::SENDER;

    validate_fabric_config_for_sockets(fabric_config, sender_device_id, recv_device_id);

    if (fabric_config == FabricConfig::FABRIC_1D or fabric_config == FabricConfig::FABRIC_1D_RING) {
        // 1D Fabric requires passing in the number of hops between the sender and receiver
        auto sender_global_coord = SystemMesh::instance().get_global_device_coordinate(sender_device_id);
        auto recv_global_coord = SystemMesh::instance().get_global_device_coordinate(recv_device_id);

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
            return control_plane.get_fabric_node_id_from_physical_chip_id(recv_device_id).chip_id;
        } else {
            return control_plane.get_fabric_node_id_from_physical_chip_id(sender_device_id).chip_id;
        }
    }
}

void validate_remote_desc(const SocketPeerDescriptor& local_desc, const SocketPeerDescriptor& remote_desc) {
    // Verify that socket connection config matches
    TT_FATAL(
        local_desc.config.socket_connection_config.size() == remote_desc.config.socket_connection_config.size(),
        "Mismatch in number of socket connections during handshake.");
    for (size_t i = 0; i < local_desc.config.socket_connection_config.size(); ++i) {
        const auto& local_conn = local_desc.config.socket_connection_config[i];
        const auto& remote_conn = remote_desc.config.socket_connection_config[i];
        TT_FATAL(local_conn.sender_core == remote_conn.sender_core, "Mismatch in sender core during handshake.");
        TT_FATAL(local_conn.receiver_core == remote_conn.receiver_core, "Mismatch in receiver core during handshake.");
    }
    // Verify that socket memory config matches
    TT_FATAL(
        local_desc.config.socket_mem_config.socket_storage_type ==
            remote_desc.config.socket_mem_config.socket_storage_type,
        "Mismatch in socket storage type during handshake.");
    TT_FATAL(
        local_desc.config.socket_mem_config.fifo_size == remote_desc.config.socket_mem_config.fifo_size,
        "Mismatch in socket FIFO size during handshake.");

    TT_FATAL(
        local_desc.config.socket_mem_config.sender_sub_device == remote_desc.config.socket_mem_config.sender_sub_device,
        "Mismatch in sender sub-device during handshake.");

    TT_FATAL(
        local_desc.config.socket_mem_config.receiver_sub_device ==
            remote_desc.config.socket_mem_config.receiver_sub_device,
        "Mismatch in receiver sub-device during handshake.");
    TT_FATAL(
        local_desc.config.socket_connection_config.size() == remote_desc.mesh_ids.size(),
        "Mismatch in number of mesh IDs during handshake.");
    TT_FATAL(
        local_desc.config.socket_connection_config.size() == remote_desc.chip_ids.size(),
        "Mismatch in number of chip IDs during handshake.");
}

uint32_t generate_descriptor_exchange_tag() {
    // Generate a unique id to tag the exchange of socket peer
    // descriptors between the sender and receiver.
    // This is used to ensure that the sender and receiver are
    // exchanging the correct descriptors.
    static uint32_t exchange_tag = 0;
    return ++exchange_tag;
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
        .sharding_args = BufferShardingArgs(shard_params, TensorMemoryLayout::HEIGHT_SHARDED),
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
        .sharding_args = BufferShardingArgs(
            ShardSpecBuffer(shard_grid, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {num_data_cores, 1}),
            TensorMemoryLayout::HEIGHT_SHARDED),
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
    const SocketPeerDescriptor& local_descriptor,
    const SocketPeerDescriptor& peer_descriptor,
    SocketEndpoint socket_endpoint) {
    auto mesh_device = config_buffer->device();
    auto& core_to_core_id = config_buffer->get_backing_buffer()->get_buffer_page_mapping()->core_to_core_id;
    bool is_sender = socket_endpoint == SocketEndpoint::SENDER;
    const auto& config = peer_descriptor.config;
    auto grouped_connections = group_socket_connections(config, socket_endpoint);
    auto peer_config_buf_addr = peer_descriptor.config_buffer_address;

    FabricConfig fabric_config = tt::tt_metal::MetalContext::instance().get_fabric_config();

    if (is_sender) {
        std::vector<sender_socket_md> config_data(config_buffer->size() / sizeof(sender_socket_md), sender_socket_md());

        for (const auto& [device_coord, indexed_connections] : grouped_connections) {
            for (const auto& [conn_idx, connection] : indexed_connections) {
                const auto& [sender_core, recv_core] = connection;
                TT_FATAL(sender_core.device_coord == device_coord, "Internal Error: Sender cores incorrectly grouped.");
                auto downstream_chip_id = get_sender_receiver_chip_fabric_encoding(
                    mesh_device->get_device(sender_core.device_coord)->id(),
                    peer_descriptor.chip_ids[conn_idx],
                    fabric_config,
                    SocketEndpoint::SENDER);
                auto downstream_mesh_id = peer_descriptor.mesh_ids[conn_idx];
                auto recv_virtual_core = mesh_device->worker_core_from_logical_core(recv_core.core_coord);

                uint32_t idx = core_to_core_id.at(sender_core.core_coord);
                auto& md = config_data[idx];
                md.write_ptr = peer_descriptor.data_buffer_address;
                md.downstream_fifo_addr = peer_descriptor.data_buffer_address;
                md.downstream_fifo_total_size = config.socket_mem_config.fifo_size;
                md.downstream_mesh_id = downstream_mesh_id;
                md.downstream_chip_id = downstream_chip_id;
                md.downstream_noc_y = recv_virtual_core.y;
                md.downstream_noc_x = recv_virtual_core.x;
                md.downstream_bytes_sent_addr = peer_config_buf_addr;
                md.is_sender = is_sender;
            }
            distributed::WriteShard(mesh_device->mesh_command_queue(0), config_buffer, config_data, device_coord, true);
        }
    } else {
        std::vector<receiver_socket_md> config_data(
            config_buffer->size() / sizeof(receiver_socket_md), receiver_socket_md());

        for (const auto& [device_coord, indexed_connections] : grouped_connections) {
            for (const auto& [conn_idx, connection] : indexed_connections) {
                const auto& [sender_core, recv_core] = connection;
                TT_FATAL(recv_core.device_coord == device_coord, "Internal Error: Receiver cores incorrectly grouped.");
                auto upstream_chip_id = get_sender_receiver_chip_fabric_encoding(
                    peer_descriptor.chip_ids[conn_idx],
                    mesh_device->get_device(recv_core.device_coord)->id(),
                    fabric_config,
                    SocketEndpoint::RECEIVER);
                auto upstream_mesh_id = peer_descriptor.mesh_ids[conn_idx];
                auto sender_virtual_core = mesh_device->worker_core_from_logical_core(sender_core.core_coord);

                uint32_t idx = core_to_core_id.at(recv_core.core_coord);
                auto& md = config_data[idx];
                md.read_ptr = local_descriptor.data_buffer_address;
                md.fifo_addr = local_descriptor.data_buffer_address;
                md.fifo_total_size = config.socket_mem_config.fifo_size;
                md.upstream_mesh_id = upstream_mesh_id;
                md.upstream_chip_id = upstream_chip_id;
                md.upstream_noc_y = sender_virtual_core.y;
                md.upstream_noc_x = sender_virtual_core.x;
                md.upstream_bytes_acked_addr = peer_config_buf_addr;
                md.is_sender = is_sender;
            }
            distributed::WriteShard(mesh_device->mesh_command_queue(0), config_buffer, config_data, device_coord, true);
        }
    }
}

SocketPeerDescriptor generate_local_endpoint_descriptor(const MeshSocket& socket_endpoint) {
    auto context = DistributedContext::get_current_world();
    const auto& config = socket_endpoint.get_config();
    bool is_sender = socket_endpoint.get_socket_endpoint_type() == SocketEndpoint::SENDER;
    SocketPeerDescriptor local_endpoint_desc = {
        .config = config,
        .config_buffer_address = socket_endpoint.get_config_buffer()->address(),
        .data_buffer_address = is_sender ? 0 : socket_endpoint.get_data_buffer()->address(),
        .exchange_tag = Tag{generate_descriptor_exchange_tag()}  // Unique tag for this exchange
    };
    auto device = socket_endpoint.get_config_buffer()->device();
    for (const auto& [sender_core, recv_core] : config.socket_connection_config) {
        const auto& device_coord = is_sender ? sender_core.device_coord : recv_core.device_coord;
        local_endpoint_desc.mesh_ids.push_back(get_physical_mesh_id(device, device_coord));
        local_endpoint_desc.chip_ids.push_back(device->get_device(device_coord)->id());
    }
    return local_endpoint_desc;
}

void forward_descriptor_to_peer(
    const SocketPeerDescriptor& desc,
    SocketEndpoint socket_endpoint_type,
    std::shared_ptr<multihost::DistributedContext> context) {
    const auto& config = desc.config;
    bool is_sender = socket_endpoint_type == SocketEndpoint::SENDER;
    auto peer_rank = is_sender ? config.receiver_rank : config.sender_rank;
    // Serialize the local endpoint descriptor
    std::vector<uint8_t> serialized_local_desc = serialize_to_bytes(desc);
    // Send size of serialized descriptor first, so that the peer knows the amount of data to expect
    int descriptor_size_bytes = serialized_local_desc.size();
    context->send(
        tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&descriptor_size_bytes), sizeof(descriptor_size_bytes)),
        Rank{peer_rank},
        desc.exchange_tag  // Forward this descriptor over the specified tag
    );
    // Send the serialized descriptor
    context->send(
        tt::stl::as_writable_bytes(tt::stl::Span<uint8_t>(serialized_local_desc.data(), serialized_local_desc.size())),
        Rank{peer_rank},
        desc.exchange_tag  // Forward this descriptor over the specified tag
    );
}

SocketPeerDescriptor receive_and_verify_descriptor_from_peer(
    const SocketPeerDescriptor& desc,
    SocketEndpoint socket_endpoint_type,
    std::shared_ptr<multihost::DistributedContext> context) {
    const auto& config = desc.config;
    bool is_sender = socket_endpoint_type == SocketEndpoint::SENDER;
    auto peer_rank = is_sender ? config.receiver_rank : config.sender_rank;

    static constexpr uint32_t NUM_SOCKET_HEADER_ELEMENTS = 1;
    // Query the size of the serialized descriptor first (this is the only element in the header)
    auto msg_header_size = context->snoop_incoming_msg_size(Rank{peer_rank}, desc.exchange_tag);
    TT_FATAL(
        msg_header_size == sizeof(int),
        "Expected {} bytes in the header for socket descriptor, but got {} bytes during multi-host handshake.",
        sizeof(int),
        msg_header_size);

    int expected_descriptor_size_bytes = 0;
    context->recv(
        tt::stl::Span<std::byte>(
            reinterpret_cast<std::byte*>(&expected_descriptor_size_bytes), sizeof(expected_descriptor_size_bytes)),
        Rank{peer_rank},
        desc.exchange_tag  // Read the descriptor over the specified tag
    );
    // Validate that the size in the header matches the descriptor message size
    auto descriptor_size_bytes = context->snoop_incoming_msg_size(Rank{peer_rank}, desc.exchange_tag);
    TT_FATAL(
        descriptor_size_bytes == expected_descriptor_size_bytes,
        "Expected {} bytes in the socket descriptor, but got {} bytes during multi-host handshake.",
        expected_descriptor_size_bytes,
        descriptor_size_bytes);

    // Allocate a buffer to receive the serialized descriptor
    std::vector<uint8_t> serialized_remote_desc(descriptor_size_bytes);
    // Receive the serialized descriptor
    context->recv(
        tt::stl::as_writable_bytes(
            tt::stl::Span<uint8_t>(serialized_remote_desc.data(), serialized_remote_desc.size())),
        Rank{peer_rank},
        desc.exchange_tag  // Read the descriptor over the specified tag
    );
    // Deserialize the received descriptor
    auto remote_desc = deserialize_from_bytes(serialized_remote_desc);
    // Validate that socket configs from remote and local descriptors match
    validate_remote_desc(desc, remote_desc);
    return remote_desc;
}

uint32_t get_physical_mesh_id(const MeshDevice* mesh_device, const MeshCoordinate& coord) {
    auto physical_device_id = mesh_device->get_device(coord)->id();
    auto global_coord = SystemMesh::instance().get_global_device_coordinate(physical_device_id);
    return SystemMesh::instance().get_physical_mesh_id(global_coord);
}

}  // namespace tt::tt_metal::distributed
