// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/mesh_socket_utils.hpp"
#include "impl/context/metal_context.hpp"
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>

using namespace tt::tt_metal::distributed::multihost;

namespace tt::tt_metal::distributed {

namespace {

void point_to_point_barrier(
    const std::vector<Rank>& ranks, const std::shared_ptr<multihost::DistributedContext>& distributed_context) {
    TT_FATAL(ranks.size() == 2, "Point-to-point barrier requires exactly two ranks.");
    TT_FATAL(ranks[0] != ranks[1], "Point-to-Point barrier cannot be used for synchronization within the same rank.");
    TT_FATAL(
        distributed_context->rank() == ranks[0] || distributed_context->rank() == ranks[1],
        "Point-to-Point barrier for ranks {} and {} cannot be called on rank {}.",
        *ranks[0],
        *ranks[1],
        *distributed_context->rank());

    if (distributed_context->rank() == ranks[0]) {
        int sync_msg = 1;
        distributed_context->ssend(
            tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&sync_msg), sizeof(sync_msg)), ranks[1], Tag{0});
    } else {
        int sync_msg = 0;
        distributed_context->recv(
            tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&sync_msg), sizeof(sync_msg)), ranks[0], Tag{0});
        TT_FATAL(sync_msg == 1, "Received unexpected message during point-to-point barrier.");
    }
}

}  // namespace

MeshSocket::MeshSocket(const std::shared_ptr<MeshDevice>& device, const SocketConfig& config) : config_(config) {
    auto context = config.distributed_context ? config.distributed_context : DistributedContext::get_current_world();
    auto local_mesh_binding = tt::tt_metal::MetalContext::instance().get_control_plane().get_local_mesh_id_bindings();
    TT_FATAL(local_mesh_binding.size() == 1, "Local mesh binding must be exactly one.");

    if (!(*local_mesh_binding[0] == config.sender_mesh_id || *local_mesh_binding[0] == config.receiver_mesh_id)) {
        log_warning(
            LogMetal,
            "Creating a null socket on Mesh ID {} with sender Mesh ID {} and receiver Mesh ID {}.",
            *local_mesh_binding[0],
            config.sender_mesh_id,
            config.receiver_mesh_id);
        return;
    }

    TT_FATAL(
        config.sender_mesh_id != config.receiver_mesh_id,
        "{} must only be used for communication between different host ranks, not within the same rank.",
        __func__);

    bool is_sender = *local_mesh_binding[0] == config.sender_mesh_id;
    // Allocate config buffers on both the sender and receiver meshes (even if the current host does not open a
    // connection)
    if (is_sender) {
        std::cout << "Creating sender buffer on Mesh ID: " << config.sender_mesh_id << " Host Rank ID: "
                  << *tt::tt_metal::MetalContext::instance().get_control_plane().get_local_host_rank_id_binding()
                  << std::endl;
        socket_endpoint_type_ = SocketEndpoint::SENDER;
        config_buffer_ = create_socket_config_buffer(device, config, socket_endpoint_type_);
    } else {
        std::cout << "Creating receiver buffer on Mesh ID: " << config.receiver_mesh_id << " Host Rank ID: "
                  << *tt::tt_metal::MetalContext::instance().get_control_plane().get_local_host_rank_id_binding()
                  << std::endl;
        socket_endpoint_type_ = SocketEndpoint::RECEIVER;
        config_buffer_ = create_socket_config_buffer(device, config, socket_endpoint_type_);
        data_buffer_ = create_socket_data_buffer(device, config);
    }
    this->connect_with_peer(context);
}

void MeshSocket::connect_with_peer(const std::shared_ptr<multihost::DistributedContext>& context) {
    auto local_endpoint_desc = generate_local_endpoint_descriptor(*this, context->id());
    SocketPeerDescriptor remote_endpoint_desc;
    // Convention:
    //  - Sender Endpoint sends its descriptor first, then receives the peer's descriptor.
    //  - Receiver Endpoint receives the peer's descriptor first, then sends its own descriptor.
    // Asymmetry ensures that the blocking send/recv do not deadlock.
    if (socket_endpoint_type_ == SocketEndpoint::SENDER) {
        forward_descriptor_to_peer(local_endpoint_desc, socket_endpoint_type_, context);
        remote_endpoint_desc =
            receive_and_verify_descriptor_from_peer(local_endpoint_desc, socket_endpoint_type_, context);
        fabric_node_id_map_ = generate_fabric_node_id_map(config_, local_endpoint_desc, remote_endpoint_desc);
    } else {
        remote_endpoint_desc =
            receive_and_verify_descriptor_from_peer(local_endpoint_desc, socket_endpoint_type_, context);
        forward_descriptor_to_peer(local_endpoint_desc, socket_endpoint_type_, context);
        fabric_node_id_map_ = generate_fabric_node_id_map(config_, remote_endpoint_desc, local_endpoint_desc);
    }
    write_socket_configs(config_buffer_, local_endpoint_desc, remote_endpoint_desc, socket_endpoint_type_);

    std::vector<Rank> sender_ranks = {};
    std::vector<Rank> recv_ranks = {};
    const auto& global_logical_bindings =
        tt::tt_metal::MetalContext::instance().get_control_plane().get_global_logical_bindings();

    for (const auto& [rank, mesh_id_and_host_rank] : global_logical_bindings) {
        if (*std::get<0>(mesh_id_and_host_rank) == config_.sender_mesh_id) {
            sender_ranks.push_back(rank);
        } else if (*std::get<0>(mesh_id_and_host_rank) == config_.receiver_mesh_id) {
            recv_ranks.push_back(rank);
        }
    }
    for (const auto& sender_rank : sender_ranks) {
        for (const auto& recv_rank : recv_ranks) {
            if (context->rank() == sender_rank || context->rank() == recv_rank) {
                point_to_point_barrier({sender_rank, recv_rank}, context);
            }
        }
    }
    // point_to_point_barrier({config_.sender_rank, config_.receiver_rank}, context);
}

std::pair<MeshSocket, MeshSocket> MeshSocket::create_socket_pair(
    const std::shared_ptr<MeshDevice>& sender,
    const std::shared_ptr<MeshDevice>& receiver,
    const SocketConfig& config) {
    auto sender_config_buffer = create_socket_config_buffer(sender, config, SocketEndpoint::SENDER);
    auto recv_config_buffer = create_socket_config_buffer(receiver, config, SocketEndpoint::RECEIVER);
    auto socket_data_buffer = create_socket_data_buffer(receiver, config);

    auto sender_socket = MeshSocket(
        nullptr,  // The sender socket does not have a data-buffer allocated
        sender_config_buffer,
        config,
        SocketEndpoint::SENDER);
    auto receiver_socket = MeshSocket(socket_data_buffer, recv_config_buffer, config, SocketEndpoint::RECEIVER);

    auto send_peer_descriptor = generate_local_endpoint_descriptor(sender_socket);
    auto recv_peer_descriptor = generate_local_endpoint_descriptor(receiver_socket);

    write_socket_configs(sender_config_buffer, send_peer_descriptor, recv_peer_descriptor, SocketEndpoint::SENDER);
    write_socket_configs(recv_config_buffer, recv_peer_descriptor, send_peer_descriptor, SocketEndpoint::RECEIVER);

    auto fabric_node_id_map = generate_fabric_node_id_map(config, send_peer_descriptor, recv_peer_descriptor);

    sender_socket.fabric_node_id_map_ = fabric_node_id_map;
    receiver_socket.fabric_node_id_map_ = fabric_node_id_map;

    return {sender_socket, receiver_socket};
}

std::shared_ptr<MeshBuffer> MeshSocket::get_data_buffer() const {
    TT_FATAL(data_buffer_, "Cannot access the data buffer for a sender socket.");
    return data_buffer_;
};

std::shared_ptr<MeshBuffer> MeshSocket::get_config_buffer() const { return config_buffer_; }

const SocketConfig& MeshSocket::get_config() const { return config_; }

tt::tt_fabric::FabricNodeId MeshSocket::get_fabric_node_id(SocketEndpoint endpoint, const MeshCoordinate& coord) const {
    return fabric_node_id_map_[static_cast<std::underlying_type_t<SocketEndpoint>>(endpoint)].at(coord);
}

}  // namespace tt::tt_metal::distributed

namespace std {

std::size_t hash<tt::tt_metal::distributed::SocketConnection>::operator()(
    const tt::tt_metal::distributed::SocketConnection& conn) const noexcept {
    return tt::stl::hash::hash_objects_with_default_seed(conn.sender_core, conn.receiver_core);
}

std::size_t hash<tt::tt_metal::distributed::MeshCoreCoord>::operator()(
    const tt::tt_metal::distributed::MeshCoreCoord& coord) const noexcept {
    return tt::stl::hash::hash_objects_with_default_seed(coord.device_coord, coord.core_coord);
}

std::size_t hash<tt::tt_metal::distributed::SocketConfig>::operator()(
    const tt::tt_metal::distributed::SocketConfig& config) const noexcept {
    std::optional<tt::tt_metal::distributed::multihost::Rank> distributed_context_rank = std::nullopt;
    std::optional<tt::tt_metal::distributed::multihost::Size> distributed_context_size = std::nullopt;
    if (config.distributed_context) {
        distributed_context_rank = config.distributed_context->rank();
        distributed_context_size = config.distributed_context->size();
    }
    return tt::stl::hash::hash_objects_with_default_seed(
        config.socket_connection_config,
        config.socket_mem_config,
        config.sender_rank,
        config.receiver_rank,
        distributed_context_rank,
        distributed_context_size);
}

std::size_t hash<tt::tt_metal::distributed::MeshSocket>::operator()(
    const tt::tt_metal::distributed::MeshSocket& socket) const noexcept {
    return tt::stl::hash::hash_objects_with_default_seed(socket.attribute_values());
}

}  // namespace std
