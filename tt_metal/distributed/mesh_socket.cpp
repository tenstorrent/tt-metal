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

void barrier_across_send_recv_ranks(
    const std::vector<Rank>& sender_ranks,
    const std::vector<Rank>& recv_ranks,
    const std::shared_ptr<multihost::DistributedContext>& distributed_context) {
    std::vector<int> ranks;
    ranks.reserve(sender_ranks.size() + recv_ranks.size());
    for (const auto& sender_rank : sender_ranks) {
        ranks.push_back(*sender_rank);
    }
    for (const auto& recv_rank : recv_ranks) {
        ranks.push_back(*recv_rank);
    }
    auto sub_context = distributed_context->create_sub_context(ranks);
    sub_context->barrier();
}

void validate_device_ownership(
    multihost::Rank global_sender_rank, multihost::Rank global_receiver_rank, const SocketConfig& config) {
    const auto& global_distributed_context = DistributedContext::get_current_world();
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    bool is_sender = global_distributed_context->rank() == global_sender_rank;
    bool is_receiver = global_distributed_context->rank() == global_receiver_rank;

    auto sender_coord_range = control_plane.get_coord_range(config.sender_mesh_id.value(), tt_fabric::MeshScope::LOCAL);
    auto receiver_coord_range =
        control_plane.get_coord_range(config.receiver_mesh_id.value(), tt_fabric::MeshScope::LOCAL);

    if (is_sender || is_receiver) {
        for (const auto& connection : config.socket_connection_config) {
            if (is_sender) {
                TT_FATAL(
                    sender_coord_range.contains(connection.sender_core.device_coord),
                    "Sender core coordinate {} is out of bounds for rank {} on mesh id {}",
                    connection.sender_core.device_coord,
                    *global_sender_rank,
                    *config.sender_mesh_id);
            } else {
                TT_FATAL(is_receiver, "Internal Error: Expected receiver rank to be set when sender rank is not set.");
                TT_FATAL(
                    receiver_coord_range.contains(connection.receiver_core.device_coord),
                    "Receiver core coordinate {} is out of bounds for rank {} on mesh id {}",
                    connection.receiver_core.device_coord,
                    *global_receiver_rank,
                    *config.receiver_mesh_id);
            }
        }
    }
}

}  // namespace

void MeshSocket::process_host_ranks() {
    multihost::Rank sender_rank = config_.sender_rank;
    multihost::Rank receiver_rank = config_.receiver_rank;
    if (config_.distributed_context) {
        std::array<int, 2> socket_ranks = {*sender_rank, *receiver_rank};
        std::array<int, 2> translated_socket_ranks = {-1, -1};
        config_.distributed_context->translate_ranks_to_other_ctx(
            socket_ranks, DistributedContext::get_current_world(), translated_socket_ranks);
        sender_rank = Rank{translated_socket_ranks[0]};
        receiver_rank = Rank{translated_socket_ranks[1]};
        rank_translation_table_[sender_rank] = config_.sender_rank;
        rank_translation_table_[receiver_rank] = config_.receiver_rank;

    } else {
        rank_translation_table_[config_.sender_rank] = config_.sender_rank;
        rank_translation_table_[config_.receiver_rank] = config_.receiver_rank;
    }
    const auto& global_logical_bindings =
        tt::tt_metal::MetalContext::instance().get_control_plane().get_global_logical_bindings();
    TT_FATAL(
        global_logical_bindings.contains(sender_rank) && global_logical_bindings.contains(receiver_rank),
        "Invalid socket sender rank {} or receiver rank {} specified.",
        *sender_rank,
        *receiver_rank);

    config_.sender_mesh_id = std::get<0>(global_logical_bindings.at(sender_rank));
    config_.receiver_mesh_id = std::get<0>(global_logical_bindings.at(receiver_rank));
    // These ranks belong to the global distributed context. Use them to ensure
    // that the hosts they correspond to own socket connection coordinates.
    validate_device_ownership(sender_rank, receiver_rank, config_);
}

void MeshSocket::process_mesh_ids() {
    const auto& global_logical_bindings =
        tt::tt_metal::MetalContext::instance().get_control_plane().get_global_logical_bindings();

    for (const auto& [rank, mesh_id_and_host_rank] : global_logical_bindings) {
        if (std::get<0>(mesh_id_and_host_rank) == config_.sender_mesh_id.value() ||
            std::get<0>(mesh_id_and_host_rank) == config_.receiver_mesh_id.value()) {
            if (config_.distributed_context) {
                std::array<int, 1> socket_ranks = {*rank};
                std::array<int, 1> translated_socket_ranks = {-1};
                DistributedContext::get_current_world()->translate_ranks_to_other_ctx(
                    socket_ranks, config_.distributed_context, translated_socket_ranks);
                // Only add to translation table if the global rank is part of the subcontext
                // i.e. the translated rank is valid (positive and less than the size of the subcontext)
                if (translated_socket_ranks[0] >= 0 &&
                    translated_socket_ranks[0] < *config_.distributed_context->size()) {
                    rank_translation_table_[rank] = Rank{translated_socket_ranks[0]};
                }
            } else {
                rank_translation_table_[rank] = rank;
            }
        }
    }
}

SocketConfig MeshSocket::populate_mesh_ids(
    const std::shared_ptr<MeshDevice>& sender,
    const std::shared_ptr<MeshDevice>& receiver,
    const SocketConfig& base_config) {
    auto config = base_config;
    std::optional<tt_fabric::MeshId> sender_mesh_id = std::nullopt;
    std::optional<tt_fabric::MeshId> receiver_mesh_id = std::nullopt;
    for (const auto& conn : config.socket_connection_config) {
        auto sender_coord = conn.sender_core.device_coord;
        auto receiver_coord = conn.receiver_core.device_coord;

        if (!sender_mesh_id.has_value()) {
            sender_mesh_id = sender->get_fabric_node_id(sender_coord).mesh_id;
        } else {
            TT_FATAL(
                sender_mesh_id.value() == sender->get_fabric_node_id(sender_coord).mesh_id,
                "Expected Mesh IDs for all Sender devices to be identical.");
        }
        if (!receiver_mesh_id.has_value()) {
            receiver_mesh_id = receiver->get_fabric_node_id(receiver_coord).mesh_id;
        } else {
            TT_FATAL(
                receiver_mesh_id.value() == receiver->get_fabric_node_id(receiver_coord).mesh_id,
                "Expected Mesh IDs for all Receiver devices to be identical.");
        }
    }

    config.sender_mesh_id = sender_mesh_id;
    config.receiver_mesh_id = receiver_mesh_id;

    return config;
}

MeshSocket::MeshSocket(const std::shared_ptr<MeshDevice>& device, const SocketConfig& config) : config_(config) {
    auto context = config_.distributed_context ? config_.distributed_context : DistributedContext::get_current_world();

    TT_FATAL(!config_.socket_connection_config.empty(), "Socket connection config cannot be empty.");

    if (config_.sender_mesh_id.has_value()) {
        TT_FATAL(
            config.receiver_mesh_id.has_value(), "Expected receiver mesh id to be set when sender mesh id is set.");
        this->process_mesh_ids();
    } else {
        this->process_host_ranks();
    }
    TT_FATAL(
        config_.sender_mesh_id.has_value() && config_.receiver_mesh_id.has_value(),
        "Unable to determine mesh ids for socket.");
    auto local_mesh_binding = tt::tt_metal::MetalContext::instance().get_control_plane().get_local_mesh_id_bindings();
    TT_FATAL(local_mesh_binding.size() == 1, "Local mesh binding must be exactly one.");

    if (!(local_mesh_binding[0] == config_.sender_mesh_id.value() ||
          local_mesh_binding[0] == config_.receiver_mesh_id.value())) {
        log_warning(
            LogMetal,
            "Creating a null socket on Mesh ID {} with sender Mesh ID {} and receiver Mesh ID {}.",
            *local_mesh_binding[0],
            *config_.sender_mesh_id.value(),
            *config_.receiver_mesh_id.value());
        return;
    }
    TT_FATAL(
        config_.sender_mesh_id.value() != config_.receiver_mesh_id.value(),
        "{} must only be used for communication between different host ranks, not within the same rank.",
        __func__);

    bool is_sender = local_mesh_binding[0] == config_.sender_mesh_id.value();
    // Allocate config buffers on both the sender and receiver meshes (even if the current host does not open a
    // connection)
    if (is_sender) {
        socket_endpoint_type_ = SocketEndpoint::SENDER;
        config_buffer_ = create_socket_config_buffer(device, config_, socket_endpoint_type_);
    } else {
        socket_endpoint_type_ = SocketEndpoint::RECEIVER;
        config_buffer_ = create_socket_config_buffer(device, config_, socket_endpoint_type_);
        data_buffer_ = create_socket_data_buffer(device, config_);
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
        forward_descriptor_to_peer(local_endpoint_desc, socket_endpoint_type_, context, rank_translation_table_);
        remote_endpoint_desc = receive_and_verify_descriptor_from_peer(
            local_endpoint_desc, socket_endpoint_type_, context, rank_translation_table_);
        fabric_node_id_map_ = generate_fabric_node_id_map(config_);
    } else {
        remote_endpoint_desc = receive_and_verify_descriptor_from_peer(
            local_endpoint_desc, socket_endpoint_type_, context, rank_translation_table_);
        forward_descriptor_to_peer(local_endpoint_desc, socket_endpoint_type_, context, rank_translation_table_);
        fabric_node_id_map_ = generate_fabric_node_id_map(config_);
    }
    write_socket_configs(config_buffer_, local_endpoint_desc, remote_endpoint_desc, socket_endpoint_type_);

    std::vector<Rank> sender_ranks = get_ranks_for_mesh_id(config_.sender_mesh_id.value(), rank_translation_table_);
    std::vector<Rank> recv_ranks = get_ranks_for_mesh_id(config_.receiver_mesh_id.value(), rank_translation_table_);
    // Barrier across all sender and receiver ranks. This ensures that the downstream workloads using this socket
    // will start after the socket is initialized across all hosts.
    execute_with_timeout([&]() { barrier_across_send_recv_ranks(sender_ranks, recv_ranks, context); });
}

std::pair<MeshSocket, MeshSocket> MeshSocket::create_socket_pair(
    const std::shared_ptr<MeshDevice>& sender,
    const std::shared_ptr<MeshDevice>& receiver,
    const SocketConfig& base_config) {
    TT_FATAL(!base_config.socket_connection_config.empty(), "Socket connection config cannot be empty.");

    auto config = populate_mesh_ids(sender, receiver, base_config);
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

    write_socket_configs(
        sender_config_buffer, send_peer_descriptor, recv_peer_descriptor, SocketEndpoint::SENDER, receiver);
    write_socket_configs(
        recv_config_buffer, recv_peer_descriptor, send_peer_descriptor, SocketEndpoint::RECEIVER, sender);

    auto fabric_node_id_map = generate_fabric_node_id_map(config, sender, receiver);

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
