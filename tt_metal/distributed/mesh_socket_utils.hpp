// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/fabric_types.hpp>
#include <tt-metalium/mesh_socket.hpp>

namespace tt::tt_metal::distributed {

// Utiity struct used for Sender and Receiver Hanshaking.
// Each endpoint (sender/receiver) needs to know the following about its peer:
// 1. The socket config used to create the connection (this is used for validation to ensure that both endpoints
// correspond to the same socket config)
// 2. Addresses of the peer's socket config and data buffers
// 3. The Mesh and Chip IDs corresponding to the peer endpoint (to compute the fabric encoding)

// For single-host sockets, this struct can be directly used when writing the socket config to the MeshDevice.
// For multi-host sockets, this struct is serialized to a FlatBuffer and sent over the network to the peer endpoint.
struct SocketPeerDescriptor {
    SocketConfig config;
    DeviceAddr config_buffer_address = 0;
    DeviceAddr data_buffer_address = 0;
    std::vector<uint32_t> mesh_ids;
    std::vector<uint32_t> chip_ids;
    multihost::Tag exchange_tag = multihost::Tag{0};
};

// Create send/receive socket config buffers
std::shared_ptr<MeshBuffer> create_socket_config_buffer(
    const std::shared_ptr<MeshDevice>& device, const SocketConfig& config, SocketEndpoint socket_endpoint);

// Create socket data buffer on receiver
std::shared_ptr<MeshBuffer> create_socket_data_buffer(
    const std::shared_ptr<MeshDevice>& receiver, const SocketConfig& config);

// Write socket config data to allocated buffers
void write_socket_configs(
    const std::shared_ptr<MeshBuffer>& config_buffer,
    const SocketPeerDescriptor& local_descriptor,
    const SocketPeerDescriptor& peer_descriptor,
    SocketEndpoint socket_endpoint);

SocketPeerDescriptor generate_local_endpoint_descriptor(
    const MeshSocket& socket_endpoint, std::optional<multihost::DistributedContextId> context_id = std::nullopt);

void forward_descriptor_to_peer(
    const SocketPeerDescriptor& desc,
    SocketEndpoint socket_endpoint_type,
    const std::shared_ptr<const multihost::DistributedContext>& context);

SocketPeerDescriptor receive_and_verify_descriptor_from_peer(
    const SocketPeerDescriptor& desc,
    SocketEndpoint socket_endpoint_type,
    const std::shared_ptr<const multihost::DistributedContext>& context);

std::array<std::unordered_map<MeshCoordinate, tt::tt_fabric::FabricNodeId>, 2> generate_fabric_node_id_map(
    const SocketConfig& config,
    const SocketPeerDescriptor& sender_descriptor,
    const SocketPeerDescriptor& receiver_descriptor);

}  // namespace tt::tt_metal::distributed
