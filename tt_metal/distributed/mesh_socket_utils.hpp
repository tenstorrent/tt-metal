// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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

// Overload that generates endpoint descriptor from buffers directly (without requiring MeshSocket)
SocketPeerDescriptor generate_local_endpoint_descriptor(
    const std::shared_ptr<MeshBuffer>& config_buffer,
    const std::shared_ptr<MeshBuffer>& data_buffer,
    const SocketConfig& config,
    SocketEndpoint socket_endpoint,
    std::optional<multihost::DistributedContextId> context_id = std::nullopt);

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

// Helper functions to perform socket initialization side effects without creating MeshSocket objects
// These functions replicate the side effects of MeshSocket constructor:
// - Create and allocate socket buffers (config and data)
// - Perform handshaking with peer endpoint
// - Write socket configuration to buffers
// - Synchronize with peer via barrier

// Initialize socket connection for a given socket config.
// This performs all the side effects of MeshSocket construction:
// creates buffers, performs handshaking, writes configs, and synchronizes.
// The socket config object will be updated with any necessary state.
void initialize_socket_connection(
    const std::shared_ptr<MeshDevice>& device,
    const SocketConfig& config,
    const std::shared_ptr<multihost::DistributedContext>& context = nullptr);

// More granular helper functions for socket initialization side effects:

// Step 1: Create socket buffers (config and optionally data buffer for receiver)
// Returns: pair of (config_buffer, data_buffer), where data_buffer is nullptr for sender
std::pair<std::shared_ptr<MeshBuffer>, std::shared_ptr<MeshBuffer>> create_socket_buffers(
    const std::shared_ptr<MeshDevice>& device, const SocketConfig& config, SocketEndpoint socket_endpoint);

// Step 2: Perform socket handshaking with peer and write configs
// This exchanges descriptors with peer, generates fabric node ID maps, writes configs, and synchronizes
void connect_socket_with_peer(
    const std::shared_ptr<MeshBuffer>& config_buffer,
    const std::shared_ptr<MeshBuffer>& data_buffer,
    const SocketConfig& config,
    SocketEndpoint socket_endpoint,
    const std::shared_ptr<multihost::DistributedContext>& context = nullptr);

// Helper function for single-host socket pair initialization (both sender and receiver on same host)
// This performs all the side effects of create_socket_pair without requiring MeshSocket objects:
// - Creates config buffers (sender and receiver)
// - Creates data buffer (receiver)
// - Generates peer descriptors
// - Writes socket configs to buffers
// - Generates fabric node ID maps
void initialize_socket_pair(
    const std::shared_ptr<MeshDevice>& sender_device,
    const std::shared_ptr<MeshDevice>& receiver_device,
    const SocketConfig& config);

}  // namespace tt::tt_metal::distributed
