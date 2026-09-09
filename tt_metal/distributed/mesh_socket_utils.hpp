// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <tt-metalium/tt_align.hpp>

#include "impl/context/metal_context.hpp"
#include "tt_metal/hw/inc/hostdev/socket.h"

#include <unordered_set>

namespace tt::tt_metal::distributed {

struct SocketSenderSize {
    const uint32_t l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
    const uint32_t md_size_bytes = tt::align(sizeof(sender_socket_md), l1_alignment);
    const uint32_t ack_size_bytes = tt::align(sizeof(uint32_t), l1_alignment);
    const uint32_t enc_size_bytes = tt::align(sizeof(sender_downstream_encoding), l1_alignment);
};

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
    multihost::Tag exchange_tag = multihost::Tag{0};
    // Fabric chip id of this endpoint's core per connection (socket_connection_config order),
    // resolved locally from its own MeshDevice. Lets the peer skip deriving the chip from a
    // submesh-local coord, which mis-resolves when a submesh spans several ranks. Empty if the
    // peer did not supply it; callers then fall back to the coordinate derivation.
    std::vector<uint32_t> local_chip_ids;
};

// True when `mesh_device` covers devices this rank does not drive, i.e. a submesh co-owned by
// several ranks.
bool mesh_is_coowned(const MeshDevice& mesh_device);

// The distinct (device, core) pairs this endpoint occupies.
std::unordered_set<MeshCoreCoord> socket_endpoint_cores(const SocketConfig& config, SocketEndpoint socket_endpoint);

// Whether this endpoint's config buffer is per-core rather than lockstep. Requires
// per_core_allocation, L1 storage, and a single (device, core) for the endpoint, since the peer
// descriptor carries one address per buffer.
bool socket_endpoint_uses_per_core_allocation(const SocketConfig& config, SocketEndpoint socket_endpoint);

// Whether every buffer of this socket is per-core and so occupies L1 only on its two endpoint
// cores, leaving nothing for a co-owning rank to reserve.
bool socket_is_fully_per_core(const SocketConfig& config);

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
    SocketEndpoint socket_endpoint,
    const std::shared_ptr<MeshDevice>& peer_device = nullptr);

SocketPeerDescriptor generate_local_endpoint_descriptor(
    const MeshSocket& socket_endpoint, std::optional<multihost::DistributedContextId> context_id = std::nullopt);

void forward_descriptor_to_peer(
    const SocketPeerDescriptor& desc,
    multihost::Rank peer_rank,
    const std::shared_ptr<const multihost::DistributedContext>& context);

void forward_descriptor_to_peer(
    const SocketPeerDescriptor& desc,
    SocketEndpoint socket_endpoint_type,
    const std::shared_ptr<const multihost::DistributedContext>& context,
    const std::unordered_map<multihost::Rank, multihost::Rank>& rank_translation_table);

SocketPeerDescriptor receive_and_verify_descriptor_from_peer(
    const SocketPeerDescriptor& desc,
    multihost::Rank peer_rank,
    const std::shared_ptr<const multihost::DistributedContext>& context);

SocketPeerDescriptor receive_and_verify_descriptor_from_peer(
    const SocketPeerDescriptor& desc,
    SocketEndpoint socket_endpoint_type,
    const std::shared_ptr<const multihost::DistributedContext>& context,
    const std::unordered_map<multihost::Rank, multihost::Rank>& rank_translation_table);

// Map each connection's endpoint coords to fabric node ids.
//
// An endpoint backed by a local MeshDevice resolves through it. For a remote endpoint, pass the chip
// ids the peer sent in its descriptor (SocketPeerDescriptor::local_chip_ids) -- the peer resolved
// them from its own device handle, so they stay correct when its submesh spans several ranks. When
// they are empty the coord is derived from the owning rank's host binding, which assumes the submesh
// begins at that rank's host slice.
std::array<std::unordered_map<MeshCoordinate, tt::tt_fabric::FabricNodeId>, 2> generate_fabric_node_id_map(
    const SocketConfig& config,
    const std::shared_ptr<MeshDevice>& sender_device = nullptr,
    const std::shared_ptr<MeshDevice>& receiver_device = nullptr,
    const std::vector<uint32_t>& peer_sender_chip_ids = {},
    const std::vector<uint32_t>& peer_receiver_chip_ids = {});

std::vector<multihost::Rank> get_ranks_for_mesh_id(
    tt_fabric::MeshId mesh_id, const std::unordered_map<multihost::Rank, multihost::Rank>& rank_translation_table);

template <typename OperationType, typename... Args>
void execute_with_timeout(OperationType&& operation, Args&&... args) {
    const auto timeout = std::chrono::duration<float>(10.0f);

    std::atomic<bool> completed{false};
    std::atomic<bool> failed{false};
    std::exception_ptr exception_ptr{nullptr};

    std::thread thread([&]() {
        try {
            operation(std::forward<Args>(args)...);
            completed = true;
        } catch (...) {
            exception_ptr = std::current_exception();
            failed = true;
        }
    });

    auto start = std::chrono::steady_clock::now();
    while (!completed && !failed) {
        std::this_thread::yield();
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration<float>(now - start).count();
        if (elapsed >= timeout.count()) {
            thread.detach();
            TT_THROW(
                "Timed out trying to establish a socket connection. Please ensure that the socket is being created on "
                "all hosts mapped to the requested meshes.");
        }
    }

    if (thread.joinable()) {
        thread.join();
    }

    if (failed && exception_ptr) {
        std::rethrow_exception(exception_ptr);
    }
}

}  // namespace tt::tt_metal::distributed
