// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/sockets/mesh_socket.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace tt::tt_metal::distributed {

class D2HSocket;
class H2DSocket;

namespace host_transport {
class HostTransport;
class RelayEndpoint;
}  // namespace host_transport

/**
 * @brief A D2D socket that reaches its peer over the host network instead of
 *        TT-Fabric.
 *
 * @code
 *   sender tensix --D2H--> host RAM --net--> host RAM --H2D--> receiver tensix
 * @endcode
 *
 * Needs no new device primitives: socket_api.h already branches on the is_d2h /
 * is_h2d discriminators this socket sets, so kernels drive an ordinary
 * Socket{Sender,Receiver}Interface. Only the payload-move call differs.
 *
 * Requires vIOMMU, an MPI build, and endpoints on PCIe x8 chips for any real
 * throughput. See
 * tech_reports/TT-Distributed/HostMeshSocket.md.
 */
class HostMeshSocket {
public:
    struct TransportConfig {
        /// Must divide fifo_size and match the page size the kernels set.
        uint32_t page_size = 0;
        /// Pages handed to the transport at once; peer ring depth is the real bound.
        uint32_t max_batch_pages = 8;
        /// False: no relay thread, caller must call poll().
        bool own_relay_thread = true;
    };

    HostMeshSocket(
        const std::shared_ptr<MeshDevice>& device, const SocketConfig& config, const TransportConfig& transport);
    ~HostMeshSocket();

    HostMeshSocket(const HostMeshSocket&) = delete;
    HostMeshSocket& operator=(const HostMeshSocket&) = delete;
    HostMeshSocket(HostMeshSocket&&) noexcept;
    HostMeshSocket& operator=(HostMeshSocket&&) noexcept;

    /// Same on every endpoint core, so a multi-core kernel takes one value.
    DeviceAddr get_config_buffer_address() const;
    std::shared_ptr<MeshBuffer> get_config_buffer() const;
    /// Receiver-side landing buffer, allocated exactly as D2D allocates its data
    /// buffer. Not the FIFO -- that is the pinned host ring -- but the same size
    /// and shape, so a receiver kernel pulls into it where a D2D kernel would
    /// find its pages. Raises on a sender, as D2D does.
    std::shared_ptr<MeshBuffer> get_data_buffer() const;
    const SocketConfig& get_config() const;
    SocketEndpoint get_socket_endpoint_type() const;
    /// Always true: the two endpoints are on different ranks by construction.
    bool is_rank_scoped_socket() const { return true; }
    std::vector<MeshCoreCoord> get_active_cores() const;
    MeshDevice* get_mesh_device() const;

    /// Only valid when own_relay_thread is false; an endpoint is single-consumer.
    bool poll();

    /// Sender: blocks until the NIC has read every page the local device produced.
    ///
    /// Receiver: returns immediately. Waiting for device consumption would
    /// deadlock -- the peer pipelines into the next round, so it would block on
    /// pages only the caller's next kernel launch reads. A receiver's sync point
    /// is its own program completion.
    ///
    /// Raises if the relay failed; this is how a relay-thread error surfaces.
    void barrier(std::optional<uint32_t> timeout_ms = std::nullopt);

    uint64_t pages_transferred() const;

    /// Sender only. Batch-handoff to peer-credit round trip. Off by default:
    /// costs a timestamp per batch.
    void set_latency_sampling(bool enabled);
    std::vector<uint64_t> take_latency_samples_ns();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tt::tt_metal::distributed
