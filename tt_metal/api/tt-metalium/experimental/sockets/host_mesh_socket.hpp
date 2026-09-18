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
class RdmaContext;
class RdmaChannel;
class RdmaRegion;
class RelayEndpoint;
}  // namespace host_transport

/**
 * @brief A socket carrying a device-to-device stream over the host interconnect.
 *
 * Presents MeshSocket's interface, but reaches its peer through host RDMA rather
 * than TT-Fabric:
 *
 * @code
 *   sender tensix --D2H--> host RAM --RDMA--> host RAM --H2D--> receiver tensix
 * @endcode
 *
 * Each leg is existing tt-metal machinery. The device side needs no new kernel
 * primitives: `socket_api.h` is transport-agnostic apart from
 * `socket_notify_receiver` / `socket_notify_sender`, which already branch on the
 * `is_d2h` / `is_h2d` discriminators this socket sets. A sender kernel therefore
 * drives an ordinary SocketSenderInterface and a receiver kernel an ordinary
 * SocketReceiverInterface; only the payload-move call differs (a PCIe write
 * instead of a NOC write, and a chunked NOC read instead of an L1 copy).
 *
 * Both FIFOs are pinned host rings holding the same number of pages, registered
 * once with the NIC, so the host hop copies nothing: the NIC reads the D2H ring
 * and writes the peer's H2D ring directly.
 *
 * Ordering and pipelining come from using one RC queue pair per connection. RC
 * delivery is in-order, so payload work requests are posted back-to-back
 * unsignaled and a trailing RDMA_WRITE_WITH_IMM doorbell cannot be observed by
 * the peer before the bytes ahead of it have landed. Payload is never striped
 * across queue pairs, which would break that guarantee.
 *
 * A single process-wide relay thread services every socket with a non-blocking
 * poll loop, so an owner incurs no polling duty. Pass
 * `TransportConfig::own_relay_thread = false` to drive `poll()` yourself.
 *
 * Requirements:
 * - vIOMMU enabled (the pinned rings must be NOC-mappable)
 * - a RoCE-capable RDMA device reachable from both hosts
 * - endpoints placed on PCIe x8 chips for any real throughput
 */
class HostMeshSocket {
public:
    struct TransportConfig {
        /// Transfer granularity, in bytes. Must divide the socket's fifo_size and
        /// match the page size the device kernels set.
        uint32_t page_size = 0;
        /// RDMA device name; empty selects the first.
        std::string rdma_device;
        /// GID index; negative auto-selects a RoCEv2 IPv4-mapped GID.
        int gid_index = -1;
        /// Pages coalesced into one work request. Larger amortises per-request
        /// cost; the peer ring depth is the real bound.
        uint32_t max_batch_pages = 8;
        /// When false, the caller must call poll(); no relay thread is started.
        bool own_relay_thread = true;
    };

    HostMeshSocket(
        const std::shared_ptr<MeshDevice>& device, const SocketConfig& config, const TransportConfig& transport);
    ~HostMeshSocket();

    HostMeshSocket(const HostMeshSocket&) = delete;
    HostMeshSocket& operator=(const HostMeshSocket&) = delete;
    HostMeshSocket(HostMeshSocket&&) noexcept;
    HostMeshSocket& operator=(HostMeshSocket&&) noexcept;

    /// L1 address of the socket config buffer. The same on every endpoint core,
    /// so a multi-core kernel takes one value.
    DeviceAddr get_config_buffer_address() const;
    std::shared_ptr<MeshBuffer> get_config_buffer() const;
    /// Always throws: unlike MeshSocket, this socket's FIFO is a pinned host ring,
    /// not a device buffer. A receiver kernel pulls into a landing buffer the
    /// caller allocates, so there is no socket-owned device data buffer to hand back.
    std::shared_ptr<MeshBuffer> get_data_buffer() const;
    const SocketConfig& get_config() const;
    SocketEndpoint get_socket_endpoint_type() const;
    std::vector<MeshCoreCoord> get_active_cores() const;
    MeshDevice* get_mesh_device() const;

    /// One non-blocking relay sweep on the calling thread. Returns whether it
    /// progressed. Only required when own_relay_thread is false.
    bool poll();

    /// On a **sender**, blocks until every page the local device produced has been
    /// read off the host by the NIC and acknowledged back to the device, so the
    /// send ring is empty and the data is on the peer.
    ///
    /// On a **receiver**, returns as soon as the relay is healthy. There is
    /// deliberately nothing to wait for: publication into the receive ring is
    /// driven by the peer, and the only step after that is the device consuming,
    /// which the owner controls by running its own program. Waiting for
    /// consumption here would deadlock, because the peer pipelines into the next
    /// round and the barrier would block on pages that only the caller's next
    /// kernel launch will read. A receiver's synchronisation point is its own
    /// program completion.
    ///
    /// Either way this raises if the relay has failed, which is how a transport
    /// error on the relay thread reaches the caller.
    void barrier(std::optional<uint32_t> timeout_ms = std::nullopt);

    /// Pages handed to the wire (sender) or published to the device (receiver).
    uint64_t pages_transferred() const;

    /// Samples, on a sender, the time from handing a batch to the NIC until the
    /// peer credits it -- the round trip through the far host and far device.
    /// Off by default; enabling it costs a timestamp per batch. No effect on a
    /// receiver. Pair with an idle round-trip measurement to separate transit
    /// from queueing delay under load.
    void set_latency_sampling(bool enabled);
    std::vector<uint64_t> take_latency_samples_ns();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tt::tt_metal::distributed
