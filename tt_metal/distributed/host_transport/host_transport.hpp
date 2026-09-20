// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace tt::tt_metal::distributed::multihost {
class DistributedContext;
}

namespace tt::tt_metal::distributed::host_transport {

// Both ends hold the same page count, so a page's source and destination index
// are the same value.
struct RingGeometry {
    uint32_t page_size = 0;
    uint32_t num_pages = 0;

    uint64_t fifo_bytes() const { return static_cast<uint64_t>(page_size) * num_pages; }
};

// Moves whole pages between two hosts' pinned rings, at the same page index on
// both sides.
//
// The seam is "how many pages are valid in the ring" rather than "post a write":
// one-sided lands the bytes with the NIC and the receiver reads a counter,
// two-sided needs the receiver to complete a receive. Both answer the former.
//
// Counts are absolute, so a duplicated or stale update is a no-op. Calls are
// non-blocking and single-consumer: one thread drives one transport.
class HostTransport {
public:
    virtual ~HostTransport() = default;

    // --- sender ---

    // Room for a batch of `pages` right now.
    virtual bool can_send(uint32_t pages) const = 0;
    // Hand `pages` pages from absolute ring page `first_page` to the peer, which
    // places them at the same index. False if nothing was sent. `first_page` stays
    // 64-bit: it is compared against an absolute counter and only narrowed after
    // the modulo, so truncating here would break once a stream passes 2^32 pages.
    virtual bool send(uint64_t first_page, uint32_t pages) = 0;
    // Source bytes the transport has finished reading, so the relay can release
    // them back to the device. Absolute.
    virtual uint64_t pages_released() const = 0;
    // What the peer reports its device has consumed. Absolute.
    virtual uint64_t peer_consumed_pages() const = 0;

    // --- receiver ---

    // Pages the peer has delivered into the local ring. Absolute.
    virtual uint64_t pages_delivered() const = 0;
    // Publish how many pages the local device has consumed. Absolute.
    virtual bool post_credit(uint64_t consumed_pages) = 0;
    // How far the local device has got, so a two-sided transport knows which
    // pages are safe to receive into again. Absolute.
    virtual void set_consumed(uint64_t consumed_pages) = 0;

    // --- both ---

    // Drive progress, then refresh the counters above.
    virtual void poll() = 0;
    virtual std::string describe() const = 0;
};

// Tags each connection consumes: payload and credit.
inline constexpr int kTagsPerConnection = 2;

struct TransportParams {
    RingGeometry geometry;
    bool is_sender = false;
    // The local pinned ring the transport reads from or writes into.
    void* ring = nullptr;
    // Used for the handshake, and for MPI the data path too.
    int peer_rank = 0;
    std::shared_ptr<multihost::DistributedContext> context;
    // Separates concurrent connections between the same rank pair. Both ends must
    // pick the same base, which the caller does by construction order.
    int tag_base = 0;
    uint32_t max_batch_pages = 8;
};

// Whether this build can move pages between hosts at all. Safe to call before the
// handshake, so both ends can agree instead of one blocking.
bool host_transport_available();

// Builds and connects a transport. Collective with the peer: both ends must call
// it, in the same order, before either returns.
std::unique_ptr<HostTransport> make_host_transport(const TransportParams& params);

}  // namespace tt::tt_metal::distributed::host_transport
