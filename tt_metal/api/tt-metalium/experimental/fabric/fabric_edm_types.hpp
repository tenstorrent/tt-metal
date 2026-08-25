// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <iostream>

namespace tt::tt_fabric {

// AllToAll: every device is a direct peer, nothing is forwarded. Distinct from
// NeighborExchange, which is 1D and reaches only the two adjacent chips.
enum class Topology { NeighborExchange = 0, Linear = 1, Ring = 2, Mesh = 3, Torus = 4, AllToAll = 5 };

// The property the route encoding turns on, not the dimensionality.
constexpr bool is_forwarding_topology(Topology topology) { return topology != Topology::AllToAll; }

// Topology classification utilities
constexpr bool is_2D_topology(Topology topology) { return topology == Topology::Mesh || topology == Topology::Torus; }

constexpr bool is_ring_or_torus(Topology topology) { return topology == Topology::Ring || topology == Topology::Torus; }

// Pull-Fabric request set. Here rather than in the ARCH_QUASAR-gated pull
// header because the host needs it too, to size the backing scratchpad.

// A multicast cannot fork -- the routing fields are a positional per-hop stream
// with no branch offsets -- so it takes one route per direction. A mesh range
// forks E/W within its own spine but cannot reach the opposite one, hence four.
template <Topology topology>
inline constexpr uint32_t fabric_max_routes = is_2D_topology(topology) ? 4u : 2u;

// Nothing forwards, so there is no stream to fork: one header carries a peer
// mask and the DE expands it into one SWQ per set bit.
template <>
inline constexpr uint32_t fabric_max_routes<Topology::AllToAll> = 1u;

// One slot per route, in the caller's scratchpad. A slot is a plain packet
// header: the pull fields ride inside it, so there is no trailer.
template <typename PacketHeader, uint32_t MaxRoutes>
struct alignas(16) FabricPullRequestSet {
    static constexpr uint32_t max_routes = MaxRoutes;

    PacketHeader routes[MaxRoutes];
    uint8_t direction[MaxRoutes];  // not in the packet; picks the outgoing link
    uint8_t used;                  // slots set-state filled; headers to publish
    // Source reads to expect back. Equals `used` only where a chain amortises;
    // AllToAll publishes one header and owes one read per peer.
    uint8_t source_read_completions;
    uint8_t include_self;  // not in the packet; served by a local NoC write
};

// sizeof(FabricPullRequestSet<...>) for a host that cannot name
// PACKET_HEADER_TYPE and only has its size. Kept beside the struct.
constexpr uint32_t fabric_pull_request_set_bytes(uint32_t packet_header_bytes, uint32_t max_routes) {
    const uint32_t unaligned = max_routes * packet_header_bytes + max_routes + 3;
    return (unaligned + 15u) & ~15u;  // alignas(16)
}

std::ostream& operator<<(std::ostream& os, const Topology& topology);

struct WorkerXY {
    uint16_t x;
    uint16_t y;

    constexpr WorkerXY() : x(0), y(0) {}

    constexpr WorkerXY(uint16_t x, uint16_t y) : x(x), y(y) {}

    constexpr uint32_t to_uint32() const { return (y << 16) | x; }
    static constexpr WorkerXY from_uint32(uint32_t v) { return WorkerXY(v & 0xFFFF, (v >> 16) & 0xFFFF); }

    constexpr bool operator==(const WorkerXY& rhs) const { return x == rhs.x && y == rhs.y; }
    constexpr bool operator!=(const WorkerXY& rhs) const { return !(*this == rhs); }
};

struct coord_t {
    coord_t(uint32_t x, uint32_t y) : x(x), y(y) {}
    uint32_t x;
    uint32_t y;
};

enum SendStatus : uint8_t {
    // Indicates that the sender was able to send the payload
    // but was not able to send the channel_sync_t at the end of the
    // buffer
    //
    // This enum should only ever be returned if we are sending less than
    // a full packet/buffer of data AND when we are trying to send the
    // channel_sync_t at the end of the buffer (which must be as a separate
    // command) but the eth_tx_cmd_q is busy for that second message
    //
    // Receiving this value indicates we
    // MUST:
    // - Eventually send the channel_sync_t before advancing to the next buffer
    // MUST NOT:
    // - Advance to the next buffer index
    // - Forward the other sender channel's data (if it has any)
    SENT_PAYLOAD_ONLY,

    // Indicates both the payload and the channel sync were sent successfully
    SENT_PAYLOAD_AND_SYNC,

    // Indicates no data was sent because the eth_tx_cmd_q was busy
    NOT_SENT,

    ERROR,
};

struct EDMChannelWorkerLocationInfo {
    uint32_t worker_semaphore_address{};
    uint32_t align_pad_0{};  // Padding added for safe reading over noc
    uint32_t align_pad_1{};
    uint32_t align_pad_2{};

    uint32_t worker_teardown_semaphore_address{};
    uint32_t align_pad_3{};  // Padding added for safe reading over noc
    uint32_t align_pad_4{};
    uint32_t align_pad_5{};

    WorkerXY worker_xy{0, 0};
    uint32_t align_pad_6{};  // Padding added for safe reading over noc
    uint32_t align_pad_7{};
    uint32_t align_pad_8{};

    uint32_t edm_read_counter = 0;
    uint32_t align_pad_9{};  // Padding added for safe reading over noc
    uint32_t align_pad_10{};
    uint32_t align_pad_11{};
};

static_assert(sizeof(EDMChannelWorkerLocationInfo) <= 64);

}  // namespace tt::tt_fabric
