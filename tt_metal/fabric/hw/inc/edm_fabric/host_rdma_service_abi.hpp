// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The host <-> service kernel contract of the host_rdma request service (tt_metal/fabric/host_rdma/host_rdma.hpp,
// tt_metal/fabric/impl/kernels/host_rdma_service.cpp): the service kernel's L1 layout and its request format.
//
// The host posts write requests into a channel's queue in the service core's L1. The kernel services them: it splits
// each into packets, reads each packet's payload from the request's source on its chip, forms the packet's header,
// and injects the packet into the channel's router (its host_rdma sender channel). The router credits the kernel in
// L1 (`Channel::credits`).

#include <cstddef>
#include <cstdint>

namespace tt::tt_fabric::host_rdma::service_abi {

inline constexpr uint32_t MAX_CHANNELS = 8;  // routers one service serves
inline constexpr uint32_t QUEUE_DEPTH = 64;  // requests per channel queue (a power of two): the largest batch
inline constexpr uint32_t WINDOW = 8;        // packets in flight, each on its own NoC transaction ID (1..WINDOW)
static_assert((QUEUE_DEPTH & (QUEUE_DEPTH - 1)) == 0);

// A NoC endpoint on the service's chip or the destination chip: a core in virtual coordinates and an address there.
struct Endpoint {
    uint32_t addr;
    uint16_t x;
    uint16_t y;
};
static_assert(sizeof(Endpoint) == 8);

// Write `size` bytes from `src` (on the service's chip) to `dst` on chip (dst_mesh_id, dst_dev_id). With
// completion_inc != 0, the request's last packet also adds completion_inc to the 32-bit counter at `completion` on the
// destination chip, after its data has landed. Addresses and the size are 16 B aligned.
struct Request {
    Endpoint src;
    Endpoint dst;
    Endpoint completion;
    uint32_t size;
    uint32_t completion_inc;
    uint16_t dst_dev_id;
    uint16_t dst_mesh_id;
    uint32_t hops;  // 1D fabrics: chips from the source to the destination (2D routes by dst_dev_id/dst_mesh_id)
    uint32_t pad[6];
};
static_assert(sizeof(Request) == 64);

struct Header {
    uint32_t terminate;       // host: nonzero once the kernel should stop (the host has drained every channel)
    uint32_t heartbeat;       // kernel: bumped every pass
    uint32_t staging_addr;    // host: the kernel's packet staging area, WINDOW slots of staging_stride bytes
    uint32_t staging_stride;  // host: the largest packet, rounded up to 64 B
    uint32_t pad[12];
};
static_assert(sizeof(Header) == 64);

// One channel: a router's host_rdma sender channel, which the kernel drives as a fabric connection (a
// WorkerToFabricEdmSender over the fields below), and its request queue. The host fills it with `active` 0, then sets
// `active`. The kernel opens the connection when it sees `active`, taking up the producer cursor the channel's last
// producer left in the router, and closes it (leaving its cursor there) when `active` clears or the kernel stops. The
// host clears `active` only once `consumed` has caught up with `produced`.
struct Channel {
    // router: slots of its host_rdma channel freed (absolute; the router writes it): the connection's credit word.
    // First, so it is 16 B aligned: the router writes it from a 16 B aligned word, and a NoC write keeps its source's
    // offset within 16 B.
    uint32_t credits;
    uint32_t pad0[3];
    uint32_t active;
    uint32_t produced;     // host: requests posted (absolute)
    uint32_t consumed;     // kernel: requests whose packets have all been taken by the router
    uint32_t max_payload;  // host: payload bytes per packet
    // host: the router's channel, as a connection to it takes it (addresses in the router's L1).
    uint32_t router_x;  // the router's core (virtual)
    uint32_t router_y;
    uint32_t slots_base;  // slot 0
    uint32_t num_slots;
    uint32_t slot_bytes;                 // the slot stride
    uint32_t free_slots_stream;          // the channel's free-slots stream
    uint32_t connection_semaphore_addr;  // the channel's connection semaphore
    uint32_t connection_info_addr;       // its HostRdmaChannelInfo, laid out as a worker's location info
    uint32_t producer_cursor_addr;       // its SenderChannelProducerCursor
    uint32_t pad1[3];
    // kernel: the connection's own words: where opening reads the producer cursor (16 B aligned), and its teardown
    // word.
    uint32_t cursor_scratch[4];
    uint32_t teardown;
    uint32_t pad2[7];
    Request queue[QUEUE_DEPTH];
};
static_assert(offsetof(Channel, cursor_scratch) % 16 == 0);
static_assert(sizeof(Channel) == 128 + QUEUE_DEPTH * sizeof(Request));
static_assert(sizeof(Header) % 64 == 0 && sizeof(Channel) % 64 == 0, "channels (and their credits) stay 64 B aligned");

struct Layout {
    Header header;
    Channel channels[MAX_CHANNELS];
};

}  // namespace tt::tt_fabric::host_rdma::service_abi
