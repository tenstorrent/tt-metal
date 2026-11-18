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
    uint32_t terminate;       // host: nonzero once the kernel should stop (it does once every active channel is idle)
    uint32_t heartbeat;       // kernel: bumped every pass
    uint32_t staging_addr;    // host: the kernel's packet staging area, WINDOW slots of staging_stride bytes
    uint32_t staging_stride;  // host: the largest packet, rounded up to 64 B
    uint32_t pad[12];
};
static_assert(sizeof(Header) == 64);

// One channel: a router's host_rdma sender channel and its request queue. The host fills it with `active` 0, then sets
// `active`; it clears `active` only once `consumed` has caught up with `produced`.
struct Channel {
    // router: slots of its host_rdma channel freed (absolute; the router writes it). First, so it is 16 B aligned: the
    // router writes it from a 16 B aligned word, and a NoC write keeps its source's offset within 16 B.
    uint32_t credits;
    uint32_t pad0[3];
    uint32_t active;
    uint32_t produced;  // host: requests posted (absolute)
    uint32_t consumed;  // kernel: requests whose packets have all been taken by the router
    uint32_t start;     // host: the router's slot count when the channel opened (credits starts there)
    uint32_t router_x;  // the router's core (virtual)
    uint32_t router_y;
    uint32_t slots_base;       // the router's host_rdma slot 0
    uint32_t free_slots_addr;  // the router channel's free-slots register, its increment-on-write address
    uint32_t num_slots;
    uint32_t slot_bytes;   // the router's slot stride
    uint32_t max_payload;  // payload bytes per packet
    uint32_t pad1;
    Request queue[QUEUE_DEPTH];
};
static_assert(sizeof(Channel) == 64 + QUEUE_DEPTH * sizeof(Request));
static_assert(sizeof(Header) % 64 == 0 && sizeof(Channel) % 64 == 0, "channels (and their credits) stay 64 B aligned");

struct Layout {
    Header header;
    Channel channels[MAX_CHANNELS];
};

}  // namespace tt::tt_fabric::host_rdma::service_abi
