// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

namespace tt::tt_fabric::host_rdma::service_abi {

inline constexpr uint32_t MAX_CHANNELS = 8;
inline constexpr uint32_t QUEUE_DEPTH = 64;
inline constexpr uint32_t WINDOW = 8;  // packets in flight, each on its own NoC transaction ID (1..WINDOW)
inline constexpr uint32_t NOC_ALIGNMENT = 16;
inline constexpr uint32_t LAYOUT_ALIGNMENT = 64;

constexpr uint32_t align_up(uint32_t value, uint32_t alignment) { return (value + alignment - 1) & ~(alignment - 1); }
constexpr uint32_t align_down(uint32_t value, uint32_t alignment) { return value & ~(alignment - 1); }
static_assert((QUEUE_DEPTH & (QUEUE_DEPTH - 1)) == 0);

struct Endpoint {
    uint32_t addr;
    uint16_t x;
    uint16_t y;
};
static_assert(sizeof(Endpoint) == 8);

struct Request {
    Endpoint src;
    Endpoint dst;
    Endpoint completion;
    uint32_t size;
    uint32_t completion_inc;  // 0: no completion
    uint16_t dst_dev_id;
    uint16_t dst_mesh_id;
    uint32_t pad[7];
};
static_assert(sizeof(Request) == 64);

struct Header {
    uint32_t terminate;
    uint32_t heartbeat;
    uint32_t staging_addr;  // WINDOW slots of staging_stride bytes
    uint32_t staging_stride;
    uint32_t pad[12];
};
static_assert(sizeof(Header) == 64);

// Written by the host with `active` 0, then activated. The host deactivates it only once `consumed` == `produced`.
struct Channel {
    uint32_t credits;
    uint32_t pad0[3];
    uint32_t active;
    uint32_t produced;
    uint32_t consumed;
    uint32_t max_payload;
    uint32_t router_x;
    uint32_t router_y;
    uint32_t slots_base;
    uint32_t num_slots;
    uint32_t slot_bytes;
    uint32_t free_slots_stream;
    uint32_t connection_semaphore_addr;
    uint32_t connection_info_addr;
    uint32_t producer_cursor_addr;
    uint32_t pad1[3];
    uint32_t cursor_scratch[4];
    uint32_t teardown;
    uint32_t pad2[7];
    Request queue[QUEUE_DEPTH];
};
static_assert(offsetof(Channel, credits) % NOC_ALIGNMENT == 0, "the router writes credits from a 16 B aligned source");
static_assert(offsetof(Channel, cursor_scratch) % NOC_ALIGNMENT == 0);
static_assert(sizeof(Channel) == 128 + QUEUE_DEPTH * sizeof(Request));
static_assert(sizeof(Header) % LAYOUT_ALIGNMENT == 0 && sizeof(Channel) % LAYOUT_ALIGNMENT == 0);

struct Layout {
    Header header;
    Channel channels[MAX_CHANNELS];
};

}  // namespace tt::tt_fabric::host_rdma::service_abi
