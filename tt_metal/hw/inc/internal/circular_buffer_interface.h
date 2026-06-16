// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "tt-metalium/circular_buffer_constants.h"
#include "hostdev/dev_msgs.h"
#include "internal/risc_attribs.h"

constexpr static std::uint32_t REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE = L1_ALIGNMENT;

// Pack/unpack helpers for RemoteSenderCBInterface::num_receivers_and_remote_pages_sent_ptr.
// The REMOTE_CB_PACKED_* constants they use live in hostdev/dev_msgs.h. Host code composes the
// same field with matching bit math in global_circular_buffer.cpp.
inline constexpr std::uint32_t remote_cb_num_receivers(std::uint32_t packed) {
    return packed >> REMOTE_CB_PACKED_COUNT_SHIFT;
}
inline constexpr std::uint32_t remote_cb_remote_pages_sent_ptr(std::uint32_t packed) {
    return packed & REMOTE_CB_PACKED_ADDR_MASK;
}
inline constexpr std::uint32_t remote_cb_pack(std::uint32_t num_receivers, std::uint32_t remote_pages_sent_ptr) {
    return (num_receivers << REMOTE_CB_PACKED_COUNT_SHIFT) | (remote_pages_sent_ptr & REMOTE_CB_PACKED_ADDR_MASK);
}

struct RemoteSenderCBInterface {
    uint32_t config_ptr;
    uint32_t fifo_start_addr;
    uint32_t fifo_limit_page_aligned;
    uint32_t fifo_page_size;

    uint32_t fifo_wr_ptr;

    // Address to an array of x, y coords
    // arranged x0, y0, x1, y1, ...
    uint32_t receiver_noc_xy_ptr;

    // Points to an array of size num_receivers, stored as pages_sent, pages_acked pairs
    // for each receiver. Per-receiver stride is REMOTE_CB_LOCAL_PAGES_STRIDE: 2 * L1_ALIGNMENT
    // for worker senders, 2 * sizeof(uint32_t) for DRISC senders (packed; NoC atomic inc
    // only needs 4-byte alignment).
    uint32_t aligned_pages_sent_ptr;

    // Packed: bits [23:0] are the remote pages_sent address on the receiver's L1,
    // bits [31:24] are num_receivers. The remote address is canonical (no fallback):
    // for a sharded GCB it equals `aligned_pages_sent_ptr` (local == remote); for a
    // DRAM-sender GCB it's the worker-side pages_sent base (the DRISC sender's local
    // counter is in DRISC L1, but the NoC target is in worker L1).
    uint32_t num_receivers_and_remote_pages_sent_ptr;
};

struct RemoteReceiverCBInterface {
    uint32_t config_ptr;
    uint32_t fifo_start_addr;
    uint32_t fifo_limit_page_aligned;
    uint32_t fifo_page_size;

    uint32_t fifo_rd_ptr;

    // NoC XY fit in 8 bits each; using u16 keeps the struct at 8 uint32 words.
    uint16_t sender_noc_x;
    uint16_t sender_noc_y;

    // These point to a single entry corresponding to receiver index
    // Each entry is L1 aligned
    uint32_t aligned_pages_acked_ptr;

    // Address ON the SENDER's L1 where the receiver's NoC inc lands for pages_acked.
    // Canonical (no fallback): for a sharded GCB it equals `aligned_pages_acked_ptr`;
    // for a DRAM-sender GCB it points into DRISC L1 (separate address space from the
    // receiver's local pages_acked).
    uint32_t remote_pages_acked_ptr;
};

// Required for update_remote_cb_config
static_assert(
    offsetof(RemoteSenderCBInterface, fifo_start_addr) == offsetof(RemoteReceiverCBInterface, fifo_start_addr),
    "fifo_start_addr must be at the same offset in RemoteSenderCBInterface and RemoteReceiverCBInterface");
static_assert(
    offsetof(RemoteSenderCBInterface, fifo_limit_page_aligned) ==
        offsetof(RemoteReceiverCBInterface, fifo_limit_page_aligned),
    "fifo_limit_page_aligned must be at the same offset in RemoteSenderCBInterface and RemoteReceiverCBInterface");
static_assert(
    offsetof(RemoteSenderCBInterface, fifo_wr_ptr) == offsetof(RemoteReceiverCBInterface, fifo_rd_ptr),
    "fifo_wr_ptr and fifo_rd_ptr must be at the same offset in RemoteSenderCBInterface and RemoteReceiverCBInterface");
static_assert(
    offsetof(RemoteSenderCBInterface, config_ptr) == offsetof(RemoteReceiverCBInterface, config_ptr),
    "config_ptr must be at the same offset in RemoteSenderCBInterface and RemoteReceiverCBInterface");

struct LocalCBInterface {
    uint32_t fifo_size;
    uint32_t fifo_limit;  // range is inclusive of the limit
    uint32_t fifo_page_size;
    uint32_t fifo_num_pages;

    uint32_t fifo_rd_ptr;
    uint32_t fifo_wr_ptr;

    // Save a cycle during init by writing 0 to the uint32 below
    union {
        uint32_t tiles_acked_received_init;
        struct {
            uint16_t tiles_acked;
            uint16_t tiles_received;
        };
    };

    // used by packer for in-order packing
    uint32_t fifo_wr_tile_ptr;
};

struct CBInterface {
    union {
        LocalCBInterface local_cb_interface;
        RemoteSenderCBInterface remote_sender_cb_interface;
        RemoteReceiverCBInterface remote_receiver_cb_interface;
    };
};

// Named this way for compatibility with existing code where existing code references local_cb_interface as cb_interface
#ifdef ARCH_QUASAR
extern thread_local CBInterface cb_interface[NUM_CIRCULAR_BUFFERS];
#else
extern CBInterface cb_interface[NUM_CIRCULAR_BUFFERS];
#endif

FORCE_INLINE LocalCBInterface& get_local_cb_interface(uint32_t cb_id) { return cb_interface[cb_id].local_cb_interface; }

FORCE_INLINE RemoteSenderCBInterface& get_remote_sender_cb_interface(uint32_t cb_id) {
    return cb_interface[cb_id].remote_sender_cb_interface;
}

FORCE_INLINE RemoteReceiverCBInterface& get_remote_receiver_cb_interface(uint32_t cb_id) {
    return cb_interface[cb_id].remote_receiver_cb_interface;
}

__attribute__((noinline)) inline bool cb_access_within_bounds(
    uint32_t cb_id, uint32_t start_tile_index, uint32_t num_tiles) {
    const auto& cb = get_local_cb_interface(cb_id);
    return cb.fifo_rd_ptr + (start_tile_index + num_tiles) * cb.fifo_page_size <= cb.fifo_limit;
}

#if defined(COMPILE_FOR_TRISC)
constexpr uint32_t cb_addr_shift = CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT;
#else
constexpr uint32_t cb_addr_shift = 0;
#endif
