// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "circular_buffer_constants.h"
#include "risc_attribs.h"

constexpr static std::uint32_t REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE = L1_ALIGNMENT;

struct RemoteSenderCBInterface {
    uint32_t config_ptr;
    uint32_t fifo_start_addr;
    uint32_t fifo_limit_page_aligned;
    uint32_t fifo_page_size;

    uint32_t fifo_wr_ptr;

    // Address to an array of x, y coords
    // arranged x0, y0, x1, y1, ...
    uint32_t receiver_noc_xy_ptr;

    // These point to an array of size num_receivers
    // It's stored as pages_sent, pages_acked pairs for each receiver
    // Each entry is L1 aligned
    uint32_t aligned_pages_sent_ptr;
    uint32_t num_receivers;
};

struct RemoteReceiverCBInterface {
    uint32_t config_ptr;
    uint32_t fifo_start_addr;
    uint32_t fifo_limit_page_aligned;
    uint32_t fifo_page_size;

    uint32_t fifo_rd_ptr;

    uint32_t sender_noc_x;
    uint32_t sender_noc_y;

    // These point to a single entry corresponding to receiver index
    // Each entry is L1 aligned
    uint32_t aligned_pages_acked_ptr;
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
extern CBInterface cb_interface[NUM_CIRCULAR_BUFFERS];

FORCE_INLINE LocalCBInterface& get_local_cb_interface(uint32_t cb_id) { return cb_interface[cb_id].local_cb_interface; }

FORCE_INLINE RemoteSenderCBInterface& get_remote_sender_cb_interface(uint32_t cb_id) {
    return cb_interface[cb_id].remote_sender_cb_interface;
}

FORCE_INLINE RemoteReceiverCBInterface& get_remote_receiver_cb_interface(uint32_t cb_id) {
    return cb_interface[cb_id].remote_receiver_cb_interface;
}

#if defined(COMPILE_FOR_TRISC)
constexpr uint32_t cb_addr_shift = CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT;
#else
constexpr uint32_t cb_addr_shift = 0;
#endif
