// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstddef>
#include "internal/risc_attribs.h"
#include "hostdev/dev_msgs.h"

// Sentinel: no local DFB relay registered for this CrossNodeDFB receiver slot.
constexpr static uint8_t RELAY_DFB_INVALID = 0xFF;

// Pack/unpack helpers for CrossNodeSenderDFBInterface::num_receivers_and_remote_pages_sent_ptr.
// Same bit layout as Remote CB (REMOTE_CB_PACKED_* in hostdev/dev_msgs.h).
// bits [31:24] = num_receivers; bits [23:0] = remote pages_sent base address.
inline constexpr uint32_t cross_node_dfb_num_receivers(uint32_t packed) {
    return packed >> REMOTE_CB_PACKED_COUNT_SHIFT;
}
inline constexpr uint32_t cross_node_dfb_remote_pages_sent_ptr(uint32_t packed) {
    return packed & REMOTE_CB_PACKED_ADDR_MASK;
}
inline constexpr uint32_t cross_node_dfb_pack(uint32_t num_receivers, uint32_t remote_pages_sent_ptr) {
    return (num_receivers << REMOTE_CB_PACKED_COUNT_SHIFT) | (remote_pages_sent_ptr & REMOTE_CB_PACKED_ADDR_MASK);
}

// Device working copies for CrossNodeDFB. Layout of the shared prefix matches Remote CB so
// credit helpers can reinterpret_cast; CrossNode owns these types so it can diverge (relays
// live on the receiver only). fifo_page_size is entry_size; pages_sent/acked are entry credits.

struct CrossNodeSenderDFBInterface {
    uint32_t config_ptr;
    uint32_t fifo_start_addr;
    uint32_t fifo_limit_page_aligned;
    uint32_t fifo_page_size;

    // Unused by the CrossNode sender: each receiver's write cursor is derived from that
    // receiver's local entries_sent counter (credits reset to zero every launch, so
    // sent % ring_units is the next free slot). The field is kept because the RemoteCB
    // overlay aliases it with the receiver's fifo_rd_ptr.
    uint32_t fifo_wr_ptr;

    // Address of receiver NOC XY table: x0, y0, x1, y1, ...
    uint32_t receiver_noc_xy_ptr;

    // Base of local pages_sent / pages_acked pairs (L1_ALIGNMENT stride per receiver).
    uint32_t aligned_pages_sent_ptr;

    // Packed: bits [23:0] = remote pages_sent base; bits [31:24] = num_receivers.
    uint32_t num_receivers_and_remote_pages_sent_ptr;
};
static_assert(sizeof(CrossNodeSenderDFBInterface) == 32);

struct CrossNodeReceiverDFBInterface {
    uint32_t config_ptr;
    uint32_t fifo_start_addr;
    uint32_t fifo_limit_page_aligned;
    uint32_t fifo_page_size;

    uint32_t fifo_rd_ptr;

    uint16_t sender_noc_x;
    uint16_t sender_noc_y;

    uint32_t aligned_pages_acked_ptr;

    // Address on the sender's L1 where this receiver's pages_acked NOC inc lands.
    uint32_t remote_pages_acked_ptr;

    // Receiver-only: one local DFB that shares this CrossNode FIFO for TRISC.
    uint8_t relay_id;  // RELAY_DFB_INVALID if none
    uint8_t pad[3];
};
static_assert(sizeof(CrossNodeReceiverDFBInterface) == 36);

// Shared prefix must overlay for the sender/receiver union (same as Remote CB).
static_assert(
    offsetof(CrossNodeSenderDFBInterface, fifo_start_addr) == offsetof(CrossNodeReceiverDFBInterface, fifo_start_addr));
static_assert(
    offsetof(CrossNodeSenderDFBInterface, fifo_limit_page_aligned) ==
    offsetof(CrossNodeReceiverDFBInterface, fifo_limit_page_aligned));
static_assert(
    offsetof(CrossNodeSenderDFBInterface, fifo_wr_ptr) == offsetof(CrossNodeReceiverDFBInterface, fifo_rd_ptr));
static_assert(offsetof(CrossNodeSenderDFBInterface, config_ptr) == offsetof(CrossNodeReceiverDFBInterface, config_ptr));

// Same overlay pattern as CBInterface for Remote CB: a core is sender XOR receiver for a
// given remote_dfb_id.
struct CrossNodeDFBInterface {
    union {
        CrossNodeSenderDFBInterface sender;
        CrossNodeReceiverDFBInterface receiver;
    };
};
static_assert(sizeof(CrossNodeDFBInterface) == sizeof(CrossNodeReceiverDFBInterface));
