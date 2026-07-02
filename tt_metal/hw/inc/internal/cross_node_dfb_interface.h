// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "internal/risc_attribs.h"
#include "internal/circular_buffer_interface.h"

// Maximum number of CrossNodeDFBs that can be attached to a single program per core.
// Indices are 0-based, ascending (distinct from GlobalCB's descending cb_interface[] scheme).
constexpr static uint32_t MAX_CROSS_NODE_DFBS = 16;

// Number of words per CrossNodeDFB kernel-config entry (config_page_addr + entry_size|flags).
constexpr static uint32_t UINT32_WORDS_PER_CROSS_NODE_DFB_CONFIG = 2;

// Maximum number of local DataflowBuffer relays a receiver can register per CrossNodeDFB.
// Deliberately capped at 2: auto_commit(1) + num_relays(1) + relay_ids[2](2) = 4 bytes,
// exactly one uint32_t — zero padding, flat 64-byte g_cross_node_dfb_metadata[] array.
constexpr static uint32_t MAX_RELAY_DFBS_PER_CROSS_NODE = 2;
constexpr static uint8_t  RELAY_DFB_INVALID = 0xFF;

// Bit flag packed into the entry_size word of the kernel-config entry (word[1]).
// Bit 31 set = auto_commit enabled for this slot; bits[30:0] = actual entry_size.
constexpr static uint32_t CROSS_NODE_DFB_AUTO_COMMIT_FLAG = (1u << 31);
constexpr static uint32_t CROSS_NODE_DFB_ENTRY_SIZE_MASK  = ~CROSS_NODE_DFB_AUTO_COMMIT_FLAG;

// Pack/unpack helpers for CrossNodeSenderDFBInterface::num_receivers_and_remote_pages_sent_ptr.
// Reuse the CB-flavored constants from circular_buffer_interface.h (identical bit layout).
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

// CrossNodeDFB sender/receiver interfaces are layout-identical to the GCB structs once
// access_pattern and sender_rr_idx are removed.  Type-alias rather than duplicate.
// DFB code reads fifo_page_size as entry_size, pages_sent/acked as entries_sent/acked.
using CrossNodeSenderDFBInterface   = RemoteSenderCBInterface;
using CrossNodeReceiverDFBInterface = RemoteReceiverCBInterface;

// Per-slot DFB-specific metadata (auto-commit flag + relay DFB registrations).
// Kept in a separate array so RemoteSender/ReceiverDFBInterface stays 8 words (32 bytes).
//
// Layout: auto_commit(1) + num_relays(1) + relay_ids[2](2) = 4 bytes = 1 uint32_t.
// sizeof == 4 ensures g_cross_node_dfb_metadata[MAX_REMOTE_DFBS] is exactly 64 bytes, no padding.
struct CrossNodeDFBMetadata {
    uint8_t auto_commit;              // 1 = firmware writes back ptr at kernel exit
    uint8_t num_relays;               // number of valid entries in relay_ids[]
    uint8_t relay_ids[MAX_RELAY_DFBS_PER_CROSS_NODE]; // local DFB logical handles; RELAY_DFB_INVALID = unused
};
static_assert(sizeof(CrossNodeDFBMetadata) == 4, "CrossNodeDFBMetadata must be 4 bytes");

// Global arrays, one entry per remote DFB index (indexed by remote_dfb_id).
// Firmware (brisc/ncrisc on WH/BH) calls setup_cross_node_dfb_interfaces() to populate these
// from the kernel config page. register_relay_dfbs() in kernel code populates g_cross_node_dfb_metadata[].
extern CrossNodeSenderDFBInterface   g_cross_node_sender_dfb_interface[MAX_CROSS_NODE_DFBS];
extern CrossNodeReceiverDFBInterface g_cross_node_receiver_dfb_interface[MAX_CROSS_NODE_DFBS];
extern CrossNodeDFBMetadata          g_cross_node_dfb_metadata[MAX_CROSS_NODE_DFBS];

// Internal accessors — use experimental::CrossNodeDFB methods in kernel code instead.
FORCE_INLINE CrossNodeSenderDFBInterface& get_cross_node_sender_dfb_interface(uint32_t dfb_id) {
    return g_cross_node_sender_dfb_interface[dfb_id];
}

FORCE_INLINE CrossNodeReceiverDFBInterface& get_cross_node_receiver_dfb_interface(uint32_t dfb_id) {
    return g_cross_node_receiver_dfb_interface[dfb_id];
}
