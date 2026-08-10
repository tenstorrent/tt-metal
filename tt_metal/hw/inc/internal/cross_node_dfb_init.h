// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "internal/cross_node_dfb_interface.h"
#include "internal/circular_buffer_interface.h"
#include "api/alignment.h"
#include "api/debug/assert.h"
#include "internal/risc_attribs.h"
#include "hostdev/cross_node_dfb_constants.h"

// Config page layout (8-word header, uniform for sender and receiver pages):
//   word[0] = is_sender (1) | is_receiver (0)
//   word[1] = num_receivers
//   word[2] = fifo_start_addr
//   word[3] = fifo_size (entry_size * num_entries)
//   word[4] = fifo_wr/rd checkpoint (reserved for GlobalDFB; CrossNode FW ignores —
//             iface ptrs always init from fifo_start_addr on program init)
//   word[5] = noc_xy_ptr: address of word[8] (NOC XY table for sender; sender_x,y for receiver)
//   word[6] = aligned_pages_sent_ptr:
//               sender page: base of entries_sent/entries_acked array in this config page
//               receiver page: address of this receiver's entries_sent slot
//   word[7] = remote_pages_acked_ptr (receiver's acked counter address on sender's core)
// Sender pages additionally store:
//   words[8..8+2N-1] = NOC XY table: x0,y0,x1,y1,... for N receivers
//   Then entries_sent[i] / entries_acked[i] pairs at L1_ALIGNMENT stride
// Receiver pages additionally store:
//   word[8]  = sender_physical_coord.x
//   word[9]  = sender_physical_coord.y

// 2-word kernel-config entry format per CrossNodeDFB, densely packed after a header word:
//   region at cross_node_dfb_offset:
//     word[0] = num_slots
//     slot i in [0, num_slots): [config_page_addr, entry_size]
// Slot index i matches the remote_dfb_id returned by AttachCrossNodeDFB (0 .. num-1).
// Launch-msg cross_node_dfb_offset == CROSS_NODE_DFB_OFFSET_NONE means no CrossNodeDFBs.

namespace experimental {

// Populate one CrossNodeDFB slot from a kernel-config entry [config_page_ptr, entry_size].
// CrossNode is same-program only: every setup resets fifo ptrs to fifo_start_addr.
FORCE_INLINE void setup_one_cross_node_dfb_slot(uint32_t dfb_id, uint32_t config_page_ptr, uint32_t entry_size_word) {
    if (config_page_ptr == 0) {
        return;
    }

    const uint32_t entry_size = entry_size_word;

    volatile tt_l1_ptr uint32_t* l1_config = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(config_page_ptr);

    const bool is_sender = static_cast<bool>(l1_config[0]);
    const uint32_t num_receivers = l1_config[1];
    const uint32_t fifo_start_addr = l1_config[2];
    const uint32_t fifo_size = l1_config[3];        // entry_size * num_entries
    const uint32_t noc_xy_addr = l1_config[5];      // pointer to word[8] in this config page
    const uint32_t aligned_cnt_ptr = l1_config[6];  // pages_sent or pages_acked base
    const uint32_t remote_cnt_ptr = l1_config[7];   // pages_acked target on sender

    // Derived: largest multiple of entry_size that fits in fifo_size.
    const uint32_t size_aligned = fifo_size - (fifo_size % entry_size);
    const uint32_t fifo_limit = fifo_start_addr + size_aligned;

    // Receiver path clears relay fields below; sender leaves the union's trailing
    // relay bytes unused (sender XOR receiver per slot).
    if (is_sender) {
        CrossNodeSenderDFBInterface& iface = get_cross_node_sender_dfb_interface(dfb_id);
        iface.config_ptr = config_page_ptr;
        iface.fifo_start_addr = fifo_start_addr;
        iface.fifo_page_size = entry_size;
        iface.fifo_wr_ptr = fifo_start_addr;
        iface.receiver_noc_xy_ptr = noc_xy_addr;  // points to word[8] of config page
        iface.aligned_pages_sent_ptr = aligned_cnt_ptr;
        iface.num_receivers_and_remote_pages_sent_ptr = cross_node_dfb_pack(num_receivers, remote_cnt_ptr);
        iface.fifo_limit_page_aligned = fifo_limit;

    } else {
        // Receiver page: sender NOC XY is stored at word[8..9], noc_xy_addr points there.
        volatile tt_l1_ptr uint32_t* xy = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(noc_xy_addr);
        const uint32_t sender_noc_x = xy[0];
        const uint32_t sender_noc_y = xy[1];

        // For receiver: aligned_cnt_ptr = this receiver's entries_sent slot;
        // entries_acked slot is at aligned_cnt_ptr + L1_ALIGNMENT (same as GlobalCB).
        const uint32_t aligned_acked_ptr = aligned_cnt_ptr + L1_ALIGNMENT;

        CrossNodeReceiverDFBInterface& iface = get_cross_node_receiver_dfb_interface(dfb_id);
        iface.config_ptr = config_page_ptr;
        iface.fifo_start_addr = fifo_start_addr;
        iface.fifo_page_size = entry_size;
        iface.fifo_rd_ptr = fifo_start_addr;
        iface.sender_noc_x = static_cast<uint16_t>(sender_noc_x);
        iface.sender_noc_y = static_cast<uint16_t>(sender_noc_y);
        iface.aligned_pages_acked_ptr = aligned_acked_ptr;
        iface.remote_pages_acked_ptr = remote_cnt_ptr;
        iface.fifo_limit_page_aligned = fifo_limit;
        // Relay registration is filled later by register_relay_dfb() from the kernel.
        iface.relay_id = RELAY_DFB_INVALID;

    }
}

// Populate g_cross_node_*_dfb_interface[0 .. num_cross_node_dfbs-1] from the dense slot
// array (pointer already past the region header word).
// Called by firmware when cross_node_dfb_offset != CROSS_NODE_DFB_OFFSET_NONE.
FORCE_INLINE void setup_cross_node_dfb_interfaces(uint32_t tt_l1_ptr* dfb_l1_base, uint32_t num_cross_node_dfbs) {
    volatile tt_l1_ptr uint32_t* dfb_config_addr = dfb_l1_base;

    for (uint32_t dfb_id = 0; dfb_id < num_cross_node_dfbs; ++dfb_id) {
        setup_one_cross_node_dfb_slot(dfb_id, dfb_config_addr[0], dfb_config_addr[1]);
        dfb_config_addr += UINT32_WORDS_PER_CROSS_NODE_DFB_CONFIG;
    }
}

}  // namespace experimental
