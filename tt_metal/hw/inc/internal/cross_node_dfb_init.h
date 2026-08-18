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
#include "hostdev/remote_dfb_constants.h"
#include "hostdev/remote_dfb_config_layout.h"

// Config page layout (8-word CrossNode header, uniform for sender and receiver pages):
//   word[0] = is_sender (1) | is_receiver (0)
//   word[1] = num_receivers
//   word[2] = fifo_start_addr
//   word[3] = fifo_size (entry_size * num_entries)
//   word[4] = fifo_wr/rd checkpoint (reserved; CrossNode ignores —
//             iface ptrs always init from fifo_start_addr on construction)
//   word[5] = noc_xy_offset (page-relative → word[8])
//   word[6] = aligned_pages_sent offset (page-relative)
//   word[7] = remote_pages_acked offset (page-relative; same numeric offset on sender)
// Sender pages additionally store:
//   words[8..8+2N-1] = NOC XY table: x0,y0,x1,y1,... for N receivers
//   Then entries_sent[i] / entries_acked[i] pairs at L1_ALIGNMENT stride.
// Receiver pages additionally store:
//   word[8] = sender_physical_coord.x
//   word[9] = sender_physical_coord.y

// 3-word kernel-config entry format per CrossNodeDFB, densely packed after a header word:
//   region at cross_node_dfb_offset:
//     word[0] = num_slots
//     slot i in [0, num_slots): [config_page_addr, entry_size, relay_dfb_id]
// Slot index i matches the remote_dfb_id returned by CreateCrossNodeDFB (0 .. num-1).
// Launch-msg cross_node_dfb_offset == CROSS_NODE_DFB_OFFSET_NONE means no CrossNodeDFBs.

namespace experimental {

// Populate a kernel-owned CrossNodeDFB interface from
// [config_page_addr, entry_size, relay_dfb_id].
// config_page_addr is the absolute L1 address of the dedicated config Buffer
// (0 means unused sparse slot).
// CrossNode is same-program only: construction resets fifo ptrs to fifo_start_addr.
FORCE_INLINE void setup_cross_node_dfb_interface(
    CrossNodeDFBInterface& interface, uint32_t config_page_addr, uint32_t entry_size_word, uint32_t relay_dfb_id_word) {
    ASSERT(config_page_addr != 0);

    const uint32_t entry_size = entry_size_word;
    const uint32_t config_page_ptr = config_page_addr;

    volatile tt_l1_ptr uint32_t* l1_config = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(config_page_ptr);

    const bool is_sender = static_cast<bool>(l1_config[REMOTE_DFB_CFG_IS_SENDER]);
    const uint32_t num_receivers = l1_config[REMOTE_DFB_CFG_NUM_RECEIVERS];
    const uint32_t fifo_start_addr = l1_config[REMOTE_DFB_CFG_FIFO_START];
    const uint32_t fifo_size = l1_config[REMOTE_DFB_CFG_FIFO_SIZE];  // entry_size * num_entries
    // Words 5–7 are page-relative offsets into this config page.
    const uint32_t noc_xy_addr = config_page_ptr + l1_config[CROSS_NODE_DFB_CFG_NOC_XY_OFFSET];
    const uint32_t aligned_cnt_ptr = config_page_ptr + l1_config[CROSS_NODE_DFB_CFG_PAGES_SENT_OFFSET];
    const uint32_t remote_cnt_ptr = config_page_ptr + l1_config[CROSS_NODE_DFB_CFG_PAGES_ACKED_OFFSET];

    // Derived: largest multiple of entry_size that fits in fifo_size.
    const uint32_t size_aligned = fifo_size - (fifo_size % entry_size);
    const uint32_t fifo_limit = fifo_start_addr + size_aligned;

    // Receiver path clears relay fields below; sender leaves the union's trailing
    // relay bytes unused (sender XOR receiver per slot).
    if (is_sender) {
        CrossNodeSenderDFBInterface& iface = interface.sender;
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

        CrossNodeReceiverDFBInterface& iface = interface.receiver;
        iface.config_ptr = config_page_ptr;
        iface.fifo_start_addr = fifo_start_addr;
        iface.fifo_page_size = entry_size;
        iface.fifo_rd_ptr = fifo_start_addr;
        iface.sender_noc_x = static_cast<uint16_t>(sender_noc_x);
        iface.sender_noc_y = static_cast<uint16_t>(sender_noc_y);
        iface.aligned_pages_acked_ptr = aligned_acked_ptr;
        iface.remote_pages_acked_ptr = remote_cnt_ptr;
        iface.fifo_limit_page_aligned = fifo_limit;
        iface.relay_id = static_cast<uint8_t>(relay_dfb_id_word);
    }
}

}  // namespace experimental
