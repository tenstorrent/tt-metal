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
// reset_credits=true (BRISC only): also zero local pages_sent/pages_acked in this core's
// config page L1. No NOC traffic — each sender/receiver BRISC clears its own counters.
template <bool reset_credits = false>
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

        // BRISC: zero all pages_sent/pages_acked pairs (N receivers, L1_ALIGNMENT stride).
        if constexpr (reset_credits) {
            volatile tt_l1_ptr uint32_t* cnt_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(aligned_cnt_ptr);
            const uint32_t pair_stride = (2 * L1_ALIGNMENT) / sizeof(uint32_t);
            const uint32_t word_stride = L1_ALIGNMENT / sizeof(uint32_t);
            for (uint32_t i = 0; i < num_receivers; ++i) {
                cnt_ptr[0] = 0;            // pages_sent
                cnt_ptr[word_stride] = 0;  // pages_acked
                cnt_ptr += pair_stride;
            }
        }

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

        // BRISC: zero this receiver's local pages_sent / pages_acked pair.
        if constexpr (reset_credits) {
            volatile tt_l1_ptr uint32_t* sent_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(aligned_cnt_ptr);
            volatile tt_l1_ptr uint32_t* acked_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(aligned_acked_ptr);
            *sent_ptr = 0;
            *acked_ptr = 0;
        }
    }
}

// Populate g_cross_node_*_dfb_interface[0 .. num_cross_node_dfbs-1] from the dense slot
// array (pointer already past the region header word).
// Called by firmware when cross_node_dfb_offset != CROSS_NODE_DFB_OFFSET_NONE.
// reset_credits: true on BRISC (zeros local L1 credit counters); false on NCRISC/TRISC
// (iface / ptr setup only — avoids triple-clearing the same L1 words).
template <bool reset_credits = false>
FORCE_INLINE void setup_cross_node_dfb_interfaces(uint32_t tt_l1_ptr* dfb_l1_base, uint32_t num_cross_node_dfbs) {
    volatile tt_l1_ptr uint32_t* dfb_config_addr = dfb_l1_base;

    for (uint32_t dfb_id = 0; dfb_id < num_cross_node_dfbs; ++dfb_id) {
        setup_one_cross_node_dfb_slot<reset_credits>(dfb_id, dfb_config_addr[0], dfb_config_addr[1]);
        dfb_config_addr += UINT32_WORDS_PER_CROSS_NODE_DFB_CONFIG;
    }
}

#if defined(KERNEL_BUILD) && !defined(COMPILE_FOR_TRISC)
#include "hostdev/dev_msgs.h"
#include "api/dataflow/dataflow_api.h"

// Kernel-side fallback when firmware has not populated g_cross_node_*_dfb_interface[] yet
// (e.g. precompiled firmware predating CrossNodeDFB setup). Reads the 2-word kernel-config
// slot from the active launch message and initializes this slot on first use.
FORCE_INLINE void ensure_cross_node_dfb_initialized(uint8_t dfb_id) {
    // config_ptr is at the same offset in sender and receiver views (unioned).
    if (get_cross_node_sender_dfb_interface(dfb_id).config_ptr != 0) {
        return;
    }

    const uint32_t launch_idx = *GET_MAILBOX_ADDRESS_DEV(launch_msg_rd_ptr);
    tt_l1_ptr launch_msg_t* launch = GET_MAILBOX_ADDRESS_DEV(launch[launch_idx]);
    const uint16_t cross_node_dfb_offset = launch->kernel_config.cross_node_dfb_offset;
    if (cross_node_dfb_offset == CROSS_NODE_DFB_OFFSET_NONE) {
        return;
    }

    const uint32_t kernel_config_base = launch->kernel_config.kernel_config_base[ProgrammableCoreType::TENSIX];
    volatile tt_l1_ptr uint32_t* region =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kernel_config_base + cross_node_dfb_offset);
    const uint32_t num_cross_node_dfbs = region[0];
    if (dfb_id >= num_cross_node_dfbs) {
        return;
    }

    volatile tt_l1_ptr uint32_t* slot = region + 1 + dfb_id * UINT32_WORDS_PER_CROSS_NODE_DFB_CONFIG;

    setup_one_cross_node_dfb_slot</*reset_credits=*/true>(dfb_id, slot[0], slot[1]);
}
#endif  // KERNEL_BUILD && !COMPILE_FOR_TRISC

// Mirror experimental::align_local_cbs_to_remote_cb for CrossNodeDFB relay DFBs.
// TRISC calls this at kernel start so its local CB interface matches the live
// CrossNodeReceiverDFBInterface (populated by firmware from the config page).
// DM-side register_relay_dfb() performs the same alignment on the DM copy only.
template <uint32_t num_local_cbs>
FORCE_INLINE void align_local_cbs_to_cross_node_receiver_dfb(
    uint8_t remote_dfb_id, const uint32_t (&local_cb_indices)[num_local_cbs]) {
    const CrossNodeReceiverDFBInterface& iface = get_cross_node_receiver_dfb_interface(remote_dfb_id);
    uint32_t fifo_limit = iface.fifo_limit_page_aligned >> cb_addr_shift;
    uint32_t fifo_size = fifo_limit - (iface.fifo_start_addr >> cb_addr_shift);
    uint32_t fifo_ptr = iface.fifo_rd_ptr >> cb_addr_shift;
    for (uint32_t i = 0; i < num_local_cbs; ++i) {
        LocalCBInterface& local_cb = get_local_cb_interface(local_cb_indices[i]);
        ASSERT(fifo_size % local_cb.fifo_page_size == 0);
        uint32_t fifo_num_pages = fifo_size / local_cb.fifo_page_size;
        local_cb.fifo_limit = fifo_limit;
        local_cb.fifo_size = fifo_size;
        local_cb.fifo_num_pages = fifo_num_pages;
        local_cb.fifo_wr_ptr = fifo_ptr;
        local_cb.fifo_rd_ptr = fifo_ptr;
    }
}

}  // namespace experimental
