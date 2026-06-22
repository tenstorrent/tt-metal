// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "internal/cross_node_dfb_interface.h"
#include "api/alignment.h"
#include "api/debug/assert.h"
#include "internal/risc_attribs.h"

// Config page layout (8-word header, uniform for sender and receiver pages):
//   word[0] = is_sender (1) | is_receiver (0)
//   word[1] = num_receivers
//   word[2] = fifo_start_addr
//   word[3] = fifo_size (entry_size * num_entries)
//   word[4] = fifo_wr_ptr / fifo_rd_ptr (cross-program checkpoint)
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

// 2-word kernel-config entry format per CrossNodeDFB, densely packed:
//   slot i in [0, num_cross_node_dfbs): [config_page_addr, entry_size | flags]
//   Bit 31 of word[1] = auto_commit flag (firmware writes back ptr at kernel exit).
//   Bits[30:0] of word[1] = entry_size in bytes.
// Slot index i matches the remote_dfb_id returned by AttachCrossNodeDFB (0 .. num-1).

namespace experimental {

// Populate one CrossNodeDFB slot from a kernel-config entry [config_page_ptr, entry_size|flags].
template <bool update_remote_over_noc = false>
FORCE_INLINE void setup_one_cross_node_dfb_slot(
    uint32_t dfb_id,
    uint32_t config_page_ptr,
    uint32_t entry_size_word,
    uint8_t noc,
    uint8_t nm,
    bool posted,
    uint8_t cmd_buf) {
    if (config_page_ptr == 0) {
        return;
    }

    // Extract entry_size (bits[30:0]) and auto_commit flag (bit 31).
    const uint32_t entry_size  = entry_size_word & CROSS_NODE_DFB_ENTRY_SIZE_MASK;
    const bool     auto_commit = (entry_size_word & CROSS_NODE_DFB_AUTO_COMMIT_FLAG) != 0;

    volatile tt_l1_ptr uint32_t* l1_config = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(config_page_ptr);

    const bool     is_sender       = static_cast<bool>(l1_config[0]);
    const uint32_t num_receivers   = l1_config[1];
    const uint32_t fifo_start_addr = l1_config[2];
    const uint32_t fifo_size       = l1_config[3];  // entry_size * num_entries
    const uint32_t fifo_ptr        = l1_config[4];  // wr_ptr or rd_ptr checkpoint
    const uint32_t noc_xy_addr     = l1_config[5];  // pointer to word[8] in this config page
    const uint32_t aligned_cnt_ptr = l1_config[6];  // pages_sent or pages_acked base
    const uint32_t remote_cnt_ptr  = l1_config[7];  // pages_acked target on sender

    // Derived: largest multiple of entry_size that fits in fifo_size.
    const uint32_t size_aligned = fifo_size - (fifo_size % entry_size);
    const uint32_t fifo_limit   = fifo_start_addr + size_aligned;

    // Initialize per-slot metadata (auto_commit; relay fields cleared here, populated
    // later by register_relay_dfbs() from the kernel if needed).
    g_cross_node_dfb_metadata[dfb_id].auto_commit = auto_commit ? 1u : 0u;
    g_cross_node_dfb_metadata[dfb_id].num_relays  = 0;
    for (uint32_t s = 0; s < MAX_RELAY_DFBS_PER_CROSS_NODE; ++s) {
        g_cross_node_dfb_metadata[dfb_id].relay_ids[s] = RELAY_DFB_INVALID;
    }

    if (is_sender) {
        CrossNodeSenderDFBInterface& iface = get_cross_node_sender_dfb_interface(dfb_id);
        iface.config_ptr                             = config_page_ptr;
        iface.fifo_start_addr                        = fifo_start_addr;
        iface.fifo_page_size                         = entry_size;
        iface.fifo_wr_ptr                            = fifo_ptr;
        iface.receiver_noc_xy_ptr                    = noc_xy_addr;  // points to word[8] of config page
        iface.aligned_pages_sent_ptr                 = aligned_cnt_ptr;
        iface.num_receivers_and_remote_pages_sent_ptr =
            cross_node_dfb_pack(num_receivers, remote_cnt_ptr);

        // Align wr_ptr to entry_size boundary and set fifo_limit.
        uint32_t new_wr_ptr = fifo_start_addr + align(fifo_ptr - fifo_start_addr, entry_size);
        if constexpr (update_remote_over_noc) {
            if (new_wr_ptr >= fifo_limit) {
                uint32_t skip_bytes = fifo_start_addr + fifo_size - fifo_ptr;
                uint32_t skip_units = skip_bytes / L1_ALIGNMENT;
                volatile tt_l1_ptr uint32_t* cnt_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(aligned_cnt_ptr);
                volatile tt_l1_ptr uint32_t* xy_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(noc_xy_addr);
                for (uint32_t i = 0; i < num_receivers; ++i) {
                    *cnt_ptr += skip_units;
                    cnt_ptr += (2 * L1_ALIGNMENT) / sizeof(uint32_t);
                    xy_ptr  += 2;
                }
                new_wr_ptr = fifo_start_addr;
            } else if (new_wr_ptr != fifo_ptr) {
                uint32_t skip_units = (new_wr_ptr - fifo_ptr) / L1_ALIGNMENT;
                volatile tt_l1_ptr uint32_t* cnt_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(aligned_cnt_ptr);
                for (uint32_t i = 0; i < num_receivers; ++i) {
                    *cnt_ptr += skip_units;
                    cnt_ptr += (2 * L1_ALIGNMENT) / sizeof(uint32_t);
                }
            }
        } else if (new_wr_ptr >= fifo_limit) {
            new_wr_ptr = fifo_start_addr;
        }
        iface.fifo_wr_ptr              = new_wr_ptr;
        iface.fifo_limit_page_aligned  = fifo_limit;

    } else {
        // Receiver page: sender NOC XY is stored at word[8..9], noc_xy_addr points there.
        volatile tt_l1_ptr uint32_t* xy = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(noc_xy_addr);
        const uint32_t sender_noc_x = xy[0];
        const uint32_t sender_noc_y = xy[1];

        // For receiver: aligned_cnt_ptr = this receiver's entries_sent slot;
        // entries_acked slot is at aligned_cnt_ptr + L1_ALIGNMENT (same as GlobalCB).
        const uint32_t aligned_acked_ptr = aligned_cnt_ptr + L1_ALIGNMENT;

        CrossNodeReceiverDFBInterface& iface = get_cross_node_receiver_dfb_interface(dfb_id);
        iface.config_ptr            = config_page_ptr;
        iface.fifo_start_addr       = fifo_start_addr;
        iface.fifo_page_size        = entry_size;
        iface.fifo_rd_ptr           = fifo_ptr;
        iface.sender_noc_x          = static_cast<uint16_t>(sender_noc_x);
        iface.sender_noc_y          = static_cast<uint16_t>(sender_noc_y);
        iface.aligned_pages_acked_ptr = aligned_acked_ptr;
        iface.remote_pages_acked_ptr  = remote_cnt_ptr;

        uint32_t new_rd_ptr = fifo_start_addr + align(fifo_ptr - fifo_start_addr, entry_size);
        if constexpr (update_remote_over_noc) {
            if (new_rd_ptr >= fifo_limit) {
                uint32_t skip_bytes = fifo_start_addr + fifo_size - fifo_ptr;
                uint32_t skip_units = skip_bytes / L1_ALIGNMENT;
                volatile tt_l1_ptr uint32_t* acked_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(aligned_acked_ptr);
                *acked_ptr += skip_units;
                new_rd_ptr = fifo_start_addr;
            } else if (new_rd_ptr != fifo_ptr) {
                uint32_t skip_units = (new_rd_ptr - fifo_ptr) / L1_ALIGNMENT;
                volatile tt_l1_ptr uint32_t* acked_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(aligned_acked_ptr);
                *acked_ptr += skip_units;
            }
        } else if (new_rd_ptr >= fifo_limit) {
            new_rd_ptr = fifo_start_addr;
        }
        iface.fifo_rd_ptr             = new_rd_ptr;
        iface.fifo_limit_page_aligned = fifo_limit;
    }
}

// Populate g_cross_node_*_dfb_interface[0 .. num_cross_node_dfbs-1] from the dense kernel-config
// region at remote_cross_node_dfb_offset. Called by firmware (brisc.cc / ncrisc.cc on WH/BH),
// guarded by `if (kernel_config->num_cross_node_dfbs != 0)`.
// template param update_remote_over_noc: set true for the BRISC/DM0 role that issues
// the alignment skip atomics; false for subordinate cores that only populate interfaces.
template <bool update_remote_over_noc = false>
FORCE_INLINE void setup_cross_node_dfb_interfaces(
    uint32_t tt_l1_ptr* dfb_l1_base,
    uint32_t num_cross_node_dfbs,
    uint8_t noc,
    uint8_t nm,
    bool posted,
    uint8_t cmd_buf) {

    volatile tt_l1_ptr uint32_t* dfb_config_addr = dfb_l1_base;

    for (uint32_t dfb_id = 0; dfb_id < num_cross_node_dfbs; ++dfb_id) {
        setup_one_cross_node_dfb_slot<update_remote_over_noc>(
            dfb_id, dfb_config_addr[0], dfb_config_addr[1], noc, nm, posted, cmd_buf);
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
    if (get_cross_node_sender_dfb_interface(dfb_id).config_ptr != 0 ||
        get_cross_node_receiver_dfb_interface(dfb_id).config_ptr != 0) {
        return;
    }

    const uint32_t launch_idx = *GET_MAILBOX_ADDRESS_DEV(launch_msg_rd_ptr);
    tt_l1_ptr launch_msg_t* launch = GET_MAILBOX_ADDRESS_DEV(launch[launch_idx]);
    const uint8_t num_cross_node_dfbs = launch->kernel_config.num_cross_node_dfbs;
    if (num_cross_node_dfbs == 0 || dfb_id >= num_cross_node_dfbs) {
        return;
    }

    const uint32_t kernel_config_base =
        launch->kernel_config.kernel_config_base[ProgrammableCoreType::TENSIX];
    const uint32_t remote_cross_node_dfb_offset =
        launch->kernel_config.remote_cross_node_dfb_offset;
    volatile tt_l1_ptr uint32_t* slot = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        kernel_config_base + remote_cross_node_dfb_offset +
        dfb_id * UINT32_WORDS_PER_CROSS_NODE_DFB_CONFIG * sizeof(uint32_t));

    setup_one_cross_node_dfb_slot<true>(
        dfb_id, slot[0], slot[1], 0, 0, true, write_at_cmd_buf);
}
#endif  // KERNEL_BUILD && !COMPILE_FOR_TRISC

// Firmware kernel-exit epilogue (brisc.cc / ncrisc.cc): for each CrossNodeDFB slot with
// auto_commit set, write fifo_wr_ptr (sender) or fifo_rd_ptr (receiver) back to config
// page word[4] for cross-program pointer persistence.
// Called under the existing `if (num_cross_node_dfbs)` guard in brisc.cc/ncrisc.cc.
FORCE_INLINE void commit_auto_dfbs(uint32_t tt_l1_ptr* dfb_l1_base, uint32_t num_cross_node_dfbs) {
    volatile tt_l1_ptr uint32_t* dfb_config_addr = dfb_l1_base;
    for (uint32_t dfb_id = 0; dfb_id < num_cross_node_dfbs; ++dfb_id) {
        const uint32_t config_page_ptr  = dfb_config_addr[0];
        const uint32_t entry_size_word  = dfb_config_addr[1];
        dfb_config_addr += UINT32_WORDS_PER_CROSS_NODE_DFB_CONFIG;

        if (!(entry_size_word & CROSS_NODE_DFB_AUTO_COMMIT_FLAG)) { continue; }

        volatile tt_l1_ptr uint32_t* l1_config =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(config_page_ptr);
        const bool is_sender = static_cast<bool>(l1_config[0]);

        if (is_sender) {
            l1_config[4] = g_cross_node_sender_dfb_interface[dfb_id].fifo_wr_ptr;
        } else {
            l1_config[4] = g_cross_node_receiver_dfb_interface[dfb_id].fifo_rd_ptr;
        }
    }
}

// Mirror experimental::align_local_cbs_to_remote_cb for CrossNodeDFB relay DFBs.
// TRISC calls this at kernel start so its local CB interface matches the live
// CrossNodeReceiverDFBInterface (populated by firmware from the config page).
// DM-side register_relay_dfbs() performs the same alignment on the DM copy only.
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
