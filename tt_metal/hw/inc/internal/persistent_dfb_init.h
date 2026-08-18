// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "internal/cross_node_dfb_interface.h"
#include "internal/circular_buffer_interface.h"
#include "api/alignment.h"
#include "api/debug/assert.h"
#include "internal/risc_attribs.h"
#include "hostdev/dev_msgs.h"
#include "hostdev/remote_dfb_constants.h"
#include "hostdev/remote_dfb_config_layout.h"

namespace experimental {

// Populate a kernel-owned PersistentDFB interface from
// [config_page_addr, entry_size, relay_dfb_id].
// Loads fifo_wr/rd_ptr from PERSISTENT_DFB_CFG_FIFO_PTR_CHECKPOINT (durable
// sender wr / receiver rd cursor stored by commit()).
FORCE_INLINE void setup_persistent_dfb_interface(
    CrossNodeDFBInterface& interface, uint32_t config_page_addr, uint32_t entry_size_word, uint32_t relay_dfb_id_word) {
    ASSERT(config_page_addr != 0);

    const uint32_t entry_size = entry_size_word;
    const uint32_t config_page_ptr = config_page_addr;

    volatile tt_l1_ptr uint32_t* l1_config = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(config_page_ptr);

    const bool is_sender = static_cast<bool>(l1_config[REMOTE_DFB_CFG_IS_SENDER]);
    const uint32_t num_receivers = l1_config[REMOTE_DFB_CFG_NUM_RECEIVERS];
    const uint32_t fifo_start_addr = l1_config[REMOTE_DFB_CFG_FIFO_START];
    const uint32_t fifo_size = l1_config[REMOTE_DFB_CFG_FIFO_SIZE];
    const uint32_t fifo_ptr_checkpoint = l1_config[PERSISTENT_DFB_CFG_FIFO_PTR_CHECKPOINT];
    const uint32_t noc_xy_addr = config_page_ptr + l1_config[PERSISTENT_DFB_CFG_NOC_XY_OFFSET];
    const uint32_t aligned_cnt_ptr = config_page_ptr + l1_config[PERSISTENT_DFB_CFG_PAGES_SENT_OFFSET];
    const uint32_t remote_cnt_ptr = config_page_ptr + l1_config[PERSISTENT_DFB_CFG_PAGES_ACKED_OFFSET];

    const uint32_t size_aligned = fifo_size - (fifo_size % entry_size);
    const uint32_t fifo_limit = fifo_start_addr + size_aligned;

    if (is_sender) {
        CrossNodeSenderDFBInterface& iface = interface.sender;
        iface.config_ptr = config_page_ptr;
        iface.fifo_start_addr = fifo_start_addr;
        iface.fifo_page_size = entry_size;
        iface.fifo_wr_ptr = fifo_ptr_checkpoint;
        iface.receiver_noc_xy_ptr = noc_xy_addr;
        iface.aligned_pages_sent_ptr = aligned_cnt_ptr;
        iface.num_receivers_and_remote_pages_sent_ptr = cross_node_dfb_pack(num_receivers, remote_cnt_ptr);
        iface.fifo_limit_page_aligned = fifo_limit;
    } else {
        volatile tt_l1_ptr uint32_t* xy = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(noc_xy_addr);
        const uint32_t sender_noc_x = xy[0];
        const uint32_t sender_noc_y = xy[1];
        const uint32_t aligned_acked_ptr = aligned_cnt_ptr + L1_ALIGNMENT;

        CrossNodeReceiverDFBInterface& iface = interface.receiver;
        iface.config_ptr = config_page_ptr;
        iface.fifo_start_addr = fifo_start_addr;
        iface.fifo_page_size = entry_size;
        iface.fifo_rd_ptr = fifo_ptr_checkpoint;
        iface.sender_noc_x = static_cast<uint16_t>(sender_noc_x);
        iface.sender_noc_y = static_cast<uint16_t>(sender_noc_y);
        iface.aligned_pages_acked_ptr = aligned_acked_ptr;
        iface.remote_pages_acked_ptr = remote_cnt_ptr;
        iface.fifo_limit_page_aligned = fifo_limit;
        iface.relay_id = static_cast<uint8_t>(relay_dfb_id_word);
    }
}

// Seed a borrowed local DFB iface from PERSISTENT_DFB_CFG_FIFO_PTR_CHECKPOINT
// (durable receiver rd cursor) + this-launch entry_size. Local snap only — same
// address math as receiver resize without credit fixup.
//
// Called from align_local_dfb_to_persistent_slot.
// Not a public compute API (TRISC kernels consume via DataflowBuffer /
// RelayDFBBindingToken after this snap).
FORCE_INLINE void align_local_dfb_to_persistent_checkpoint(
    uint32_t relay_dfb_id, uint32_t config_page_addr, uint32_t entry_size) {
    ASSERT(config_page_addr != 0);
    ASSERT(entry_size != 0);
    ASSERT(entry_size % REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE == 0);

    volatile tt_l1_ptr uint32_t* l1_config = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(config_page_addr);
    const uint32_t fifo_start_addr = l1_config[REMOTE_DFB_CFG_FIFO_START];
    const uint32_t fifo_size = l1_config[REMOTE_DFB_CFG_FIFO_SIZE];
    const uint32_t fifo_ptr_checkpoint = l1_config[PERSISTENT_DFB_CFG_FIFO_PTR_CHECKPOINT];

    const uint32_t cb_size_page_aligned = fifo_size - (fifo_size % entry_size);
    const uint32_t fifo_limit_page_aligned = fifo_start_addr + cb_size_page_aligned;
    uint32_t next_fifo_rd_ptr = fifo_start_addr + align(fifo_ptr_checkpoint - fifo_start_addr, entry_size);
    if (next_fifo_rd_ptr >= fifo_limit_page_aligned) {
        next_fifo_rd_ptr = fifo_start_addr;
    }

    LocalCBInterface& local = get_local_cb_interface(relay_dfb_id);
    const uint32_t fifo_limit = fifo_limit_page_aligned >> cb_addr_shift;
    const uint32_t fifo_size_units = fifo_limit - (fifo_start_addr >> cb_addr_shift);
    const uint32_t fifo_ptr_units = next_fifo_rd_ptr >> cb_addr_shift;
    const uint32_t page_size_units = entry_size >> cb_addr_shift;
    ASSERT(page_size_units != 0);
    ASSERT(fifo_size_units % page_size_units == 0);
    local.fifo_limit = fifo_limit;
    local.fifo_size = fifo_size_units;
    local.fifo_page_size = page_size_units;
    local.fifo_num_pages = fifo_size_units / page_size_units;
    local.fifo_wr_ptr = fifo_ptr_units;
    local.fifo_rd_ptr = fifo_ptr_units;
}

// launch-msg lookup + checkpoint snap for a Persistent relay local DFB.
// Called from the DataflowBuffer(RelayDFBBindingToken) constructor on TRISC:
// the token bakes persistent_dfb_id at compile time, so this indexes the dense
// launch-msg persistent region directly and snaps get_local_cb_interface(relay_dfb_id)
// to the durable checkpoint using this launch's [config_page_addr, entry_size] from the slot.
FORCE_INLINE void align_local_dfb_to_persistent_slot(uint32_t relay_dfb_id, uint32_t persistent_dfb_id) {
    const uint32_t launch_index = *GET_MAILBOX_ADDRESS_DEV(launch_msg_rd_ptr);
    const auto* launch_msg = GET_MAILBOX_ADDRESS_DEV(launch[launch_index]);
    const auto& kernel_config = launch_msg->kernel_config;
    ASSERT(kernel_config.persistent_dfb_offset != PERSISTENT_DFB_OFFSET_NONE);

    const uint32_t kernel_config_base = kernel_config.kernel_config_base[PROGRAMMABLE_CORE_TYPE];
    volatile tt_l1_ptr uint32_t* region =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kernel_config_base + kernel_config.persistent_dfb_offset);
    ASSERT(persistent_dfb_id < region[0]);

    volatile tt_l1_ptr uint32_t* slot =
        region + PERSISTENT_DFB_REGION_HEADER_WORDS + persistent_dfb_id * UINT32_WORDS_PER_PERSISTENT_DFB_CONFIG;
    // Host must have registered this local DFB as the relay for this persistent slot
    // (CreatePersistentRelayDataflowBuffer); catches a mismatched token.
    ASSERT(slot[2] == relay_dfb_id);
    align_local_dfb_to_persistent_checkpoint(relay_dfb_id, /*config_page_addr=*/slot[0], /*entry_size=*/slot[1]);
}

#if defined(KERNEL_BUILD) && !defined(COMPILE_FOR_TRISC)

// Align DM's private local relay-DFB iface to the Persistent receiver's post-ctor
// (post-resize) rd_ptr / page size / limit. Called from PersistentDFB::bind_relay().
// No NOC — copies from the already-hydrated receiver iface into get_local_cb_interface(relay_id).
FORCE_INLINE void align_local_dfb_to_persistent_receiver_iface(
    uint32_t relay_dfb_id, const CrossNodeReceiverDFBInterface& iface) {
    LocalCBInterface& local = get_local_cb_interface(relay_dfb_id);
    const uint32_t fifo_limit = iface.fifo_limit_page_aligned >> cb_addr_shift;
    const uint32_t fifo_size_units = fifo_limit - (iface.fifo_start_addr >> cb_addr_shift);
    const uint32_t fifo_ptr_units = iface.fifo_rd_ptr >> cb_addr_shift;
    const uint32_t page_size_units = iface.fifo_page_size >> cb_addr_shift;
    ASSERT(page_size_units != 0);
    ASSERT(fifo_size_units % page_size_units == 0);
    local.fifo_limit = fifo_limit;
    local.fifo_size = fifo_size_units;
    local.fifo_page_size = page_size_units;
    local.fifo_num_pages = fifo_size_units / page_size_units;
    local.fifo_wr_ptr = fifo_ptr_units;
    local.fifo_rd_ptr = fifo_ptr_units;
}

#endif  // KERNEL_BUILD && !COMPILE_FOR_TRISC

}  // namespace experimental
