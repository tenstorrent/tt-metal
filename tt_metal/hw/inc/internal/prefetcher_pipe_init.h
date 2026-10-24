// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "internal/cross_node_dfb_interface.h"
#include "internal/circular_buffer_interface.h"
#ifdef ARCH_QUASAR
#include "internal/tt-2xx/dataflow_buffer/dataflow_buffer_interface.h"
#endif
#include "api/alignment.h"
#include "api/debug/assert.h"
#include "internal/risc_attribs.h"
#include "hostdev/dev_msgs.h"
#include "hostdev/remote_dfb_constants.h"
#include "hostdev/remote_dfb_config_layout.h"

namespace experimental {

FORCE_INLINE volatile tt_l1_ptr uint32_t* prefetcher_pipe_config_word_ptr(
    volatile tt_l1_ptr uint32_t* l1_config, uint32_t word_idx) {
#ifdef ARCH_QUASAR
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        reinterpret_cast<uintptr_t>(l1_config + word_idx) + MEM_L1_UNCACHED_BASE);
#else
    return l1_config + word_idx;
#endif
}

FORCE_INLINE uint32_t load_prefetcher_pipe_config_word(volatile tt_l1_ptr uint32_t* l1_config, uint32_t word_idx) {
    return *prefetcher_pipe_config_word_ptr(l1_config, word_idx);
}

FORCE_INLINE void store_prefetcher_pipe_config_word(
    volatile tt_l1_ptr uint32_t* l1_config, uint32_t word_idx, uint32_t value) {
    *prefetcher_pipe_config_word_ptr(l1_config, word_idx) = value;
}

// Populate a kernel-owned PrefetcherPipe interface from
// [config_page_addr, entry_size, relay_dfb_id].
// Loads fifo_wr/rd_ptr from PREFETCHER_PIPE_CFG_FIFO_PTR_CHECKPOINT (durable
// sender wr / receiver rd cursor stored by commit()). `known_sender` is set when the caller already
// knows the page is a sender's, so the receiver setup compiles out.
template <bool known_sender = false>
FORCE_INLINE void setup_prefetcher_pipe_interface(
    CrossNodeDFBInterface& interface, uint32_t config_page_addr, uint32_t entry_size_word, uint32_t relay_dfb_id_word) {
    ASSERT(config_page_addr != 0);
    ASSERT(entry_size_word != 0);
    ASSERT(entry_size_word % REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE == 0);

    const uint32_t entry_size = entry_size_word;
    const uint32_t config_page_ptr = config_page_addr;

    volatile tt_l1_ptr uint32_t* l1_config = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(config_page_ptr);

    const bool is_sender =
        known_sender || static_cast<bool>(load_prefetcher_pipe_config_word(l1_config, REMOTE_DFB_CFG_IS_SENDER));
    const uint32_t num_receivers = load_prefetcher_pipe_config_word(l1_config, REMOTE_DFB_CFG_NUM_RECEIVERS);
    const uint32_t fifo_start_addr = load_prefetcher_pipe_config_word(l1_config, REMOTE_DFB_CFG_FIFO_START);
    const uint32_t fifo_size = load_prefetcher_pipe_config_word(l1_config, REMOTE_DFB_CFG_FIFO_SIZE);
    const uint32_t fifo_ptr_checkpoint =
        load_prefetcher_pipe_config_word(l1_config, PREFETCHER_PIPE_CFG_FIFO_PTR_CHECKPOINT);
    const uint32_t noc_xy_addr =
        config_page_ptr + load_prefetcher_pipe_config_word(l1_config, PREFETCHER_PIPE_CFG_NOC_XY_OFFSET);
    // Sender page: block bases. Receiver page: this receiver's lane-0 slot in each block. Both
    // are local; where the peer's mirror counters sit comes from word[9] below.
    const uint32_t sent_ptr =
        config_page_ptr + load_prefetcher_pipe_config_word(l1_config, PREFETCHER_PIPE_CFG_PAGES_SENT_OFFSET);
    const uint32_t acked_ptr =
        config_page_ptr + load_prefetcher_pipe_config_word(l1_config, PREFETCHER_PIPE_CFG_PAGES_ACKED_OFFSET);
    // Active lane count (kernel-config slot) is cached by the PrefetcherPipe ctor, not the iface.

    const uint32_t size_aligned = fifo_size - (fifo_size % entry_size);
    const uint32_t fifo_limit = fifo_start_addr + size_aligned;

    if (is_sender) {
        CrossNodeSenderDFBInterface& iface = interface.sender;
        iface.config_ptr = config_page_ptr;
        iface.fifo_start_addr = fifo_start_addr;
        iface.fifo_page_size = entry_size;
        // Receiver 0's checkpoint, for parity with the shared interface. A sender addresses
        // through the per-receiver cursors in the credit slots, not through this field.
        iface.fifo_wr_ptr = fifo_ptr_checkpoint;
        iface.receiver_noc_xy_ptr = noc_xy_addr;
        iface.aligned_pages_sent_ptr = sent_ptr;    // local, cached stores
        iface.aligned_pages_acked_ptr = acked_ptr;  // receivers' NoC atomics
        // Receivers' SENT block base, page-relative to this page (word[9]); every receiver page
        // sits at one address. A worker sender's page shares that address, a DRAM sender's does not.
        iface.num_receivers_and_remote_pages_sent_ptr = cross_node_dfb_pack(
            num_receivers,
            config_page_ptr + load_prefetcher_pipe_config_word(l1_config, PREFETCHER_PIPE_CFG_PEER_COUNTER_OFFSET));
        iface.fifo_limit_page_aligned = fifo_limit;
    } else {
        volatile tt_l1_ptr uint32_t* xy = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(noc_xy_addr);
        const uint32_t sender_noc_x = xy[0];
        const uint32_t sender_noc_y = xy[1];

        CrossNodeReceiverDFBInterface& iface = interface.receiver;
        iface.config_ptr = config_page_ptr;
        iface.fifo_start_addr = fifo_start_addr;
        iface.fifo_page_size = entry_size;
        iface.fifo_rd_ptr = fifo_ptr_checkpoint;
        iface.sender_noc_x = static_cast<uint16_t>(sender_noc_x);
        iface.sender_noc_y = static_cast<uint16_t>(sender_noc_y);
        // Lane 0 slots; PrefetcherPipe ctor offsets all three by tid * L1_ALIGNMENT when lanes > 1.
        iface.aligned_pages_sent_ptr = sent_ptr;    // sender's NoC atomics
        iface.aligned_pages_acked_ptr = acked_ptr;  // local, cached stores
        // Ack target on the sender core, page-relative to this page (word[9]). A DRAM sender keeps
        // its page in DRISC L1 rather than at this address, so the target is not simply acked_ptr.
        iface.remote_pages_acked_ptr =
            config_page_ptr + load_prefetcher_pipe_config_word(l1_config, PREFETCHER_PIPE_CFG_PEER_COUNTER_OFFSET);
        iface.fifo_limit_page_aligned = fifo_limit;
        // Low byte only; the slot word also carries the active lane count (remote_dfb_constants.h).
        iface.relay_id = static_cast<uint8_t>(prefetcher_pipe_slot_relay_id(relay_dfb_id_word));
    }
}

// The relay's own entry (page) size in bytes, as firmware configured it from the relay's DFB config.
FORCE_INLINE uint32_t prefetcher_pipe_relay_page_size(uint32_t relay_dfb_id) {
#ifdef ARCH_QUASAR
    return get_local_dfb_interface(relay_dfb_id).entry_size << cb_addr_shift;
#else
    return get_local_cb_interface(relay_dfb_id).fifo_page_size << cb_addr_shift;
#endif
}

// Relay pages per pipe entry: how many of the relay's pages one pipe entry holds. The Program fixes
// it as the relayed pipe's entry size over the relay's entry size, so it is read at launch, while
// the relay still carries the page size firmware set from the relay's config. 1 when the relay
// pages exactly like the pipe; more when a consumer pages one entry finer (a K-block as tiles, or
// one relay entry per consumer).
FORCE_INLINE uint32_t prefetcher_pipe_relay_pages_per_entry(uint32_t relay_dfb_id, uint32_t pipe_entry_size) {
    const uint32_t page_size = prefetcher_pipe_relay_page_size(relay_dfb_id);
    ASSERT(page_size != 0);
    ASSERT(pipe_entry_size % page_size == 0);
    return pipe_entry_size / page_size;
}

#ifdef ARCH_QUASAR
// How far apart a relay counter's operations start. Each operation covers `per_op` entries and then
// the counter jumps, so operation starts repeat every (per_op - 1) * stride + jump. On a STRIDED
// ring that is just the stride. `share` is this thread's entries per operation; when one operation
// is split across the thread's counters, each counter takes a slice of it.
FORCE_INLINE uint32_t
relay_counter_period(const LocalDFBInterface& local, uint32_t share, uint32_t stride, uint32_t jump) {
    const uint32_t per_op = local.split_tc != 0 ? share / local.num_tcs_to_rr : share;
    return (per_op - 1u) * stride + jump;
}

// The first operation start at or after `cursor`, for a counter that starts at `base` and repeats
// every `period`. Units are whatever the caller uses (entries on TRISC, bytes on DM).
FORCE_INLINE uint32_t relay_counter_next(uint32_t base, uint32_t cursor, uint32_t period) {
    if (cursor <= base) {
        return base;
    }
    return base + ((cursor - base + period - 1u) / period) * period;
}

#if !defined(COMPILE_FOR_TRISC)
// Re-paging a bound relay (set_receiver_entry_size) keeps its block layout: byte values (counter
// bases, stride, jump) scale with the new page size, entry counts stay the same. Each counter's
// window is one ring length from its base, so the window is moved, not scaled.
FORCE_INLINE void rescale_relay_counter_geometry(
    LocalDFBInterface& local,
    uint32_t ring_start,
    uint32_t ring_bytes,
    uint32_t old_entry_size,
    uint32_t new_entry_size) {
    ASSERT(old_entry_size != 0);
    auto scale = [&](uint32_t bytes) { return (bytes / old_entry_size) * new_entry_size; };
    local.stride_size = scale(local.stride_size);
    local.jump = scale(local.jump);
    for (uint8_t t = 0; t < local.num_tcs_to_rr; ++t) {
        DFBTCSlot& slot = local.tc_slots[t];
        slot.base_addr = ring_start + scale(slot.base_addr - ring_start);
        slot.limit = slot.base_addr + ring_bytes;
    }
}
#endif
#endif  // ARCH_QUASAR

#ifndef ARCH_QUASAR

// Point a relay's local CB at a PrefetcherPipe's ring window [fifo_start_addr, fifo_limit_addr) with
// its cursor at fifo_ptr, paged at relay_page_size. All arguments are bytes. relay_page_size divides
// the pipe entry and the window is whole pipe entries, so the window is whole relay pages.
FORCE_INLINE void set_local_relay_cb_to_pipe_window(
    LocalCBInterface& local,
    uint32_t fifo_start_addr,
    uint32_t fifo_limit_addr,
    uint32_t fifo_ptr,
    uint32_t pipe_entry_size,
    uint32_t relay_page_size) {
    const uint32_t fifo_limit_units = fifo_limit_addr >> cb_addr_shift;
    const uint32_t fifo_size_units = fifo_limit_units - (fifo_start_addr >> cb_addr_shift);
    const uint32_t fifo_ptr_units = fifo_ptr >> cb_addr_shift;
    const uint32_t page_size_units = relay_page_size >> cb_addr_shift;
    ASSERT(page_size_units != 0);
    ASSERT(pipe_entry_size % relay_page_size == 0);
    ASSERT(fifo_size_units % (pipe_entry_size >> cb_addr_shift) == 0);
    local.fifo_limit = fifo_limit_units;
    local.fifo_size = fifo_size_units;
    local.fifo_page_size = page_size_units;
    local.fifo_num_pages = fifo_size_units / page_size_units;
    local.fifo_wr_ptr = fifo_ptr_units;
    local.fifo_rd_ptr = fifo_ptr_units;
}
#endif

// Seed a borrowed local DFB iface from PREFETCHER_PIPE_CFG_FIFO_PTR_CHECKPOINT
// (durable receiver rd cursor) + this-launch entry_size. Local snap only — same
// address math as receiver resize without credit fixup.
//
// Called from align_local_dfb_to_prefetcher_pipe_slot.
// Not a public compute API (TRISC kernels consume via DataflowBuffer /
// RelayDFBBindingToken after this snap).
FORCE_INLINE void align_local_dfb_to_prefetcher_pipe_checkpoint(
    uint32_t relay_dfb_id, uint32_t config_page_addr, uint32_t entry_size, uint32_t share) {
    ASSERT(config_page_addr != 0);
    ASSERT(entry_size != 0);
    ASSERT(entry_size % REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE == 0);

    volatile tt_l1_ptr uint32_t* l1_config = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(config_page_addr);
    const uint32_t fifo_start_addr = load_prefetcher_pipe_config_word(l1_config, REMOTE_DFB_CFG_FIFO_START);
    const uint32_t fifo_size = load_prefetcher_pipe_config_word(l1_config, REMOTE_DFB_CFG_FIFO_SIZE);
    const uint32_t fifo_ptr_checkpoint =
        load_prefetcher_pipe_config_word(l1_config, PREFETCHER_PIPE_CFG_FIFO_PTR_CHECKPOINT);

    const uint32_t cb_size_page_aligned = fifo_size - (fifo_size % entry_size);
    const uint32_t fifo_limit_page_aligned = fifo_start_addr + cb_size_page_aligned;
    uint32_t next_fifo_rd_ptr = fifo_start_addr + align(fifo_ptr_checkpoint - fifo_start_addr, entry_size);
    if (next_fifo_rd_ptr >= fifo_limit_page_aligned) {
        next_fifo_rd_ptr = fifo_start_addr;
    }

#ifdef ARCH_QUASAR
    // Quasar relay DFB state lives in LocalDFBInterface / TC slots (not LocalCBInterface). The relay
    // keeps the entry firmware set from its config, one pipe entry or a whole fraction of one, and so
    // does its TC geometry (stride, per-TC base/limit): the host sized both over the pipe's whole
    // entries. Only each TC's cursor moves, to the first relay entry it owns at or after the
    // checkpoint, and the TC that owns the checkpoint entry goes next.
    LocalDFBInterface& local = get_local_dfb_interface(relay_dfb_id);
    ASSERT(local.num_tcs_to_rr >= 1);
    const uint32_t relay_entry_size = prefetcher_pipe_relay_page_size(relay_dfb_id);
    ASSERT(relay_entry_size != 0);
    ASSERT(entry_size % relay_entry_size == 0);
    const uint32_t window_entries = cb_size_page_aligned / relay_entry_size;
    const uint32_t cursor_entry = (next_fifo_rd_ptr - fifo_start_addr) / relay_entry_size;
#if defined(COMPILE_FOR_TRISC) && (defined(UCK_CHLKC_UNPACK) || defined(UCK_CHLKC_PACK))
    // A STRIDED consumer hart has num_tcs_to_rr==1 but a non-zero base_entry_idx that must be kept.
    ASSERT(local.stride_size_tiles != 0);
    local.stride_size = static_cast<uint16_t>(local.entry_size * local.stride_size_tiles);
    local.num_entries = static_cast<uint16_t>(window_entries);
    // A BLOCKED relay moves whole blocks, so the checkpoint lands on a block boundary.
    ASSERT(local.block_size <= 1 || cursor_entry % local.block_size == 0);
    // cursor_entry is measured from the ring start, but this thread's entry indices start at its own
    // first counter window, which the host placed first_entry entries into the ring. Convert the
    // cursor to that index space first. Counter bases are in tile units here, like entry_size.
    const uint32_t first_entry = (local.tc_slots[0].base_addr - (fifo_start_addr >> cb_addr_shift)) / local.entry_size;
    ASSERT(first_entry < window_entries);
    const uint32_t local_cursor = cursor_entry > first_entry ? cursor_entry - first_entry : 0u;
    const uint32_t local_window = window_entries - first_entry;
    // Move each counter to its first operation at or after the checkpoint; the earliest one goes next.
    const uint32_t period = relay_counter_period(local, share, local.stride_size_tiles, local.jump);
    uint32_t first_next = 0xFFFFFFFFu;
    local.tc_idx = 0;
    for (uint8_t t = 0; t < local.num_tcs_to_rr; ++t) {
        DFBTCSlot& slot = local.tc_slots[t];
        const uint32_t next = relay_counter_next(slot.base_entry_idx, local_cursor, period);
        if (next < first_next) {
            first_next = next;
            local.tc_idx = t;
        }
        const uint32_t idx = next < local_window ? next : slot.base_entry_idx;
        slot.cursor_entry_idx = static_cast<uint16_t>(idx);
    }
#elif !defined(COMPILE_FOR_TRISC)
    // DM: ptrs/limits are raw bytes; entry_size is address units (cb_addr_shift==0 → bytes).
    local.num_entries = static_cast<uint16_t>(window_entries);
    local.tc_idx = 0;
    if (local.num_tcs_to_rr == 1) {
        local.stride_size = relay_entry_size;  // 1-entry stride in bytes
        local.jump = local.stride_size;        // plain ring: the DM cursor jump equals the stride
        DFBTCSlot& slot = local.tc_slots[0];
        slot.base_addr = fifo_start_addr;
        slot.limit = fifo_limit_page_aligned;
        slot.ptr = next_fifo_rd_ptr;
    } else {
        ASSERT(local.stride_size != 0);
        ASSERT(local.stride_size % relay_entry_size == 0);
        for (uint8_t t = 0; t < local.num_tcs_to_rr; ++t) {
            DFBTCSlot& slot = local.tc_slots[t];
            uint32_t ptr = slot.base_addr;
            if (next_fifo_rd_ptr > slot.base_addr) {
                const uint32_t delta = next_fifo_rd_ptr - slot.base_addr;
                const uint32_t aligned = ((delta + local.stride_size - 1) / local.stride_size) * local.stride_size;
                ptr = slot.base_addr + aligned;
                if (ptr >= slot.limit) {
                    ptr = slot.base_addr;
                }
            }
            if (ptr == next_fifo_rd_ptr) {
                local.tc_idx = t;
            }
            slot.ptr = ptr;
        }
    }
#endif
    // TRISC MATH owns no relay state; the DM path works in byte pointers rather than entry indices.
    (void)window_entries;
    (void)cursor_entry;
    (void)share;
#else
    // Launch-time snap: only the window and cursor move. The relay keeps the page firmware set from its
    // config, one pipe entry or a whole fraction of one.
    LocalCBInterface& local = get_local_cb_interface(relay_dfb_id);
    set_local_relay_cb_to_pipe_window(
        local,
        fifo_start_addr,
        fifo_limit_page_aligned,
        next_fifo_rd_ptr,
        entry_size,
        local.fifo_page_size << cb_addr_shift);
    (void)share;
#endif
}

// launch-msg lookup + checkpoint snap for a PrefetcherPipe relay local DFB.
// Called from the DataflowBuffer(RelayDFBBindingToken) constructor on TRISC:
// the token bakes prefetcher_pipe_id at compile time, so this indexes the dense
// launch-msg persistent region directly and snaps the borrowed local iface to the
// durable checkpoint using this launch's [config_page_addr, entry_size] from the slot.
FORCE_INLINE void align_local_dfb_to_prefetcher_pipe_slot(
    uint32_t relay_dfb_id, uint32_t prefetcher_pipe_id, uint32_t share) {
    const uint32_t launch_index = *GET_MAILBOX_ADDRESS_DEV(launch_msg_rd_ptr);
    const auto* launch_msg = GET_MAILBOX_ADDRESS_DEV(launch[launch_index]);
    const auto& kernel_config = launch_msg->kernel_config;
    ASSERT(kernel_config.prefetcher_pipe_offset != REMOTE_DFB_OFFSET_NONE);

    const uint32_t kernel_config_base = kernel_config.kernel_config_base[PROGRAMMABLE_CORE_TYPE];
    volatile tt_l1_ptr uint32_t* region =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kernel_config_base + kernel_config.prefetcher_pipe_offset);
    // Host rewrites this dense region every launch; Quasar must read via the uncached alias.
    ASSERT(prefetcher_pipe_id < load_prefetcher_pipe_config_word(region, 0));

    volatile tt_l1_ptr uint32_t* slot =
        region + REMOTE_DFB_REGION_HEADER_WORDS + prefetcher_pipe_id * UINT32_WORDS_PER_REMOTE_DFB_CONFIG;
    const uint32_t config_page_addr = load_prefetcher_pipe_config_word(slot, 0);
    const uint32_t entry_size = load_prefetcher_pipe_config_word(slot, 1);
    // The host registered this DFB as the slot's relay (DFBAdvancedOptions::prefetcher_pipe_relays);
    // a mismatched token fails here. The slot word also holds the lane count, so compare only the
    // relay-id byte.
    ASSERT(prefetcher_pipe_slot_relay_id(load_prefetcher_pipe_config_word(slot, 2)) == relay_dfb_id);
    align_local_dfb_to_prefetcher_pipe_checkpoint(relay_dfb_id, config_page_addr, entry_size, share);
}

#if defined(KERNEL_BUILD) && !defined(COMPILE_FOR_TRISC)

// Align DM's private local relay-DFB iface to the PrefetcherPipe receiver's
// rd_ptr / page size / limit. Called from PrefetcherPipe::bind_relay() and from
// set_receiver_entry_size() so a mid-kernel page-size change does not leave the
// local CB/DFB on the old stride. No NOC — copies from the receiver iface.
// relay_pages_per_entry (see prefetcher_pipe_relay_pages_per_entry) sets the relay's page to
// the pipe entry divided by it. `share` is how many relay entries this thread pushes at a time.
FORCE_INLINE void align_local_dfb_to_prefetcher_pipe_receiver_iface(
    uint32_t relay_dfb_id, const CrossNodeReceiverDFBInterface& iface, uint32_t relay_pages_per_entry, uint32_t share) {
#ifdef ARCH_QUASAR
    LocalDFBInterface& local = get_local_dfb_interface(relay_dfb_id);
    ASSERT(local.num_tcs_to_rr >= 1);
    const uint32_t entry_size = iface.fifo_page_size;
    ASSERT(entry_size != 0);
    ASSERT(entry_size % REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE == 0);
    ASSERT(relay_pages_per_entry != 0);
    ASSERT(entry_size % relay_pages_per_entry == 0);
    const uint32_t relay_entry_size = entry_size / relay_pages_per_entry;
    const uint32_t ring_bytes = iface.fifo_limit_page_aligned - iface.fifo_start_addr;
    ASSERT(ring_bytes % entry_size == 0);
    // Several TCs keep the stride and per-TC base/limit DFB init set for the relay's configured
    // entry, so they cannot follow a mid-kernel change of it.
    ASSERT(local.num_tcs_to_rr == 1 || prefetcher_pipe_relay_page_size(relay_dfb_id) == relay_entry_size);
    // DM: ptrs are raw bytes; entry_size field is address units (shift 0 → bytes).
    const uint32_t old_entry_size = local.entry_size;
    local.entry_size = relay_entry_size;
    local.num_entries = static_cast<uint16_t>(ring_bytes / relay_entry_size);
    local.tc_idx = 0;
    const bool is_plain_ring = local.num_tcs_to_rr == 1 && local.block_size <= 1 &&
                               local.stride_size == old_entry_size && local.jump == old_entry_size;
    if (is_plain_ring) {
        local.stride_size = relay_entry_size;
        local.jump = local.stride_size;  // plain ring: the DM cursor jump equals the stride
        DFBTCSlot& slot = local.tc_slots[0];
        slot.base_addr = iface.fifo_start_addr;
        slot.limit = iface.fifo_limit_page_aligned;
        slot.ptr = iface.fifo_rd_ptr;
    } else {
        if (relay_entry_size != old_entry_size) {
            // A re-page must keep this relay's layout, so scale its byte values instead of
            // resetting it to a plain ring.
            rescale_relay_counter_geometry(local, iface.fifo_start_addr, ring_bytes, old_entry_size, relay_entry_size);
        }
        // Move each counter to its first operation at or after the cursor; the earliest one takes
        // the next push.
        ASSERT(local.stride_size != 0);
        ASSERT(local.stride_size % relay_entry_size == 0);
        const uint32_t period = relay_counter_period(local, share, local.stride_size, local.jump);
        uint32_t first_next = 0xFFFFFFFFu;
        for (uint8_t t = 0; t < local.num_tcs_to_rr; ++t) {
            DFBTCSlot& slot = local.tc_slots[t];
            const uint32_t next = relay_counter_next(slot.base_addr, iface.fifo_rd_ptr, period);
            if (next < first_next) {
                first_next = next;
                local.tc_idx = t;
            }
            slot.ptr = next < slot.limit ? next : slot.base_addr;
        }
    }
#else
    ASSERT(relay_pages_per_entry != 0);
    ASSERT(iface.fifo_page_size % relay_pages_per_entry == 0);
    set_local_relay_cb_to_pipe_window(
        get_local_cb_interface(relay_dfb_id),
        iface.fifo_start_addr,
        iface.fifo_limit_page_aligned,
        iface.fifo_rd_ptr,
        iface.fifo_page_size,
        iface.fifo_page_size / relay_pages_per_entry);
    (void)share;
#endif
}

#endif  // KERNEL_BUILD && !COMPILE_FOR_TRISC

}  // namespace experimental
