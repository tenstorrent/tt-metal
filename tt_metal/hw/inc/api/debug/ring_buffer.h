// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(KERNEL_BUILD) || defined(FW_BUILD)

#include "hostdev/dev_msgs.h"

#if defined(WATCHER_ENABLED) && !defined(WATCHER_DISABLE_RING_BUFFER) && !defined(FORCE_WATCHER_OFF)

// Ring buffer modes:
// - Quasar: MPSC (32-bit atomics, lock-free) - works on both DM (tt-qsr64) and TRISC (tt-qsr32)
// - WH/BH: SPSC

#if defined(ARCH_QUASAR)
#include "internal/tt-2xx/quasar/overlay/overlay_addresses.h"
#include "internal/hw_thread.h"
#include "risc_common.h"

// MPSC ring buffer for Quasar watcher debug
// Strategy: Use cached memory for atomic slot reservation (atomics on RISC-V require cache),
// but write data to uncached memory for immediate host visibility without flush overhead
// Uncached writes are much faster than cache + flush per entry
inline __attribute__((always_inline)) void push_to_ring_buffer(uint32_t val) {
    auto* wrapper = GET_MAILBOX_ADDRESS_DEV(watcher.debug_ring_buf);
    auto* buf = reinterpret_cast<debug_mpsc_ring_buf_msg_t*>(wrapper->data);
    uintptr_t addr = reinterpret_cast<uintptr_t>(buf);

    // Cached for atomics (required by hardware), uncached for data (host visibility)
    auto* cached_buf = (addr >= MEM_L1_UNCACHED_BASE)
                           ? reinterpret_cast<debug_mpsc_ring_buf_msg_t*>(addr - MEM_L1_UNCACHED_BASE)
                           : buf;
    auto* uncached_buf =
        (addr < MEM_L1_UNCACHED_BASE) ? reinterpret_cast<debug_mpsc_ring_buf_msg_t*>(addr + MEM_L1_UNCACHED_BASE) : buf;

    // Atomically claim a slot
    uint32_t pos = __atomic_fetch_add(&cached_buf->head, 1, __ATOMIC_RELAXED);
    uint32_t idx = pos & DEBUG_RING_BUFFER_MPSC_MASK;

    // Write to uncached: immediately visible to host, no flush needed
    uncached_buf->slots[idx].data = val;

    // Publish write_id (host uses this to detect valid vs in-flight entries)
    uint32_t thread_idx = internal_::get_hw_thread_idx();
    uint32_t write_id =
        (thread_idx << DEBUG_RING_BUFFER_MPSC_THREAD_ID_SHIFT) | ((pos + 1) & DEBUG_RING_BUFFER_MPSC_POS_MASK);
    uncached_buf->slots[idx].write_id = write_id;
}

#else  // WH/BH: SPSC ring buffer

inline __attribute__((always_inline)) void push_to_ring_buffer(uint32_t val) {
    auto* wrapper = GET_MAILBOX_ADDRESS_DEV(watcher.debug_ring_buf);
    auto* buf = reinterpret_cast<debug_spsc_ring_buf_msg_t tt_l1_ptr*>(wrapper->data);
    volatile tt_l1_ptr int16_t* curr_ptr = &buf->current_ptr;
    volatile tt_l1_ptr uint16_t* wrapped = &buf->wrapped;
    uint32_t* data = buf->data;

    // Bounds check, set to -1 to wrap since we increment before using
    if (*curr_ptr >= static_cast<int16_t>(DEBUG_RING_BUFFER_ELEMENTS - 1)) {
        *curr_ptr = DEBUG_RING_BUFFER_STARTING_INDEX;
        *wrapped = 1;
    }
    data[++(*curr_ptr)] = val;
}

#endif  // ARCH_QUASAR

#define WATCHER_RING_BUFFER_PUSH(x) push_to_ring_buffer(x)
#else  // !defined(WATCHER_ENABLED)
#define WATCHER_RING_BUFFER_PUSH(x)
#endif  // defined(WATCHER_ENABLED)

#endif  // KERNEL_BUILD || FW_BUILD
