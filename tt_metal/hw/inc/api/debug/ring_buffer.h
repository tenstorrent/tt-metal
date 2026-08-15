// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(KERNEL_BUILD) || defined(FW_BUILD)

#include "hostdev/dev_msgs.h"

#if defined(WATCHER_ENABLED) && !defined(WATCHER_DISABLE_RING_BUFFER) && !defined(FORCE_WATCHER_OFF)

// Ring buffer modes:
// - Quasar, Blackhole: MPSC (32-bit atomics, lock-free)
// - WH: SPSC

#if defined(ARCH_QUASAR) || defined(ARCH_BLACKHOLE)
#include "internal/hw_thread.h"
#include "risc_common.h"

// Must be inline - DM stack is only 1KB, can't afford function call overhead
inline __attribute__((always_inline)) void push_to_ring_buffer(uint32_t val) {
    auto* wrapper = GET_MAILBOX_ADDRESS_DEV(watcher.debug_ring_buf);
    auto* buf = reinterpret_cast<debug_mpsc_ring_buf_msg_t*>(wrapper->data);

#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)
    // Remap to cached for atomics
    uintptr_t addr = reinterpret_cast<uintptr_t>(buf);
    if (addr >= MEM_L1_UNCACHED_BASE) {
        buf = reinterpret_cast<debug_mpsc_ring_buf_msg_t*>(addr - MEM_L1_UNCACHED_BASE);
    }
#endif  // ARCH_QUASAR

    // Atomically claim a slot
    uint32_t pos = __atomic_fetch_add(&buf->head, 1, __ATOMIC_RELAXED);
    uint32_t idx = pos & DEBUG_RING_BUFFER_MASK;

    // Write data
    buf->slots[idx].data = val;

    uint32_t thread_idx = internal_::get_hw_thread_idx();
    __atomic_store_n(&buf->slots[idx].write_id, thread_idx + 1, __ATOMIC_RELEASE);

#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)
    // Flush cache line for host visibility
    // TODO: can this be optimized by flushing only after N amount of heads
    flush_l2_cache_line(reinterpret_cast<uintptr_t>(&buf->slots[idx]));
    // Head needs to be flushed separately since it lies on a different cache line
    flush_l2_cache_line(reinterpret_cast<uintptr_t>(&buf->head));
#endif  // ARCH_QUASAR
}

#else  // WH: SPSC ring buffer

inline __attribute__((always_inline)) void push_to_ring_buffer(uint32_t val) {
    auto* wrapper = GET_MAILBOX_ADDRESS_DEV(watcher.debug_ring_buf);
    auto* buf = reinterpret_cast<debug_spsc_ring_buf_msg_t tt_l1_ptr*>(wrapper->data);
    volatile tt_l1_ptr int16_t* curr_ptr = &buf->current_ptr;
    volatile tt_l1_ptr uint16_t* wrapped = &buf->wrapped;
    uint32_t* data = buf->data;

    // Bounds check, set to -1 to wrap since we increment before using
    if (*curr_ptr >= DEBUG_RING_BUFFER_ELEMENTS - 1) {
        *curr_ptr = DEBUG_RING_BUFFER_STARTING_INDEX;
        *wrapped = 1;
    }
    data[++(*curr_ptr)] = val;
}

#endif  // ARCH_QUASAR || ARCH_BLACKHOLE

#define WATCHER_RING_BUFFER_PUSH(x) push_to_ring_buffer(x)
#else  // !defined(WATCHER_ENABLED)
#define WATCHER_RING_BUFFER_PUSH(x)
#endif  // defined(WATCHER_ENABLED)

#endif  // KERNEL_BUILD || FW_BUILD
