// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(KERNEL_BUILD) || defined(FW_BUILD)

#include "hostdev/dev_msgs.h"

#if defined(WATCHER_ENABLED) && !defined(WATCHER_DISABLE_RING_BUFFER) && !defined(FORCE_WATCHER_OFF)

// Ring buffer modes (see DEBUG_RING_BUFFER_MPSC for which cores get which):
// - Quasar Tensix:   MPSC, slot claimed via a NEO cluster semaphore (DM and TRISC caches are not
//                    coherent, so a RISC-V atomic on an L1 word would not serialize between them)
// - Quasar dispatch: MPSC, slot claimed via a RISC-V atomic on a cached head (no NEO registers, and
//                    only DMs, whose caches are coherent)
// - Blackhole:       MPSC, slot claimed via a 32-bit RISC-V atomic on the in-mailbox head
// - Wormhole:        SPSC, no synchronization between RISCs
// Quasar's host never reads the head (a host read of the semaphore can post to it); write_id carries
// the claimed position instead.

#if defined(DEBUG_RING_BUFFER_MPSC)
#include "internal/hw_thread.h"
#include "risc_common.h"
#if defined(ARCH_QUASAR) && !defined(COMPILE_FOR_DISPATCH_ENGINE)
#include "api/debug/waypoint.h"
#include "tensix_neo_reg.h"
// NEO cluster semaphore 31 is reserved for the MPSC head; kernels must not use it. Semaphores 27–30
// are reserved for kernel thread barriers.
constexpr uintptr_t watcher_ring_buf_sem = TENSIX_GLOBAL_REGS_SEMAPHORE_REGS_SEMAPHORE_31__REG_ADDR;
// The semaphore saturates at watcher_ring_buf_sem_max instead of wrapping, and a post past it raises a
// fatal GLOBAL_SEMAPHORES error. The push that claims watcher_ring_buf_reset_at rewinds the head to 0,
// which leaves room for every other processor to claim once more before the rewind lands.
constexpr uint32_t watcher_ring_buf_sem_max = 0xFFFF;
constexpr uint32_t watcher_ring_buf_reset_at = DEBUG_RING_BUFFER_QUASAR_SEM_POS_WRAP - 1;
static_assert(watcher_ring_buf_reset_at == watcher_ring_buf_sem_max - DEBUG_RING_BUFFER_ELEMENTS);
static_assert(DEBUG_RING_BUFFER_ELEMENTS >= MaxProcessorsPerCoreType);
// The rewind keeps slots in order: the resetter takes the last slot and the next claim the first.
static_assert((watcher_ring_buf_reset_at + 1) % DEBUG_RING_BUFFER_ELEMENTS == 0);

// SEM_INIT: a write at +0 overwrites the head.
inline __attribute__((always_inline)) void watcher_ring_buf_sem_init(uint32_t value) {
    *reinterpret_cast<volatile uint32_t*>(watcher_ring_buf_sem) = value;
}

// A read at +4*(inc+8) posts `inc` and returns the pre-increment value.
inline __attribute__((always_inline)) uint32_t watcher_ring_buf_sem_post() {
    return *reinterpret_cast<volatile uint32_t*>(watcher_ring_buf_sem + 4 * (1 + 8));
}

// Posts 0: returns the current value without changing it.
inline __attribute__((always_inline)) uint32_t watcher_ring_buf_sem_peek() {
    return *reinterpret_cast<volatile uint32_t*>(watcher_ring_buf_sem + 4 * (0 + 8));
}
#endif

inline __attribute__((always_inline)) void push_to_ring_buffer(uint32_t val) {
    auto* wrapper = GET_MAILBOX_ADDRESS_DEV(watcher.debug_ring_buf);
    auto* buf = reinterpret_cast<debug_mpsc_ring_buf_msg_t tt_l1_ptr*>(wrapper->data);

#if defined(COMPILE_FOR_DISPATCH_ENGINE)
    // AMOs hang on the uncached alias, so the head is claimed through the cached view and never flushed.
    uint32_t pos = __atomic_fetch_add(reinterpret_cast<uint32_t*>(MEM_DISPATCH_WATCHER_RING_HEAD), 1, __ATOMIC_RELAXED);
#elif defined(ARCH_QUASAR)
    // A claim past reset_at landed before the rewind, which erases it: wait for the rewind and claim again.
    uint32_t pos;
    while (true) {
        pos = watcher_ring_buf_sem_post();
        if (__builtin_expect(pos <= watcher_ring_buf_reset_at, 1)) {
            break;
        }
        WAYPOINT("WRBW");
        while (watcher_ring_buf_sem_peek() > watcher_ring_buf_reset_at) {
        }
        WAYPOINT("WRBD");
    }
    if (__builtin_expect(pos == watcher_ring_buf_reset_at, 0)) {
        watcher_ring_buf_sem_init(0);
    }
#else
    uint32_t pos = __atomic_fetch_add(&buf->head, 1, __ATOMIC_RELAXED);
#endif
    uint32_t idx = pos & DEBUG_RING_BUFFER_MASK;

    buf->slots[idx].data = val;

    uint32_t write_id = internal_::get_hw_thread_idx() + 1;
#if defined(ARCH_QUASAR)
    write_id |= pos << DEBUG_RING_BUFFER_WRITE_ID_POS_SHIFT;
#endif
    __atomic_store_n(&buf->slots[idx].write_id, write_id, __ATOMIC_RELEASE);
}

#else  // SPSC ring buffer

inline __attribute__((always_inline)) void push_to_ring_buffer(uint32_t val) {
    auto* wrapper = GET_MAILBOX_ADDRESS_DEV(watcher.debug_ring_buf);
    auto* buf = reinterpret_cast<debug_spsc_ring_buf_msg_t tt_l1_ptr*>(wrapper->data);
    volatile tt_l1_ptr int16_t* curr_ptr = &buf->current_ptr;
    volatile tt_l1_ptr uint16_t* wrapped = &buf->wrapped;
    uint32_t* data = buf->data;

    // Bounds check, set to -1 to wrap since we increment before using.
    if (*curr_ptr >= DEBUG_RING_BUFFER_ELEMENTS - 1) {
        *curr_ptr = DEBUG_RING_BUFFER_STARTING_INDEX;
        *wrapped = 1;
    }
    data[++(*curr_ptr)] = val;
}

#endif  // DEBUG_RING_BUFFER_MPSC

// Quasar Tensix: hardware raises GLOBAL_SEMAPHORES/POST_ON_UNINITIALIZED if a semaphore is posted before
// it is initialized. DM0 firmware does it, before any other processor can push.
inline __attribute__((always_inline)) void init_ring_buffer() {
#if defined(ARCH_QUASAR)
    // The host orders slots by the position in write_id, so slots from before a head reset would sort as newest.
    auto* buf =
        reinterpret_cast<debug_mpsc_ring_buf_msg_t tt_l1_ptr*>(GET_MAILBOX_ADDRESS_DEV(watcher.debug_ring_buf)->data);
    for (uint32_t i = 0; i < DEBUG_RING_BUFFER_ELEMENTS; i++) {
        buf->slots[i].write_id = 0;
    }
#endif
#if defined(COMPILE_FOR_DISPATCH_ENGINE)
    __atomic_store_n(reinterpret_cast<uint32_t*>(MEM_DISPATCH_WATCHER_RING_HEAD), 0, __ATOMIC_RELAXED);
#elif defined(ARCH_QUASAR)
    watcher_ring_buf_sem_init(0);
#endif
}

#define WATCHER_RING_BUFFER_PUSH(x) push_to_ring_buffer(x)
#define WATCHER_RING_BUFFER_INIT() init_ring_buffer()
#else  // !defined(WATCHER_ENABLED)
#define WATCHER_RING_BUFFER_PUSH(x)
#define WATCHER_RING_BUFFER_INIT()
#endif  // defined(WATCHER_ENABLED)

#endif  // KERNEL_BUILD || FW_BUILD
