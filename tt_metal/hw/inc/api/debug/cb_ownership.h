// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//
// debug/cb_ownership.h
//
// Tracks which RISCs touched each circular buffer via cb_wait_front /
// cb_pop_front. Host-side watcher reader OR-reduces the per-RISC bytes into
// masks and fatally flags mixed consumer wait-without-pop patterns.
//
// Per-launch reset: BRISC clears all slots at firmware boot before releasing
// the other RISCs. See WATCHER_CB_OWNERSHIP_RESET.
//
// Enabled only for kernels that define WATCHER_ENABLE_CB_OWNERSHIP_RECORDING
// while the sanitizer is narrowed to the SDPA decode c_8 repro.
//
// Scope: Tensix only (BRISC, NCRISC, TRISC0/1/2 — 5-RISC mask). Eth/DRAM/
// Quasar are no-ops.
//
#pragma once

#if defined(WATCHER_ENABLED) && defined(WATCHER_ENABLE_CB_OWNERSHIP_RECORDING) &&                                \
    !defined(WATCHER_DISABLE_CB_OWNERSHIP) && !defined(COMPILE_FOR_ERISC) && !defined(COMPILE_FOR_IDLE_ERISC) && \
    !defined(COMPILE_FOR_DRISC) && !defined(ARCH_QUASAR) && !defined(FORCE_WATCHER_OFF)

#include "hostdev/dev_msgs.h"

// Must match the literal sizes in debug_cb_ownership_msg_t (40 bytes/category).
// 8 bytes per RISC × 5 RISCs = 40 bytes. See dev_msgs.h for why these are
// not exposed as named constants there.
constexpr static uint32_t CB_OWNERSHIP_BITMAP_BYTES_PER_RISC = 8;
constexpr static uint32_t CB_OWNERSHIP_BITMAP_SIZE = 40;
static_assert(MaxProcessorsPerCoreType * CB_OWNERSHIP_BITMAP_BYTES_PER_RISC <= CB_OWNERSHIP_BITMAP_SIZE);
static_assert((NUM_CIRCULAR_BUFFERS + 7) / 8 <= CB_OWNERSHIP_BITMAP_BYTES_PER_RISC);

// RISC index into the 5-wide per-RISC byte arrays. Matches
// TensixProcessorTypes (DM0=0, DM1=1, MATH0=2, MATH1=3, MATH2=4).
#if defined(COMPILE_FOR_BRISC)
#define WATCHER_CB_RISC_INDEX 0
#elif defined(COMPILE_FOR_NCRISC)
#define WATCHER_CB_RISC_INDEX 1
#elif defined(COMPILE_FOR_TRISC)
// COMPILE_FOR_TRISC is defined to 0/1/2 by hal_1xx_common.cpp.
#define WATCHER_CB_RISC_INDEX (2 + COMPILE_FOR_TRISC)
#else
#error "cb_ownership.h: no COMPILE_FOR_* RISC define"
#endif

// Bit-packed layout: each RISC owns the slice
//   [WATCHER_CB_RISC_INDEX * CB_OWNERSHIP_BITMAP_BYTES_PER_RISC,
//    (WATCHER_CB_RISC_INDEX + 1) * CB_OWNERSHIP_BITMAP_BYTES_PER_RISC)
// within each bitmap. Disjoint per-RISC → no cross-RISC race. Within-RISC
// writes are sequential (one hart) → no race.
//
// GET_MAILBOX_ADDRESS_DEV returns a pointer-to-array (`uint8_t(*)[40]`);
// dereferencing decays to `volatile uint8_t*` for element access.
inline void watcher_cb_record_bit(volatile tt_l1_ptr uint8_t* bitmap, uint32_t cb_id) {
    // Keep the default checker focused on the SDPA decode c_8 repro while
    // CB ownership semantics are tightened for broader kernel coverage.
    if (cb_id != 8) {
        return;
    }
    uint32_t byte_idx = WATCHER_CB_RISC_INDEX * CB_OWNERSHIP_BITMAP_BYTES_PER_RISC + (cb_id >> 3);
    uint8_t bit = static_cast<uint8_t>(1u << (cb_id & 0x7));
    if ((bitmap[byte_idx] & bit) == 0) {
        bitmap[byte_idx] |= bit;
    }
}

inline void watcher_cb_record_push(uint32_t cb_id) {
    watcher_cb_record_bit(*GET_MAILBOX_ADDRESS_DEV(watcher.cb_ownership.producer_bitmap), cb_id);
}

inline void watcher_cb_record_wait(uint32_t cb_id) {
    watcher_cb_record_bit(*GET_MAILBOX_ADDRESS_DEV(watcher.cb_ownership.consumer_bitmap), cb_id);
}

// Popper tracking is used to distinguish legitimate split-reader patterns
// (multi-consumer + every consumer also pops, e.g. conv2d halo gather) from
// the c_8-style anti-pattern (multi-consumer but one consumer never pops →
// deadlock if the other consumer pops the only token first).
inline void watcher_cb_record_pop(uint32_t cb_id) {
    watcher_cb_record_bit(*GET_MAILBOX_ADDRESS_DEV(watcher.cb_ownership.popper_bitmap), cb_id);
}

// Reset hook — called by BRISC firmware before releasing NCRISC/TRISCs.
// Wipes all bitmaps. Race-free because no other RISC is running yet
// (they're held in reset until BRISC issues start_ncrisc_kernel_run /
// run_triscs). Doing it BRISC-only also handles the cross-program case
// where a RISC that touched a CB in program A doesn't run in program B —
// without a global reset, its stale bits would falsely accuse it in B.
inline void watcher_cb_ownership_reset() {
#if defined(COMPILE_FOR_BRISC)
    volatile tt_l1_ptr uint8_t* prod = *GET_MAILBOX_ADDRESS_DEV(watcher.cb_ownership.producer_bitmap);
    volatile tt_l1_ptr uint8_t* cons = *GET_MAILBOX_ADDRESS_DEV(watcher.cb_ownership.consumer_bitmap);
    volatile tt_l1_ptr uint8_t* pop = *GET_MAILBOX_ADDRESS_DEV(watcher.cb_ownership.popper_bitmap);
    for (uint32_t i = 0; i < CB_OWNERSHIP_BITMAP_SIZE; ++i) {
        prod[i] = 0;
        cons[i] = 0;
        pop[i] = 0;
    }
#endif
}

#define WATCHER_CB_RECORD_PUSH(id) ((void)0)
#define WATCHER_CB_RECORD_WAIT(id) watcher_cb_record_wait(id)
#define WATCHER_CB_RECORD_POP(id) watcher_cb_record_pop(id)
#define WATCHER_CB_OWNERSHIP_RESET() watcher_cb_ownership_reset()

#else  // not enabled / wrong RISC type

// ((void)0) (not empty) keeps comma-expression call sites valid:
//   UNPACK((WATCHER_CB_RECORD_WAIT(id), llk_wait_tiles(id, n)))
// must remain syntactically valid when the macro disables.
#define WATCHER_CB_RECORD_PUSH(id) ((void)0)
#define WATCHER_CB_RECORD_WAIT(id) ((void)0)
#define WATCHER_CB_RECORD_POP(id) ((void)0)
#define WATCHER_CB_OWNERSHIP_RESET() ((void)0)

#endif
