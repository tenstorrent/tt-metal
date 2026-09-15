// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// TENSIX cores only; ethernet and DRAM use watcher_ringbuf.cpp.

#include <cstdint>
#include "api/debug/ring_buffer.h"
#include "experimental/kernel_args.h"
#if defined(DEBUG_RING_BUFFER_MPSC)
// get_hw_thread_idx(), not get_my_thread_id(), must match ring_buffer.h's MPSC write_id encoding.
#include "internal/hw_thread.h"
#endif

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/common.h"
#endif

// Which processors push: all data movement (COMPILE_FOR_DM is Quasar, BRISC/NCRISC are WH/BH), plus
// TRISCs selected either individually by WATCHER_RINGBUF_TRISCn or collectively by MULTI_DM_TEST.
#if defined(COMPILE_FOR_DM) || defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC) || \
    (defined(COMPILE_FOR_TRISC) && defined(MULTI_DM_TEST)) ||                               \
    (defined(UCK_CHLKC_UNPACK) && defined(WATCHER_RINGBUF_TRISC0)) ||                       \
    (defined(UCK_CHLKC_MATH) && defined(WATCHER_RINGBUF_TRISC1)) ||                         \
    (defined(UCK_CHLKC_PACK) && defined(WATCHER_RINGBUF_TRISC2)) ||                         \
    (defined(UCK_CHLKC_ISOLATE_SFPU) && defined(WATCHER_RINGBUF_TRISC3))
#define WATCHER_RINGBUF_PUSHER
#endif

void kernel_main() {
#if defined(WATCHER_RINGBUF_PUSHER)
    constexpr uint32_t num_pushes = get_arg(args::num_pushes);

#if defined(DEBUG_RING_BUFFER_MPSC)
    const uint32_t thread_idx = internal_::get_hw_thread_idx();
#if defined(COMPILE_FOR_DM) && !defined(MULTI_DM_TEST)
    // Single-DM test: all 6 user DMs are launched, only the requested one pushes.
    if (get_arg(args::dm_id) != thread_idx) {
        return;
    }
#endif
#endif

    for (uint32_t i = 0; i < num_pushes; i++) {
#if defined(DEBUG_RING_BUFFER_MPSC)
        WATCHER_RING_BUFFER_PUSH((thread_idx << 16) | i);
#else
        // WH slots carry no write_id, so the payload itself encodes the sequence.
        WATCHER_RING_BUFFER_PUSH((i + 1) + (i << 16));
#endif
    }
#endif
}
