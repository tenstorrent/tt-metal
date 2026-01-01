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

#if defined(WATCHER_RINGBUF_SEM_WRAP)
#include "api/kernel_thread_globals.h"
#include "dev_mem_map.h"
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

#if defined(WATCHER_RINGBUF_SEM_WRAP)
    // Quasar only. One DM starts the head so that half of all pushes land before the rewind, and no writer
    // pushes until it is set.
    constexpr uint32_t num_dm_threads = get_arg(args::num_dm_threads);
    constexpr uint32_t num_tensixes = get_arg(args::num_tensixes);
#if defined(COMPILE_FOR_DM)
    if (get_my_thread_id() == 0) {
        constexpr uint32_t total_pushes = (num_dm_threads + num_tensixes * NUM_TRISC_CORES) * num_pushes;
        watcher_ring_buf_sem_init(watcher_ring_buf_reset_at + 1 - total_pushes / 2);
    }
#endif
    sync_dm_compute_threads(num_dm_threads, num_tensixes);
#endif

    for (uint32_t i = 0; i < num_pushes; i++) {
#if defined(DEBUG_RING_BUFFER_MPSC)
        WATCHER_RING_BUFFER_PUSH((thread_idx << 16) | i);
#else
        // WH slots carry no write_id, so the payload itself encodes the sequence.
        WATCHER_RING_BUFFER_PUSH((i + 1) + (i << 16));
#endif
    }

#if defined(WATCHER_RINGBUF_SEM_WRAP)
    // Once every writer has pushed, one DM hands the head to the host through L1.
    sync_dm_compute_threads(num_dm_threads, num_tensixes);
#if defined(COMPILE_FOR_DM)
    if (get_my_thread_id() == 0) {
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg(args::head_addr) + MEM_L1_UNCACHED_BASE) =
            watcher_ring_buf_sem_peek();
    }
#endif
#endif
#endif
}
