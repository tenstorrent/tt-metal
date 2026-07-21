// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef COMPILE_FOR_TRISC
#include "api/dataflow/dataflow_api.h"
#include "api/kernel_thread_globals.h"
#include "api/debug/dprint.h"
#endif

void kernel_main() {
    uint32_t arrivals_addr = get_arg_val<uint32_t>(0);
    volatile tt_l1_ptr uint32_t* arrivals = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrivals_addr);
    volatile tt_l1_ptr uint32_t* observed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(1));
    volatile tt_l1_ptr uint32_t* post = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(2));
    uint32_t rounds = get_arg_val<uint32_t>(3);
    uint32_t skew_iters = get_arg_val<uint32_t>(4);
    uint32_t total_words = get_arg_val<uint32_t>(5);

    const uint32_t num_threads = get_num_threads();
    const uint32_t thread_id = get_my_thread_id();

    sync_threads();

    for (uint32_t r = 0; r < rounds; r++) {
        // Stagger arrivals so the barrier must actively wait for slower threads.
        uint32_t delay = (thread_id + 1) * skew_iters;
        for (uint32_t d = 0; d < delay; d++) {
            asm volatile("nop");
        }

#ifdef ARCH_QUASAR
        __atomic_add_fetch(reinterpret_cast<uint32_t*>(&arrivals[r]), 1u, __ATOMIC_SEQ_CST);
#else
        arrivals[r] += 1;
#endif
        sync_threads();

        // Thread 0 snapshots the arrival count after the barrier. All threads have passed sync_threads(), observed[r] must equal num_threads.
        if (thread_id == 0) {
#ifdef ARCH_QUASAR
            observed[r] = __atomic_load_n(reinterpret_cast<uint32_t*>(&arrivals[r]), __ATOMIC_SEQ_CST);
#else
            observed[r] = arrivals[r];
#endif
        }
        sync_threads();

        // Each thread increments post[r] to prove it reached the post-barrier phase of this
        // round. The host checks post[r] == num_threads to confirm no thread skipped ahead.
#ifdef ARCH_QUASAR
        __atomic_add_fetch(reinterpret_cast<uint32_t*>(&post[r]), 1u, __ATOMIC_SEQ_CST);
#else
        post[r] += 1;
#endif
        sync_threads();
    }

    if (thread_id == 0) {
        observed[rounds] = num_threads;
    }

#ifdef ARCH_QUASAR
    flush_l2_cache_range(arrivals_addr, total_words * sizeof(uint32_t));
#endif
}
