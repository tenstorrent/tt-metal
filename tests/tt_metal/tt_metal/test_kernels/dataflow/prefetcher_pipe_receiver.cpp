// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PrefetcherPipe receiver kernel: wait for entries and pop.
//
// Compile-time parameters (via kernel compile_args):
//   [0] prefetcher_pipe_id
//   [1] entry_size
//   [2] num_entries        - total dense entries in this epoch (must be divisible by
//                           get_num_threads() when multi-DM lane credits are armed)
//   [3] receiver_idx       - unused (reserved for test harness symmetry)
//
// Multi-DM: host AttachPrefetcherPipe(..., num_pipe_consumer_threads=P) must match
// num_threads_per_cluster. Each hart owns lane tid and wait_front/pop_front one
// owned stride per loop iteration (num_entries / P iterations).

#include "api/dataflow/prefetcher_pipe.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"

void kernel_main() {
    constexpr uint8_t prefetcher_pipe_id = get_compile_time_arg_val(0);
    constexpr uint32_t num_entries = get_compile_time_arg_val(2);

    Noc noc;
    experimental::PrefetcherPipe gdfb(prefetcher_pipe_id);

    const uint32_t num_threads = get_num_threads();
    const uint32_t tid = get_my_thread_id();
    ASSERT(num_threads >= 1);
    ASSERT(tid < num_threads);
    ASSERT(num_entries % num_threads == 0);
    const uint32_t entries_this_thread = num_entries / num_threads;

    for (uint32_t i = 0; i < entries_this_thread; ++i) {
        gdfb.wait_front(1);
        gdfb.pop_front(1, noc);
    }
}
