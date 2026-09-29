// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PrefetcherPipe receiver kernel: wait for entries and pop.
//
// Bindings:
//   pipe::in               — KernelAdvancedOptions::PrefetcherPipeBinding accessor (program slot id baked in)
// Args (named CTAs):
//   args::num_entries      - total dense entries in this epoch (must be divisible by
//                            get_num_threads() when multi-DM lane credits are armed)
//
// Multi-DM: the receiver KernelSpec's num_threads is the pipe's credit lane count P. Each hart
// owns lane tid and wait_front/pop_front one owned stride per loop iteration
// (num_entries / P iterations).

#include "api/dataflow/prefetcher_pipe.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_entries = get_arg(args::num_entries);

    Noc noc;
    experimental::PrefetcherPipe gdfb(pipe::in);

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
