// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 PrefetcherPipe receiver bridging the pipe ring to TRISC through a relay DFB.
//
// Bindings:
//   pipe::in               — KernelSpec::PrefetcherPipeBinding accessor (program slot id baked in)
//   (relay DFB)            — bound as PRODUCER; reached through pipe.bind_relay(), no dfb:: token
// Args:
//   args::total_entries    — named CTA, pipe entries this core consumes over the run
//   args::batch_size       — named CTA, pipe entries per wait_front / pop_front
//
// Each hart owns total_entries / P pipe entries (P = num_threads = the pipe's credit lanes) and
// publishes one relay entry per pipe entry (see prefetcher_pipe_relay_receiver.cpp for why the
// relay push is never batched).

#include "api/dataflow/prefetcher_pipe.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t total_entries = get_arg(args::total_entries);
    constexpr uint16_t batch_size = get_arg(args::batch_size);

    Noc noc;
    experimental::PrefetcherPipe pipe(pipe::in);
    auto relay = pipe.bind_relay();

    const uint32_t num_threads = get_num_threads();
    ASSERT(get_my_thread_id() < num_threads);
    ASSERT(total_entries % num_threads == 0);
    ASSERT((total_entries / num_threads) % batch_size == 0);
    const uint32_t entries_this_thread = total_entries / num_threads;
    for (uint32_t offset = 0; offset < entries_this_thread; offset += batch_size) {
        pipe.wait_front(batch_size);
        for (uint32_t i = 0; i < batch_size; ++i) {
            relay.reserve_back(1);
            relay.push_back(1);
        }
        pipe.pop_front(batch_size, noc);
    }
}
