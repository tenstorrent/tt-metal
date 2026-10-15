// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PrefetcherPipe DM receiver bridging the pipe ring to TRISC through a relay DFB.
//
// Bindings:
//   pipe::in               — KernelAdvancedOptions::PrefetcherPipeBinding accessor (program slot id baked in)
//   dfb::relay             — PRODUCER accessor of the relay DFB; built here and handed to pipe.bind_relay()
// Args (named CTAs):
//   args::total_entries    - pipe entries this core consumes over the run
//   args::batch_size       - pipe entries per wait_front / pop_front
//   args::relay_pages_per_entry - relay entries per pipe entry (the relay may page an entry finer)
//
// Pipe consumers are the relay producers. Each hart owns total_entries / P pipe entries
// (P = num_threads = the pipe's credit lanes; P=1: all of them, dense; P>1: entries tid, tid+P,
// ...) and consumes them batch_size at a time with wait_front(n) / pop_front(n).
//
// Relay publish is one producer share per push_back: one entry on a STRIDED producer, one block
// (one pipe entry) on a BLOCKED one. The relay may fan this producer out over several consumer
// TCs (cap=STRIDED with more consumers than producers); DFB push_back(n) posts n entries on the
// *current* TC at the DFB stride and only then round-robins, so an over-sized push would publish
// slots {0, C, ...} while the pipe delivered {0, 1, ...}: one consumer reads an unfilled slot and
// pop_front then acks an entry nobody published. Pushing one share at a time lands relay entry k
// on TC k % C, which is the DFB's own layout; a pipe entry is relay_pages_per_entry consecutive
// relay entries.
// Consequently batch_size here is the *pipe* batch: with C TCs per producer and a TRISC
// batch of b, it must be b * C so every TC receives a full consumer batch before pop_front
// waits for the consumers to catch up.

#include "api/dataflow/prefetcher_pipe.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t total_entries = get_arg(args::total_entries);
    constexpr uint16_t batch_size = get_arg(args::batch_size);
    constexpr uint32_t relay_pages_per_entry = get_arg(args::relay_pages_per_entry);

    Noc noc;
    experimental::PrefetcherPipe pipe(pipe::in);
    DataflowBuffer relay(dfb::relay);
    pipe.bind_relay(relay);
    const uint32_t share = relay.get_producer_share();

    const uint32_t tid = get_my_thread_id();
    const uint32_t num_threads = get_num_threads();
    ASSERT(tid < num_threads);
    ASSERT(total_entries % num_threads == 0);
    ASSERT((total_entries / num_threads) % batch_size == 0);
    const uint32_t entries_this_thread = total_entries / num_threads;
    ASSERT((batch_size * relay_pages_per_entry) % share == 0);
    for (uint32_t offset = 0; offset < entries_this_thread; offset += batch_size) {
        pipe.wait_front(batch_size);
        for (uint32_t i = 0; i < batch_size * relay_pages_per_entry; i += share) {
            relay.reserve_back(share);
            relay.push_back(share);
        }
        pipe.pop_front(batch_size, noc);
    }
}
