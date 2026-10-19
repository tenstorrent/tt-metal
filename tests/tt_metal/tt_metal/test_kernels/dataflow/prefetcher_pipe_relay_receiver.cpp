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
//   args::repage_entry_size - 0, or a pipe entry size to switch to in place once everything at the
//                             old size is consumed (set_receiver_entry_size on the bound relay)
//   args::probe_addr       - with repage_entry_size: L1 address of per-hart [write_ptr, stride, entry,
//                             window, span] words recorded before and after the re-page
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
#ifdef ARCH_QUASAR
#include "internal/tt-2xx/quasar/dev_mem_map.h"
#endif

void kernel_main() {
    constexpr uint32_t total_entries = get_arg(args::total_entries);
    constexpr uint16_t batch_size = get_arg(args::batch_size);
    constexpr uint32_t relay_pages_per_entry = get_arg(args::relay_pages_per_entry);
    constexpr uint32_t repage_entry_size = get_arg(args::repage_entry_size);
    constexpr uint32_t probe_addr = get_arg(args::probe_addr);

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

    if constexpr (repage_entry_size != 0) {
        // Once everything at the old size is consumed, re-page the bound relay in place and record this
        // thread's relay layout before and after, so the host can check that the block layout survived.
        // Written through the uncached alias so the host sees the words.
#ifdef ARCH_QUASAR
        volatile tt_l1_ptr uint32_t* probe =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(probe_addr + MEM_L1_UNCACHED_BASE) + tid * 10;
#else
        volatile tt_l1_ptr uint32_t* probe = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(probe_addr) + tid * 10;
#endif
        auto record = [&](uint32_t at) {
#ifdef ARCH_QUASAR
            probe[at + 0] = relay.get_write_ptr() - MEM_L1_UNCACHED_BASE;
#else
            probe[at + 0] = relay.get_write_ptr();
#endif
            probe[at + 1] = relay.get_stride_size();
            probe[at + 2] = relay.get_entry_size();
            probe[at + 3] = relay.get_local_size_bytes();
            probe[at + 4] = relay.get_ring_span_bytes();
        };
        record(0);
        pipe.set_receiver_entry_size(repage_entry_size);
        record(5);
    }
}
