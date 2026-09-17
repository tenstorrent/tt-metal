// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PrefetcherPipe DM receiver with relay DFB bridging to TRISC.
//
// Pipe consumers are the relay producers. Each hart owns total_entries / P pipe entries
// (P=1: all of them, dense; P>1: pipe lane credits, entries tid, tid+P, ...) and consumes
// them batch_size at a time with wait_front(n) / pop_front(n).
//
// Relay publish is always one entry per push_back. The relay may fan this producer out over
// several consumer TCs (cap=STRIDED with more consumers than producers); DFB push_back(n)
// posts n entries on the *current* TC at the DFB stride and only then round-robins, so a
// batched push would publish slots {0, C, ...} while the pipe delivered {0, 1, ...}: one
// consumer reads an unfilled slot and pop_front then acks an entry nobody published.
// Pushing one at a time lands pipe entry k on TC k % C, which is the DFB's own layout.
// Consequently batch_size here is the *pipe* batch: with C TCs per producer and a TRISC
// batch of b, it must be b * C so every TC receives a full consumer batch before pop_front
// waits for the consumers to catch up.
//
// Compile-time parameters:
//   [0] prefetcher_pipe_id
//   [1] total_entries
//   [2] batch_size (pipe entries per wait_front/pop_front)

#include "api/dataflow/prefetcher_pipe.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"

void kernel_main() {
    constexpr uint8_t prefetcher_pipe_id = get_compile_time_arg_val(0);
    constexpr uint32_t total_entries = get_compile_time_arg_val(1);
    constexpr uint16_t batch_size = get_compile_time_arg_val(2);

    Noc noc;
    experimental::PrefetcherPipe pipe(prefetcher_pipe_id);
    auto relay = pipe.bind_relay();

    const uint32_t tid = get_my_thread_id();
    const uint32_t num_threads = get_num_threads();
    ASSERT(tid < num_threads);
    ASSERT(total_entries % batch_size == 0);

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
