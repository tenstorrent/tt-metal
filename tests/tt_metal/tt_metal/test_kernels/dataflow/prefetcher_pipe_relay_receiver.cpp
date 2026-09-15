// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PrefetcherPipe DM receiver with relay DFB bridging to TRISC.
//
// Pipe consumers are the relay producers:
//   - P=1: batched reserve → wait_front → push → pop (legacy path).
//   - P>1 (pap=STRIDED, pipe lane credits): each hart's wait_front(n)/pop_front(n)
//     are n owned strides; same n on relay.reserve/push.
//
// Compile-time parameters:
//   [0] prefetcher_pipe_id
//   [1] total_entries
//   [2] batch_size

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

    if (num_threads == 1) {
        for (uint32_t offset = 0; offset < total_entries; offset += batch_size) {
            relay.reserve_back(batch_size);
            pipe.wait_front(batch_size);
            relay.push_back(batch_size);
            pipe.pop_front(batch_size, noc);
        }
        return;
    }

    // Multi-RISC: lane credits + STRIDED relay. Each hart consumes total/P entries
    // in batches of batch_size owned strides (no unit-step barrier loop).
    ASSERT(total_entries % num_threads == 0);
    ASSERT((total_entries / num_threads) % batch_size == 0);
    const uint32_t entries_this_thread = total_entries / num_threads;
    for (uint32_t offset = 0; offset < entries_this_thread; offset += batch_size) {
        relay.reserve_back(batch_size);
        pipe.wait_front(batch_size);
        relay.push_back(batch_size);
        pipe.pop_front(batch_size, noc);
    }
}
