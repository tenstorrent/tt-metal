// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PrefetcherPipe receiver kernel: wait for entries and pop
//
// Compile-time parameters (via kernel compile_args):
//   [0] prefetcher_pipe_id
//   [1] entry_size
//   [2] num_entries
//   [3] receiver_idx       - unused (reserved for test harness symmetry)

#include "api/dataflow/prefetcher_pipe.h"
#include "api/dataflow/noc.h"

void kernel_main() {
    constexpr uint8_t prefetcher_pipe_id = get_compile_time_arg_val(0);
    constexpr uint32_t num_entries = get_compile_time_arg_val(2);

    Noc noc;

    experimental::PrefetcherPipe gdfb(prefetcher_pipe_id);

    for (uint32_t i = 0; i < num_entries; ++i) {
        DPRINT("Doing wait front\n");
        gdfb.wait_front(1);
        DPRINT("Done wait front\n");
        gdfb.pop_front(1, noc);
        DPRINT("Done pop front\n");
    }
}
