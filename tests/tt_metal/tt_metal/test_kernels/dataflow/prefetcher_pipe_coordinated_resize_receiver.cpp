// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Receiver half of a coordinated PrefetcherPipe page-size change:
// consume all E1 traffic, then resize to E2 and consume the sender's pad
// credits. The sender may already be configured for E2 while E1 is consumed.
//
// Bindings:
//   pipe::in               — KernelAdvancedOptions::PrefetcherPipeBinding accessor (program slot id baked in)
// Args (named CTAs):
//   args::num_entries_e1
//   args::entry_size_e2

#include "api/dataflow/prefetcher_pipe.h"
#include "api/dataflow/noc.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_entries_e1 = get_arg(args::num_entries_e1);
    constexpr uint32_t entry_size_e2 = get_arg(args::entry_size_e2);

    Noc noc;
    experimental::PrefetcherPipe dfb(pipe::in);
    for (uint32_t i = 0; i < num_entries_e1; ++i) {
        dfb.wait_front(1);
        dfb.pop_front(1, noc);
    }
    dfb.set_receiver_entry_size(entry_size_e2);
}
