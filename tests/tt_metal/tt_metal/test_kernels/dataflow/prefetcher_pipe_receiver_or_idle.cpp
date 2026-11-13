// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PrefetcherPipe receiver kernel that also runs on extra nodes (none of the pipe's receivers): on a receiver it waits
// for entries and pops them; on an extra node it returns without constructing the pipe.
//
// Bindings:
//   pipe::in             — KernelAdvancedOptions::PrefetcherPipeBinding accessor (program slot id baked in)
// Args (named CTAs):
//   args::num_entries    - entries to pop on each receiver
// Args (named RTAs):
//   args::is_receiver    - 1 on the pipe's receivers, 0 on extra nodes

#include "api/dataflow/prefetcher_pipe.h"
#include "api/dataflow/noc.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    if (get_arg(args::is_receiver) == 0) {
        return;
    }
    constexpr uint32_t num_entries = get_arg(args::num_entries);

    Noc noc;
    experimental::PrefetcherPipe pipe(pipe::in);
    for (uint32_t i = 0; i < num_entries; ++i) {
        pipe.wait_front(1);
        pipe.pop_front(1, noc);
    }
}
