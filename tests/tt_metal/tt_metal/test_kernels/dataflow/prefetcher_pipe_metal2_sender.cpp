// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 PrefetcherPipe sender: reserve / write_broadcast / push_back `num_entries` dense
// entries from a host-populated L1 staging area (multicast counter layout).
//
// Bindings:
//   pipe::out              — KernelSpec::PrefetcherPipeBinding accessor (program slot id baked in)
// Args:
//   args::entry_size       — named CTA, bytes per entry
//   args::num_entries      — named CTA, entries to push
//   args::staging_addr     — named RTA, sender-local L1 staging base
//
// Quasar multi-DM: launched with num_threads > 1 every hart runs this body; the pipe APIs
// partition receivers by hart (Flow C), so the same loop is correct for any thread count.

#include "api/dataflow/prefetcher_pipe.h"
#include "api/dataflow/noc.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t entry_size = get_arg(args::entry_size);
    constexpr uint32_t num_entries = get_arg(args::num_entries);
    const uint32_t staging_base = get_arg(args::staging_addr);
    const CoreLocalMem<uint8_t> staging(staging_base);

    Noc noc;
    experimental::PrefetcherPipe pipe(pipe::out);

    for (uint32_t i = 0; i < num_entries; ++i) {
        pipe.reserve_back(1);
        pipe.write_broadcast(noc, staging, 1, {.offset_bytes = i * entry_size});
        pipe.flush_writes(noc);
        pipe.push_back(1, noc);
    }
}
