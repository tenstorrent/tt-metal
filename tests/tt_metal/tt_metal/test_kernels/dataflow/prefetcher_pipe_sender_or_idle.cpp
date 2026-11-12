// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PrefetcherPipe sender kernel that also runs on extra nodes (not the pipe's sender): on the sender it broadcasts
// `num_entries` entries from a host-populated L1 staging area, one at a time; on an extra node it returns without
// constructing the pipe.
//
// Bindings:
//   pipe::out            — KernelAdvancedOptions::PrefetcherPipeBinding accessor (program slot id baked in)
// Args (named CTAs):
//   args::entry_size     - bytes per entry
//   args::num_entries    - entries to push
// Args (named RTAs):
//   args::is_sender      - 1 on the pipe's sender, 0 on extra nodes
//   args::staging_addr   - sender-local L1 staging (multicast counter layout, prefetcher_pipe_test_utils.hpp)

#include "api/dataflow/prefetcher_pipe.h"
#include "api/dataflow/noc.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    if (get_arg(args::is_sender) == 0) {
        return;
    }
    constexpr uint32_t entry_size = get_arg(args::entry_size);
    constexpr uint32_t num_entries = get_arg(args::num_entries);
    const CoreLocalMem<uint8_t> staging(get_arg(args::staging_addr));

    Noc noc;
    experimental::PrefetcherPipe pipe(pipe::out);
    for (uint32_t i = 0; i < num_entries; ++i) {
        pipe.reserve_back(1);
        pipe.write_broadcast(noc, staging, 1, {.offset_bytes = i * entry_size});
        pipe.flush_writes(noc);
        pipe.push_back(1, noc);
    }
}
