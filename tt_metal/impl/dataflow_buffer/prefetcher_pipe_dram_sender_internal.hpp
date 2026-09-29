// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// DRAM-sender endpoints for a PrefetcherPipeSpace.
//
// The public PrefetcherPipeSpace surface only carries DRAM-sender CAPACITY
// (PrefetcherPipeSpaceConfig::num_dram_senders); it never names DRAM cores, because each DRAM
// bank has several candidate sender cores and picking one is a routing decision the tensor
// prefetcher makes. The exact cores are bound here, from tt_metal/ only (TensorPrefetcherManager,
// tests), mirroring the GCB split between the public factory and
// global_circular_buffer_dram_sender_internal.hpp.
//
// Not implemented yet: DRISC L1 for the sender ring / config page is blocked on #55285. Both
// functions validate their arguments and then TT_FATAL.

#include <span>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/prefetcher_pipe.hpp>

namespace tt::tt_metal::experimental {

// Pin sender L1 on `dram_senders` (at most space.num_dram_senders() cores) before any DRISC
// kernel runs on them.
void set_dram_sender_cores(PrefetcherPipeSpace& space, std::span<const CoreCoord> dram_senders);

// Carve a pipe whose sender is one of the cores passed to set_dram_sender_cores.
PrefetcherPipe create_dram_sender_pipe(
    PrefetcherPipeSpace& space, CoreCoord dram_sender, const CoreRangeSet& receivers);

}  // namespace tt::tt_metal::experimental
