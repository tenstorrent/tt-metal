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
// tests).
//
#include <span>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/prefetcher_pipe.hpp>

namespace tt::tt_metal::experimental {

// Pin sender L1 on `dram_senders` (at most space.num_dram_senders() cores) before any DRISC
// kernel runs on them.
void set_dram_sender_cores(PrefetcherPipeSpace& space, std::span<const CoreCoord> dram_senders);

}  // namespace tt::tt_metal::experimental
