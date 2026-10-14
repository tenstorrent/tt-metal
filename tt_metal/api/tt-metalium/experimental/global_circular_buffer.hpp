// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Experimental sender-domain query for GlobalCircularBuffer. Programmable DRAM cores (Blackhole
// DRISCs) send through DRAM-sender PrefetcherPipes (CreatePrefetcherPipesForTensorPrefetcher in
// tt-metalium/experimental/prefetcher_pipe.hpp), not through a GlobalCircularBuffer, so every
// GlobalCircularBuffer has worker senders.

#pragma once

#include <tt-metalium/global_circular_buffer.hpp>
#include <tt-metalium/experimental/sender_core_type.hpp>

namespace tt::tt_metal {

namespace experimental {

// Sender domain of a GlobalCircularBuffer: SenderCoreType::Worker, the only kind there is.
SenderCoreType sender_core_type(const GlobalCircularBuffer& gcb);

}  // namespace experimental
}  // namespace tt::tt_metal
