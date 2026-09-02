// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Wire contract for the host-local layer-completion SHM ring. One POSIX
// shared-memory region per host carries a LayerCompletionRingHeader
// followed by kLayerCompletionRingCapacity LayerCompletionCells. The
// ring is a Vyukov bounded MPMC queue: multiple producer processes (the
// prefill runners) push; a single consumer (the host's
// LayerCompletionRouter thread) pops.
//
// Cross-process atomics: the segment lives in shared memory mapped into
// every participant. std::atomic<uint64_t> is lock-free on the target
// and usable across processes when the storage is shared — the same
// guarantee inter_process_counter_layout.hpp relies on.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

#include <internal/disaggregation/layer_completion_message.hpp>

namespace tt::tt_metal::internal {

inline constexpr std::size_t kLayerCompletionCacheLine = 64;
inline constexpr uint32_t kLayerCompletionRingMagic = 0x4C435131u;  // 'LCQ1'

// One ring slot. `sequence` gates ownership (Vyukov): producers wait for
// sequence==pos, consumers wait for sequence==pos+1.
struct LayerCompletionCell {
    std::atomic<uint64_t> sequence;
    LayerCompletionMessage msg;
};

struct LayerCompletionRingHeader {
    // Producers CAS to claim the next enqueue slot.
    alignas(kLayerCompletionCacheLine) std::atomic<uint64_t> enqueue_pos;
    // The single consumer CAS-advances this (CAS keeps the algorithm
    // uniform; there is never real contention on the consumer side).
    alignas(kLayerCompletionCacheLine) std::atomic<uint64_t> dequeue_pos;
    // Sanity fields validated by connectors at attach.
    uint32_t capacity;
    uint32_t magic;
};

// Cells start on the first LayerCompletionCell-aligned offset past the header.
inline constexpr std::size_t layer_completion_cells_offset() {
    return ((sizeof(LayerCompletionRingHeader) + alignof(LayerCompletionCell) - 1) / alignof(LayerCompletionCell)) *
           alignof(LayerCompletionCell);
}

inline constexpr std::size_t kLayerCompletionRingBytes =
    layer_completion_cells_offset() +
    static_cast<std::size_t>(kLayerCompletionRingCapacity) * sizeof(LayerCompletionCell);

}  // namespace tt::tt_metal::internal
