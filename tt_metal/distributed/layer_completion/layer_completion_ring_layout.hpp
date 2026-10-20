// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Layout of the layer-completion SHM ring: a LayerCompletionRingHeader followed by kLayerCompletionRingCapacity
// cells, a Vyukov bounded MPMC queue. Runners push into the host-local ring and the router pops; the master
// router pushes into the scheduler-facing ring and the scheduler pops. A participant that dies between claiming
// a cell and committing it wedges the ring; recreate the segment. The header magic identifies the wire format and
// connect() refuses anything else.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

#include <internal/disaggregation/layer_completion_message.hpp>

namespace tt::tt_metal::internal {

inline constexpr std::size_t kLayerCompletionCacheLine = 64;
inline constexpr uint32_t kLayerCompletionRingMagic = 0x4C435152u;  // 'LCQR'

// One cell. `sequence` gates ownership: producers wait for sequence == pos, consumers for pos + 1.
struct alignas(kLayerCompletionCacheLine) LayerCompletionCell {
    std::atomic<uint64_t> sequence;
    LayerCompletionMessage msg;
};

// Header; the offsets below are the cross-process contract.
struct LayerCompletionRingHeader {
    // Producers CAS to claim the next cell.
    alignas(kLayerCompletionCacheLine) std::atomic<uint64_t> enqueue_pos;
    // Consumer cursor.
    alignas(kLayerCompletionCacheLine) std::atomic<uint64_t> dequeue_pos;
    // Validated by connectors at attach.
    uint32_t capacity;
    uint32_t magic;
    // Layers one chunk covers across all ranks, summed over its completions' spans; the consumer retires a
    // chunk once its spans cover this many.
    uint32_t completions_per_chunk;
};

// Cells start at the first cell-aligned offset past the header.
inline constexpr std::size_t kLayerCompletionCellsOffset =
    ((sizeof(LayerCompletionRingHeader) + alignof(LayerCompletionCell) - 1) / alignof(LayerCompletionCell)) *
    alignof(LayerCompletionCell);

inline constexpr std::size_t kLayerCompletionRingBytes =
    kLayerCompletionCellsOffset + static_cast<std::size_t>(kLayerCompletionRingCapacity) * sizeof(LayerCompletionCell);

// Drift in the shared layout is a compile error.
static_assert(offsetof(LayerCompletionRingHeader, enqueue_pos) == 0);
static_assert(offsetof(LayerCompletionRingHeader, dequeue_pos) == kLayerCompletionCacheLine);
static_assert(offsetof(LayerCompletionRingHeader, capacity) == kLayerCompletionCacheLine + 8);
static_assert(offsetof(LayerCompletionRingHeader, completions_per_chunk) == kLayerCompletionCacheLine + 16);
static_assert(sizeof(LayerCompletionRingHeader) == 2 * kLayerCompletionCacheLine);

static_assert(alignof(LayerCompletionCell) == kLayerCompletionCacheLine);
static_assert(sizeof(LayerCompletionCell) == kLayerCompletionCacheLine);
static_assert(offsetof(LayerCompletionCell, msg) == 8);
static_assert(sizeof(LayerCompletionMessage) == 32);
static_assert(kLayerCompletionCellsOffset == 128);
static_assert(kLayerCompletionRingBytes % kLayerCompletionCacheLine == 0);
static_assert(kLayerCompletionRingBytes == 65664);

}  // namespace tt::tt_metal::internal
