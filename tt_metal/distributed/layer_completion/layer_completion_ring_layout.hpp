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

template <typename MsgT>
struct LayerCompletionRingTraits;

template <>
struct LayerCompletionRingTraits<LayerCompletionMessage> {
    static constexpr uint32_t magic = 0x4C435131u;
    static constexpr std::size_t cell_alignment = alignof(LayerCompletionMessage);
};

template <>
struct LayerCompletionRingTraits<LayerCompletionMessageV2> {
    static constexpr uint32_t magic = 0x4C435132u;
    static constexpr std::size_t cell_alignment = kLayerCompletionCacheLine;
};

inline constexpr uint32_t kLayerCompletionRingMagic = LayerCompletionRingTraits<LayerCompletionMessage>::magic;

// One ring slot. `sequence` gates ownership (Vyukov): producers wait for
// sequence==pos, consumers wait for sequence==pos+1.
template <typename MsgT>
struct alignas(LayerCompletionRingTraits<MsgT>::cell_alignment) LayerCompletionCellT {
    std::atomic<uint64_t> sequence;
    MsgT msg;
};

using LayerCompletionCell = LayerCompletionCellT<LayerCompletionMessage>;
using LayerCompletionCellV2 = LayerCompletionCellT<LayerCompletionMessageV2>;

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

template <typename MsgT>
constexpr std::size_t layer_completion_cells_offset() {
    using Cell = LayerCompletionCellT<MsgT>;
    return ((sizeof(LayerCompletionRingHeader) + alignof(Cell) - 1) / alignof(Cell)) * alignof(Cell);
}

template <typename MsgT>
inline constexpr std::size_t kLayerCompletionRingBytes =
    layer_completion_cells_offset<MsgT>() +
    static_cast<std::size_t>(kLayerCompletionRingCapacity) * sizeof(LayerCompletionCellT<MsgT>);

static_assert(offsetof(LayerCompletionRingHeader, enqueue_pos) == 0);
static_assert(offsetof(LayerCompletionRingHeader, dequeue_pos) == kLayerCompletionCacheLine);
static_assert(offsetof(LayerCompletionRingHeader, capacity) == kLayerCompletionCacheLine + 8);
static_assert(sizeof(LayerCompletionRingHeader) == 2 * kLayerCompletionCacheLine);

static_assert(alignof(LayerCompletionCell) == 8);
static_assert(sizeof(LayerCompletionCell) == 32);
static_assert(offsetof(LayerCompletionCell, msg) == 8);
static_assert(layer_completion_cells_offset<LayerCompletionMessage>() == 128);
static_assert(kLayerCompletionRingBytes<LayerCompletionMessage> == 32896);

static_assert(alignof(LayerCompletionCellV2) == kLayerCompletionCacheLine);
static_assert(sizeof(LayerCompletionCellV2) == kLayerCompletionCacheLine);
static_assert(offsetof(LayerCompletionCellV2, msg) == 8);
static_assert(layer_completion_cells_offset<LayerCompletionMessageV2>() % kLayerCompletionCacheLine == 0);
static_assert(layer_completion_cells_offset<LayerCompletionMessageV2>() == 128);
static_assert(kLayerCompletionRingBytes<LayerCompletionMessageV2> % kLayerCompletionCacheLine == 0);
static_assert(kLayerCompletionRingBytes<LayerCompletionMessageV2> == 65664);

}  // namespace tt::tt_metal::internal
