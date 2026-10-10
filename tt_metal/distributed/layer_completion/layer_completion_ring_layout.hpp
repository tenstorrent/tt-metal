// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Wire contract for the host-local layer-completion SHM ring. One POSIX
// shared-memory region per host carries a LayerCompletionRingHeader
// followed by kLayerCompletionRingCapacity cells. The ring is a Vyukov
// bounded MPMC queue: multiple producer processes (the prefill runners)
// push; a single consumer (the host's LayerCompletionRouter thread) pops.
// The same layout also backs the v2 scheduler-facing ring (master router
// pushes; the scheduler process pops).
//
// A participant that dies between claiming a cell and committing it leaves the
// ring wedged (producers see it full, or the consumer waits on that cell);
// there is no recovery short of recreating the segment.
//
// Cross-process atomics: the segment lives in shared memory mapped into
// every participant. std::atomic<uint64_t> is lock-free on the target
// and usable across processes when the storage is shared — the same
// guarantee inter_process_counter_layout.hpp relies on.
//
// LayerCompletionRingTraits binds each message version to its magic (validated
// by connect()) and cell alignment: v1 cells stay packed at 32 B (frozen), v2
// cells are one cache line each.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

#include <internal/disaggregation/layer_completion_message.hpp>

namespace tt::tt_metal::internal {

inline constexpr std::size_t kLayerCompletionCacheLine = 64;

template <typename MsgT>
struct LayerCompletionRingTraits;  // no primary definition: unregistered message types do not compile

template <>
struct LayerCompletionRingTraits<LayerCompletionMessage> {
    static constexpr uint32_t magic = 0x4C435131u;  // 'LCQ1'
    static constexpr std::size_t cell_alignment = alignof(LayerCompletionMessage);
};

template <>
struct LayerCompletionRingTraits<LayerCompletionMessageV2> {
    static constexpr uint32_t magic = 0x4C435132u;  // 'LCQ2'
    static constexpr std::size_t cell_alignment = kLayerCompletionCacheLine;
};

// One ring slot. `sequence` gates ownership (Vyukov): producers wait for
// sequence==pos, consumers wait for sequence==pos+1.
template <typename MsgT>
struct alignas(LayerCompletionRingTraits<MsgT>::cell_alignment) LayerCompletionCellT {
    std::atomic<uint64_t> sequence;
    MsgT msg;
};

using LayerCompletionCell = LayerCompletionCellT<LayerCompletionMessage>;      // v1
using LayerCompletionCellV2 = LayerCompletionCellT<LayerCompletionMessageV2>;  // v2

// Ring header — shared by both protocol versions; do not change its layout
// (magic identifies the version, so the header can stay common).
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

// Cells start on the first cell-aligned offset past the header.
template <typename MsgT>
inline constexpr std::size_t layer_completion_cells_offset() {
    using Cell = LayerCompletionCellT<MsgT>;
    return ((sizeof(LayerCompletionRingHeader) + alignof(Cell) - 1) / alignof(Cell)) * alignof(Cell);
}

template <typename MsgT>
inline constexpr std::size_t kLayerCompletionRingBytes =
    layer_completion_cells_offset<MsgT>() +
    static_cast<std::size_t>(kLayerCompletionRingCapacity) * sizeof(LayerCompletionCellT<MsgT>);

// Wire geometry: these offsets are the cross-process contract, so a drift is a compile error.
static_assert(offsetof(LayerCompletionRingHeader, enqueue_pos) == 0);
static_assert(offsetof(LayerCompletionRingHeader, dequeue_pos) == kLayerCompletionCacheLine);
static_assert(offsetof(LayerCompletionRingHeader, capacity) == kLayerCompletionCacheLine + 8);
static_assert(sizeof(LayerCompletionRingHeader) == 2 * kLayerCompletionCacheLine);

// V1 (frozen)
static_assert(alignof(LayerCompletionCell) == 8);
static_assert(sizeof(LayerCompletionCell) == 32);
static_assert(offsetof(LayerCompletionCell, msg) == 8);
static_assert(layer_completion_cells_offset<LayerCompletionMessage>() == 128);
static_assert(kLayerCompletionRingBytes<LayerCompletionMessage> == 32896);

// V2 (one cache line per cell)
static_assert(alignof(LayerCompletionCellV2) == kLayerCompletionCacheLine);
static_assert(sizeof(LayerCompletionCellV2) == kLayerCompletionCacheLine);
static_assert(offsetof(LayerCompletionCellV2, msg) == 8);
static_assert(layer_completion_cells_offset<LayerCompletionMessageV2>() % kLayerCompletionCacheLine == 0);
static_assert(layer_completion_cells_offset<LayerCompletionMessageV2>() == 128);
static_assert(kLayerCompletionRingBytes<LayerCompletionMessageV2> % kLayerCompletionCacheLine == 0);
static_assert(kLayerCompletionRingBytes<LayerCompletionMessageV2> == 65664);

}  // namespace tt::tt_metal::internal
