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
// Cross-process atomics: the segment lives in shared memory mapped into
// every participant. std::atomic<uint64_t> is lock-free on the target
// and usable across processes when the storage is shared — the same
// guarantee inter_process_counter_layout.hpp relies on.
//
// The ring is templated on the message type; LayerCompletionRingTraits
// binds each message version to a magic (the on-disk version key that
// connect() validates) and a cell alignment. V1 cells stay naturally
// packed (32B, two per cache line — frozen wire format). V2 cells are
// padded to a full cache line: a packed 48B cell would straddle lines,
// so every v2 push/pop would touch two lines for half the cells and
// adjacent-cell false sharing would stop being a boundary-only case.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

#include <internal/disaggregation/layer_completion_message.hpp>

namespace tt::tt_metal::internal {

inline constexpr std::size_t kLayerCompletionCacheLine = 64;

// Per-message-version wire traits. No primary definition: an unregistered
// message type is a compile error, not a silent format.
template <typename MsgT>
struct LayerCompletionRingTraits;

template <>
struct LayerCompletionRingTraits<LayerCompletionMessage> {
    static constexpr uint32_t magic = 0x4C435131u;  // 'LCQ1'
    // Natural alignment — cells stay packed 32B (frozen v1 wire format).
    static constexpr std::size_t cell_alignment = alignof(LayerCompletionMessage);
};

template <>
struct LayerCompletionRingTraits<LayerCompletionMessageV2> {
    static constexpr uint32_t magic = 0x4C435132u;  // 'LCQ2'
    // One cache line per cell (see header comment).
    static constexpr std::size_t cell_alignment = kLayerCompletionCacheLine;
};

// v1 spelling, kept for existing references.
inline constexpr uint32_t kLayerCompletionRingMagic = LayerCompletionRingTraits<LayerCompletionMessage>::magic;

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

// ---------------------------------------------------------------------------
// Wire-geometry contract. These values ARE the protocol: connect() validates
// magic+capacity, and every other byte offset follows from the layouts below.
// A drift here means silent cross-process corruption, so it is a compile
// error. The v1 values additionally guard the byte-compat promise — v1
// geometry must not move while v2 is added alongside it.
// ---------------------------------------------------------------------------

// Header (shared). The alignas(64) members force alignof == 64, so sizeof
// rounds up to a full two cache lines; capacity/magic live in the second
// line's tail.
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
