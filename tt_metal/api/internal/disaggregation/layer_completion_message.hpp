// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Wire payloads for prefill layer-completion events. Trivially copyable,
// standard-layout — byte-copied verbatim through the host-local SHM ring and
// across MPI to the master rank. The transport is agnostic to the fields'
// meaning.
//
// Two protocol versions coexist; the version is selected once per job
// (PREFILL_LAYER_COMPLETION_PROTOCOL) and keyed by the ring magic, so a
// mismatched peer fails at connect() rather than corrupting:
//
//   V1 (LayerCompletionMessage, 24B, magic 'LCQ1'): the master reorders by
//   the globally-dense `seq` and injects a bare COUNT into the
//   scheduler-facing InterProcessCounterChannel. The scheduler correlates
//   ticks with its own in-order chunk FIFO, which forces per-request
//   in-order consumption (head-of-line blocking — issue #54632). This
//   format is FROZEN: the static asserts below make byte-compat executable.
//
//   V2 (LayerCompletionMessageV2, 40B, magic 'LCQ2'): each completion is
//   fully self-describing — (request_id, slot_id, position range, layer
//   range) — so the master forwards as-arrived (no reorder) into a
//   scheduler-facing structured ring and the consumer keys work on content,
//   not arrival order. One message per completion EVENT at whatever span
//   the model reports: per-layer hooks emit [l, l+1); a whole-stage
//   completion emits one [first, end) message — never split, never
//   coalesced.

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace tt::tt_metal::internal {

// Fixed ring capacity (slots). Power of two so index = pos & mask. Sized
// well above the in-flight completion count; the consumer drains
// continuously, so this is headroom, not a steady-state depth.
inline constexpr uint32_t kLayerCompletionRingCapacity = 1024;
inline constexpr uint64_t kLayerCompletionRingMask = kLayerCompletionRingCapacity - 1;

struct LayerCompletionMessage {
    // Globally-dense ordering key, supplied by the producer. The master
    // re-emits completions strictly in ascending `seq` (0,1,2,…). For
    // pipelined prefill the runner computes e.g. request_index*num_layers
    // + global_layer_idx; the C++ layer never interprets it.
    uint64_t seq = 0;
    // World rank of the host whose runner produced this completion.
    uint32_t source_rank = 0;
    // Layer that completed (global layer index).
    uint32_t layer_idx = 0;
    // Request/chunk this completion belongs to (diagnostic in step 1).
    uint32_t request_id = 0;
    // Explicit pad → 24 bytes, 8-byte alignment, no implementation-
    // defined tail padding across the wire.
    uint32_t reserved = 0;
};

// V1 wire contract — frozen. Offset drift between producer/router/scheduler
// builds would silently corrupt SHM and MPI traffic, so it is a compile error.
static_assert(sizeof(LayerCompletionMessage) == 24, "LayerCompletionMessage wire size changed");
static_assert(alignof(LayerCompletionMessage) == 8, "LayerCompletionMessage alignment changed");
static_assert(std::is_trivially_copyable_v<LayerCompletionMessage>);
static_assert(std::is_standard_layout_v<LayerCompletionMessage>);
static_assert(offsetof(LayerCompletionMessage, seq) == 0);
static_assert(offsetof(LayerCompletionMessage, source_rank) == 8);
static_assert(offsetof(LayerCompletionMessage, layer_idx) == 12);
static_assert(offsetof(LayerCompletionMessage, request_id) == 16);
static_assert(offsetof(LayerCompletionMessage, reserved) == 20);
static_assert(
    offsetof(LayerCompletionMessage, reserved) + sizeof(uint32_t) == sizeof(LayerCompletionMessage),
    "LayerCompletionMessage grew tail padding");

struct LayerCompletionMessageV2 {
    // Producer-supplied ordering key, request_id*num_layers + layer_start.
    // DIAGNOSTIC ONLY in v2: the master forwards as-arrived and never
    // reorders on it; consumers must not assume density (a multi-layer
    // completion skips the seqs of the layers it covers).
    uint64_t seq = 0;
    // World rank of the host whose runner produced this completion.
    uint32_t source_rank = 0;
    // Request/chunk ordinal this completion belongs to.
    uint32_t request_id = 0;
    // Cache user slot the KV was written into.
    uint32_t slot_id = 0;
    // Absolute KV-position range [pos_start, pos_end) this completion covers.
    uint32_t pos_start = 0;
    uint32_t pos_end = 0;
    // Global layer range [layer_start, layer_end) this completion covers.
    // Per-layer emission is the degenerate case [l, l+1); a stage-level
    // event covers [first_layer_idx, first_layer_idx + stage_layers).
    uint32_t layer_start = 0;
    uint32_t layer_end = 0;
    // 0 for real completions; kLayerCompletionSentinel marks end-of-stream.
    uint32_t flags = 0;
};

// V2 wire contract — pinned for the same reason as v1.
static_assert(sizeof(LayerCompletionMessageV2) == 40, "LayerCompletionMessageV2 wire size changed");
static_assert(alignof(LayerCompletionMessageV2) == 8, "LayerCompletionMessageV2 alignment changed");
static_assert(std::is_trivially_copyable_v<LayerCompletionMessageV2>);
static_assert(std::is_standard_layout_v<LayerCompletionMessageV2>);
static_assert(offsetof(LayerCompletionMessageV2, seq) == 0);
static_assert(offsetof(LayerCompletionMessageV2, source_rank) == 8);
static_assert(offsetof(LayerCompletionMessageV2, request_id) == 12);
static_assert(offsetof(LayerCompletionMessageV2, slot_id) == 16);
static_assert(offsetof(LayerCompletionMessageV2, pos_start) == 20);
static_assert(offsetof(LayerCompletionMessageV2, pos_end) == 24);
static_assert(offsetof(LayerCompletionMessageV2, layer_start) == 28);
static_assert(offsetof(LayerCompletionMessageV2, layer_end) == 32);
static_assert(offsetof(LayerCompletionMessageV2, flags) == 36);
static_assert(
    offsetof(LayerCompletionMessageV2, flags) + sizeof(uint32_t) == sizeof(LayerCompletionMessageV2),
    "LayerCompletionMessageV2 has tail padding");

// A message whose sentinel slot equals this is an end-of-stream SENTINEL, not a real completion:
// a subordinate router sends exactly one as its final message at teardown so the master knows no more
// completions will arrive from that rank and can stop without cancelling a live receive (coordinated
// teardown — see LayerCompletionRouter). Real completions always carry 0 there.
inline constexpr uint32_t kLayerCompletionSentinel = 0xFFFFFFFFu;

inline bool is_layer_completion_sentinel(const LayerCompletionMessage& m) noexcept {
    return m.reserved == kLayerCompletionSentinel;
}

inline bool is_layer_completion_sentinel(const LayerCompletionMessageV2& m) noexcept {
    return m.flags == kLayerCompletionSentinel;
}

// Build the end-of-stream sentinel for the message version in use.
template <typename MsgT>
MsgT layer_completion_sentinel(uint32_t source_rank);

template <>
inline LayerCompletionMessage layer_completion_sentinel<LayerCompletionMessage>(uint32_t source_rank) {
    LayerCompletionMessage m{};
    m.source_rank = source_rank;
    m.reserved = kLayerCompletionSentinel;
    return m;
}

template <>
inline LayerCompletionMessageV2 layer_completion_sentinel<LayerCompletionMessageV2>(uint32_t source_rank) {
    LayerCompletionMessageV2 m{};
    m.source_rank = source_rank;
    m.flags = kLayerCompletionSentinel;
    return m;
}

}  // namespace tt::tt_metal::internal
