// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Wire payload for a single prefill layer-completion event. Trivially
// copyable, standard-layout — byte-copied verbatim through the host-local SHM
// ring and across MPI to the master rank. The transport is agnostic to
// the fields' meaning; `seq` is a producer-supplied, globally-dense
// ordering key the master sequences on before re-emitting completions
// into the scheduler-facing counter channel.

#pragma once

#include <cstdint>

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

static_assert(sizeof(LayerCompletionMessage) == 24, "LayerCompletionMessage wire size changed");

// A message whose `reserved` field equals this is an end-of-stream SENTINEL, not a real completion:
// a subordinate router sends exactly one as its final message at teardown so the master knows no more
// completions will arrive from that rank and can stop without cancelling a live receive (coordinated
// teardown — see LayerCompletionRouter). Real completions always carry reserved == 0.
inline constexpr uint32_t kLayerCompletionSentinel = 0xFFFFFFFFu;

inline bool is_layer_completion_sentinel(const LayerCompletionMessage& m) noexcept {
    return m.reserved == kLayerCompletionSentinel;
}

}  // namespace tt::tt_metal::internal
