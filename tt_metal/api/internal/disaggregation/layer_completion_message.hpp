// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Wire payload for prefill layer-completion events, byte-copied through the host-local SHM ring and over MPI
// to the master rank. Each message names its chunk (slot, position range) and the layers now resident for it,
// so the master forwards as-arrived and consumers key on content. One message per completion event at the
// span the model reports: per-layer hooks emit [l, l+1); a stage-level completion emits one [first, end).

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace tt::tt_metal::internal {

// Ring capacity in cells; a power of two so index = pos & mask.
inline constexpr uint32_t kLayerCompletionRingCapacity = 1024;
inline constexpr uint64_t kLayerCompletionRingMask = kLayerCompletionRingCapacity - 1;

struct LayerCompletionMessage {
    // Cache slot the KV was written into.
    uint32_t slot_id = 0;
    // KV positions [pos_start, pos_end) this completion covers.
    uint32_t pos_start = 0;
    uint32_t pos_end = 0;
    // Global layers [layer_start, layer_end); per-layer emission is [l, l+1).
    uint32_t layer_start = 0;
    uint32_t layer_end = 0;
    // 0 for completions; kLayerCompletionSentinel for end-of-stream.
    uint32_t flags = 0;
    // Host wall clock (ns since the Unix epoch) when a host first saw the completion; forwarding keeps it.
    uint64_t host_timestamp_ns = 0;
};

// Offsets are shared across processes, so drift is a compile error.
static_assert(sizeof(LayerCompletionMessage) == 32, "LayerCompletionMessage wire size changed");
static_assert(std::is_trivially_copyable_v<LayerCompletionMessage>);
static_assert(std::is_standard_layout_v<LayerCompletionMessage>);
static_assert(offsetof(LayerCompletionMessage, slot_id) == 0);
static_assert(offsetof(LayerCompletionMessage, pos_start) == 4);
static_assert(offsetof(LayerCompletionMessage, pos_end) == 8);
static_assert(offsetof(LayerCompletionMessage, layer_start) == 12);
static_assert(offsetof(LayerCompletionMessage, layer_end) == 16);
static_assert(offsetof(LayerCompletionMessage, flags) == 20);
static_assert(offsetof(LayerCompletionMessage, host_timestamp_ns) == 24);

inline constexpr uint32_t kLayerCompletionSentinel = 0xFFFFFFFFu;

inline uint64_t layer_completion_host_timestamp_ns() noexcept {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch())
            .count());
}

inline bool is_layer_completion_sentinel(const LayerCompletionMessage& m) noexcept {
    return m.flags == kLayerCompletionSentinel;
}

inline LayerCompletionMessage layer_completion_sentinel() {
    LayerCompletionMessage m{};
    m.flags = kLayerCompletionSentinel;
    return m;
}

}  // namespace tt::tt_metal::internal
