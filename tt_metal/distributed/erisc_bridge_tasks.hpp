// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// What a bridge leg hands its consumer; mirrors host_tasks.hpp with `arena` in place of `core`.
#pragma once

#include <cstdint>
#include <optional>

namespace tt::tt_metal::experimental {

struct BridgeSendTask {
    uint32_t arena = 0;        // bridge_arena_index(link, sender channel, chans_per_link)
    uint64_t page_offset = 0;  // region-relative
    uint32_t page_bytes = 0;
    uint32_t ordering_cntr = 0;  // absolute and monotonic per sender
    uint32_t length = 0;
    uint32_t origin = 0;
    uint64_t elapsed = 0;      // ERISC cycles, same-hart delta; never converted to time
    uint32_t release_cyc = 0;  // sender clock at release; only differences are meaningful
    // The far receiver ring's slot, as the sending router computed it. Carried untouched to H2E,
    // which checks it against where it is about to inject. Empty when no router produced it.
    std::optional<uint16_t> far_slot;
};

// What the RX host hands its consumer. `slot` is this host's own Rx ring slot, NOT far_slot.
struct BridgeDeliverTask {
    uint32_t arena = 0;
    uint32_t slot = 0;  // slot within the arena's segment
    uint64_t page_offset = 0;
    uint32_t page_bytes = 0;
    uint32_t ordering_cntr = 0;
    uint32_t length = 0;
    uint32_t origin = 0;
    uint64_t elapsed = 0;
    // Carried the whole way: in the pipeline this is the originating ERISC's release time, and
    // the hop that drops it makes the end-to-end span unattributable. Zero when unstamped.
    uint32_t release_cyc = 0;
    std::optional<uint16_t> far_slot;
};

}  // namespace tt::tt_metal::experimental
