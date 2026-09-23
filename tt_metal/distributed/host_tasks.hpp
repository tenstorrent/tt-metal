// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// What a frame becomes between the legs: one on its way out, one that has arrived. Here
// rather than in a leg's own header so no leg has to include another's to name them.
#pragma once

#include <cstdint>

#include <tt-metalium/experimental/sockets/host_uva.hpp>

namespace tt::tt_metal::experimental {

// One frame ready to leave this host. `page_offset`/`page_bytes` name the whole
// [payload|trailer] unit, which the transport forwards verbatim.
struct SendTask {
    uint32_t core = 0;
    uint64_t page_offset = 0;
    uint32_t page_bytes = 0;
    tt_uva_t dst = kUvaNone;
    uint32_t length = 0;
    uint32_t origin = 0;
    uint64_t elapsed = 0;
};

// A frame that arrived from a peer and is sitting in this host's RX arena.
struct DeliverTask {
    uint32_t core = 0;
    uint32_t slot = 0;
    uint64_t page_offset = 0;
    uint32_t page_bytes = 0;
    tt_uva_t dst = kUvaNone;
    uint32_t length = 0;
    uint32_t origin = 0;
    uint64_t elapsed = 0;
};

}  // namespace tt::tt_metal::experimental
