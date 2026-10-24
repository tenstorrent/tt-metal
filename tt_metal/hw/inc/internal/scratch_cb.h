// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "tt-metalium/circular_buffer_constants.h"

namespace experimental {

// Scratch channels use the stream counters of the last CB IDs. The host does not reserve these IDs, so a program
// that uses channel Ch must not also create CB kScratchCbFirstId + Ch.
inline constexpr std::uint32_t kScratchCbChannels = 2;
inline constexpr std::uint32_t kScratchCbFirstId = NUM_CIRCULAR_BUFFERS - kScratchCbChannels;

namespace scratch_cb_detail {

template <std::uint32_t Ch>
constexpr std::uint32_t cb_id() {
    static_assert(Ch < kScratchCbChannels, "scratch CB channel must be 0 or 1");
    return kScratchCbFirstId + Ch;
}

}  // namespace scratch_cb_detail

}  // namespace experimental
