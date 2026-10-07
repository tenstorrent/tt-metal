// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "tt-metalium/circular_buffer_constants.h"

namespace experimental {

inline constexpr std::uint32_t kScratchCbFirstId = BLACKHOLE_NUM_CIRCULAR_BUFFERS;
inline constexpr std::uint32_t kScratchCbChannels = BLACKHOLE_NUM_SCRATCH_SYNC_CHANNELS;

namespace scratch_cb_detail {

template <std::uint32_t Ch>
constexpr std::uint32_t cb_id() {
    static_assert(Ch < kScratchCbChannels, "scratch CB channel must be 0 or 1");
    return kScratchCbFirstId + Ch;
}

}  // namespace scratch_cb_detail

}  // namespace experimental
