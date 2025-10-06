// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/core_coord.hpp>

#include <vector>
#include <limits>

namespace tt::tt_metal {

struct UncompressedBufferPageMapping {
    // Represents a page on device which doesn't match any host page within core_host_page_indices.
    static constexpr uint32_t PADDING = std::numeric_limits<uint32_t>::max();

    std::vector<CoreCoord> all_cores;
    // For each core, a vector of host page indices (or PADDING if there's no corresponding host page).
    std::vector<std::vector<uint32_t>> core_host_page_indices;
};

}  // namespace tt::tt_metal
