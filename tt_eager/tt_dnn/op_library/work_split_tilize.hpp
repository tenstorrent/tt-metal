// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

//
// Contains utility functions for partitioning work between multiple cores.
//

#pragma once

#include "tt_metal/common/core_coord.h"


namespace tt::tt_metal {

struct BlockSplit {
    uint32_t ncores;
    CoreRangeSet all_cores;
    CoreRangeSet core_range;
    CoreRangeSet core_range_cliff;
    uint32_t nblocks_per_core;
    uint32_t nblocks_per_core_cliff;
};

inline BlockSplit split_blocks_for_tilize(CoreCoord grid_size, uint32_t nblocks) {
    const uint32_t nblocks_per_core = std::ceil(static_cast<float>(nblocks) / (grid_size.x * grid_size.y));
    const uint32_t ncores = std::ceil(static_cast<float>(nblocks) / nblocks_per_core);
    const uint32_t nblocks_per_core_cliff = nblocks % nblocks_per_core;
    const uint32_t ncores_x = grid_size.x;
    const uint32_t ncores_y = std::ceil(static_cast<float>(ncores) / ncores_x);
    const uint32_t ncores_x_cliff = ncores - (ncores_y - 1) * ncores_x;

    std::set<CoreRange> core_range, cliff_core_range;
    std::optional<CoreCoord> cliff_core;

    // Top non-cliff range (full rows)
    uint32_t top_range_end_y = ncores_y - (ncores_x_cliff > 0 || nblocks_per_core_cliff > 0 ? 1 : 0);
    if (top_range_end_y > 0) {
        auto range = CoreRange{CoreCoord{0, 0}, CoreCoord{ncores_x - 1, top_range_end_y - 1}};
        core_range.insert(range);
    }

    if (ncores_x_cliff > 0 && nblocks_per_core_cliff == 0) {
        // Last partial row (non-cliff)
        auto range = CoreRange{CoreCoord{0, ncores_y - 1}, CoreCoord{ncores_x_cliff - 1, ncores_y - 1}};
        core_range.insert(range);
    } else if (nblocks_per_core_cliff > 0) {
        // Last partial row (excluding last core) and single cliff core
        if (ncores_x_cliff > 1) {  // Add range only if there are cores before the cliff core
            auto range = CoreRange{CoreCoord{0, ncores_y - 1}, CoreCoord{ncores_x_cliff - 2, ncores_y - 1}};
            core_range.insert(range);
        }
        cliff_core = CoreCoord{ncores_x_cliff - 1, ncores_y - 1};
    }

    std::set<CoreRange> all_cores = core_range;

    if (cliff_core.has_value()) {
        cliff_core_range.insert(CoreRange{*cliff_core, *cliff_core});
        if (all_cores.size() == 1) {
            // Cliff core is in a new row, insert it into all_cores
            all_cores.insert(cliff_core_range.begin(), cliff_core_range.end());
        } else {
            // Cliff core is in the same row as the last core range, increment its end
            auto last_range = *all_cores.rbegin();
            auto node = all_cores.extract(last_range);
            node.value().end = *cliff_core;
            all_cores.insert(std::move(node));
        }
    }

    return BlockSplit{ncores, all_cores, core_range, cliff_core_range, nblocks_per_core, nblocks_per_core_cliff};
}

} // namespace tt::tt_metal
