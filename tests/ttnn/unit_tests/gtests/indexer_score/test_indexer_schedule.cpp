// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host-side checks of indexer_schedule::for_core, the band placement shared by the indexer_score kernels.

#include <algorithm>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "ttnn/operations/experimental/indexer_score/device/kernels/indexer_schedule.hpp"

namespace {

using indexer_schedule::for_core;

constexpr indexer_ring_schedule::Geometry unfused_geometry(uint32_t units, uint32_t num_blocks, uint32_t cols) {
    return {1, units, units, 1, num_blocks, cols, false};
}

// Core 5 of a 2-block x 4-column grid (one row per block) is block 1, column 1: lane 1 + 1 * 2.
static_assert(for_core<false>(5, 1, unfused_geometry(20, 2, 4)).band_start == 3);
static_assert(for_core<false>(5, 1, unfused_geometry(20, 2, 4)).band_stride == 8);
static_assert(for_core<false>(5, 1, unfused_geometry(20, 2, 4)).band_count == 3);  // bands 3, 11, 19

// Band j of a cell is lane + j * lanes, so every band lands on exactly one cell, per-cell counts differ by at
// most one, and any valid prefix of the bands (a runtime kv_len) spreads over every cell.
TEST(IndexerSchedule, UnfusedDealsBandsRoundRobinOverLanes) {
    for (uint32_t num_blocks = 1; num_blocks <= 4; ++num_blocks) {
        for (uint32_t cols = 1; cols <= 8; ++cols) {
            for (uint32_t group_rows = 1; group_rows <= 2; ++group_rows) {
                for (uint32_t units = 0; units <= 70; ++units) {
                    SCOPED_TRACE(
                        ::testing::Message() << "blocks=" << num_blocks << " cols=" << cols
                                             << " group_rows=" << group_rows << " units=" << units);
                    const indexer_ring_schedule::Geometry g = unfused_geometry(units, num_blocks, cols);
                    const uint32_t lanes = num_blocks * cols;
                    const uint32_t rows = num_blocks * group_rows;
                    std::vector<uint32_t> owner_count(units, 0);
                    uint32_t min_count = UINT32_MAX;
                    uint32_t max_count = 0;
                    for (uint32_t core_id = 0; core_id < rows * cols; ++core_id) {
                        const auto s = for_core<false>(core_id, group_rows, g);
                        const uint32_t lane = s.block + s.col * num_blocks;
                        EXPECT_EQ(s.row, core_id / cols);
                        EXPECT_EQ(s.col, core_id % cols);
                        EXPECT_EQ(s.block, s.row / group_rows);
                        EXPECT_EQ(s.row_group, s.row % group_rows);
                        EXPECT_EQ(s.band_start, lane);
                        EXPECT_EQ(s.band_stride, lanes);
                        min_count = std::min(min_count, s.band_count);
                        max_count = std::max(max_count, s.band_count);
                        for (uint32_t j = 0; j < s.band_count; ++j) {
                            const uint32_t band = s.band_start + j * s.band_stride;
                            ASSERT_LT(band, units);
                            if (s.row_group == 0) {  // the block's other rows repeat its bands
                                ++owner_count[band];
                            }
                        }
                    }
                    for (uint32_t band = 0; band < units; ++band) {
                        EXPECT_EQ(owner_count[band], 1u) << "band " << band;
                    }
                    EXPECT_LE(max_count - min_count, 1u);
                    // The host pads every cell to this many bands (max_bands).
                    EXPECT_EQ(max_count, (units + lanes - 1) / lanes);

                    for (uint32_t prefix = 1; prefix <= units; ++prefix) {
                        uint32_t min_in_prefix = UINT32_MAX;
                        uint32_t max_in_prefix = 0;
                        for (uint32_t core_id = 0; core_id < rows * cols; ++core_id) {
                            const auto s = for_core<false>(core_id, group_rows, g);
                            uint32_t n = 0;
                            for (uint32_t j = 0; j < s.band_count; ++j) {
                                n += s.band_start + j * s.band_stride < prefix ? 1 : 0;
                            }
                            min_in_prefix = std::min(min_in_prefix, n);
                            max_in_prefix = std::max(max_in_prefix, n);
                        }
                        EXPECT_LE(max_in_prefix - min_in_prefix, 1u) << "prefix " << prefix;
                    }
                }
            }
        }
    }
}

// The fused ring path keeps its lane identity and the ring-arrival band count.
TEST(IndexerSchedule, RingKeepsLaneAndRingBandCount) {
    for (uint32_t ring_size = 1; ring_size <= 8; ++ring_size) {
        for (uint32_t num_blocks = 1; num_blocks <= 3; ++num_blocks) {
            for (uint32_t cols = 1; cols <= 6; ++cols) {
                for (uint32_t units = 0; units <= 24; ++units) {
                    for (bool rotate : {false, true}) {
                        SCOPED_TRACE(
                            ::testing::Message() << "ring=" << ring_size << " blocks=" << num_blocks << " cols=" << cols
                                                 << " units=" << units << " rotate=" << rotate);
                        const indexer_ring_schedule::Geometry g{ring_size, units, units, 1, num_blocks, cols, rotate};
                        for (uint32_t core_id = 0; core_id < num_blocks * cols; ++core_id) {
                            const auto s = for_core<true>(core_id, 1, g);
                            const uint32_t lane = s.block + s.col * num_blocks;
                            EXPECT_EQ(s.band_start, lane);
                            EXPECT_EQ(s.band_count, indexer_ring_schedule::band_count(g, lane));
                            indexer_ring_schedule::Iterator it(g, lane);
                            uint32_t walked = 0;
                            uint32_t physical_start = 0;
                            while (it.next([](uint32_t shard) { return shard; }, physical_start)) {
                                ++walked;
                            }
                            EXPECT_EQ(s.band_count, walked);
                        }
                    }
                }
            }
        }
    }
}

}  // namespace
