// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host-side checks of the rotating KV window's slot geometry (ring_attention_kv_window.hpp): the gather packs each
// readiness part's ranges into a slot in send order, and the SDPA reader finds a chunk there with kv_window_locate.

#include <algorithm>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/ring_attention_kv_window.hpp"

namespace {

using namespace ttnn::ring_attention_all_gather;

TEST(RingJointKvWindow, PartsPackEveryRowOnceAndLocateFindsEveryChunk) {
    for (uint32_t chunk_tiles : {4u, 8u, 16u}) {
        for (uint32_t bands = 1; bands <= 4; ++bands) {
            for (uint32_t valid_rows = 0; valid_rows <= 40 * chunk_tiles + 3; valid_rows += 3) {
                for (uint32_t key = 0; key < 8; ++key) {
                    SCOPED_TRACE(
                        ::testing::Message() << "chunk_tiles=" << chunk_tiles << " bands=" << bands
                                             << " valid_rows=" << valid_rows << " key=" << key);
                    const uint32_t num_chunks = readiness_split_chunks(valid_rows, chunk_tiles);
                    const uint32_t slot_rows = kv_window_slot_rows(valid_rows, bands, chunk_tiles);
                    EXPECT_LE(slot_rows, kv_window_slot_rows_bound(valid_rows, 4, chunk_tiles));
                    // The writer's packing: (part, slot row) of every source row.
                    std::vector<uint32_t> part_of_row(valid_rows, kReadinessParts);
                    std::vector<uint32_t> slot_row_of_row(valid_rows, 0);
                    for (uint32_t part = 0; part < kReadinessParts; ++part) {
                        uint32_t slot_row = 0;
                        for (uint32_t range = 0; range < bands; ++range) {
                            const RowRange rows =
                                readiness_rows(num_chunks, bands, chunk_tiles, valid_rows, part, range, key);
                            if (rows.end <= rows.begin) {
                                continue;
                            }
                            EXPECT_EQ(
                                slot_row,
                                kv_window_range_row(num_chunks, bands, chunk_tiles, valid_rows, part, range, key));
                            for (uint32_t row = rows.begin; row < rows.end; ++row) {
                                EXPECT_EQ(part_of_row[row], kReadinessParts) << "row " << row << " sent twice";
                                part_of_row[row] = part;
                                slot_row_of_row[row] = slot_row + row - rows.begin;
                            }
                            slot_row += rows.end - rows.begin;
                        }
                        EXPECT_LE(slot_row, slot_rows);
                    }
                    for (uint32_t row = 0; row < valid_rows; ++row) {
                        EXPECT_LT(part_of_row[row], kReadinessParts) << "row " << row << " never sent";
                    }
                    // The reader: a chunk's rows sit where the writer put them, and its range ends at row_end.
                    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
                        const KvWindowChunk loc =
                            kv_window_locate(chunk, num_chunks, bands, chunk_tiles, valid_rows, key);
                        EXPECT_EQ(loc.part, readiness_part_of(chunk, num_chunks, bands, key));
                        const uint32_t first = chunk * chunk_tiles;
                        const uint32_t last = std::min(first + chunk_tiles, valid_rows);
                        EXPECT_EQ(loc.row_end - loc.row >= last - first, true);
                        for (uint32_t row = first; row < last; ++row) {
                            EXPECT_EQ(part_of_row[row], loc.part);
                            EXPECT_EQ(slot_row_of_row[row], loc.row + row - first);
                            EXPECT_LT(slot_row_of_row[row], loc.row_end);
                        }
                    }
                }
            }
        }
    }
}

TEST(RingJointKvWindow, LayoutCutsTheWindowIntoThisLaunchsParts) {
    constexpr uint32_t window = kReadinessParts + 1;
    for (uint32_t chunk_tiles : {8u, 16u}) {
        for (uint32_t bands = 1; bands <= 4; ++bands) {
            const uint32_t max_rows = 4160;  // a 133,120-token cache slab
            const uint32_t cap_rows = kv_window_slot_rows_bound(max_rows, 4, chunk_tiles);
            const uint32_t window_rows = kv_window_rows(window, cap_rows);
            for (uint32_t valid_rows : {0u, 7u, 160u, 2080u, max_rows}) {
                SCOPED_TRACE(
                    ::testing::Message() << "chunk_tiles=" << chunk_tiles << " bands=" << bands
                                         << " valid_rows=" << valid_rows);
                const KvWindowLayout layout = kv_window_layout(valid_rows, bands, chunk_tiles, window, cap_rows);
                // Never fewer slots than the window promises; every part fits its slot; slots start on whole chunks.
                EXPECT_GE(layout.rot_slots, window);
                EXPECT_GE(layout.diam_slots, kKvWindowDiamParts);
                EXPECT_GE(layout.part_rows, kv_window_slot_rows(valid_rows, bands, chunk_tiles));
                EXPECT_EQ(layout.part_rows % chunk_tiles, 0u);
                for (uint32_t dir = 0; dir < 2; ++dir) {
                    for (uint32_t unit = 0; unit < 4 * kReadinessParts; ++unit) {
                        const uint32_t row = layout.rot_row(dir, unit);
                        EXPECT_GE(row, dir * layout.region_rows);
                        EXPECT_LE(row + layout.part_rows, (dir + 1) * layout.region_rows);
                        // A unit's slot is next used by the unit `rot_slots` later, never by one in between.
                        for (uint32_t later = unit + 1; later < unit + layout.rot_slots && later < unit + 64; ++later) {
                            EXPECT_NE(layout.rot_row(dir, later), row);
                        }
                    }
                }
                for (uint32_t part = 0; part < kReadinessParts; ++part) {
                    EXPECT_GE(layout.diam_row(part), 2 * layout.region_rows);
                    EXPECT_LE(layout.diam_row(part) + layout.part_rows, window_rows);
                }
            }
            // The largest slice gets exactly the window; a short one never reuses a slot.
            EXPECT_EQ(kv_window_layout(max_rows, 4, chunk_tiles, window, cap_rows).rot_slots >= window, true);
            EXPECT_GE(kv_window_layout(160, bands, chunk_tiles, window, cap_rows).rot_slots, 4 * kReadinessParts);
        }
    }
}

TEST(RingJointKvWindow, ProgressFieldsRoundTrip) {
    const uint32_t progress = kv_window_progress(3, 17, kKvWindowDone);
    EXPECT_EQ(kv_window_field(progress, 0), 3u);
    EXPECT_EQ(kv_window_field(progress, 1), 17u);
    EXPECT_EQ(kv_window_field(progress, 2), kKvWindowDone);
}

}  // namespace
