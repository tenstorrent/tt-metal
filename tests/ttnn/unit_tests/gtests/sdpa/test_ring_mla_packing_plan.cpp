// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host-side checks of the split-KV ring MLA packed-source schedule and geometry, pure integer math shared by the
// reader, writer and compute kernels and by the program factory.

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "ttnn/operations/transformer/sdpa/device/kernels/ring_mla_geometry.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/ring_mla_packing_plan.hpp"

namespace {

using namespace ttnn::operations::transformer::sdpa::ring_joint;

struct ScheduleCase {
    uint32_t ring_size;
    uint32_t group;
    uint32_t region_tiles;
    uint32_t slabs;
    uint32_t chunk_tiles;
};

// Production (SP8 x TP4, q32/k640 -> region 5 tiles), small LB geometries, single-slab caches, chunks narrower
// than a region, wider than a row, and partial final chunks.
const ScheduleCase kScheduleCases[] = {
    {32, 4, 5, 11, 20},
    {32, 4, 5, 2, 20},
    {32, 4, 5, 1, 20},
    {8, 4, 2, 3, 11},
    {8, 4, 2, 1, 4},
    {8, 2, 3, 4, 1},
    {8, 2, 3, 4, 64},
    {16, 8, 1, 5, 7},
    {4, 4, 4, 3, 6},
};

std::vector<std::vector<uint32_t>> arrival_orders(uint32_t ring_size) {
    std::vector<uint32_t> identity(ring_size);
    std::iota(identity.begin(), identity.end(), 0u);
    std::vector<std::vector<uint32_t>> orders{identity};
    orders.emplace_back(identity.rbegin(), identity.rend());
    // Rotated: every row straddles two passes.
    std::vector<uint32_t> rotated(ring_size);
    for (uint32_t i = 0; i < ring_size; ++i) {
        rotated[i] = (i + 1) % ring_size;
    }
    orders.push_back(rotated);
    std::mt19937 rng(20261007);
    for (int i = 0; i < 4; ++i) {
        std::shuffle(identity.begin(), identity.end(), rng);
        orders.push_back(identity);
    }
    return orders;
}

uint32_t block_cyclic_global(const ScheduleCase& c, uint32_t rank, uint32_t local) {
    const uint32_t slab = local / c.region_tiles;
    return slab * c.region_tiles * c.ring_size + rank * c.region_tiles + local % c.region_tiles;
}

// Every tile of every source is attended exactly once across the passes; head tiles come from sources the
// chunk's readiness covers; newest tiles only from ranks already arrived; and mask runs, segments and the
// largest slab agree with locate() and the block-cyclic layout.
TEST(RingMLAPackingPlan, PassesCoverEverySourceTileOnceAndOnlyAfterItArrives) {
    for (const auto& c : kScheduleCases) {
        const uint32_t source_tiles = c.slabs * c.region_tiles;
        const uint32_t global_chunk_tiles = c.region_tiles * c.ring_size;
        for (const auto& order : arrival_orders(c.ring_size)) {
            SCOPED_TRACE(
                ::testing::Message() << "ring=" << c.ring_size << " group=" << c.group << " region=" << c.region_tiles
                                     << " slabs=" << c.slabs << " chunk=" << c.chunk_tiles
                                     << " first_arrival=" << order[0] << " second_arrival=" << order[1]);
            uint32_t next = 0;
            const PackedKVSchedule schedule = packed_kv_schedule(
                c.group,
                c.ring_size,
                source_tiles,
                source_tiles * c.ring_size,
                all_sources_mask(c.ring_size),
                c.region_tiles,
                c.chunk_tiles,
                /*sliding_window=*/false,
                [&] { return order[next++]; });
            ASSERT_TRUE(schedule.packed());
            ASSERT_EQ(schedule.ring_iterations, c.ring_size / c.group);
            ASSERT_EQ(next, c.ring_size);

            std::vector<uint32_t> arrival_index(c.ring_size);
            for (uint32_t i = 0; i < c.ring_size; ++i) {
                arrival_index[order[i]] = i;
            }
            std::vector<uint32_t> seen(c.ring_size * source_tiles, 0);
            uint32_t rows_seen = 0;
            for (uint32_t pass = 0; pass < schedule.ring_iterations; ++pass) {
                const PackedKVGroupPlan plan = schedule.pass_plan(pass);
                EXPECT_EQ(rows_seen & plan.newest_rows, 0u);
                rows_seen |= plan.newest_rows;
                const uint32_t* ids = &order[pass * c.group];
                std::vector<PackedKVMaskRun> runs(plan.chunk_tiles);
                for (uint32_t chunk = 0; chunk < plan.chunk_count(); ++chunk) {
                    const uint32_t valid = plan.valid_tiles(chunk);
                    ASSERT_GT(valid, 0u);
                    const uint32_t run_count = plan.mask_runs(chunk, ids, global_chunk_tiles, runs.data());
                    ASSERT_LE(run_count, plan.chunk_tiles);
                    ASSERT_EQ(runs[run_count - 1].column_end, valid);
                    uint32_t run = 0;
                    uint32_t run_begin = 0;
                    uint32_t max_slab = 0;
                    for (uint32_t column = 0; column < valid; ++column) {
                        const uint32_t stream_tile = chunk * plan.chunk_tiles + column;
                        const auto at = plan.locate(stream_tile, ids);
                        ASSERT_LT(at.rank, c.ring_size);
                        ASSERT_LT(at.local, source_tiles);
                        ++seen[at.rank * source_tiles + at.local];
                        max_slab = std::max(max_slab, at.local / c.region_tiles);
                        if (stream_tile < plan.head_stream_tiles()) {
                            const uint32_t member = stream_tile / plan.head_tiles();
                            EXPECT_EQ(at.rank, ids[member]);
                            EXPECT_LE(member, plan.last_source(chunk));
                        } else {
                            // A newest row joins a pass only after all its ranks arrived.
                            EXPECT_LT(arrival_index[at.rank], (pass + 1) * c.group);
                            EXPECT_EQ(plan.last_source(chunk), c.group - 1);
                        }
                        while (column >= runs[run].column_end) {
                            run_begin = runs[run++].column_end;
                        }
                        EXPECT_EQ(
                            runs[run].global_start_tile + (column - run_begin),
                            block_cyclic_global(c, at.rank, at.local));
                    }
                    EXPECT_EQ(plan.max_slab(chunk), max_slab);
                    for (uint32_t offset = 0; offset < valid;) {
                        const uint32_t length = plan.segment_tiles(chunk, offset);
                        ASSERT_GT(length, 0u);
                        const auto first = plan.locate(chunk * plan.chunk_tiles + offset, ids);
                        for (uint32_t t = 1; t < length; ++t) {
                            const auto at = plan.locate(chunk * plan.chunk_tiles + offset + t, ids);
                            EXPECT_EQ(at.rank, first.rank);
                            EXPECT_EQ(at.local, first.local + t);
                        }
                        offset += length;
                    }
                }
            }
            if (c.slabs > 1) {
                EXPECT_EQ(rows_seen, all_sources_mask(c.ring_size / c.group));
            }
            for (uint32_t tile = 0; tile < seen.size(); ++tile) {
                ASSERT_EQ(seen[tile], 1u) << "rank " << tile / source_tiles << " local " << tile % source_tiles;
            }
        }
    }
}

TEST(RingMLAPackingPlan, ReadinessWaitsEachSourceOnceInOrder) {
    const PackedKVGroupPlan plan = packed_kv_pass_plan({15, 4, 4, 5}, 0b1);
    PackedKVSourceReadiness readiness;
    std::vector<uint32_t> waits;
    const auto wait = [&](uint32_t source) { waits.push_back(source); };
    for (uint32_t chunk = 0; chunk < plan.chunk_count(); ++chunk) {
        readiness.wait_for_chunk(plan, chunk, wait);
        EXPECT_EQ(readiness.ready_sources, plan.last_source(chunk) + 1);
    }
    readiness.drain(4, wait);
    EXPECT_EQ(waits, (std::vector<uint32_t>{0, 1, 2, 3}));
}

TEST(RingMLAPackingPlan, GroupSizeFallsBackForUnsupportedSchedules) {
    const uint32_t all = all_sources_mask(32);
    EXPECT_EQ(packed_kv_source_group_size(4, 32, 55, 55 * 32, all), 4u);
    EXPECT_EQ(packed_kv_source_group_size(1, 32, 55, 55 * 32, all), 1u);
    EXPECT_EQ(packed_kv_source_group_size(3, 32, 55, 55 * 32, all), 1u);        // group does not divide the ring
    EXPECT_EQ(packed_kv_source_group_size(4, 32, 55, 55 * 32 + 1, all), 1u);    // logical length past capacity
    EXPECT_EQ(packed_kv_source_group_size(4, 32, 55, 0, all), 1u);              // empty
    EXPECT_EQ(packed_kv_source_group_size(4, 32, 55, 55 * 32, all & ~1u), 1u);  // inactive source
    EXPECT_EQ(packed_kv_source_group_size(4, 33, 55, 55 * 33, ~0u), 1u);        // ring too large
}

TEST(RingMLAPackingPlan, SourceTilesClipToTouchedSlabs) {
    // Region 5, 32 sources: one global chunk is 160 tiles.
    EXPECT_EQ(packed_kv_source_tiles(55, 160, 5, 32), 5u);
    EXPECT_EQ(packed_kv_source_tiles(55, 161, 5, 32), 10u);
    EXPECT_EQ(packed_kv_source_tiles(55, 1760, 5, 32), 55u);
    EXPECT_EQ(packed_kv_source_tiles(55, 100000, 5, 32), 55u);
}

TEST(RingMLAPackingPlan, RowBoundsSkipPaddedRowsAndClipToLogicalEnd) {
    const auto rows_from = [](const std::vector<uint32_t>& tiles) {
        return [tiles](uint32_t row) { return tiles[row]; };
    };
    auto bounds = packed_kv_row_bounds(3, 100, 40, rows_from({50, 51, 52}));
    EXPECT_EQ(bounds.visible_end, 50u);
    EXPECT_EQ(bounds.masked_from, 53u);
    EXPECT_TRUE(bounds.heads_visible);

    bounds = packed_kv_row_bounds(3, 52, 51, rows_from({50, 51, 52}));
    EXPECT_EQ(bounds.visible_end, 50u);
    EXPECT_EQ(bounds.masked_from, 52u);
    EXPECT_FALSE(bounds.heads_visible);

    // A padded row sees nothing, so nothing is visible to every row; it does not widen masked_from.
    bounds = packed_kv_row_bounds(3, 100, 0, rows_from({50, kPackedKVInvalidRowTile, 51}));
    EXPECT_EQ(bounds.visible_end, 0u);
    EXPECT_EQ(bounds.masked_from, 52u);
    EXPECT_TRUE(bounds.heads_visible);
}

// For every chunk and every consistent pair of row bounds: dropped columns are masked for every row, the live
// width is whole subblocks (or the caller's width), an unmasked chunk is fully visible, and a masked chunk's
// runs cover exactly the live columns with their block-cyclic global tiles.
TEST(RingMLAPackingPlan, ChunkPlanDropsOnlyFullyMaskedColumnsAndStampsOnlyWhenNeeded) {
    for (const auto& c : kScheduleCases) {
        if (c.chunk_tiles > kMaxPackedKVChunkTiles) {
            continue;
        }
        const uint32_t source_tiles = c.slabs * c.region_tiles;
        const uint32_t global_chunk_tiles = c.region_tiles * c.ring_size;
        const uint32_t logical_tiles = source_tiles * c.ring_size;
        std::vector<uint32_t> order(c.ring_size);
        std::iota(order.begin(), order.end(), 0u);
        uint32_t next = 0;
        const PackedKVSchedule schedule = packed_kv_schedule(
            c.group,
            c.ring_size,
            source_tiles,
            logical_tiles,
            all_sources_mask(c.ring_size),
            c.region_tiles,
            c.chunk_tiles,
            false,
            [&] { return order[next++]; });
        const uint32_t subblock = std::min(4u, c.chunk_tiles);
        // Q rows of the newest global chunk, as one Q chunk of each shard sees them, plus a few prefix rows.
        const uint32_t newest_start = (c.slabs - 1) * global_chunk_tiles;
        std::vector<uint32_t> probes;
        for (uint32_t q = 0; q < global_chunk_tiles; q += std::max(1u, c.region_tiles / 2)) {
            probes.push_back(newest_start + q);
        }
        probes.push_back(1);
        probes.push_back(logical_tiles - 1);
        for (uint32_t pass = 0; pass < schedule.ring_iterations; ++pass) {
            const PackedKVGroupPlan plan = schedule.pass_plan(pass);
            const uint32_t* ids = &order[pass * c.group];
            const uint32_t head_global_end = plan.head_tiles() / c.region_tiles * global_chunk_tiles;
            for (uint32_t q_first : probes) {
                for (uint32_t q_rows : {1u, 2u, 7u}) {
                    const uint32_t q_last = std::min(q_first + q_rows - 1, logical_tiles - 1);
                    const auto bounds =
                        packed_kv_row_bounds(q_last - q_first + 1, logical_tiles, head_global_end, [&](uint32_t row) {
                            return q_first + row;
                        });
                    for (uint32_t chunk = 0; chunk < plan.chunk_count(); ++chunk) {
                        SCOPED_TRACE(
                            ::testing::Message() << "ring=" << c.ring_size << " group=" << c.group << " pass=" << pass
                                                 << " chunk=" << chunk << " q=[" << q_first << "," << q_last << "]");
                        const uint32_t valid = plan.valid_tiles(chunk);
                        std::vector<PackedKVMaskRun> runs(plan.chunk_tiles);
                        const PackedKVChunkPlan chunk_plan = packed_kv_chunk_plan(
                            plan, chunk, ids, global_chunk_tiles, bounds, subblock, valid, runs.data());
                        ASSERT_TRUE(chunk_plan.mask.packed());
                        ASSERT_GT(chunk_plan.active_tiles, 0u);
                        ASSERT_LE(chunk_plan.active_tiles, valid);
                        EXPECT_TRUE(chunk_plan.active_tiles == valid || chunk_plan.active_tiles % subblock == 0);
                        std::vector<uint32_t> global(valid);
                        for (uint32_t column = 0; column < valid; ++column) {
                            const auto at = plan.locate(chunk * plan.chunk_tiles + column, ids);
                            global[column] = block_cyclic_global(c, at.rank, at.local);
                        }
                        for (uint32_t column = chunk_plan.active_tiles; column < valid; ++column) {
                            EXPECT_GE(global[column], bounds.masked_from) << "dropped column " << column;
                        }
                        if (chunk_plan.mask.mode == PackedKVMaskMode::PackedUnmasked) {
                            EXPECT_EQ(chunk_plan.mask.run_count, 0u);
                            for (uint32_t column = 0; column < chunk_plan.active_tiles; ++column) {
                                EXPECT_LT(global[column], bounds.visible_end) << "unstamped column " << column;
                            }
                            continue;
                        }
                        ASSERT_EQ(chunk_plan.mask.mode, PackedKVMaskMode::PackedMasked);
                        ASSERT_EQ(chunk_plan.mask.runs, runs.data());
                        const auto& last = chunk_plan.mask.runs[chunk_plan.mask.run_count - 1];
                        EXPECT_EQ(last.column_end, chunk_plan.active_tiles);
                        bool any_partially_visible = false;
                        uint32_t begin = 0;
                        for (uint32_t run = 0; run < chunk_plan.mask.run_count; ++run) {
                            const auto& interval = chunk_plan.mask.runs[run];
                            for (uint32_t column = begin; column < interval.column_end; ++column) {
                                EXPECT_EQ(interval.global_start_tile + (column - begin), global[column]);
                                any_partially_visible |= global[column] >= bounds.visible_end;
                            }
                            begin = interval.column_end;
                        }
                        EXPECT_TRUE(any_partially_visible);
                    }
                }
            }
        }
    }
}

TEST(RingMLAGeometry, AcceptsWholeRegionsAndRejectsTheRest) {
    // Galaxy SP8 x TP4 production: Q slab 20 tiles, four stripes of 5-tile regions.
    EXPECT_TRUE((RingMLAGeometry{8, 32, 20, 55}.valid()));
    // LB SP2 x TP4.
    EXPECT_TRUE((RingMLAGeometry{2, 8, 8, 16}.valid()));
    // Legacy equal-shard layout: one stripe.
    EXPECT_TRUE((RingMLAGeometry{32, 32, 2, 4}.valid()));

    EXPECT_FALSE((RingMLAGeometry{0, 32, 20, 55}.valid()));
    EXPECT_FALSE((RingMLAGeometry{8, 0, 20, 55}.valid()));
    EXPECT_FALSE((RingMLAGeometry{8, 64, 20, 55}.valid()));  // more than 32 sources
    EXPECT_FALSE((RingMLAGeometry{3, 32, 20, 55}.valid()));  // sources not a multiple of Q shards
    EXPECT_FALSE((RingMLAGeometry{8, 32, 0, 55}.valid()));
    EXPECT_FALSE((RingMLAGeometry{8, 32, 22, 55}.valid()));  // Q slab not a whole number of stripes
    EXPECT_FALSE((RingMLAGeometry{8, 32, 20, 0}.valid()));
    EXPECT_FALSE((RingMLAGeometry{8, 32, 20, 56}.valid()));  // capacity not whole regions
    // Whole regions, but 32 sources of 134217730 tiles overflow uint32_t.
    EXPECT_FALSE((RingMLAGeometry{8, 32, 20, 134217730}.valid()));
}

}  // namespace
