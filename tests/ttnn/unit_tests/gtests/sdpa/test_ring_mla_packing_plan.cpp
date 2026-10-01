// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host-side checks of the split-KV ring MLA contracts shared by the RingJointSDPA program factory,
// reader, compute and writer kernels (ring_mla_packing_plan.hpp, ring_mla_geometry.hpp). Both are
// pure integer math over the block-cyclic KV layout, so they are pinned here without a device.

#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "ttnn/operations/transformer/sdpa/device/kernels/ring_mla_geometry.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/ring_mla_packing_plan.hpp"

namespace {

using ttnn::operations::transformer::sdpa::ring_joint::RingMLAGeometry;

struct StreamTile {
    uint32_t member;  // group member whose readiness covers the tile
    uint32_t rank;
    uint32_t local;
};

// One pass's stream built independently of the plan: every member's head (all slabs but the
// newest when there are several) member by member, then the newest slab of each row in `rows`
// (group consecutive ranks) in row order.
std::vector<StreamTile> reference_stream(
    uint32_t source_tiles, const std::vector<uint32_t>& ids, uint32_t region, uint32_t ring, uint32_t rows) {
    const bool defers = source_tiles > region;
    const uint32_t head = defers ? source_tiles - region : source_tiles;
    std::vector<StreamTile> stream;
    for (uint32_t m = 0; m < ids.size(); ++m) {
        for (uint32_t local = 0; local < head; ++local) {
            stream.push_back({m, ids[m], local});
        }
    }
    const uint32_t group = ids.size();
    for (uint32_t row = 0; defers && row < ring / group; ++row) {
        if (((rows >> row) & 1u) == 0) {
            continue;
        }
        for (uint32_t rank = row * group; rank < (row + 1) * group; ++rank) {
            for (uint32_t local = head; local < source_tiles; ++local) {
                stream.push_back({group - 1, rank, local});
            }
        }
    }
    return stream;
}

// Each source stores its regions of successive global chunks back to back.
uint32_t reference_global_tile(const StreamTile& tile, uint32_t region, uint32_t global_chunk_tiles) {
    return (tile.local / region) * global_chunk_tiles + tile.rank * region + tile.local % region;
}

// A non-trivial member order, as the route delivers it.
std::vector<uint32_t> route_ids(uint32_t group, uint32_t ring) {
    std::vector<uint32_t> ids;
    for (uint32_t m = 0; m < group; ++m) {
        ids.push_back((ring - 1 - m * 3) % ring);
    }
    return ids;
}

TEST(RingMLAPackingPlan, AddressingAndMaskRunsMatchReferenceStream) {
    for (uint32_t region : {1u, 2u, 5u}) {
        for (uint32_t ring : {8u, 32u}) {
            for (uint32_t group : {2u, 4u}) {
                for (uint32_t slabs : {1u, 3u, 11u}) {
                    for (uint32_t chunk_tiles : {1u, 3u, 11u, 20u}) {
                        for (uint32_t row_pattern : {0u, 1u, 0b101u, ~0u}) {
                            const uint32_t rows = row_pattern & ((1u << (ring / group)) - 1);
                            const uint32_t source_tiles = slabs * region;
                            const uint32_t global_chunk = region * ring;
                            const auto ids = route_ids(group, ring);
                            const PackedKVGroupPlan plan =
                                packed_kv_pass_plan({source_tiles, group, chunk_tiles, region}, rows);
                            const auto stream = reference_stream(source_tiles, ids, region, ring, rows);
                            ASSERT_EQ(plan.tile_count(), stream.size());
                            uint32_t covered = 0;
                            for (uint32_t chunk = 0; chunk < plan.chunk_count(); ++chunk) {
                                SCOPED_TRACE(
                                    ::testing::Message() << "region=" << region << " ring=" << ring
                                                         << " group=" << group << " slabs=" << slabs << " chunk_tiles="
                                                         << chunk_tiles << " rows=" << rows << " chunk=" << chunk);
                                const uint32_t first = chunk * chunk_tiles;
                                uint32_t max_member = 0;
                                uint32_t max_slab = 0;
                                for (uint32_t col = 0; col < plan.valid_tiles(chunk);) {
                                    const uint32_t rows = plan.segment_tiles(chunk, col);
                                    ASSERT_GE(rows, 1u);
                                    // A segment is one contiguous read from a single source.
                                    const StreamTile head = stream[first + col];
                                    for (uint32_t i = 0; i < rows; ++i) {
                                        const StreamTile expected = stream[first + col + i];
                                        EXPECT_EQ(expected.rank, head.rank);
                                        EXPECT_EQ(expected.local, head.local + i);
                                        const auto at = plan.locate(first + col + i, ids.data());
                                        EXPECT_EQ(at.rank, expected.rank);
                                        EXPECT_EQ(at.local, expected.local);
                                        max_member = expected.member > max_member ? expected.member : max_member;
                                        const uint32_t slab = expected.local / region;
                                        max_slab = slab > max_slab ? slab : max_slab;
                                    }
                                    col += rows;
                                }
                                EXPECT_EQ(plan.last_source(chunk), max_member);
                                EXPECT_EQ(plan.max_slab(chunk), max_slab);

                                std::vector<PackedKVMaskRun> runs(chunk_tiles);
                                const uint32_t count = plan.mask_runs(chunk, ids.data(), global_chunk, runs.data());
                                ASSERT_GE(count, 1u);
                                ASSERT_LE(count, chunk_tiles);
                                uint32_t begin = 0;
                                for (uint32_t r = 0; r < count; ++r) {
                                    ASSERT_GT(runs[r].column_end, begin);
                                    for (uint32_t col = begin; col < runs[r].column_end; ++col) {
                                        EXPECT_EQ(
                                            runs[r].global_start_tile + (col - begin),
                                            reference_global_tile(stream[first + col], region, global_chunk));
                                    }
                                    begin = runs[r].column_end;
                                }
                                EXPECT_EQ(begin, plan.valid_tiles(chunk));
                                covered += begin;
                            }
                            EXPECT_EQ(covered, plan.tile_count());
                        }
                    }
                }
            }
        }
    }
}

TEST(RingMLAPackingPlan, NewestRowsChunkLikeTheClassicRing) {
    // Galaxy SP8 x TP4 at 50k+5k: four 55-tile sources (11 slabs of 5 tiles), 20-tile K chunks,
    // 32 ranks, so one global chunk is 160 tiles and one row of newest slabs is one K chunk.
    const PackedKVGroupPlan base{/*source_tiles=*/55, /*source_count=*/4, /*chunk_tiles=*/20, /*region_tiles=*/5};
    EXPECT_EQ(packed_kv_pass_plan(base, 0).chunk_count(), 10u);
    const auto pass = packed_kv_pass_plan(base, 0b10010000u);  // rows 4 and 7
    ASSERT_EQ(pass.chunk_count(), 12u);
    for (uint32_t chunk = 0; chunk < 10; ++chunk) {
        EXPECT_LT(pass.max_slab(chunk), 10u) << "chunk=" << chunk;
    }
    const std::vector<uint32_t> ids{7, 6, 5, 4};
    for (uint32_t j = 0; j < 2; ++j) {
        // Row r covers global tiles [1600 + 20r, 1620 + 20r) in one run: the classic ring's
        // diagonal chunk of SP rank r.
        const uint32_t row = j == 0 ? 4 : 7;
        PackedKVMaskRun runs[20];
        ASSERT_EQ(pass.mask_runs(10 + j, ids.data(), 160, runs), 1u);
        EXPECT_EQ(runs[0].global_start_tile, 1600u + 20 * row);
        EXPECT_EQ(runs[0].column_end, 20u);
        EXPECT_EQ(pass.max_slab(10 + j), 10u);
        EXPECT_EQ(pass.last_source(10 + j), 3u);
        EXPECT_EQ(pass.locate(200 + 20 * j + 7, ids.data()).rank, row * 4 + 1);
    }
}

TEST(RingMLAPackingPlan, PassRowsJoinWhenTheirLastRankArrives) {
    // 8 ranks in rows of 2, delivered in a snake that splits rows across passes.
    const std::vector<uint32_t> arrival{3, 4, 5, 6, 7, 0, 1, 2};
    uint32_t rows[4];
    packed_kv_pass_rows(8, 2, [&](uint32_t i) { return arrival[i]; }, rows);
    // Row 2 = {4, 5} completes in pass 1; row 3 = {6, 7} in pass 2; row 0 = {0, 1} and
    // row 1 = {2, 3} in pass 3.
    EXPECT_EQ(rows[0], 0u);
    EXPECT_EQ(rows[1], 0b0100u);
    EXPECT_EQ(rows[2], 0b1000u);
    EXPECT_EQ(rows[3], 0b0011u);
}

TEST(RingMLAPackingPlan, SingleSlabStreamsWholeSources) {
    const PackedKVGroupPlan base{/*source_tiles=*/5, /*source_count=*/4, /*chunk_tiles=*/8, /*region_tiles=*/5};
    const auto final_pass = packed_kv_pass_plan(base, 0xFFu);
    EXPECT_FALSE(final_pass.defers_newest());
    EXPECT_EQ(final_pass.tile_count(), 20u);
    EXPECT_EQ(final_pass.chunk_count(), 3u);
    EXPECT_EQ(final_pass.valid_tiles(2), 4u);
    EXPECT_EQ(final_pass.valid_tiles(3), 0u);
}

TEST(RingMLAPackingPlan, ReadinessWaitsOnlyThroughRequiredSourceInOrder) {
    // Heads of 4 tiles fill chunks 0-1 ([0, 16)); the final pass's chunk 2 holds newest slabs.
    const auto plan =
        packed_kv_pass_plan({/*source_tiles=*/6, /*source_count=*/4, /*chunk_tiles=*/8, /*region_tiles=*/2}, 0b1u);
    PackedKVSourceReadiness readiness;
    std::vector<uint32_t> waited;
    auto wait = [&](uint32_t source) { waited.push_back(source); };
    readiness.wait_for_chunk(plan, 0, wait);
    EXPECT_EQ(waited, (std::vector<uint32_t>{0, 1}));
    readiness.wait_for_chunk(plan, 0, wait);  // reused across Q chunks: no new waits
    EXPECT_EQ(waited.size(), 2u);
    readiness.wait_for_chunk(plan, 2, wait);
    EXPECT_EQ(waited, (std::vector<uint32_t>{0, 1, 2, 3}));
    readiness.drain(4, wait);  // idle cores drain the rest: already complete
    EXPECT_EQ(waited.size(), 4u);
}

TEST(RingMLAPackingPlan, SourceTilesClipToTouchedSlabs) {
    // 8 sources, 2-tile regions: one global chunk is 16 tiles.
    EXPECT_EQ(packed_kv_source_tiles(/*capacity=*/10, /*logical=*/16, 2, 8), 2u);
    EXPECT_EQ(packed_kv_source_tiles(10, 17, 2, 8), 4u);  // a partial second chunk touches slab 2
    EXPECT_EQ(packed_kv_source_tiles(10, 1000, 2, 8), 10u);
    EXPECT_EQ(packed_kv_source_tiles(10, 16, 0, 8), 10u);
}

TEST(RingMLAPackingPlan, GroupOnlyWithEveryConfiguredSourceActive) {
    constexpr uint32_t kAll8 = 0xFFu;
    EXPECT_EQ(packed_kv_source_group_size(4, 8, 2, 16, kAll8), 4u);
    EXPECT_EQ(packed_kv_source_group_size(4, 8, 2, 16, 0x7Fu), 1u);  // one inactive source
    EXPECT_EQ(packed_kv_source_group_size(1, 8, 2, 16, kAll8), 1u);  // not configured
    EXPECT_EQ(packed_kv_source_group_size(3, 8, 2, 16, kAll8), 1u);  // does not divide the ring
    EXPECT_EQ(packed_kv_source_group_size(4, 8, 2, 17, kAll8), 1u);  // logical beyond capacity
    EXPECT_EQ(packed_kv_source_group_size(4, 8, 2, 0, kAll8), 1u);
    EXPECT_EQ(packed_kv_source_group_size(4, 32, 2, 64, 0xFFFFFFFFu), 4u);
}

TEST(RingMLAGeometry, ValidRequiresWholeRegionsPerSource) {
    // Galaxy SP8 x TP4: 20-tile Q slab, 5-tile regions, 11 slabs of cache per source.
    EXPECT_TRUE((RingMLAGeometry{8, 32, 20, 55}.valid()));
    EXPECT_TRUE((RingMLAGeometry{8, 8, 20, 20}.valid()));    // classic: one region per Q slab
    EXPECT_FALSE((RingMLAGeometry{8, 32, 20, 54}.valid()));  // capacity is not whole regions
    EXPECT_FALSE((RingMLAGeometry{8, 32, 18, 54}.valid()));  // Q slab does not split into 4 stripes
    EXPECT_FALSE((RingMLAGeometry{8, 12, 20, 20}.valid()));  // KV sources not a multiple of Q shards
    EXPECT_FALSE((RingMLAGeometry{8, 64, 16, 16}.valid()));  // beyond the 32-source mask
    EXPECT_FALSE((RingMLAGeometry{0, 32, 20, 55}.valid()));
    EXPECT_FALSE((RingMLAGeometry{8, 32, 0, 55}.valid()));
    EXPECT_FALSE((RingMLAGeometry{8, 32, 20, 0}.valid()));
}

}  // namespace
