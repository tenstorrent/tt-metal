// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/chunked_prefill_utils.hpp"

TEST(RingMLAPackingPlan, PreservesEveryTileAcrossSourcesAndPartialChunks) {
    for (uint32_t ring : {8u, 32u}) {
        for (uint32_t group_size : {2u, 4u, 8u}) {
            for (uint32_t region : {1u, 2u, 5u}) {
                for (uint32_t depth : {1u, 5u, 6u, 11u, 205u}) {
                    for (uint32_t k : {1u, 4u, 11u, 20u, 32u, 128u}) {
                        const uint32_t source_tiles = region * depth;
                        const PackedKVGroupPlan plan{source_tiles, group_size, k};
                        std::vector<uint32_t> sources(ring);
                        std::iota(sources.begin(), sources.end(), 0);
                        std::reverse(sources.begin(), sources.end());
                        std::rotate(sources.begin(), sources.begin() + 3, sources.end());
                        std::vector<uint32_t> visits(ring * source_tiles, 0);
                        for (uint32_t group = 0; group < ring; group += group_size) {
                            uint32_t stream = 0;
                            for (uint32_t chunk = 0; chunk < plan.chunk_count(); ++chunk) {
                                const uint32_t valid = plan.valid_tiles(chunk);
                                ASSERT_GT(valid, 0u);
                                ASSERT_LE(valid, k);
                                ASSERT_LT(plan.last_source(chunk), group_size);
                                std::vector<PackedKVMaskRun> runs(k);
                                const uint32_t count =
                                    plan.mask_runs(chunk, sources.data() + group, region, ring * region, runs.data());
                                ASSERT_GT(count, 0u);
                                ASSERT_LE(count, valid);
                                uint32_t column = 0;
                                uint32_t max_global_tile = 0;
                                for (uint32_t run = 0; run < count; ++run) {
                                    const uint32_t begin = column;
                                    ASSERT_GT(runs[run].column_end, begin);
                                    ASSERT_LE(runs[run].column_end, valid);
                                    for (; column < runs[run].column_end; ++column) {
                                        const uint32_t flat = chunk * k + column;
                                        const uint32_t source = sources[group + flat / source_tiles];
                                        const uint32_t local = flat % source_tiles;
                                        const uint32_t expected =
                                            (local / region) * (ring * region) + source * region + local % region;
                                        ASSERT_EQ(runs[run].global_start_tile + column - begin, expected);
                                        max_global_tile = std::max(max_global_tile, expected);
                                    }
                                }
                                ASSERT_EQ(column, valid);
                                EXPECT_TRUE(packed_kv_runs_need_causal_mask(runs.data(), count, 0));
                                EXPECT_TRUE(packed_kv_runs_need_causal_mask(runs.data(), count, max_global_tile));
                                EXPECT_FALSE(packed_kv_runs_need_causal_mask(runs.data(), count, max_global_tile + 1));
                                for (uint32_t dst = 0; dst < valid;) {
                                    const uint32_t segment = plan.segment_tiles(chunk, dst);
                                    ASSERT_GT(segment, 0u);
                                    ASSERT_LE(dst + segment, valid);
                                    ASSERT_LE(plan.source_offset(stream) + segment, source_tiles);
                                    const uint32_t source = sources[group + plan.source_index(stream)];
                                    for (uint32_t tile = 0; tile < segment; ++tile) {
                                        const uint32_t global = plan.global_tile(stream, source, region, ring * region);
                                        ASSERT_LT(global, visits.size());
                                        ++visits[global];
                                        ++stream;
                                    }
                                    dst += segment;
                                }
                            }
                            EXPECT_EQ(stream, source_tiles * group_size);
                            EXPECT_EQ(plan.valid_tiles(plan.chunk_count()), 0u);
                        }
                        EXPECT_TRUE(
                            std::all_of(visits.begin(), visits.end(), [](uint32_t count) { return count == 1; }));
                    }
                }
            }
        }
    }
}

TEST(RingMLAPackingPlan, UsesRuntimeLengthAndActivityOnProgramReuse) {
    EXPECT_EQ(packed_kv_source_group_size(4, 8, 25, 200, 0xff), 4u);
    EXPECT_EQ(packed_kv_source_group_size(4, 8, 25, 199, 0xff), 1u);
    EXPECT_EQ(packed_kv_source_group_size(4, 8, 25, 200, 0x7f), 1u);
    EXPECT_EQ(packed_kv_source_group_size(4, 32, 25, 800, 0xffffffff), 4u);
    EXPECT_EQ(packed_kv_source_group_size(4, 32, 25, 800, 0x7fffffff), 1u);
    EXPECT_EQ(packed_kv_source_group_size(1, 8, 25, 200, 0xff), 1u);
    EXPECT_EQ(packed_kv_source_group_size(3, 8, 25, 200, 0xff), 1u);
    EXPECT_EQ(packed_kv_source_group_size(4, 8, 0, 0, 0xff), 1u);
}

TEST(RingMLAPackingPlan, GroupsWhenCacheIsOverAllocated) {
    // 102-chunk allocation (510 tiles per source) with 1 chunk live: 160 logical tiles over a
    // 32-ring. The cache is far from full, but the live region still splits evenly per source.
    EXPECT_EQ(packed_kv_source_group_size(4, 32, 510, 160, 0xffffffff), 4u);
    EXPECT_EQ(packed_kv_live_tiles_per_source(160, 32), 5u);

    // A live region that does not divide across the ring cannot be packed uniformly.
    EXPECT_EQ(packed_kv_source_group_size(4, 32, 510, 161, 0xffffffff), 1u);
    // Live must still fit inside the allocation.
    EXPECT_EQ(packed_kv_source_group_size(4, 32, 510, 510u * 32 + 32, 0xffffffff), 1u);
    EXPECT_EQ(packed_kv_source_group_size(4, 32, 510, 0, 0xffffffff), 1u);

    // The plan spans only the live region, so an over-allocated cache costs no extra K chunks.
    const PackedKVGroupPlan live{packed_kv_live_tiles_per_source(160, 32), 4, 20};
    EXPECT_EQ(live.tile_count(), 20u);
    EXPECT_EQ(live.chunk_count(), 1u);
    EXPECT_EQ(live.source_index(0), 0u);
    EXPECT_EQ(live.source_index(5), 1u);
    EXPECT_EQ(live.source_offset(6), 1u);
}
