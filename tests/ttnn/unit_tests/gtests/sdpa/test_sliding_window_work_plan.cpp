// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host-side checks of the constexpr sliding-window work plan shared by the RingJointSDPA program
// factory, reader and compute kernels (sliding_window_work_plan.hpp). The plan is pure integer math
// over the chunked block-cyclic K/V layout, so its contract is pinned here without a device.

#include <algorithm>
#include <cstdint>
#include <set>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "ttnn/operations/transformer/sdpa/device/kernels/chunked_q_mapping.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/sliding_window_work_plan.hpp"

namespace {

using namespace ttnn::operations::transformer::sdpa::ring_joint;

constexpr uint32_t kTileHeight = 32;
constexpr uint32_t kWindowTokens = 128;

TEST(ChunkedQMapping, MatchesPackedAbsoluteTiles) {
    for (uint32_t ring : {1u, 2u, 4u, 8u}) {
        for (uint32_t local : {2u, 8u, 32u}) {
            const uint32_t group = ring * local;
            for (uint32_t start = 0; start < 2 * group; ++start) {
                for (uint32_t length : {1u, local - 1, local, group - 1, group}) {
                    for (uint32_t device = 0; device < ring; ++device) {
                        SCOPED_TRACE(
                            ::testing::Message() << "ring=" << ring << " local=" << local << " start=" << start
                                                 << " length=" << length << " device=" << device);
                        std::vector<uint32_t> positions;
                        uint32_t pre_wrap_count = 0;
                        for (uint32_t tile = start; tile < start + length; ++tile) {
                            if (tile / local % ring == device) {
                                positions.push_back(tile);
                                pre_wrap_count += tile / group == start / group;
                            }
                        }
                        const auto mapping = build_chunked_q_mapping(start, start + length, local, ring, device);
                        ASSERT_EQ(mapping.q_valid_tile_count, positions.size());
                        EXPECT_LE(mapping.q_valid_tile_count, local);
                        EXPECT_EQ(mapping.q_pre_wrap_tile_count, pre_wrap_count);
                        EXPECT_EQ(mapping.q_pre_wrap_start_tile, pre_wrap_count ? positions.front() : 0u);
                        EXPECT_EQ(
                            mapping.q_post_wrap_start_tile,
                            pre_wrap_count < positions.size() ? positions[pre_wrap_count] : 0u);
                        for (uint32_t row = 0; row < positions.size(); ++row) {
                            const uint32_t absolute =
                                row < mapping.q_pre_wrap_tile_count
                                    ? mapping.q_pre_wrap_start_tile + row
                                    : mapping.q_post_wrap_start_tile + row - mapping.q_pre_wrap_tile_count;
                            EXPECT_EQ(absolute, positions[row]);
                        }
                    }
                }
            }
        }
    }
}

TEST(ChunkedQMapping, WrapDetectionMatchesPackedSegments) {
    for (uint32_t ring : {4u, 8u}) {
        for (uint32_t local : {8u, 32u}) {
            const uint32_t group = ring * local;
            for (uint32_t start = 0; start < 2 * group; ++start) {
                for (uint32_t length : {1u, local, group - 1, group}) {
                    bool has_split_device = false;
                    for (uint32_t device = 0; device < ring; ++device) {
                        const auto mapping = build_chunked_q_mapping(start, start + length, local, ring, device);
                        has_split_device |= mapping.q_pre_wrap_tile_count > 0 &&
                                            mapping.q_valid_tile_count > mapping.q_pre_wrap_tile_count;
                    }
                    EXPECT_EQ(chunked_q_wraps(start, start + length, local, ring), has_split_device);
                }
            }
        }
    }
    EXPECT_FALSE(chunked_q_wraps(47, 288, 32, 8));  // Stops before the second segment.
    EXPECT_TRUE(chunked_q_wraps(47, 289, 32, 8));
}

TEST(ChunkedQMapping, Offset1504SplitsOneDeviceSlab) {
    constexpr auto mapping = build_chunked_q_mapping(67040 / 32, 75232 / 32, 1024 / 32, 8, 1);
    static_assert(mapping.q_pre_wrap_start_tile == 67040 / 32);
    static_assert(mapping.q_pre_wrap_tile_count == 544 / 32);
    static_assert(mapping.q_post_wrap_start_tile == 74752 / 32);
    static_assert(mapping.q_valid_tile_count == 1024 / 32);
}

struct Geometry {
    uint32_t ring_size;
    uint32_t q_local_tile_rows;
    uint32_t k_chunk_tile_rows;
    uint32_t groups_written;  // chunk groups written so far (the current one included)
    uint32_t slabs;           // circular slab count; 0 = unbounded
};

std::vector<Geometry> geometries() {
    std::vector<Geometry> out;
    for (uint32_t ring : {2u, 4u, 8u}) {
        for (uint32_t q_local : {8u, 16u}) {
            for (uint32_t k_chunk : {4u, 8u}) {
                for (uint32_t groups : {1u, 2u, 3u, 5u, 6u}) {
                    for (uint32_t slabs : {2u, 3u, 4u}) {
                        out.push_back({ring, q_local, k_chunk, groups, slabs});
                    }
                }
            }
        }
    }
    return out;
}

// The unbounded reference plan for the same geometry: the local cache holds every group written.
SlidingQWorkPlan unbounded_plan(const Geometry& g, uint32_t device, uint32_t q_local_start) {
    return build_sliding_q_work_plan(
        q_local_start,
        g.k_chunk_tile_rows,
        device,
        g.q_local_tile_rows,
        g.ring_size,
        kWindowTokens,
        kTileHeight,
        g.groups_written * g.q_local_tile_rows,
        g.k_chunk_tile_rows,
        g.groups_written * g.ring_size * g.q_local_tile_rows,
        0);
}

// The circular plan: the local cache holds only `slabs` Q-sized slabs, chunk group j in slab j % slabs.
SlidingQWorkPlan circular_plan(const Geometry& g, uint32_t device, uint32_t q_local_start) {
    return build_sliding_q_work_plan(
        q_local_start,
        g.k_chunk_tile_rows,
        device,
        g.q_local_tile_rows,
        g.ring_size,
        kWindowTokens,
        kTileHeight,
        g.slabs * g.q_local_tile_rows,
        g.k_chunk_tile_rows,
        g.groups_written * g.ring_size * g.q_local_tile_rows,
        g.slabs);
}

std::string describe(const Geometry& g, uint32_t device, uint32_t q_local_start) {
    return "ring=" + std::to_string(g.ring_size) + " q_local=" + std::to_string(g.q_local_tile_rows) +
           " k_chunk=" + std::to_string(g.k_chunk_tile_rows) + " groups=" + std::to_string(g.groups_written) +
           " slabs=" + std::to_string(g.slabs) + " device=" + std::to_string(device) +
           " q_start=" + std::to_string(q_local_start);
}

TEST(SlidingWindowWorkPlan, LocalSlabWrapsOnlyWhenCircular) {
    for (uint32_t group = 0; group < 12; ++group) {
        EXPECT_EQ(circular_kv_local_slab(group, 0), group);
        EXPECT_EQ(circular_kv_local_slab(group, 1), group);
        for (uint32_t slabs = 2; slabs <= 5; ++slabs) {
            EXPECT_EQ(circular_kv_local_slab(group, slabs), group % slabs);
        }
    }
}

// The circular plan is the unbounded plan with only the local slab base wrapped: same ranges, same
// absolute origins, same compact indices; local K-chunk indices differ by whole wrapped slabs.
TEST(SlidingWindowWorkPlan, CircularMatchesUnboundedUpToSlabWrap) {
    for (const auto& g : geometries()) {
        for (uint32_t device = 0; device < g.ring_size; ++device) {
            for (uint32_t q_local_start = 0; q_local_start < g.q_local_tile_rows;
                 q_local_start += g.k_chunk_tile_rows) {
                const auto u = unbounded_plan(g, device, q_local_start);
                const auto c = circular_plan(g, device, q_local_start);
                const auto where = describe(g, device, q_local_start);
                ASSERT_TRUE(u.is_valid) << where;
                ASSERT_TRUE(c.is_valid) << where;
                ASSERT_EQ(c.source_range_count, u.source_range_count) << where;
                ASSERT_EQ(c.total_k_chunk_count, u.total_k_chunk_count) << where;
                for (uint32_t r = 0; r < u.source_range_count; ++r) {
                    const auto& ru = u.source_ranges[r];
                    const auto& rc = c.source_ranges[r];
                    EXPECT_EQ(rc.source_ring_id, ru.source_ring_id) << where;
                    EXPECT_EQ(rc.k_chunk_count(), ru.k_chunk_count()) << where;
                    EXPECT_EQ(rc.first_compact_k_chunk, ru.first_compact_k_chunk) << where;
                    EXPECT_EQ(rc.first_global_k_chunk, ru.first_global_k_chunk) << where;
                    // Unbounded local rows are slab-major: group = local_row / q_local.
                    const uint32_t u_local_row = ru.first_k_chunk * g.k_chunk_tile_rows;
                    const uint32_t group = u_local_row / g.q_local_tile_rows;
                    const uint32_t wrapped_row =
                        u_local_row - (group - circular_kv_local_slab(group, g.slabs)) * g.q_local_tile_rows;
                    EXPECT_EQ(rc.first_k_chunk * g.k_chunk_tile_rows, wrapped_row) << where;
                    EXPECT_LT(rc.first_k_chunk * g.k_chunk_tile_rows, g.slabs * g.q_local_tile_rows) << where;
                }
            }
        }
    }
}

// first_global_k_chunk is the absolute (sequence-global) K-chunk index of the range: the slab-major
// inversion of the unbounded local row, so the compute mask never has to invert local rows itself.
TEST(SlidingWindowWorkPlan, GlobalOriginIsAbsolute) {
    for (const auto& g : geometries()) {
        for (uint32_t device = 0; device < g.ring_size; ++device) {
            const auto u = unbounded_plan(g, device, 0);
            const auto where = describe(g, device, 0);
            ASSERT_TRUE(u.is_valid) << where;
            for (uint32_t r = 0; r < u.source_range_count; ++r) {
                const auto& ru = u.source_ranges[r];
                const uint32_t local_row = ru.first_k_chunk * g.k_chunk_tile_rows;
                const uint32_t group = local_row / g.q_local_tile_rows;
                const uint32_t global_row = (group * g.ring_size + ru.source_ring_id) * g.q_local_tile_rows +
                                            (local_row - group * g.q_local_tile_rows);
                EXPECT_EQ(ru.first_global_k_chunk * g.k_chunk_tile_rows, global_row) << where;
            }
        }
    }
}

// Remote ranges start inside the one-hop halo the neighbour sends, and the compact index is the
// distance from the halo origin; the halo origin wraps through the same slab base as the range.
TEST(SlidingWindowWorkPlan, RemoteRangesStartInsideTheHalo) {
    for (const auto& g : geometries()) {
        const uint32_t halo = chunked_sliding_halo_tile_rows(kWindowTokens, kTileHeight, g.k_chunk_tile_rows);
        const uint32_t logical_k = g.groups_written * g.ring_size * g.q_local_tile_rows;
        for (uint32_t device = 0; device < g.ring_size; ++device) {
            const auto c = circular_plan(g, device, 0);
            const auto where = describe(g, device, 0);
            for (uint32_t r = 0; r < c.source_range_count; ++r) {
                const auto& rc = c.source_ranges[r];
                if (rc.source_ring_id == device) {
                    EXPECT_EQ(rc.first_compact_k_chunk, 0u) << where;
                    continue;
                }
                const auto mapping = build_chunked_q_mapping(
                    logical_k - g.ring_size * g.q_local_tile_rows, logical_k, g.q_local_tile_rows, g.ring_size, device);
                const uint32_t halo_start =
                    sliding_halo_sources(mapping, g.q_local_tile_rows, g.ring_size, halo, g.slabs).first_start_tile;
                const uint32_t range_start = rc.first_k_chunk * g.k_chunk_tile_rows;
                EXPECT_GE(range_start, halo_start) << where;
                EXPECT_EQ(rc.first_compact_k_chunk, (range_start - halo_start) / g.k_chunk_tile_rows) << where;
            }
        }
    }
}

// k_chunk_at / global_k_chunk_at walk the ranges in order: local index, compact index and absolute
// index all advance together, and the local index never leaves the circular cache.
TEST(SlidingWindowWorkPlan, WorkItemsWalkTheRangesInOrder) {
    for (const auto& g : geometries()) {
        for (uint32_t device = 0; device < g.ring_size; ++device) {
            const auto c = circular_plan(g, device, 0);
            const auto where = describe(g, device, 0);
            uint32_t work = 0;
            for (uint32_t r = 0; r < c.source_range_count; ++r) {
                const auto& rc = c.source_ranges[r];
                for (uint32_t i = 0; i < rc.k_chunk_count(); ++i, ++work) {
                    const auto ref = c.k_chunk_at(work);
                    EXPECT_EQ(ref.source_ring_id, rc.source_ring_id) << where;
                    EXPECT_EQ(ref.source_k_chunk, rc.first_k_chunk + i) << where;
                    EXPECT_EQ(ref.compact_k_chunk, rc.first_compact_k_chunk + i) << where;
                    EXPECT_EQ(c.global_k_chunk_at(work), rc.first_global_k_chunk + i) << where;
                    EXPECT_LT(ref.source_k_chunk * g.k_chunk_tile_rows, g.slabs * g.q_local_tile_rows) << where;
                }
            }
            EXPECT_EQ(work, c.total_k_chunk_count) << where;
        }
    }
}

// One hand-computed geometry: sp=4 ring, 8-tile Q slabs, 4-tile K chunks, 128-token window, 5 chunk
// groups written, Q chunk at local tile 0 on device 0. Two slabs: group 3 lands in slab 1, group 4
// in slab 0; absolute origins are unaffected by the wrap.
TEST(SlidingWindowWorkPlan, PinnedDevice0Geometry) {
    constexpr auto circular = build_sliding_q_work_plan(0, 4, 0, 8, 4, 128, 32, 16, 4, 160, 2);
    static_assert(circular.source_range_count == 2);
    EXPECT_EQ(circular.source_ranges[0].source_ring_id, 3u);
    EXPECT_EQ(circular.source_ranges[0].first_k_chunk, 3u);
    EXPECT_EQ(circular.source_ranges[1].first_k_chunk, 0u);
    EXPECT_EQ(circular.source_ranges[0].first_global_k_chunk, 31u);
    EXPECT_EQ(circular.source_ranges[1].first_global_k_chunk, 32u);
    constexpr auto unbounded = build_sliding_q_work_plan(0, 4, 0, 8, 4, 128, 32, 40, 4, 160, 0);
    EXPECT_EQ(unbounded.source_ranges[0].first_k_chunk, 7u);
    EXPECT_EQ(unbounded.source_ranges[1].first_k_chunk, 8u);
    EXPECT_EQ(unbounded.source_ranges[0].first_global_k_chunk, 31u);
    EXPECT_EQ(unbounded.source_ranges[1].first_global_k_chunk, 32u);
}

TEST(SlidingWindowWorkPlan, RotatedQueriesCoverExactlyTheirCausalWindows) {
    for (uint32_t ring : {4u, 8u}) {
        for (uint32_t local : {8u, 32u, 64u}) {
            const uint32_t group = ring * local;
            const uint32_t capacity = 3 * group;
            for (uint32_t window : {128u, std::min(1024u, local * 32)}) {
                const uint32_t halo = chunked_sliding_halo_tile_rows(window, 32, 4);
                for (uint32_t start : {0u, 1u, local - 1, local, group - 1, group + 1, capacity - 3}) {
                    for (uint32_t length : {1u, local + 1, group}) {
                        const uint32_t end = std::min(start + length, capacity);
                        for (uint32_t device = 0; device < ring; ++device) {
                            std::vector<uint32_t> positions;
                            for (uint32_t token = start; token < end; ++token) {
                                if (token / local % ring == device) {
                                    positions.push_back(token);
                                }
                            }
                            const auto mapping = build_chunked_q_mapping(start, end, local, ring, device);
                            const auto sources = sliding_halo_sources(mapping, local, ring, halo);
                            ASSERT_EQ(mapping.q_valid_tile_count, positions.size());
                            EXPECT_EQ(
                                sources.count,
                                mapping.q_pre_wrap_tile_count &&
                                        mapping.q_valid_tile_count > mapping.q_pre_wrap_tile_count
                                    ? 2u
                                    : 1u);
                            for (uint32_t qsize : {2u, 4u}) {
                                for (uint32_t q = 0; q < local; q += qsize) {
                                    SCOPED_TRACE(
                                        ::testing::Message()
                                        << "ring=" << ring << " local=" << local << " start=" << start << " end=" << end
                                        << " device=" << device << " q=" << q << " qsize=" << qsize
                                        << " window=" << window);
                                    const auto plan = build_sliding_q_work_plan(
                                        q,
                                        qsize,
                                        device,
                                        local,
                                        ring,
                                        window,
                                        32,
                                        capacity / ring,
                                        4,
                                        end,
                                        0,
                                        &mapping);
                                    ASSERT_TRUE(plan.is_valid);
                                    if (q >= positions.size()) {
                                        EXPECT_EQ(plan.total_k_chunk_count, 1u);
                                        continue;
                                    }
                                    std::set<std::pair<uint32_t, uint32_t>> expected, actual;
                                    for (uint32_t row = q; row < std::min<uint32_t>(q + qsize, positions.size());
                                         ++row) {
                                        const uint32_t pos = positions[row];
                                        const uint32_t left = (window - 1 + 31) / 32;
                                        for (uint32_t k = pos > left ? pos - left : 0; k <= pos; ++k) {
                                            expected.emplace(k / local % ring, (k / group * local + k % local) / 4);
                                        }
                                    }
                                    for (uint32_t work = 0; work < plan.total_k_chunk_count; ++work) {
                                        const auto ref = plan.k_chunk_at(work);
                                        EXPECT_TRUE(actual.emplace(ref.source_ring_id, ref.source_k_chunk).second);
                                        if (ref.source_ring_id != device) {
                                            const uint32_t compact = ref.compact_k_chunk * 4;
                                            ASSERT_LT(compact, sources.count * halo);
                                            const uint32_t origin =
                                                compact < halo ? sources.first_start_tile : sources.second_start_tile;
                                            EXPECT_EQ(origin + compact % halo, ref.source_k_chunk * 4);
                                        }
                                    }
                                    EXPECT_EQ(actual, expected);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// Multi-hop halo geometry, hand-computed. A halo wider than one Q slab is split across cyclic
// predecessors: hop d carries the tail of the slab d positions back, oldest block first in the compact
// buffer, and the farthest hop carries only the remainder.
TEST(SlidingWindowWorkPlan, MultiHopGeometry) {
    // 10-tile halo over 4-tile slabs: 3 hops of 4, 4 and 2 rows, landing at rows 6, 2 and 0.
    static_assert(chunked_sliding_halo_hop_count(10, 4) == 3);
    EXPECT_EQ(chunked_sliding_halo_hop_rows(10, 4, 0), 0u);
    EXPECT_EQ(chunked_sliding_halo_hop_rows(10, 4, 1), 4u);
    EXPECT_EQ(chunked_sliding_halo_hop_rows(10, 4, 3), 2u);
    EXPECT_EQ(chunked_sliding_halo_hop_rows(10, 4, 4), 0u);
    EXPECT_EQ(chunked_sliding_halo_hop_dest_row(10, 4, 1), 6u);
    EXPECT_EQ(chunked_sliding_halo_hop_dest_row(10, 4, 2), 2u);
    EXPECT_EQ(chunked_sliding_halo_hop_dest_row(10, 4, 3), 0u);

    // Gemma4 at CP8, 1024-token window (32 tiles): chunk 2048 gives 8-tile slabs and 4 full hops;
    // chunk 4096 gives 16-tile slabs and 2.
    static_assert(chunked_sliding_halo_hop_count(32, 8) == 4);
    EXPECT_EQ(chunked_sliding_halo_hop_dest_row(32, 8, 1), 24u);
    EXPECT_EQ(chunked_sliding_halo_hop_dest_row(32, 8, 4), 0u);
    EXPECT_EQ(chunked_sliding_halo_remote_hop_count(32, 8, 8), 4u);
    static_assert(chunked_sliding_halo_hop_count(32, 16) == 2);
    EXPECT_EQ(chunked_sliding_halo_hop_dest_row(32, 16, 1), 16u);

    // A halo reaching all the way round the ring wraps onto this device's own earlier slab, which is a
    // local read, so only ring_size - 1 hops cross the fabric.
    EXPECT_EQ(chunked_sliding_halo_remote_hop_count(16, 4, 4), 3u);

    // Source tail for the 2-row remainder hop (ring 4, 4-tile slabs, second chunk group). Device 0's
    // hop-3 predecessor is device 1 in the previous group, so the tail starts 2 rows before the end of
    // device 1's first slab.
    EXPECT_EQ(sliding_halo_sources(build_chunked_q_mapping(16, 32, 4, 4, 0), 4, 4, 10, 0, 3).first_start_tile, 2u);
    // In the first group there is no previous group, so that hop sends from tile 0.
    EXPECT_EQ(sliding_halo_sources(build_chunked_q_mapping(0, 16, 4, 4, 0), 4, 4, 10, 0, 3).first_start_tile, 0u);
}

// Multi-hop counterpart of RotatedQueriesCoverExactlyTheirCausalWindows: a window wider than one slab,
// for aligned and non-wrapping unaligned chunk starts. Every K chunk in each Q block's causal window is
// read exactly once, and each remote one from its hop's block at the offset that hop's sender wrote it.
TEST(SlidingWindowWorkPlan, MultiHopQueriesCoverExactlyTheirCausalWindows) {
    for (uint32_t ring : {4u, 8u}) {
        for (uint32_t local : {8u, 16u}) {
            const uint32_t group = ring * local;
            const uint32_t capacity = 3 * group;
            for (uint32_t hops_wanted : {2u, 3u, 4u}) {
                const uint32_t window = hops_wanted * local * 32;
                const uint32_t halo = chunked_sliding_halo_tile_rows(window, 32, 4);
                if (chunked_sliding_halo_hop_count(halo, local) > ring) {
                    continue;
                }
                for (uint32_t start : {0u, group, 2 * group, group + 1, group + local - 1}) {
                    for (uint32_t length : {1u, local + 1, group}) {
                        const uint32_t end = std::min(start + length, capacity);
                        if (chunked_q_wraps(start, end, local, ring)) {
                            continue;
                        }
                        for (uint32_t device = 0; device < ring; ++device) {
                            std::vector<uint32_t> positions;
                            for (uint32_t token = start; token < end; ++token) {
                                if (token / local % ring == device) {
                                    positions.push_back(token);
                                }
                            }
                            const auto mapping = build_chunked_q_mapping(start, end, local, ring, device);
                            ASSERT_EQ(mapping.q_valid_tile_count, positions.size());
                            for (uint32_t q = 0; q < local; q += 4) {
                                SCOPED_TRACE(
                                    ::testing::Message()
                                    << "ring=" << ring << " local=" << local << " start=" << start << " end=" << end
                                    << " device=" << device << " q=" << q << " window=" << window);
                                const auto plan = build_sliding_q_work_plan(
                                    q, 4, device, local, ring, window, 32, capacity / ring, 4, end, 0, &mapping);
                                ASSERT_TRUE(plan.is_valid);
                                if (q >= positions.size()) {
                                    EXPECT_EQ(plan.total_k_chunk_count, 1u);
                                    continue;
                                }
                                std::set<std::pair<uint32_t, uint32_t>> expected, actual;
                                for (uint32_t row = q; row < std::min<uint32_t>(q + 4, positions.size()); ++row) {
                                    const uint32_t pos = positions[row];
                                    const uint32_t left = (window - 1 + 31) / 32;
                                    for (uint32_t k = pos > left ? pos - left : 0; k <= pos; ++k) {
                                        expected.emplace(k / local % ring, (k / group * local + k % local) / 4);
                                    }
                                }
                                for (uint32_t work = 0; work < plan.total_k_chunk_count; ++work) {
                                    const auto ref = plan.k_chunk_at(work);
                                    EXPECT_TRUE(actual.emplace(ref.source_ring_id, ref.source_k_chunk).second);
                                    if (ref.source_ring_id == device) {
                                        continue;
                                    }
                                    const uint32_t compact = ref.compact_k_chunk * 4;
                                    ASSERT_LT(compact, halo);
                                    // The hop whose block holds this row, and what that hop's sender shipped.
                                    uint32_t hop = 1;
                                    while (compact < chunked_sliding_halo_hop_dest_row(halo, local, hop)) {
                                        ++hop;
                                    }
                                    const auto sources = sliding_halo_sources(mapping, local, ring, halo, 0, hop);
                                    EXPECT_EQ(sources.count, 1u);
                                    EXPECT_EQ(ref.source_ring_id, (device + ring - hop) % ring);
                                    EXPECT_EQ(
                                        sources.first_start_tile + compact -
                                            chunked_sliding_halo_hop_dest_row(halo, local, hop),
                                        ref.source_k_chunk * 4);
                                }
                                EXPECT_EQ(actual, expected);
                            }
                        }
                    }
                }
            }
        }
    }
}

}  // namespace
