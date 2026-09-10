// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <set>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "ttnn/operations/transformer/sdpa/device/kernels/sliding_window_work_plan.hpp"

namespace ttnn::operations::transformer::sdpa::ring_joint {

TEST(RotatedSlidingWorkPlan, CoversAbsoluteWindowsAndBoundsHaloReads) {
    constexpr uint32_t local_tiles = 32;
    constexpr uint32_t cache_tiles = 4 * local_tiles;
    constexpr uint32_t k_tiles = 4;
    for (uint32_t ring_size : {2u, 8u}) {
        const uint32_t capacity = cache_tiles * ring_size;
        const uint32_t group_tiles = local_tiles * ring_size;
        for (uint32_t start = 0; start < capacity; ++start) {
            const uint32_t end = std::min(start + group_tiles, capacity);
            for (uint32_t rank = 0; rank < ring_size; ++rank) {
                // Independent reference: filter the absolute interval by cache ownership.
                std::vector<uint32_t> positions;
                for (uint32_t pos = start; pos < end; ++pos) {
                    if ((pos / local_tiles) % ring_size == rank) {
                        positions.push_back(pos);
                    }
                }
                uint32_t pre_count = 0;
                while (pre_count < positions.size() && positions[pre_count] / group_tiles == start / group_tiles) {
                    ++pre_count;
                }
                const uint32_t pre_start = pre_count ? positions.front() : 0;
                const uint32_t post_start = pre_count < positions.size() ? positions[pre_count] : 0;
                for (uint32_t q_tiles : {2u, 4u}) {
                    for (uint32_t window_tokens : {128u, 1024u}) {
                        for (uint32_t q = 0; q < local_tiles; q += q_tiles) {
                            SCOPED_TRACE(
                                testing::Message() << "start=" << start << " rank=" << rank << " q=" << q
                                                   << " window=" << window_tokens << " q_tiles=" << q_tiles);
                            const auto plan = build_rotated_sliding_q_work_plan(
                                q,
                                q_tiles,
                                rank,
                                local_tiles,
                                ring_size,
                                window_tokens,
                                32,
                                cache_tiles,
                                k_tiles,
                                end,
                                pre_start,
                                pre_count,
                                post_start,
                                positions.size());
                            ASSERT_TRUE(plan.is_valid);
                            std::set<std::pair<uint32_t, uint32_t>> actual, expected;
                            for (uint32_t i = 0; i < plan.total_k_chunk_count; ++i) {
                                const auto ref = plan.k_chunk_at(i);
                                ASSERT_LT(ref.source_k_chunk * k_tiles, cache_tiles);
                                ASSERT_TRUE(actual.emplace(ref.source_ring_id, ref.source_k_chunk).second);
                                if (ref.source_ring_id != rank) {
                                    ASSERT_EQ(ref.source_ring_id, (rank + ring_size - 1) % ring_size);
                                    ASSERT_LT(ref.compact_k_chunk * k_tiles, 2 * local_tiles);
                                    const auto origin = rotated_sliding_halo_start(
                                        positions.front(), rank, local_tiles, ring_size, cache_tiles);
                                    ASSERT_EQ(origin + ref.compact_k_chunk * k_tiles, ref.source_k_chunk * k_tiles);
                                }
                            }
                            for (uint32_t row = q; row < std::min<uint32_t>(q + q_tiles, positions.size()); ++row) {
                                // Include all key tiles touched by any of the 32 token queries in this tile.
                                const uint32_t first_token = positions[row] * 32;
                                const uint32_t begin =
                                    first_token + 1 > window_tokens ? first_token + 1 - window_tokens : 0;
                                const uint32_t stop = first_token + 32;
                                for (uint32_t key = begin / 32; key < (stop + 31) / 32; ++key) {
                                    const uint32_t source = (key / local_tiles) % ring_size;
                                    const uint32_t local = (key / group_tiles) * local_tiles + key % local_tiles;
                                    expected.emplace(source, local / k_tiles);
                                }
                            }
                            if (!expected.empty()) {
                                ASSERT_EQ(actual, expected);
                            }
                        }
                    }
                }
            }
        }
    }
}

}  // namespace ttnn::operations::transformer::sdpa::ring_joint
