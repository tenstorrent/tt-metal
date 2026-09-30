// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <exception>
#include <vector>

#include "gtest/gtest.h"
#include "ttnn/operations/experimental/ccl/moe/selective_reduce_combine/device/selective_reduce_combine_program_factory.hpp"
#include "ttnn/operations/experimental/ccl/moe_compute/device/kernels/moe_ring_common.h"
#include "ttnn/operations/experimental/ccl/moe_compute/moe_core_placement.hpp"

namespace {

using ttnn::experimental::prim::detail::compute_fused_source_buffer_layout;

constexpr uint32_t kBf16Bytes = 2;

TEST(MoEComputeHostHelpers, FusedSourceBufferLayoutMatchesPhysicalShard) {
    // Producer shard already split to one data-parallel column: [2 buffers x 32 rows, 640] in BF16
    // for hidden 2560 over four combine columns. One shard row is one token segment, so the ring
    // entry is half the shard height.
    constexpr uint32_t hidden_size = 2560;
    constexpr uint32_t data_parallel_cores = 4;
    constexpr uint32_t token_segment_width = hidden_size / data_parallel_cores;  // 640
    constexpr uint32_t source_shard_height = 64;
    constexpr uint32_t source_shard_width = token_segment_width;
    constexpr uint32_t num_buffers = 2;
    constexpr uint32_t token_segment_size_bytes = token_segment_width * kBf16Bytes;                       // 1280
    constexpr uint32_t source_buffer_size_bytes = source_shard_height * source_shard_width * kBf16Bytes;  // 81920

    const auto layout = compute_fused_source_buffer_layout(
        source_shard_height,
        source_shard_width,
        token_segment_width,
        source_buffer_size_bytes,
        token_segment_size_bytes,
        num_buffers);

    EXPECT_EQ(layout.rows_per_buffer, 32u);
    EXPECT_EQ(layout.buffer_block_size_bytes, 40960u);
    EXPECT_EQ(layout.circular_buffer_size_bytes, 81920u);
    EXPECT_LE(layout.circular_buffer_size_bytes, source_buffer_size_bytes);

    // Producer and consumer toggle between offsets 0 and buffer_block_size_bytes; the last token
    // segment of either block stays inside the circular buffer.
    EXPECT_EQ(layout.buffer_block_size_bytes, layout.rows_per_buffer * token_segment_size_bytes);
    EXPECT_EQ(layout.circular_buffer_size_bytes, num_buffers * layout.buffer_block_size_bytes);
    EXPECT_LE(
        layout.buffer_block_size_bytes + layout.rows_per_buffer * token_segment_size_bytes,
        layout.circular_buffer_size_bytes);
}

TEST(MoEComputeHostHelpers, FusedSourceBufferLayoutFullWidthShardCountsSegmentRows) {
    // moe_compute's own tilize-output shard, [2 buffers x 32 rows, hidden 7168] in BF16 (the deepseek
    // single-card nightly shape), consumed by four 1792-element combine columns. Each shard row holds
    // four token segments, so a ring entry is 32 x 4 = 128 token-segment rows and buffer 1 starts
    // half-way through the shard (458752 B), where the host readback expects it.
    constexpr uint32_t hidden_size = 7168;
    constexpr uint32_t data_parallel_cores = 4;
    constexpr uint32_t token_segment_width = hidden_size / data_parallel_cores;  // 1792
    constexpr uint32_t source_shard_height = 64;
    constexpr uint32_t source_shard_width = hidden_size;
    constexpr uint32_t num_buffers = 2;
    constexpr uint32_t token_segment_size_bytes = token_segment_width * kBf16Bytes;                       // 3584
    constexpr uint32_t source_buffer_size_bytes = source_shard_height * source_shard_width * kBf16Bytes;  // 917504

    const auto layout = compute_fused_source_buffer_layout(
        source_shard_height,
        source_shard_width,
        token_segment_width,
        source_buffer_size_bytes,
        token_segment_size_bytes,
        num_buffers);

    EXPECT_EQ(layout.rows_per_buffer, 128u);
    EXPECT_EQ(layout.buffer_block_size_bytes, 458752u);
    EXPECT_EQ(layout.circular_buffer_size_bytes, 917504u);
    EXPECT_EQ(layout.circular_buffer_size_bytes, source_buffer_size_bytes);
    EXPECT_EQ(layout.buffer_block_size_bytes, source_buffer_size_bytes / num_buffers);
    EXPECT_EQ(layout.buffer_block_size_bytes, layout.rows_per_buffer * token_segment_size_bytes);
}

TEST(MoEComputeHostHelpers, FusedSourceBufferLayoutSingleBufferUsesWholeShard) {
    const auto layout = compute_fused_source_buffer_layout(
        /*source_shard_height=*/32,
        /*source_shard_width=*/512,
        /*token_segment_width=*/512,
        /*source_buffer_size_bytes=*/32 * 1024,
        /*token_segment_size_bytes=*/1024,
        /*num_buffers=*/1);
    EXPECT_EQ(layout.rows_per_buffer, 32u);
    EXPECT_EQ(layout.buffer_block_size_bytes, 32u * 1024u);
    EXPECT_EQ(layout.circular_buffer_size_bytes, 32u * 1024u);
}

TEST(MoEComputeHostHelpers, FusedSourceBufferLayoutRejectsBadInputs) {
    // Shard height not divisible by the buffer count.
    EXPECT_THROW(compute_fused_source_buffer_layout(33, 512, 512, 33 * 1024, 1024, 2), std::exception);
    // Shard width not divisible by the token segment width.
    EXPECT_THROW(compute_fused_source_buffer_layout(64, 1000, 640, 64 * 1000 * 2, 1280, 2), std::exception);
    // Token segment size not a whole number of bytes per element over its width.
    EXPECT_THROW(compute_fused_source_buffer_layout(64, 640, 640, 64 * 640 * 2, 1281, 2), std::exception);
    // Circular buffer larger than the L1 bank that backs it.
    EXPECT_THROW(compute_fused_source_buffer_layout(64, 512, 512, 64 * 1024 - 1, 1024, 2), std::exception);
    // Zero arguments.
    EXPECT_THROW(compute_fused_source_buffer_layout(0, 512, 512, 1024, 1024, 2), std::exception);
    EXPECT_THROW(compute_fused_source_buffer_layout(64, 0, 512, 64 * 1024, 1024, 2), std::exception);
    EXPECT_THROW(compute_fused_source_buffer_layout(64, 512, 0, 64 * 1024, 1024, 2), std::exception);
    EXPECT_THROW(compute_fused_source_buffer_layout(64, 512, 512, 0, 1024, 2), std::exception);
    EXPECT_THROW(compute_fused_source_buffer_layout(64, 512, 512, 64 * 1024, 0, 2), std::exception);
    EXPECT_THROW(compute_fused_source_buffer_layout(64, 512, 512, 64 * 1024, 1024, 0), std::exception);
}

}  // namespace

// Packed token lists of the local output path (moe_ring_common.h, moe_ring::token_list), mirrored by
// ttnn/ttnn/_experimental/moe_compute_utils.py (token_list_*).
TEST(MoEComputeHostHelpers, TokenListLayoutForThePrefillSlab) {
    using namespace moe_ring::token_list;
    // 2048 tokens, K = 10, 128 experts per device, 32-token chunks.
    EXPECT_EQ(header_words(128), 144u);                       // 129 words padded to 16
    EXPECT_EQ(entry_capacity(2048, 10, 128, 32), 22432u);     // align16(20480 + 15 x 128) + 32
    EXPECT_EQ(page_words(2048, 10, 128, 32), 144u + 22432u);  // 90,304 B
    EXPECT_EQ(page_words(128, 10, 128, 32), 144u + 3232u);    // the 128-token form: 13,504 B
    EXPECT_EQ(page_words(1, 8, 16, 32), 32u + 288u);          // the 1-token / 16-expert unit-test rows
}

TEST(MoEComputeHostHelpers, TokenListSegmentsAndEntries) {
    using namespace moe_ring::token_list;
    // Segment starts follow the counts on 16-entry boundaries; a zero count keeps an aligned start in place.
    EXPECT_EQ(next_segment_start(0, 0), 0u);
    EXPECT_EQ(next_segment_start(0, 1), 16u);
    EXPECT_EQ(next_segment_start(16, 16), 32u);
    EXPECT_EQ(next_segment_start(32, 17), 64u);
    EXPECT_EQ(next_segment_start(64, 2048), 2112u);
    // The entry packs the k slot above a 24-bit token id and unpacks to the same pair.
    constexpr uint32_t entry = pack_entry(TOKEN_MASK, MAX_K_SLOTS - 1);
    EXPECT_EQ(entry_token(entry), TOKEN_MASK);
    EXPECT_EQ(entry_k_slot(entry), MAX_K_SLOTS - 1);
    EXPECT_EQ(entry_token(pack_entry(2047, 9)), 2047u);
    EXPECT_EQ(entry_k_slot(pack_entry(2047, 9)), 9u);
}

// Chunk ownership over R rings (moe_ring_common.h, moe_ring::rings::ChunkOwners): the natural-text histogram of
// PLAN-20260925 3.2 (head 28, 19, 13, 10, 8, 7, 6, 5, 4, 4, 3, 3, 3, 3, then 22 x 2 and 62 x 1 = 222 chunks).
namespace {
constexpr uint32_t kHead[] = {28, 19, 13, 10, 8, 7, 6, 5, 4, 4, 3, 3, 3, 3};

template <uint32_t R>
struct OwnerStats {
    uint32_t load[R] = {};
    uint32_t longest_run = 0;
    uint32_t chunks = 0;
};

template <uint32_t R, typename Counts>
OwnerStats<R> assign(const Counts& counts) {
    moe_ring::rings::ChunkOwners<R> owners;
    OwnerStats<R> stats;
    uint32_t run = 0, prev = R;
    for (uint32_t n : counts) {
        owners.begin_expert(n);
        for (uint32_t c = 0; c < n; ++c) {
            const uint32_t o = owners.owner(c);
            EXPECT_LT(o, R);
            stats.load[o] += 1;
            run = (o == prev) ? run + 1 : 1;
            prev = o;
            stats.longest_run = std::max(stats.longest_run, run);
            ++stats.chunks;
        }
    }
    return stats;
}

std::vector<uint32_t> natural_text_histogram() {
    std::vector<uint32_t> counts(kHead, kHead + sizeof(kHead) / sizeof(kHead[0]));
    counts.insert(counts.end(), 22, 2);
    counts.insert(counts.end(), 62, 1);
    counts.insert(counts.end(), 30, 0);  // inactive experts
    return counts;
}
}  // namespace

TEST(MoEComputeHostHelpers, ChunkOwnersSingleRingIsToday) {
    const auto stats = assign<1>(natural_text_histogram());
    EXPECT_EQ(stats.chunks, 222u);
    EXPECT_EQ(stats.load[0], 222u);
}

TEST(MoEComputeHostHelpers, ChunkOwnersBalanceTheNaturalTextHistogramAndBoundTheRun) {
    {
        const auto stats = assign<2>(natural_text_histogram());
        EXPECT_EQ(stats.load[0] + stats.load[1], 222u);
        EXPECT_LE(std::max(stats.load[0], stats.load[1]), 113u);  // 222 / 2 plus a chunk of skew
        EXPECT_LE(stats.longest_run, 1u);
    }
    {
        const auto stats = assign<3>(natural_text_histogram());
        EXPECT_EQ(stats.load[0] + stats.load[1] + stats.load[2], 222u);
        EXPECT_LE(*std::max_element(stats.load, stats.load + 3), 77u);
        EXPECT_LE(stats.longest_run, 2u);
    }
    {
        const auto stats = assign<4>(natural_text_histogram());
        EXPECT_LE(*std::max_element(stats.load, stats.load + 4), 59u);
        EXPECT_LE(stats.longest_run, 3u);
    }
}

TEST(MoEComputeHostHelpers, ChunkOwnersSpreadOneChunkExpertsAsDecodeRoutesThem) {
    // Batched decode: ~20 distinct experts with one chunk each and no slab; the rings take them least-loaded first.
    std::vector<uint32_t> counts(128, 0);
    for (uint32_t e = 3; e < 128; e += 6) {
        counts[e] = 1;
    }
    const auto stats = assign<2>(counts);
    EXPECT_EQ(stats.chunks, 21u);
    EXPECT_LE(std::max(stats.load[0], stats.load[1]) - std::min(stats.load[0], stats.load[1]), 1u);
    EXPECT_LE(stats.longest_run, 1u);
}

TEST(MoEComputeHostHelpers, ChunkHalvesAreTwoTodayAndRingsPlusOne) {
    EXPECT_EQ(moe_ring::rings::chunk_halves(0), 2u);
    EXPECT_EQ(moe_ring::rings::chunk_halves(1), 2u);
    EXPECT_EQ(moe_ring::rings::chunk_halves(2), 3u);
    EXPECT_EQ(moe_ring::rings::chunk_halves(3), 4u);
    EXPECT_EQ(moe_ring::rings::chunk_halves(4), 5u);
    EXPECT_LE(moe_ring::rings::chunk_halves(4), moe_ring::rings::MAX_CHUNK_HALVES);
}

// The ring cores' NoC virtual channels (moe_core_placement.hpp, ring_core_vchannels): the Blackhole 8-bank ring's
// cores sit two per row at x = 0 and x = 6 (positions 0..7 = banks); a second ring adds their neighbours at x = 1
// and x = 5. Ring 0 keeps the historical channels (p & 3, bumped once when the row's first core holds it); every
// row's cores end up on distinct channels.
TEST(MoEComputeHostHelpers, RingCoreVchannelsAreDistinctWithinARowAndUnchangedOnRingZero) {
    using tt::tt_metal::CoreCoord;
    const std::vector<CoreCoord> ring0 = {{6, 9}, {0, 9}, {0, 7}, {6, 6}, {6, 4}, {0, 3}, {6, 1}, {0, 0}};
    const auto one_ring = ttnn::operations::ccl::common::ring_core_vchannels(ring0, 8);
    EXPECT_EQ(one_ring, (std::vector<uint32_t>{0, 1, 2, 3, 0, 1, 2, 3}));

    std::vector<CoreCoord> two_rings = ring0;
    for (const auto& c : ring0) {
        two_rings.emplace_back(c.x == 0 ? 1 : 5, c.y);
    }
    const auto both = ttnn::operations::ccl::common::ring_core_vchannels(two_rings, 8);
    ASSERT_EQ(both.size(), 16u);
    EXPECT_EQ(std::vector<uint32_t>(both.begin(), both.begin() + 8), one_ring);
    for (std::size_t i = 0; i < both.size(); ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            if (two_rings[i].y == two_rings[j].y) {
                EXPECT_NE(both[i], both[j]) << "cores " << i << " and " << j << " share row " << two_rings[i].y;
            }
        }
        EXPECT_LT(both[i], 4u);
    }

    // three rings: six cores on a row, the four channels repeat at most twice on any row
    std::vector<CoreCoord> three_rings = two_rings;
    for (const auto& c : ring0) {
        three_rings.emplace_back(c.x == 0 ? 2 : 4, c.y);
    }
    const auto three = ttnn::operations::ccl::common::ring_core_vchannels(three_rings, 8);
    ASSERT_EQ(three.size(), 24u);
    EXPECT_EQ(std::vector<uint32_t>(three.begin(), three.begin() + 16), both);
    for (std::size_t i = 0; i < three.size(); ++i) {
        uint32_t same = 0;
        for (std::size_t j = 0; j < three.size(); ++j) {
            same += three_rings[j].y == three_rings[i].y && three[j] == three[i];
        }
        EXPECT_LE(same, 2u) << "channel " << three[i] << " on row " << three_rings[i].y;
        EXPECT_LT(three[i], 4u);
    }
}
