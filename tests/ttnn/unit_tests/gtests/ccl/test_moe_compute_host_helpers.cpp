// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <exception>

#include "gtest/gtest.h"
#include "ttnn/operations/experimental/ccl/moe/selective_reduce_combine/device/selective_reduce_combine_program_factory.hpp"
#include "ttnn/operations/experimental/ccl/moe_compute/device/kernels/moe_ring_common.h"

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
// tests/nightly/tg/ccl/moe/test_moe_compute_6U.py (token_list_*).
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
