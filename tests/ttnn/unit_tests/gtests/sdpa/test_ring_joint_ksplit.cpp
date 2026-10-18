// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host-side checks of the RingJointSDPA K-split slices and the dense chunked causal K skip, both pure integer math
// shared by the reader and compute kernels.

#include <cstdint>

#include "gtest/gtest.h"
#include "ttnn/operations/transformer/sdpa/device/kernels/ring_joint_ksplit.hpp"

namespace {

using namespace ttnn::operations::transformer::sdpa::ring_joint;

TEST(RingJointKSplit, SlicesPartitionValidChunksAndReducerOwnsTheLast) {
    for (uint32_t splits = 2; splits <= kKSplitMaxCount; ++splits) {
        for (uint32_t valid = 0; valid <= 64; ++valid) {
            SCOPED_TRACE(::testing::Message() << "splits=" << splits << " valid=" << valid);
            uint32_t next = 0;
            for (uint32_t split = 0; split < splits; ++split) {
                const KSplitRange range = ksplit_range(valid, split, splits);
                EXPECT_EQ(range.begin, next);
                EXPECT_LE(range.begin, range.end);
                next = range.end;
            }
            EXPECT_EQ(next, valid);
            if (valid > 0) {
                EXPECT_TRUE(ksplit_range(valid, splits - 1, splits).contains(valid - 1));
            }
        }
    }
}

// Compute derives which senders have state from the largest valid count alone, which relies on this.
TEST(RingJointKSplit, NonEmptySliceStaysNonEmptyAsValidChunksGrow) {
    for (uint32_t splits = 2; splits <= kKSplitMaxCount; ++splits) {
        for (uint32_t split = 0; split < splits; ++split) {
            bool seen_non_empty = false;
            for (uint32_t valid = 0; valid <= 256; ++valid) {
                const bool non_empty = !ksplit_range(valid, split, splits).empty();
                EXPECT_TRUE(non_empty || !seen_non_empty)
                    << "splits=" << splits << " split=" << split << " valid=" << valid;
                seen_non_empty |= non_empty;
            }
        }
    }
}

TEST(RingJointKSplit, ValidChunkCountIsTheLogicalPrefix) {
    constexpr uint32_t q_local = 8;  // tiles per device per prefill chunk
    constexpr uint32_t ring = 8;
    constexpr uint32_t chunk = q_local * ring;
    constexpr uint32_t k_chunk = 8;
    constexpr uint32_t kv_local = 16 * q_local;
    constexpr uint32_t num_k_chunks = kv_local / k_chunk;
    for (uint32_t prefill_chunks = 1; prefill_chunks <= 16; ++prefill_chunks) {
        for (uint32_t ring_id = 0; ring_id < ring; ++ring_id) {
            const uint32_t count = ksplit_valid_local_k_chunks<false, true, kv_local, chunk, q_local, k_chunk>(
                num_k_chunks, ring_id, prefill_chunks * chunk, prefill_chunks * chunk);
            EXPECT_EQ(count, prefill_chunks) << "prefill_chunks=" << prefill_chunks << " ring_id=" << ring_id;
        }
    }
}

TEST(ChunkedCausalSkip, KeepsVisibleChunksAndChunkZero) {
    constexpr uint32_t q_local = 8;
    constexpr uint32_t ring = 8;
    constexpr uint32_t chunk = q_local * ring;
    constexpr uint32_t kv_local = 16 * q_local;
    for (uint32_t prefill_chunks = 1; prefill_chunks <= 8; ++prefill_chunks) {
        const uint32_t logical_nt = prefill_chunks * chunk;
        for (uint32_t ring_index = 0; ring_index < ring; ++ring_index) {
            const uint32_t q_end = chunked_q_global_end_tile<false, q_local>(logical_nt, ring_index, ring, 0, 0, 0, 0);
            EXPECT_EQ(q_end, logical_nt - chunk + (ring_index + 1) * q_local);
            for (uint32_t ring_id = 0; ring_id < ring; ++ring_id) {
                for (uint32_t k = 0; k < prefill_chunks; ++k) {
                    SCOPED_TRACE(
                        ::testing::Message() << "chunks=" << prefill_chunks << " ring_index=" << ring_index
                                             << " ring_id=" << ring_id << " k=" << k);
                    // K chunk k of ring_id starts at global tile k * chunk + ring_id * q_local.
                    const bool visible = k * chunk + ring_id * q_local < q_end;
                    EXPECT_EQ(
                        (chunked_kv_chunk_is_live<false, true, kv_local, chunk, q_local>(
                            ring_id, k, q_local, logical_nt, q_end)),
                        visible || k == 0);
                }
                // The K split counts exactly the live chunks, which form a prefix.
                uint32_t live = 0;
                while (live < prefill_chunks && (live == 0 || live * chunk + ring_id * q_local < q_end)) {
                    ++live;
                }
                EXPECT_EQ(
                    (ksplit_valid_local_k_chunks<false, true, kv_local, chunk, q_local, q_local>(
                        kv_local / q_local, ring_id, logical_nt, q_end)),
                    live)
                    << "chunks=" << prefill_chunks << " ring_index=" << ring_index << " ring_id=" << ring_id;
            }
        }
    }
}

}  // namespace
