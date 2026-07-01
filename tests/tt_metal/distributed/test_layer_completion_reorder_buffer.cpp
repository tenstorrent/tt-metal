// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>

#include <layer_completion_message.hpp>
#include <layer_completion_reorder_buffer.hpp>

namespace tt::tt_metal::distributed::test {

namespace {
LayerCompletionMessage m(uint64_t seq) { return LayerCompletionMessage{seq, 0u, 0u, 0u, 0u}; }
}  // namespace

TEST(LayerCompletionReorderBuffer, InOrderDrainsOneEach) {
    LayerCompletionReorderBuffer rb;
    std::vector<LayerCompletionMessage> drained;
    for (uint64_t i = 0; i < 5; ++i) {
        EXPECT_EQ(rb.insert(m(i), drained), 1u);
        ASSERT_EQ(drained.size(), 1u);
        EXPECT_EQ(drained[0].seq, i);
    }
    EXPECT_EQ(rb.next_expected(), 5u);
    EXPECT_EQ(rb.buffered(), 0u);
}

TEST(LayerCompletionReorderBuffer, OutOfOrderBuffersThenBurstDrains) {
    LayerCompletionReorderBuffer rb;
    std::vector<LayerCompletionMessage> drained;

    EXPECT_EQ(rb.insert(m(2), drained), 0u);  // gap: expecting 0
    EXPECT_EQ(rb.insert(m(1), drained), 0u);  // still gap: expecting 0
    EXPECT_EQ(rb.buffered(), 2u);

    const uint32_t n = rb.insert(m(0), drained);  // fills the gap → 0,1,2 all ready
    EXPECT_EQ(n, 3u);
    ASSERT_EQ(drained.size(), 3u);
    EXPECT_EQ(drained[0].seq, 0u);
    EXPECT_EQ(drained[1].seq, 1u);
    EXPECT_EQ(drained[2].seq, 2u);
    EXPECT_EQ(rb.next_expected(), 3u);
    EXPECT_EQ(rb.buffered(), 0u);
}

TEST(LayerCompletionReorderBuffer, DropsStaleAndDuplicateSeqs) {
    LayerCompletionReorderBuffer rb;
    std::vector<LayerCompletionMessage> drained;
    EXPECT_EQ(rb.insert(m(0), drained), 1u);
    EXPECT_EQ(rb.insert(m(0), drained), 0u);  // stale (< next_expected) → dropped
    EXPECT_EQ(rb.insert(m(3), drained), 0u);  // buffered
    EXPECT_EQ(rb.insert(m(3), drained), 0u);  // duplicate of buffered → ignored
    EXPECT_EQ(rb.buffered(), 1u);
    EXPECT_EQ(rb.next_expected(), 1u);
}

TEST(LayerCompletionReorderBuffer, RespectsNonZeroStartSeq) {
    LayerCompletionReorderBuffer rb(/*start_seq=*/100);
    std::vector<LayerCompletionMessage> drained;
    EXPECT_EQ(rb.insert(m(99), drained), 0u);   // stale
    EXPECT_EQ(rb.insert(m(101), drained), 0u);  // buffered
    EXPECT_EQ(rb.insert(m(100), drained), 2u);  // drains 100,101
    EXPECT_EQ(rb.next_expected(), 102u);
}

}  // namespace tt::tt_metal::distributed::test
