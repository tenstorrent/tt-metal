// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdio>
#include <string>

#include <internal/disaggregation/layer_completion_message.hpp>
#include <internal/disaggregation/layer_completion_queue.hpp>

namespace tt::tt_metal::distributed::test {

namespace {
LayerCompletionMessage make_msg(uint64_t seq) {
    return LayerCompletionMessage{
        seq,
        /*source_rank=*/0u,
        /*layer_idx=*/static_cast<uint32_t>(seq % 61),
        /*request_id=*/static_cast<uint32_t>(seq / 61),
        /*reserved=*/0u};
}
void unlink_if_exists(const std::string& shm_name) { std::remove(("/dev/shm" + shm_name).c_str()); }
}  // namespace

TEST(LayerCompletionQueue, FifoRoundtrip) {
    const std::string name = "/tt_lcq_test_fifo";
    unlink_if_exists(name);
    auto owner = LayerCompletionQueue::create(name);
    auto conn = LayerCompletionQueue::connect(name, 5'000);

    for (uint64_t i = 0; i < 8; ++i) {
        EXPECT_TRUE(owner->try_push(make_msg(i)));
    }
    LayerCompletionMessage out{};
    for (uint64_t i = 0; i < 8; ++i) {
        ASSERT_TRUE(conn->try_pop(out));
        EXPECT_EQ(out.seq, i);  // FIFO
    }
    EXPECT_FALSE(conn->try_pop(out));  // empty
    owner->shutdown();
}

TEST(LayerCompletionQueue, RejectsPushWhenFull) {
    const std::string name = "/tt_lcq_test_full";
    unlink_if_exists(name);
    auto owner = LayerCompletionQueue::create(name);
    for (uint32_t i = 0; i < LayerCompletionQueue::capacity(); ++i) {
        ASSERT_TRUE(owner->try_push(make_msg(i)));
    }
    EXPECT_FALSE(owner->try_push(make_msg(9999)));  // full → reject, no overwrite
    owner->shutdown();
}

TEST(LayerCompletionQueue, WrapsAroundPastCapacity) {
    const std::string name = "/tt_lcq_test_wrap";
    unlink_if_exists(name);
    auto owner = LayerCompletionQueue::create(name);
    auto conn = LayerCompletionQueue::connect(name, 5'000);

    LayerCompletionMessage out{};
    const uint64_t total = static_cast<uint64_t>(LayerCompletionQueue::capacity()) * 3 + 7;
    uint64_t pushed = 0, popped = 0;
    while (popped < total) {
        while (pushed < total && owner->try_push(make_msg(pushed))) {
            ++pushed;
        }
        while (conn->try_pop(out)) {
            EXPECT_EQ(out.seq, popped);
            ++popped;
        }
    }
    EXPECT_EQ(popped, total);
    owner->shutdown();
}

}  // namespace tt::tt_metal::distributed::test
