// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include "tt_metal/impl/dispatch/worker_config_buffer.hpp"

using std::vector;
using namespace tt::tt_metal;

namespace worker_config_buffer_tests {

TEST(WorkerConfigBuffer, MarkCompletelyFull) {
    WorkerConfigBufferMgr mgr;
    mgr.init_add_buffer(1024, 1024);
    mgr.init_add_buffer(2, 1024);

    auto reservation = mgr.reserve({12, 12});
    mgr.alloc(1);

    mgr.mark_completely_full(5);

    // Allocation would suceed, except buffer is marked completely full.
    auto new_reservation = mgr.reserve({12, 0});
    EXPECT_TRUE(new_reservation.first.need_sync);
    EXPECT_EQ(new_reservation.first.sync_count, 5u);
    EXPECT_EQ(new_reservation.second[0].size, 12u);
    EXPECT_EQ(new_reservation.second[1].size, 0u);
    mgr.free(5);

    mgr.alloc(1);

    auto next_reservation = mgr.reserve({12, 12});

    EXPECT_FALSE(next_reservation.first.need_sync);
}

// Test that small-sized, tightly-packed ringbuffers work.
TEST(WorkerConfigBuffer, SmallSize) {
    WorkerConfigBufferMgr mgr;
    mgr.init_add_buffer(0, 5);
    for (size_t i = 0; i < 5; i++) {
        auto reservation = mgr.reserve({1});
        EXPECT_FALSE(reservation.first.need_sync);
        mgr.alloc(i + 1);
    }

    for (size_t i = 5; i < 15; i++) {
        auto reservation = mgr.reserve({1});
        EXPECT_TRUE(reservation.first.need_sync);
        EXPECT_EQ(reservation.first.sync_count, i - 4);
        mgr.free(reservation.first.sync_count);
        mgr.alloc(i + 1);
    }
}

}  // namespace worker_config_buffer_tests
