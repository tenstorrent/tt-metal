// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>

#include "gtest/gtest.h"
#include "tt_metal/impl/dispatch/worker_config_buffer.hpp"

using std::vector;
using namespace tt::tt_metal;

namespace working_config_buffer_tests {

TEST(WorkingConfigBuffer, MarkCompletelyFull) {
    WorkerConfigBufferMgr mgr;
    mgr.init_add_core(1024, 1024);
    mgr.init_add_core(2, 1024);

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
}  // namespace working_config_buffer_tests
