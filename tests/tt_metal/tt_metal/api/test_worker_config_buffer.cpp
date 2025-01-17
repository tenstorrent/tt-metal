// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <sys/types.h>
#include "gtest/gtest.h"
#include <tt-metalium/worker_config_buffer.hpp>
#include <tt-metalium/env_lib.hpp>

#include <cstddef>
#include <deque>

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

// Test that allocating buffers of size 0 doesn't eventually cause some to be ignored.
TEST(WorkerConfigBuffer, SizeOne) {
    WorkerConfigBufferMgr mgr;
    mgr.init_add_buffer(0, 100);
    mgr.init_add_buffer(0, 100);
    mgr.init_add_buffer(0, 5);
    mgr.init_add_buffer(0, 1);

    mgr.mark_completely_full(0);

    {
        auto new_reservation = mgr.reserve({1, 0, 1, 0});
        EXPECT_TRUE(new_reservation.first.need_sync);
        mgr.free(0);
        mgr.alloc(1);
    }
    {
        auto new_reservation = mgr.reserve({1, 0, 1, 0});
        EXPECT_FALSE(new_reservation.first.need_sync);
        mgr.alloc(2);
    }
    {
        auto new_reservation = mgr.reserve({1, 0, 1, 0});
        EXPECT_FALSE(new_reservation.first.need_sync);
        mgr.alloc(3);
    }
    {
        auto new_reservation = mgr.reserve({1, 1, 1, 1});
        EXPECT_FALSE(new_reservation.first.need_sync);
        mgr.free(1);
        mgr.alloc(4);
    }
    {
        auto new_reservation = mgr.reserve({1, 0, 1, 0});
        EXPECT_FALSE(new_reservation.first.need_sync);
        mgr.free(2);
        mgr.alloc(5);
    }
    {
        auto new_reservation = mgr.reserve({0, 1, 0, 1});
        EXPECT_TRUE(new_reservation.first.need_sync);
        mgr.alloc(6);
    }
}

// Test that we don't throw away the old sync counts when the number of outstanding buffers is >
// kernel_config_entry_count.
TEST(WorkerConfigBuffer, LoopAround) {
    WorkerConfigBufferMgr mgr;
    mgr.init_add_buffer(0, 10);

    mgr.mark_completely_full(0);
    for (size_t i = 0; i < 12; i++) {
        auto reservation = mgr.reserve({1});
        if (i >= 10) {
            EXPECT_TRUE(reservation.first.need_sync) << i;
        }
        if (reservation.first.need_sync) {
            mgr.free(reservation.first.sync_count);
        }
        mgr.alloc(i + 1);
    }
}

TEST(WorkerConfigBuffer, Randomized) {
    const uint32_t seed = tt::parse_env("TT_METAL_SEED", static_cast<uint32_t>(time(nullptr)));
    log_info(tt::LogTest, "Using seed: {}", seed);
    srand(seed);
    const std::vector<uint32_t> kSizes = {1024, 1024, 10, 1};
    WorkerConfigBufferMgr mgr;
    for (uint32_t size : kSizes) {
        mgr.init_add_buffer(0, size);
    }

    struct UsedRegion {
        uint32_t sync_count;
        uint32_t offset;
        uint32_t size;
    };

    auto regions_intersect = [](const UsedRegion& a, const UsedRegion& b) {
        if (b.size == 0 || a.size == 0) {
            return false;
        }
        return a.offset < b.offset + b.size && b.offset < a.offset + a.size;
    };

    std::vector<std::deque<UsedRegion>> used_regions(kSizes.size());

    for (size_t i = 0; i < 1000; i++) {
        std::vector<uint32_t> sizes;
        for (uint32_t size : kSizes) {
            sizes.push_back(rand() % (size + 1));
        }
        auto reservation = mgr.reserve(sizes);
        if (reservation.first.need_sync) {
            EXPECT_GE(reservation.first.sync_count, 1u);
            EXPECT_LT(reservation.first.sync_count, i + 1);
            mgr.free(reservation.first.sync_count);
            for (auto& region_type : used_regions) {
                while (!region_type.empty() && region_type.front().sync_count <= reservation.first.sync_count) {
                    region_type.pop_front();
                }
            }
        }
        for (size_t j = 0; j < kSizes.size(); j++) {
            UsedRegion new_region = {i + 1, reservation.second[j].addr, reservation.second[j].size};
            for (const auto& region : used_regions[j]) {
                ASSERT_FALSE(regions_intersect(region, new_region))
                    << "Region " << new_region.offset << " " << new_region.size << " intersects with " << region.offset
                    << " " << region.size;
            }
            used_regions[j].push_back(new_region);
        }
        mgr.alloc(i + 1);
    }
}

// Test reserving when one buffer type is completely empty.
TEST(WorkerConfigBuffer, VeryBasic) {
    WorkerConfigBufferMgr mgr;
    mgr.init_add_buffer(0, 1024);
    mgr.init_add_buffer(0, 10);
    auto reservation = mgr.reserve({934, 5});
    EXPECT_FALSE(reservation.first.need_sync);
    mgr.alloc(1);

    auto reservation2 = mgr.reserve({561, 0});
    EXPECT_TRUE(reservation2.first.need_sync);
    EXPECT_EQ(reservation2.first.sync_count, 1u);
    mgr.free(1);
    mgr.alloc(2);

    auto reservation3 = mgr.reserve({35, 7});
    EXPECT_FALSE(reservation3.first.need_sync);
}

}  // namespace worker_config_buffer_tests
