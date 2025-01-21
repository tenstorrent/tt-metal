// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/impl/allocator/algorithms/free_list.hpp"

// TODO: Add a variant with randomized allocations and deallocations
TEST(FreeListAllocator, TestDirectedSeriesOfAllocDealloc) {
    constexpr uint32_t max_size_bytes = 1024;
    constexpr uint32_t min_allocation_size_bytes = 32;
    constexpr uint32_t alignment = 32;

    tt::tt_metal::allocator::FreeList free_list_allocator = tt::tt_metal::allocator::FreeList(
        max_size_bytes,
        /*offset*/ 0,
        min_allocation_size_bytes,
        alignment,
        tt::tt_metal::allocator::FreeList::SearchPolicy::FIRST);

    std::optional<uint64_t> addr_0 = free_list_allocator.allocate(32, true);
    ASSERT_TRUE(addr_0.has_value());
    EXPECT_EQ(addr_0.value(), 0);

    std::optional<uint64_t> addr_1 = free_list_allocator.allocate_at_address(64, 32);
    ASSERT_TRUE(addr_1.has_value());
    EXPECT_EQ(addr_1.value(), 64);

    std::optional<uint64_t> addr_2 = free_list_allocator.allocate(48, true);
    ASSERT_TRUE(addr_2.has_value());
    EXPECT_EQ(addr_2.value(), 96);

    std::optional<uint64_t> addr_3 = free_list_allocator.allocate(16, true);
    ASSERT_TRUE(addr_3.has_value());
    EXPECT_EQ(addr_3.value(), 32);

    std::optional<uint64_t> addr_4 = free_list_allocator.allocate_at_address(512, 128);
    ASSERT_TRUE(addr_4.has_value());
    EXPECT_EQ(addr_4.value(), 512);

    free_list_allocator.deallocate(96);  // coalesce with next block
    // After deallocating check that memory between the coalesced blocks
    // is free to be allocated
    std::optional<uint64_t> addr_5 = free_list_allocator.allocate_at_address(128, 64);
    ASSERT_TRUE(addr_5.has_value());
    EXPECT_EQ(addr_5.value(), 128);

    std::optional<uint64_t> addr_6 = free_list_allocator.allocate(32, true);
    ASSERT_TRUE(addr_6.has_value());
    EXPECT_EQ(addr_6.value(), 96);

    free_list_allocator.deallocate(32);
    free_list_allocator.deallocate(64);  // coalesce with prev block
    // After deallocating check that memory between the coalesced blocks
    // is free to be allocated
    std::optional<uint64_t> addr_7 = free_list_allocator.allocate(64, true);
    ASSERT_TRUE(addr_7.has_value());
    EXPECT_EQ(addr_7.value(), 32);

    std::optional<uint64_t> addr_8 = free_list_allocator.allocate(316, true);
    ASSERT_TRUE(addr_8.has_value());
    EXPECT_EQ(addr_8.value(), 192);

    free_list_allocator.deallocate(32);
    free_list_allocator.deallocate(128);
    free_list_allocator.deallocate(96);  // coalesce with prev and next block
    // After deallocating check that memory between the coalesced blocks
    // is free to be allocated
    std::optional<uint64_t> addr_9 = free_list_allocator.allocate_at_address(64, 96);
    ASSERT_TRUE(addr_9.has_value());
    EXPECT_EQ(addr_9.value(), 64);

    free_list_allocator.deallocate(192);
    std::optional<uint64_t> addr_10 = free_list_allocator.allocate_at_address(256, 128);
    ASSERT_TRUE(addr_10.has_value());
    EXPECT_EQ(addr_10.value(), 256);

    free_list_allocator.deallocate(0);
    std::optional<uint64_t> addr_11 = free_list_allocator.allocate(28, true);
    ASSERT_TRUE(addr_11.has_value());
    EXPECT_EQ(addr_11.value(), 0);

    std::optional<uint64_t> addr_12 = free_list_allocator.allocate(64, false);
    ASSERT_TRUE(addr_12.has_value());
    EXPECT_EQ(addr_12.value(), 960);

    std::optional<uint64_t> addr_13 = free_list_allocator.allocate(128, false);
    ASSERT_TRUE(addr_13.has_value());
    EXPECT_EQ(addr_13.value(), 832);

    std::optional<uint64_t> addr_14 = free_list_allocator.allocate_at_address(736, 96);
    ASSERT_TRUE(addr_14.has_value());
    EXPECT_EQ(addr_14.value(), 736);

    std::optional<uint64_t> addr_15 = free_list_allocator.allocate(96, false);
    ASSERT_TRUE(addr_15.has_value());
    EXPECT_EQ(addr_15.value(), 640);

    std::optional<uint64_t> addr_16 = free_list_allocator.allocate(96, false);
    ASSERT_TRUE(addr_16.has_value());
    EXPECT_EQ(addr_16.value(), 416);

    free_list_allocator.deallocate(416);
    free_list_allocator.deallocate(512);
    // After deallocating check that memory between the coalesced blocks
    // is free to be allocated
    std::optional<uint64_t> addr_17 = free_list_allocator.allocate(224, true);
    ASSERT_TRUE(addr_17.has_value());
    EXPECT_EQ(addr_17.value(), 384);

    free_list_allocator.deallocate(736);

    // Allocate entire region
    free_list_allocator.clear();
    std::optional<uint64_t> addr_18 = free_list_allocator.allocate(max_size_bytes, true);
    ASSERT_TRUE(addr_18.has_value());
    EXPECT_EQ(addr_18.value(), 0);

    free_list_allocator.deallocate(0);

    std::optional<uint64_t> addr_19 = free_list_allocator.allocate(64, true);
    ASSERT_TRUE(addr_19.has_value());
    EXPECT_EQ(addr_19.value(), 0);

    std::optional<uint64_t> addr_20 = free_list_allocator.allocate(max_size_bytes - 64, true);
    ASSERT_TRUE(addr_20.has_value());
    EXPECT_EQ(addr_20.value(), 64);
}

TEST(FreeListAllocator, TestResizeAllocator) {
    constexpr uint32_t max_size_bytes = 1024;
    constexpr uint32_t min_allocation_size_bytes = 32;
    constexpr uint32_t alignment = 32;

    tt::tt_metal::allocator::FreeList free_list_allocator = tt::tt_metal::allocator::FreeList(
        max_size_bytes,
        /*offset*/ 0,
        min_allocation_size_bytes,
        alignment,
        tt::tt_metal::allocator::FreeList::SearchPolicy::FIRST);

    std::optional<uint64_t> addr_0 = free_list_allocator.allocate(32, false);
    ASSERT_TRUE(addr_0.has_value());
    EXPECT_EQ(addr_0.value(), 992);

    free_list_allocator.shrink_size(64, true);

    std::optional<uint64_t> addr_1 = free_list_allocator.allocate(32, false);
    ASSERT_TRUE(addr_1.has_value());
    EXPECT_EQ(addr_1.value(), 960);

    std::optional<uint64_t> addr_2 = free_list_allocator.allocate(32, true);
    ASSERT_TRUE(addr_2.has_value());
    EXPECT_EQ(addr_2.value(), 64);

    free_list_allocator.reset_size();

    std::optional<uint64_t> addr_3 = free_list_allocator.allocate(32, true);
    ASSERT_TRUE(addr_3.has_value());
    EXPECT_EQ(addr_3.value(), 0);

    std::optional<uint64_t> addr_4 = free_list_allocator.allocate(32, true);
    ASSERT_TRUE(addr_4.has_value());
    EXPECT_EQ(addr_4.value(), 32);

    free_list_allocator.deallocate(0);

    std::optional<uint64_t> addr_5 = free_list_allocator.allocate(64, true);
    ASSERT_TRUE(addr_5.has_value());
    EXPECT_EQ(addr_5.value(), 96);

    free_list_allocator.shrink_size(32, true);

    free_list_allocator.deallocate(32);

    std::optional<uint64_t> addr_6 = free_list_allocator.allocate(32, true);
    ASSERT_TRUE(addr_6.has_value());
    EXPECT_EQ(addr_6.value(), 32);
}

TEST(FreeListAllocator, TestDirectedResizeAllocator) {
    constexpr uint32_t max_size_bytes = 1024;
    constexpr uint32_t min_allocation_size_bytes = 32;
    constexpr uint32_t alignment = 32;

    tt::tt_metal::allocator::FreeList free_list_allocator = tt::tt_metal::allocator::FreeList(
        max_size_bytes,
        /*offset*/ 0,
        min_allocation_size_bytes,
        alignment,
        tt::tt_metal::allocator::FreeList::SearchPolicy::FIRST);

    std::optional<uint64_t> addr_0 = free_list_allocator.allocate_at_address(32, 992);
    ASSERT_TRUE(addr_0.has_value());
    EXPECT_EQ(addr_0.value(), 32);

    free_list_allocator.shrink_size(32, true);

    std::optional<uint64_t> addr_1 = free_list_allocator.allocate(32, false);
    ASSERT_TRUE(!addr_1.has_value());

    std::optional<uint64_t> addr_2 = free_list_allocator.allocate_at_address(0, 32);
    ASSERT_TRUE(!addr_2.has_value());

    free_list_allocator.deallocate(32);

    std::optional<uint64_t> addr_3 = free_list_allocator.allocate(32, true);
    ASSERT_TRUE(addr_3.has_value());
    EXPECT_EQ(addr_3.value(), 32);

    std::optional<uint64_t> addr_4 = free_list_allocator.allocate(32, false);
    ASSERT_TRUE(addr_4.has_value());
    EXPECT_EQ(addr_4.value(), 992);

    free_list_allocator.reset_size();

    std::optional<uint64_t> addr_5 = free_list_allocator.allocate(32, true);
    ASSERT_TRUE(addr_5.has_value());
    EXPECT_EQ(addr_5.value(), 0);

    free_list_allocator.deallocate(32);

    std::optional<uint64_t> addr_6 = free_list_allocator.allocate(32, true);
    ASSERT_TRUE(addr_6.has_value());
    EXPECT_EQ(addr_6.value(), 32);
}
