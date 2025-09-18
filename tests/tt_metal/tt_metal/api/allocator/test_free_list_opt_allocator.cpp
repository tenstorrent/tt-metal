// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <stddef.h>
#include <optional>
#include <utility>
#include <vector>

#include <tt-metalium/allocator_types.hpp>
#include <tt-metalium/hal_types.hpp>
#include "tt_metal/impl/allocator/algorithms/free_list_opt.hpp"

// UDL to convert integer literals to SI units
constexpr size_t operator""_KiB(unsigned long long x) { return x * 1024; }
constexpr size_t operator""_MiB(unsigned long long x) { return x * 1024 * 1024; }
constexpr size_t operator""_GiB(unsigned long long x) { return x * 1024 * 1024 * 1024; }

TEST(FreeListOptTest, Allocation) {
    auto allocator = tt::tt_metal::allocator::FreeListOpt(1_GiB, 0, 1_KiB, 1_KiB);
    auto a = allocator.allocate(1_KiB);
    ASSERT_TRUE(a.has_value());
    ASSERT_EQ(a.value(), 0);

    auto b = allocator.allocate(1_KiB);
    ASSERT_TRUE(b.has_value());
    ASSERT_EQ(b.value(), 1_KiB);
}

TEST(FreeListOptTest, Alignment) {
    auto allocator = tt::tt_metal::allocator::FreeListOpt(1_GiB, 0, 1, 1_KiB);
    auto a = allocator.allocate(64);
    ASSERT_TRUE(a.has_value());
    ASSERT_EQ(a.value(), 0);
    auto b = allocator.allocate(64);
    ASSERT_TRUE(b.has_value());
    ASSERT_EQ(b.value(), 1_KiB);
}

TEST(FreeListOptTest, MinAllocationSize) {
    auto allocator = tt::tt_metal::allocator::FreeListOpt(1_GiB, 0, 1_KiB, 1);
    auto a = allocator.allocate(1);
    ASSERT_TRUE(a.has_value());
    ASSERT_EQ(a.value(), 0);
    auto b = allocator.allocate(1);
    ASSERT_TRUE(b.has_value());
    ASSERT_EQ(b.value(), 1_KiB);
}

TEST(FreeListOptTest, Clear) {
    auto allocator = tt::tt_metal::allocator::FreeListOpt(1_GiB, 0, 1_KiB, 1_KiB);
    auto a = allocator.allocate(1_KiB);
    auto b = allocator.allocate(1_KiB);
    ASSERT_TRUE(a.has_value());
    ASSERT_TRUE(b.has_value());
    allocator.clear();
    auto c = allocator.allocate(1_KiB);
    ASSERT_TRUE(c.has_value());
    ASSERT_EQ(c.value(), 0);
}

TEST(FreeListOptTest, AllocationAndDeallocation) {
    auto allocator = tt::tt_metal::allocator::FreeListOpt(1_GiB, 0, 1_KiB, 1_KiB);
    std::vector<std::optional<tt::tt_metal::DeviceAddr>> allocations(10);

    // Deallocate in order
    for(size_t i = 0; i < allocations.size(); i++) {
        allocations[i] = allocator.allocate(1_KiB);
        ASSERT_TRUE(allocations[i].has_value());
    }

    for(size_t i = allocations.size(); i > 0; i--) {
        allocator.deallocate(allocations[i - 1].value());
    }

    // Deallocate in reverse order
    for(size_t i = 0; i < allocations.size(); i++) {
        allocations[i] = allocator.allocate(1_KiB);
        ASSERT_TRUE(allocations[i].has_value());
    }

    for(size_t i = 0; i < allocations.size(); i++) {
        allocator.deallocate(allocations[i].value());
    }
}

TEST(FreeListOptTest, AllocateAtAddress) {
    auto allocator = tt::tt_metal::allocator::FreeListOpt(1_GiB, 0, 1_KiB, 1_KiB);
    auto a = allocator.allocate(1_KiB);
    ASSERT_TRUE(a.has_value());
    ASSERT_EQ(a.value(), 0);

    auto b = allocator.allocate_at_address(1_KiB, 1_KiB);
    ASSERT_TRUE(b.has_value());
    ASSERT_EQ(b.value(), 1_KiB);

    // Address is already allocated
    auto c = allocator.allocate_at_address(1_KiB, 1_KiB);
    ASSERT_FALSE(c.has_value());

    auto d = allocator.allocate_at_address(2_KiB, 1_KiB);
    ASSERT_TRUE(d.has_value());
    ASSERT_EQ(d.value(), 2_KiB);

    allocator.deallocate(a.value());
    auto e = allocator.allocate_at_address(0, 1_KiB);
    ASSERT_TRUE(e.has_value());
    ASSERT_EQ(e.value(), 0);
}

TEST(FreeListOptTest, AllocateAtAddressInteractions) {
    auto allocator = tt::tt_metal::allocator::FreeListOpt(1_GiB, 0, 1_KiB, 1_KiB);
    allocator.allocate_at_address(32_KiB, 1_KiB);

    auto a = allocator.allocate(1_KiB);
    ASSERT_TRUE(a.has_value());
    ASSERT_EQ(a.value(), 0);

    auto z = allocator.allocate(1_KiB, false);
    ASSERT_TRUE(z.has_value());
    ASSERT_EQ(z.value(), 32_KiB - 1_KiB); // Counterintuitive, but because we use BestFit, it will find the smaller block at the beginning

    auto b = allocator.allocate(1_KiB);
    ASSERT_TRUE(b.has_value());
    ASSERT_EQ(b.value(), 1_KiB);
}

TEST(FreeListOptTest, ShrinkAndReset) {
    auto allocator = tt::tt_metal::allocator::FreeListOpt(1_GiB, 0, 1_KiB, 1_KiB);
    auto a = allocator.allocate(1_KiB);
    auto b = allocator.allocate(1_KiB);
    ASSERT_TRUE(a.has_value());
    ASSERT_TRUE(b.has_value());
    allocator.deallocate(a.value());

    allocator.shrink_size(1_KiB);
    auto c = allocator.allocate_at_address(0, 1_KiB);
    ASSERT_FALSE(c.has_value());

    auto d = allocator.allocate_at_address(1_KiB, 1_KiB);
    ASSERT_FALSE(d.has_value());

    allocator.reset_size();
    allocator.deallocate(b.value());

    auto e = allocator.allocate(2_KiB);
    ASSERT_TRUE(e.has_value());
}

TEST(FreeListOptTest, Statistics) {
    auto allocator = tt::tt_metal::allocator::FreeListOpt(1_GiB, 0, 1_KiB, 1_KiB);
    auto a = allocator.allocate(1_KiB);
    auto b = allocator.allocate(1_KiB);
    ASSERT_TRUE(a.has_value());
    ASSERT_TRUE(b.has_value());
    allocator.deallocate(a.value());

    auto stats = allocator.get_statistics();
    ASSERT_EQ(stats.total_allocated_bytes, 1_KiB);
}

TEST(FreeListOptTest, AllocateFromTop) {
    auto allocator = tt::tt_metal::allocator::FreeListOpt(1_GiB, 0, 1_KiB, 1_KiB);
    auto a = allocator.allocate(1_KiB, false);
    ASSERT_TRUE(a.has_value());
    ASSERT_EQ(a.value(), 1_GiB - 1_KiB);

    auto b = allocator.allocate(1_KiB, false);
    ASSERT_TRUE(b.has_value());
    ASSERT_EQ(b.value(), 1_GiB - 2_KiB);

    auto c = allocator.allocate(1_KiB);
    ASSERT_TRUE(c.has_value());
    ASSERT_EQ(c.value(), 0);
}

TEST(FreeListOptTest, Coalescing) {
    auto allocator = tt::tt_metal::allocator::FreeListOpt(1_GiB, 0, 1_KiB, 1_KiB);
    auto a = allocator.allocate(1_KiB);
    auto b = allocator.allocate(1_KiB);
    auto c = allocator.allocate(1_KiB);
    ASSERT_TRUE(a.has_value());
    ASSERT_TRUE(b.has_value());
    ASSERT_TRUE(c.has_value());
    allocator.deallocate(b.value());
    allocator.deallocate(a.value());

    auto d = allocator.allocate(2_KiB);
    ASSERT_TRUE(d.has_value());
    ASSERT_EQ(d.value(), 0);
}

TEST(FreeListOptTest, CoalescingAfterResetShrink) {
    auto allocator = tt::tt_metal::allocator::FreeListOpt(1_GiB, 0, 1_KiB, 1_KiB);
    auto a = allocator.allocate(1_KiB);
    auto b = allocator.allocate(1_KiB);
    auto c = allocator.allocate(1_KiB);
    ASSERT_TRUE(a.has_value());
    ASSERT_TRUE(b.has_value());
    ASSERT_TRUE(c.has_value());
    allocator.deallocate(b.value());
    allocator.deallocate(a.value());

    allocator.shrink_size(1_KiB);
    allocator.reset_size();
    auto e = allocator.allocate(2_KiB);
    ASSERT_TRUE(e.has_value());
    ASSERT_EQ(e.value(), 0);
}

TEST(FreeListOptTest, OutOfMemory) {
    auto allocator = tt::tt_metal::allocator::FreeListOpt(1_GiB, 0, 1_KiB, 1_KiB);
    auto a = allocator.allocate(1_GiB);
    ASSERT_TRUE(a.has_value());
    auto b = allocator.allocate(1_KiB);
    ASSERT_FALSE(b.has_value());

    allocator.clear();
    auto c = allocator.allocate(1_GiB - 1_KiB);
    ASSERT_TRUE(c.has_value());
    auto d = allocator.allocate(2_KiB);
    ASSERT_FALSE(d.has_value());
}

TEST(FreeListOptTest, AvailableAddresses) {
    auto allocator = tt::tt_metal::allocator::FreeListOpt(1_GiB, 0, 1_KiB, 1_KiB);
    auto a = allocator.allocate(1_KiB);
    auto aval = allocator.available_addresses(1_KiB);
    ASSERT_EQ(aval.size(), 1);
    ASSERT_EQ(aval[0].first, 1_KiB); // Start address
    ASSERT_EQ(aval[0].second, 1_GiB); // End address
    allocator.clear();

    a = allocator.allocate(1_KiB);
    auto b = allocator.allocate(1_KiB);
    auto c = allocator.allocate(1_KiB);
    ASSERT_TRUE(a.has_value());
    ASSERT_EQ(a.value(), 0);
    ASSERT_TRUE(b.has_value());
    ASSERT_EQ(b.value(), 1_KiB);
    ASSERT_TRUE(c.has_value());
    ASSERT_EQ(c.value(), 2_KiB);
    allocator.deallocate(b.value());
    aval = allocator.available_addresses(1_KiB);
    ASSERT_EQ(aval.size(), 2);
    ASSERT_EQ(aval[0].first, 1_KiB); // Start address
    ASSERT_EQ(aval[0].second, 2_KiB); // End address
    ASSERT_EQ(aval[1].first, 3_KiB); // Start address
    ASSERT_EQ(aval[1].second, 1_GiB); // End address

    allocator.clear();
    a = allocator.allocate(1_KiB);
    b = allocator.allocate(1_KiB);
    c = allocator.allocate(1_KiB);
    ASSERT_TRUE(a.has_value());
    ASSERT_EQ(a.value(), 0);
    ASSERT_TRUE(b.has_value());
    ASSERT_EQ(b.value(), 1_KiB);
    ASSERT_TRUE(c.has_value());
    ASSERT_EQ(c.value(), 2_KiB);
    allocator.deallocate(b.value());
    aval = allocator.available_addresses(10_KiB);
    ASSERT_EQ(aval.size(), 1);
    ASSERT_EQ(aval[0].first, 3_KiB); // Start address
    ASSERT_EQ(aval[0].second, 1_GiB); // End address
}

TEST(FreeListOptTest, LowestOccupiedAddress) {
    auto allocator = tt::tt_metal::allocator::FreeListOpt(1_GiB, 0, 1_KiB, 1_KiB);
    auto a = allocator.allocate(1_KiB);
    auto b = allocator.allocate(1_KiB);
    auto c = allocator.allocate(1_KiB);
    ASSERT_TRUE(a.has_value());
    ASSERT_EQ(a.value(), 0);
    ASSERT_TRUE(b.has_value());
    ASSERT_EQ(b.value(), 1_KiB);
    ASSERT_TRUE(c.has_value());
    ASSERT_EQ(c.value(), 2_KiB);
    auto loa = allocator.lowest_occupied_address();
    ASSERT_EQ(loa.value(), 0);
    allocator.deallocate(a.value());
    loa = allocator.lowest_occupied_address();
    ASSERT_EQ(loa.value(), 1_KiB);
    allocator.deallocate(b.value());
    loa = allocator.lowest_occupied_address();
    ASSERT_EQ(loa.value(), 2_KiB);
    allocator.deallocate(c.value());
    loa = allocator.lowest_occupied_address();
    ASSERT_FALSE(loa.has_value());
}

TEST(FreeListOptTest, LowestOccupiedAddressWithAllocateAt) {
    auto allocator = tt::tt_metal::allocator::FreeListOpt(1_GiB, 0, 1_KiB, 1_KiB);
    auto a = allocator.allocate_at_address(1_KiB, 1_KiB);
    ASSERT_TRUE(a.has_value());
    ASSERT_EQ(a.value(), 1_KiB);
    auto loa = allocator.lowest_occupied_address();
    ASSERT_EQ(loa.value(), 1_KiB);
    allocator.deallocate(a.value());
    loa = allocator.lowest_occupied_address();
    ASSERT_FALSE(loa.has_value());
}

TEST(FreeListOptTest, FirstFit) {
    auto allocator = tt::tt_metal::allocator::FreeListOpt(1_GiB, 0, 1_KiB, 1_KiB, tt::tt_metal::allocator::FreeListOpt::SearchPolicy::FIRST);
    auto a = allocator.allocate(1_KiB);
    auto b = allocator.allocate(3_KiB);
    auto c = allocator.allocate(1_KiB);
    auto d = allocator.allocate(1_KiB);
    auto e = allocator.allocate(1_KiB);

    ASSERT_TRUE(a.has_value());
    ASSERT_EQ(a.value(), 0);
    ASSERT_TRUE(b.has_value());
    ASSERT_EQ(b.value(), 1_KiB);
    ASSERT_TRUE(c.has_value());
    ASSERT_EQ(c.value(), 4_KiB);
    ASSERT_TRUE(d.has_value());
    ASSERT_EQ(d.value(), 5_KiB);
    ASSERT_TRUE(e.has_value());
    ASSERT_EQ(e.value(), 6_KiB);

    allocator.deallocate(b.value());
    allocator.deallocate(d.value());

    auto f = allocator.allocate(1_KiB);
    ASSERT_TRUE(f.has_value());
    ASSERT_EQ(f.value(), 1_KiB);
}

TEST(FreeListOptTest, FirstFitAllocateAtAddressInteractions) {
    auto allocator = tt::tt_metal::allocator::FreeListOpt(1_GiB, 0, 1_KiB, 1_KiB, tt::tt_metal::allocator::FreeListOpt::SearchPolicy::FIRST);
    allocator.allocate_at_address(32_KiB, 1_KiB);

    auto a = allocator.allocate(1_KiB);
    ASSERT_TRUE(a.has_value());
    ASSERT_EQ(a.value(), 0);

    auto z = allocator.allocate(1_KiB, false);
    ASSERT_TRUE(z.has_value());
    ASSERT_EQ(z.value(), 1_GiB - 1_KiB);

    auto b = allocator.allocate(1_KiB);
    ASSERT_TRUE(b.has_value());
    ASSERT_EQ(b.value(), 1_KiB);
}

TEST(FreeListOptTest, ReallocateAtSameAddressWithAllocateAtAddress) {
    auto allocator = tt::tt_metal::allocator::FreeListOpt(1_GiB, 0, 1_KiB, 1_KiB);

    /*
     * Any non-zero address stress tests necessary guard against stale metablocks in allocate_at_address impl:
     * 1. Allocate, then deallocate results in two block_addresses_
          * If address is 0, then first block_address_ is the fully merged and alive metablock
          * If address is non-zero, then first block_address_ is the stale (previously allocated) metablock
     * 2. segregated_list will fail to find the reallocated address if the stale metablock is not skipped
          * You can build in Debug mode or switch to TT_FATAL for this assert:
            TT_FATAL(it != segregated_list.end(), "Block not found in size segregated list");
    */
    const size_t alloc_address = 1_KiB;

    // Allocate with allocate_at_address
    auto a = allocator.allocate_at_address(alloc_address, 1_KiB);
    ASSERT_THAT(a, ::testing::Optional(alloc_address));

    allocator.deallocate(a.value());

    // Try to reallocate at the same address
    auto a_realloc = allocator.allocate_at_address(alloc_address, 1_KiB);
    ASSERT_THAT(a_realloc, ::testing::Optional(alloc_address));
}

TEST(FreeListOptTest, AllocatedAddresses) {
    auto allocator = tt::tt_metal::allocator::FreeListOpt(1_GiB, 0, 1_KiB, 1_KiB);

    // Check that allocated addresses is empty
    auto empty_allocated_addresses = allocator.allocated_addresses();
    ASSERT_TRUE(empty_allocated_addresses.empty());

    // Allocate some blocks and validate allocated addresses
    auto a = allocator.allocate(512_KiB, /*bottom_up=*/false);
    ASSERT_THAT(a, ::testing::Optional(1_GiB - 512_KiB));

    auto b = allocator.allocate(2_KiB);
    ASSERT_THAT(b, ::testing::Optional(0));

    // Unaligned size should be aligned to the next multiple of 1_KiB
    auto c = allocator.allocate(500);
    ASSERT_THAT(c, ::testing::Optional(2_KiB));

    auto allocated_addresses = allocator.allocated_addresses();
    ASSERT_EQ(allocated_addresses.size(), 3);

    // Allocated addresses are not sorted by start address; in this case, it should be in order of: a, b, c
    ASSERT_EQ(allocated_addresses[0], (std::pair<tt::tt_metal::DeviceAddr, tt::tt_metal::DeviceAddr>{1_GiB - 512_KiB, 1_GiB}));
    ASSERT_EQ(allocated_addresses[1], (std::pair<tt::tt_metal::DeviceAddr, tt::tt_metal::DeviceAddr>{0, 2_KiB}));
    ASSERT_EQ(allocated_addresses[2], (std::pair<tt::tt_metal::DeviceAddr, tt::tt_metal::DeviceAddr>{2_KiB, 3_KiB}));

    /*********************************************************
     * Check allocated_addresses is correct after other APIs *
     *********************************************************/
    // Deallocate first block
    allocator.deallocate(a.value());
    auto after_free = allocator.allocated_addresses();
    ASSERT_EQ(after_free.size(), 2);
    ASSERT_EQ(after_free[0], (std::pair<tt::tt_metal::DeviceAddr, tt::tt_metal::DeviceAddr>{0_KiB, 2_KiB}));
    ASSERT_EQ(after_free[1], (std::pair<tt::tt_metal::DeviceAddr, tt::tt_metal::DeviceAddr>{2_KiB, 3_KiB}));

    // Clear -> empty again
    allocator.clear();
    auto after_clear = allocator.allocated_addresses();
    ASSERT_TRUE(after_clear.empty());

    // Allocate from top to leave space at bottom, then shrink and reset
    allocator.allocate(1_KiB, /*bottom_up=*/false);
    auto after_top = allocator.allocated_addresses();
    ASSERT_EQ(after_top.size(), 1);
    ASSERT_EQ(after_top[0], (std::pair<tt::tt_metal::DeviceAddr, tt::tt_metal::DeviceAddr>{1_GiB - 1_KiB, 1_GiB}));

    // Shrink from bottom (should not affect allocated block near top)
    allocator.shrink_size(1_KiB);
    auto after_shrink = allocator.allocated_addresses();
    ASSERT_EQ(after_shrink, after_top);

    // Reset size back
    allocator.reset_size();
    auto after_reset = allocator.allocated_addresses();
    ASSERT_EQ(after_reset, after_top);
}
