// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include <tt-metalium/hal_types.hpp>
#include "tt_metal/impl/allocator/algorithms/free_list_opt.hpp"

// UDL to convert integer literals to SI units
constexpr size_t operator""_KiB(unsigned long long x) { return x * 1024; }
constexpr size_t operator""_MiB(unsigned long long x) { return x * 1024 * 1024; }
constexpr size_t operator""_GiB(unsigned long long x) { return x * 1024 * 1024 * 1024; }

TEST(FreeListOptTest, CPU_Allocation) {
    auto allocator = tt::tt_metal::allocator::FreeListOpt(1_GiB, 0, 1_KiB, 1_KiB);
    auto a = allocator.allocate(1_KiB);
    ASSERT_TRUE(a.has_value());
    ASSERT_EQ(a.value(), 0);

    auto b = allocator.allocate(1_KiB);
    ASSERT_TRUE(b.has_value());
    ASSERT_EQ(b.value(), 1_KiB);
}

TEST(FreeListOptTest, CPU_Alignment) {
    auto allocator = tt::tt_metal::allocator::FreeListOpt(1_GiB, 0, 1, 1_KiB);
    auto a = allocator.allocate(64);
    ASSERT_TRUE(a.has_value());
    ASSERT_EQ(a.value(), 0);
    auto b = allocator.allocate(64);
    ASSERT_TRUE(b.has_value());
    ASSERT_EQ(b.value(), 1_KiB);
}

TEST(FreeListOptTest, CPU_MinAllocationSize) {
    auto allocator = tt::tt_metal::allocator::FreeListOpt(1_GiB, 0, 1_KiB, 1);
    auto a = allocator.allocate(1);
    ASSERT_TRUE(a.has_value());
    ASSERT_EQ(a.value(), 0);
    auto b = allocator.allocate(1);
    ASSERT_TRUE(b.has_value());
    ASSERT_EQ(b.value(), 1_KiB);
}

TEST(FreeListOptTest, CPU_Clear) {
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

TEST(FreeListOptTest, CPU_AllocationAndDeallocation) {
    auto allocator = tt::tt_metal::allocator::FreeListOpt(1_GiB, 0, 1_KiB, 1_KiB);
    std::vector<std::optional<tt::tt_metal::DeviceAddr>> allocations(10);

    // Deallocate in order
    for(auto & allocation : allocations) {
        allocation = allocator.allocate(1_KiB);
        ASSERT_TRUE(allocation.has_value());
    }

    for(size_t i = allocations.size(); i > 0; i--) {
        allocator.deallocate(allocations[i - 1].value());
    }

    // Deallocate in reverse order
    for(auto & allocation : allocations) {
        allocation = allocator.allocate(1_KiB);
        ASSERT_TRUE(allocation.has_value());
    }

    for(auto & allocation : allocations) {
        allocator.deallocate(allocation.value());
    }
}

TEST(FreeListOptTest, CPU_AllocateAtAddress) {
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

TEST(FreeListOptTest, CPU_AllocateAtAddressInteractions) {
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

TEST(FreeListOptTest, CPU_ShrinkAndReset) {
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

TEST(FreeListOptTest, CPU_RejectFullCapacityShrink) {
    auto allocator = tt::tt_metal::allocator::FreeListOpt(1_GiB, 0, 1_KiB, 1_KiB);
    const auto stats = allocator.get_statistics();
    const auto blocks = allocator.get_memory_block_table();

    EXPECT_THAT(
        [&]() { allocator.shrink_size(1_GiB); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("must be smaller than max size")));

    EXPECT_EQ(allocator.get_statistics().total_allocatable_size_bytes, stats.total_allocatable_size_bytes);
    EXPECT_EQ(allocator.get_statistics().total_allocated_bytes, stats.total_allocated_bytes);
    EXPECT_EQ(allocator.get_statistics().total_free_bytes, stats.total_free_bytes);
    EXPECT_EQ(allocator.get_statistics().largest_free_block_bytes, stats.largest_free_block_bytes);
    EXPECT_EQ(allocator.get_memory_block_table(), blocks);
}

// Full-capacity shrink is rejected, so pin 1_KiB at the top and consume the entire leading free
// block. That takes the size-becomes-0 unlink path (the original OOB wrote through a -1 next-block
// sentinel when that block also had no successor).
TEST(FreeListOptTest, CPU_ShrinkEntireLeadingFreeBlockAndReset) {
    auto allocator = tt::tt_metal::allocator::FreeListOpt(1_GiB, 0, 1_KiB, 1_KiB);

    auto pinned = allocator.allocate(1_KiB, /*bottom_up=*/false);
    ASSERT_TRUE(pinned.has_value());
    ASSERT_EQ(pinned.value(), 1_GiB - 1_KiB);

    allocator.shrink_size(1_GiB - 1_KiB);
    auto a = allocator.allocate(1_KiB);
    ASSERT_FALSE(a.has_value());

    allocator.reset_size();
    auto b = allocator.allocate(1_GiB - 1_KiB);
    ASSERT_TRUE(b.has_value());
    ASSERT_EQ(b.value(), 0);
}

TEST(FreeListOptTest, CPU_Statistics) {
    auto allocator = tt::tt_metal::allocator::FreeListOpt(1_GiB, 0, 1_KiB, 1_KiB);
    auto a = allocator.allocate(1_KiB);
    auto b = allocator.allocate(1_KiB);
    ASSERT_TRUE(a.has_value());
    ASSERT_TRUE(b.has_value());
    allocator.deallocate(a.value());

    auto stats = allocator.get_statistics();
    ASSERT_EQ(stats.total_allocated_bytes, 1_KiB);
}

TEST(FreeListOptTest, CPU_AllocateFromTop) {
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

TEST(FreeListOptTest, CPU_Coalescing) {
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

TEST(FreeListOptTest, CPU_CoalescingAfterResetShrink) {
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

TEST(FreeListOptTest, CPU_OutOfMemory) {
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

TEST(FreeListOptTest, CPU_AvailableAddresses) {
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

TEST(FreeListOptTest, CPU_LowestOccupiedAddress) {
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

TEST(FreeListOptTest, CPU_LowestOccupiedAddressWithAllocateAt) {
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

TEST(FreeListOptTest, CPU_FirstFit) {
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

TEST(FreeListOptTest, CPU_FirstFitAllocateAtAddressInteractions) {
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

TEST(FreeListOptTest, CPU_ReallocateAtSameAddressWithAllocateAtAddress) {
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

TEST(FreeListOptTest, CPU_AllocatedAddresses) {
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

TEST(FreeListOptTest, CPU_AddressesAPIWithNonzeroOffset) {
    // Test APIs that expose addresses as inputs/outputs correctly expose absolute addresses with offset added
    const size_t offset = 2_KiB;
    const size_t alloc_size = 1_GiB;
    auto allocator = tt::tt_metal::allocator::FreeListOpt(alloc_size, offset, 1_KiB, 1_KiB);

    auto available_addresses = allocator.available_addresses(1_KiB);
    ASSERT_EQ(available_addresses.size(), 1);
    ASSERT_EQ(available_addresses[0].first, offset);
    ASSERT_EQ(available_addresses[0].second, alloc_size + offset);

    // Allocate a block with allocate_at_address
    // The way offsets are used is essentially to manually block off some lower region of address space
    // So we convert absolute addresses to local allocator address by subtracting the offset
    // This means, in practice, we should only be allocating above the offset; otherwise UB
    auto a = allocator.allocate_at_address(offset, 3_KiB);
    // This assert is not that interesting; it will always return input absolute address
    ASSERT_THAT(a, ::testing::Optional(offset));

    auto allocated_addresses = allocator.allocated_addresses();
    ASSERT_EQ(allocated_addresses.size(), 1);
    ASSERT_EQ(allocated_addresses[0].first, offset);
    ASSERT_EQ(allocated_addresses[0].second, 3_KiB + offset);

    available_addresses = allocator.available_addresses(1_KiB);
    ASSERT_EQ(available_addresses.size(), 1);
    ASSERT_EQ(available_addresses[0].first, 3_KiB + offset);
    ASSERT_EQ(available_addresses[0].second, alloc_size + offset);

    // Allocate another block above using allocate
    auto b = allocator.allocate(2_KiB, /*bottom_up=*/false);
    ASSERT_THAT(b, ::testing::Optional(alloc_size - 2_KiB + offset));

    allocated_addresses = allocator.allocated_addresses();
    ASSERT_EQ(allocated_addresses.size(), 2);
    ASSERT_EQ(allocated_addresses[0].first, offset);
    ASSERT_EQ(allocated_addresses[0].second, 3_KiB + offset);
    ASSERT_EQ(allocated_addresses[1].first, alloc_size - 2_KiB + offset);
    ASSERT_EQ(allocated_addresses[1].second, alloc_size + offset);

    available_addresses = allocator.available_addresses(1_KiB);
    ASSERT_EQ(available_addresses.size(), 1);
    ASSERT_EQ(available_addresses[0].first, 3_KiB + offset);
    ASSERT_EQ(available_addresses[0].second, alloc_size - 2_KiB + offset);
}
