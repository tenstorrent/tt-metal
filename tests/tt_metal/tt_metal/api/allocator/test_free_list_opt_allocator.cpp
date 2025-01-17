// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <tt-metalium/allocator.hpp>
#include "tt_metal/impl/allocator/algorithms/free_list_opt.hpp"

// UDL to convert integer literals to SI units
constexpr size_t operator"" _KiB(unsigned long long x) { return x * 1024; }
constexpr size_t operator"" _MiB(unsigned long long x) { return x * 1024 * 1024; }
constexpr size_t operator"" _GiB(unsigned long long x) { return x * 1024 * 1024 * 1024; }

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
    auto wedge = allocator.allocate_at_address(32_KiB, 1_KiB);

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
    auto d = allocator.allocate(2_KiB);
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
    auto wedge = allocator.allocate_at_address(32_KiB, 1_KiB);

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
