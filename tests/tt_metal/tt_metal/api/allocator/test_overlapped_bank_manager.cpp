// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "gmock/gmock.h"
#include <cstdint>
#include <tt-metalium/allocator.hpp>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/device.hpp>
#include "device_fixture.hpp"
#include <tt-metalium/hal_types.hpp>
#include "impl/context/metal_context.hpp"
#include "tt_metal/impl/allocator/bank_manager.hpp"

namespace overlapped_bank_manager_tests {
struct AllocatorDependenciesParam {
    std::unordered_map<
        tt::tt_metal::BankManager::AllocatorDependencies::AllocatorID,
        ttsl::SmallVector<tt::tt_metal::BankManager::AllocatorDependencies::AllocatorID>>
        input;
    // Expected dependencies per allocator (missing keys imply empty list)
    tt::tt_metal::BankManager::AllocatorDependencies::AdjacencyList expected_dependencies;
};

tt::tt_metal::BankManager get_bank_manager_with_allocator_dependencies(
    const uint64_t size,
    const uint32_t alignment,
    const tt::tt_metal::BankManager::AllocatorDependencies& allocator_dependencies) {
    std::vector<int64_t> bank_desc = {0};
    const uint64_t unreserved_base = 0;
    return tt::tt_metal::BankManager(
        tt::tt_metal::BufferType::DRAM, bank_desc, size, alignment, unreserved_base, false, allocator_dependencies);
}

}  // namespace overlapped_bank_manager_tests

using namespace overlapped_bank_manager_tests;
using namespace tt::tt_metal;
using AllocatorID = BankManager::AllocatorDependencies::AllocatorID;

/*******************************
 * AllocatorDependencies Tests *
 *******************************/
TEST(AllocatorDependencies, DefaultAllocatorDependencies) {
    BankManager::AllocatorDependencies allocator_dependencies;
    EXPECT_EQ(allocator_dependencies.dependencies, BankManager::AllocatorDependencies::AdjacencyList{{}});
}

TEST(AllocatorDependencies, DuplicateDependencies) {
    const std::unordered_map<AllocatorID, ttsl::SmallVector<AllocatorID>> dependencies_map = {
        {AllocatorID{0}, ttsl::SmallVector<AllocatorID>{AllocatorID{1}, AllocatorID{1}}}};

    EXPECT_THAT(
        [&]() { BankManager::AllocatorDependencies allocator_dependencies{dependencies_map}; },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Duplicate dependency for allocator 0: 1 appears more than once!")));
}

TEST(AllocatorDependencies, EquivalentDependenciesMaps) {
    const std::unordered_map<AllocatorID, ttsl::SmallVector<AllocatorID>> dependencies_map1 = {
        {AllocatorID{0}, ttsl::SmallVector<AllocatorID>{AllocatorID{1}}},
        {AllocatorID{1}, ttsl::SmallVector<AllocatorID>{AllocatorID{0}}}};
    const std::unordered_map<AllocatorID, ttsl::SmallVector<AllocatorID>> dependencies_map2 = {
        {AllocatorID{0}, ttsl::SmallVector<AllocatorID>{AllocatorID{1}}}};
    const std::unordered_map<AllocatorID, ttsl::SmallVector<AllocatorID>> dependencies_map3 = {
        {AllocatorID{1}, ttsl::SmallVector<AllocatorID>{AllocatorID{0}}}};

    EXPECT_EQ(
        BankManager::AllocatorDependencies(dependencies_map1), BankManager::AllocatorDependencies(dependencies_map2));
    EXPECT_EQ(
        BankManager::AllocatorDependencies(dependencies_map1), BankManager::AllocatorDependencies(dependencies_map3));
}

class AllocatorDependenciesParamTest : public ::testing::TestWithParam<AllocatorDependenciesParam> {};

TEST_P(AllocatorDependenciesParamTest, ValidateDependencies) {
    const auto& params = GetParam();

    BankManager::AllocatorDependencies allocator_dependencies{params.input};

    // Validate dependencies
    EXPECT_EQ(allocator_dependencies.dependencies, params.expected_dependencies);
}

INSTANTIATE_TEST_SUITE_P(
    AllocatorDependencies,
    AllocatorDependenciesParamTest,
    ::testing::Values(
        // Empty input
        AllocatorDependenciesParam{
            /*input=*/{},
            /*expected_dependencies=*/{{}}},
        // Single allocator default behavior (explicit input)
        AllocatorDependenciesParam{
            /*input=*/{{AllocatorID{0}, {}}},
            /*expected_dependencies=*/{{}}},
        // Two-way dependency
        AllocatorDependenciesParam{
            /*input=*/{{AllocatorID{0}, {AllocatorID{1}}}, {AllocatorID{1}, {AllocatorID{0}}}},
            /*expected_dependencies=*/{{AllocatorID{1}}, {AllocatorID{0}}}},
        // Sparse keys with one dependency specified
        AllocatorDependenciesParam{
            /*input=*/{{AllocatorID{3}, {AllocatorID{0}, AllocatorID{1}}}},
            /*expected_dependencies=*/{{AllocatorID{3}}, {AllocatorID{3}}, {}, {AllocatorID{0}, AllocatorID{1}}}},
        // Sparse keys with one dependency specified (value of dependents imply more allocators)
        AllocatorDependenciesParam{
            /*input=*/{{AllocatorID{1}, {AllocatorID{3}}}},
            /*expected_dependencies=*/{{}, {AllocatorID{3}}, {}, {AllocatorID{1}}}},
        // Fan-in: 1,2,3 depend on 0
        AllocatorDependenciesParam{
            /*input=*/{
                {AllocatorID{1}, {AllocatorID{0}}},
                {AllocatorID{2}, {AllocatorID{0}}},
                {AllocatorID{3}, {AllocatorID{0}}}},
            /*expected_dependencies=*/
            {{AllocatorID{1}, AllocatorID{2}, AllocatorID{3}}, {AllocatorID{0}}, {AllocatorID{0}}, {AllocatorID{0}}}},
        // Fan-out: 0 depends on 1,2,3
        AllocatorDependenciesParam{
            /*input=*/{{AllocatorID{0}, {AllocatorID{1}, AllocatorID{2}, AllocatorID{3}}}},
            /*expected_dependencies=*/
            {{AllocatorID{1}, AllocatorID{2}, AllocatorID{3}}, {AllocatorID{0}}, {AllocatorID{0}}, {AllocatorID{0}}}},
        // Chain: 0->1->2->3
        AllocatorDependenciesParam{
            /*input=*/{
                {AllocatorID{0}, {AllocatorID{1}}},
                {AllocatorID{1}, {AllocatorID{2}}},
                {AllocatorID{2}, {AllocatorID{3}}}},
            /*expected_dependencies=*/
            {{AllocatorID{1}}, {AllocatorID{0}, AllocatorID{2}}, {AllocatorID{1}, AllocatorID{3}}, {AllocatorID{2}}}},
        // Cycle: 0->1->2->0
        AllocatorDependenciesParam{
            /*input=*/{
                {AllocatorID{0}, {AllocatorID{1}}},
                {AllocatorID{1}, {AllocatorID{2}}},
                {AllocatorID{2}, {AllocatorID{0}}}},
            /*expected_dependencies=*/{
                {AllocatorID{1}, AllocatorID{2}},
                {AllocatorID{0}, AllocatorID{2}},
                {AllocatorID{0}, AllocatorID{1}}}}));

/********************************
 * Overlapped BankManager Tests *
 ********************************/
TEST(OverlappedAllocators, InvalidAllocator) {
    // Create bank manager with 2 allocators (0 and 1)
    BankManager::AllocatorDependencies deps{{{AllocatorID{0}, {}}, {AllocatorID{1}, {}}}};
    BankManager bank_manager = get_bank_manager_with_allocator_dependencies(1024 * 1024, 1024, deps);

    // Test accessing invalid allocator that's greater than num_allocators
    AllocatorID invalid_allocator{2};  // Should fail since we only have allocators 0 and 1

    // Test non-const overload of get_allocator_from_id
    EXPECT_THAT(
        [&]() {
            bank_manager.allocate_buffer(
                1024, 1024, true, CoreRangeSet(std::vector<CoreRange>{}), std::nullopt, invalid_allocator);
        },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Invalid allocator ID 2 (num_allocators=2)")));

    // Test const overload of get_allocator_from_id
    EXPECT_THAT(
        [&]() { bank_manager.lowest_occupied_address(0, invalid_allocator); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Invalid allocator ID 2 (num_allocators=2)")));
}

TEST(OverlappedAllocators, InvalidAPIsForOverlappedAllocators) {
    // Create bank manager with 2 allocators (0 and 1)
    BankManager::AllocatorDependencies deps{{{AllocatorID{0}, {}}, {AllocatorID{1}, {}}}};
    BankManager bank_manager = get_bank_manager_with_allocator_dependencies(1024 * 1024, 1024, deps);

    // Test accessing an API that only works for single allocator
    EXPECT_THAT(
        [&]() { bank_manager.reset_size(AllocatorID{0}); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("Expected single allocator!")));
    EXPECT_THAT(
        [&]() { bank_manager.reset_size(AllocatorID{1}); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("Expected single allocator!")));
    EXPECT_THAT(
        [&]() { bank_manager.shrink_size(1024, true, AllocatorID{0}); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("Expected single allocator!")));
    EXPECT_THAT(
        [&]() { bank_manager.shrink_size(1024, true, AllocatorID{1}); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("Expected single allocator!")));
}

TEST(OverlappedAllocators, DeallocateAllAndClear) {
    // Two independent allocators (0 and 1); allocator 2 overlaps both 0 and 1
    BankManager::AllocatorDependencies deps{{{AllocatorID{0}, {AllocatorID{2}}}, {AllocatorID{1}, {AllocatorID{2}}}}};
    BankManager bank_manager = get_bank_manager_with_allocator_dependencies(1024 * 1024, 1024, deps);
    const uint32_t bank_id = 0;

    // Hard-coded allocation sizes
    const uint32_t alloc_size_1K = 1024;
    const uint32_t alloc_size_2K = 2048;

    // Verify no allocations exist initially
    EXPECT_EQ(bank_manager.lowest_occupied_address(bank_id, AllocatorID{0}), std::nullopt);
    EXPECT_EQ(bank_manager.lowest_occupied_address(bank_id, AllocatorID{1}), std::nullopt);
    EXPECT_EQ(bank_manager.lowest_occupied_address(bank_id, AllocatorID{2}), std::nullopt);

    // Allocate in each allocator
    bank_manager.allocate_buffer(
        alloc_size_1K, alloc_size_1K, true, CoreRangeSet(std::vector<CoreRange>{}), std::nullopt, AllocatorID{0});
    auto addr1 = bank_manager.allocate_buffer(
        alloc_size_2K, alloc_size_2K, true, CoreRangeSet(std::vector<CoreRange>{}), std::nullopt, AllocatorID{1});
    bank_manager.allocate_buffer(
        alloc_size_1K, alloc_size_1K, true, CoreRangeSet(std::vector<CoreRange>{}), std::nullopt, AllocatorID{2});

    // Verify allocations exist in all allocators
    EXPECT_EQ(bank_manager.lowest_occupied_address(bank_id, AllocatorID{0}), 0);
    EXPECT_EQ(bank_manager.lowest_occupied_address(bank_id, AllocatorID{1}), 0);
    EXPECT_EQ(bank_manager.lowest_occupied_address(bank_id, AllocatorID{2}), addr1 + alloc_size_2K);

    // Clear all allocations
    bank_manager.deallocate_all();
    bank_manager.clear();

    // Verify all allocations are cleared in all allocators
    EXPECT_EQ(bank_manager.lowest_occupied_address(bank_id, AllocatorID{0}), std::nullopt);
    EXPECT_EQ(bank_manager.lowest_occupied_address(bank_id, AllocatorID{1}), std::nullopt);
    EXPECT_EQ(bank_manager.lowest_occupied_address(bank_id, AllocatorID{2}), std::nullopt);

    // Verify we can allocate from the beginning again in all allocators
    auto new_addr0 = bank_manager.allocate_buffer(
        alloc_size_1K, alloc_size_1K, true, CoreRangeSet(std::vector<CoreRange>{}), std::nullopt, AllocatorID{0});
    auto new_addr1 = bank_manager.allocate_buffer(
        alloc_size_2K, alloc_size_2K, true, CoreRangeSet(std::vector<CoreRange>{}), std::nullopt, AllocatorID{1});
    auto new_addr2 = bank_manager.allocate_buffer(
        alloc_size_1K, alloc_size_1K, true, CoreRangeSet(std::vector<CoreRange>{}), std::nullopt, AllocatorID{2});

    EXPECT_EQ(new_addr0, 0);
    EXPECT_EQ(new_addr1, 0);
    EXPECT_EQ(new_addr2, new_addr1 + alloc_size_2K);
}

TEST(OverlappedAllocators, IndependentAllocAndDeallocBottomUp) {
    // 2 independent allocators, no overlaps
    BankManager::AllocatorDependencies deps{{{AllocatorID{0}, {}}, {AllocatorID{1}, {}}}};
    BankManager bank_manager = get_bank_manager_with_allocator_dependencies(1024 * 1024, 1024, deps);
    const uint32_t bank_id = 0;

    // Hard-coded allocation sizes
    uint32_t alloc_size_1K = 1024;
    uint32_t alloc_size_2K = 2048;

    // Allocate 1K in allocator 0
    // - Alloc0: | 1K | free |
    // - Alloc1: |   free    |
    auto alloc0_addr0 = bank_manager.allocate_buffer(
        alloc_size_1K,
        alloc_size_1K,
        /*bottom_up=*/true,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{0});
    EXPECT_EQ(alloc0_addr0, 0);

    // Allocate 2K in allocator 1
    // - Alloc0: | 1K |   free   |
    // - Alloc1: |   2K   | free |
    auto alloc1_addr0 = bank_manager.allocate_buffer(
        alloc_size_2K,
        alloc_size_2K,
        /*bottom_up=*/true,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{1});
    EXPECT_EQ(alloc1_addr0, 0);

    // Allocate another 1K in allocator 0
    // - Alloc0: | 1K | 1K | free |
    // - Alloc1: |   2K    | free |
    auto alloc0_addr1 = bank_manager.allocate_buffer(
        alloc_size_1K,
        alloc_size_1K,
        /*bottom_up=*/true,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{0});
    EXPECT_EQ(alloc0_addr1, alloc0_addr0 + alloc_size_1K);

    // Allocate another 2K in allocator 1
    // - Alloc0: | 1K | 1K |     free      |
    // - Alloc1: |   2K    |   2K   | free |
    auto alloc1_addr1 = bank_manager.allocate_buffer(
        alloc_size_2K,
        alloc_size_2K,
        /*bottom_up=*/true,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{1});
    EXPECT_EQ(alloc1_addr1, alloc1_addr0 + alloc_size_2K);

    // Deallocate first 2K in allocator 1
    // - Alloc0: | 1K | 1K |     free      |
    // - Alloc1: | 2K free |   2K   | free |
    bank_manager.deallocate_buffer(alloc1_addr0, AllocatorID{1});
    EXPECT_EQ(bank_manager.lowest_occupied_address(bank_id, AllocatorID{1}), alloc1_addr1);

    // Allocate 1K in allocator 1
    // - Alloc0: |   1K    |   1K    |       free        |
    // - Alloc1: |   1K    | 1K free |     2K     | free |
    auto alloc1_addr3 = bank_manager.allocate_buffer(
        alloc_size_1K,
        alloc_size_1K,
        /*bottom_up=*/true,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{1});
    EXPECT_EQ(alloc1_addr3, 0);

    // Deallocate first 1K in allocator 0
    // - Alloc0: | 1K free |   1K    |       free        |
    // - Alloc1: |   1K    | 1K free |     2K     | free |
    bank_manager.deallocate_buffer(alloc0_addr0, AllocatorID{0});
    EXPECT_EQ(bank_manager.lowest_occupied_address(bank_id, AllocatorID{0}), alloc0_addr1);
}

TEST(OverlappedAllocators, IndependentAllocAndDeallocTopDown) {
    // 2 independent allocators, no overlaps
    const uint64_t total_size = 1024 * 1024;
    BankManager::AllocatorDependencies deps{{{AllocatorID{0}, {}}, {AllocatorID{1}, {}}}};
    BankManager bank_manager = get_bank_manager_with_allocator_dependencies(total_size, 1024, deps);
    const uint32_t bank_id = 0;

    // Hard-coded allocation sizes
    uint32_t alloc_size_1K = 1024;
    uint32_t alloc_size_2K = 2048;

    // Allocate 1K in allocator 0
    // - Alloc0: | free | 1K |
    // - Alloc1: |   free    |
    auto alloc0_addr0 = bank_manager.allocate_buffer(
        alloc_size_1K,
        alloc_size_1K,
        /*bottom_up=*/false,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{0});
    EXPECT_EQ(alloc0_addr0, total_size - alloc_size_1K);

    // Allocate 2K in allocator 1
    // - Alloc0: |   free   | 1K |
    // - Alloc1: | free |   2K   |
    auto alloc1_addr0 = bank_manager.allocate_buffer(
        alloc_size_2K,
        alloc_size_2K,
        /*bottom_up=*/false,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{1});
    EXPECT_EQ(alloc1_addr0, total_size - alloc_size_2K);

    // Allocate another 1K in allocator 0
    // - Alloc0: | free | 1K | 1K |
    // - Alloc1: | free |   2K    |
    auto alloc0_addr1 = bank_manager.allocate_buffer(
        alloc_size_1K,
        alloc_size_1K,
        /*bottom_up=*/false,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{0});
    EXPECT_EQ(alloc0_addr1, alloc0_addr0 - alloc_size_1K);

    // Allocate another 2K in allocator 1
    // - Alloc0: |     free      | 1K | 1K |
    // - Alloc1: | free |   2K   |   2K    |
    auto alloc1_addr1 = bank_manager.allocate_buffer(
        alloc_size_2K,
        alloc_size_2K,
        /*bottom_up=*/false,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{1});
    EXPECT_EQ(alloc1_addr1, alloc1_addr0 - alloc_size_2K);

    // Deallocate first 2K in allocator 1
    // - Alloc0: |      free      | 1K | 1K |
    // - Alloc1: | free |   2K    | 2K free |
    bank_manager.deallocate_buffer(alloc1_addr0, AllocatorID{1});
    EXPECT_EQ(bank_manager.lowest_occupied_address(bank_id, AllocatorID{1}), alloc1_addr1);

    // Allocate 1K in allocator 1
    // - Alloc0: |       free        |   1K    |   1K    |
    // - Alloc1: | free |     2K     | 1K free |   1K    |
    auto alloc1_addr3 = bank_manager.allocate_buffer(
        alloc_size_1K,
        alloc_size_1K,
        /*bottom_up=*/false,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{1});
    EXPECT_EQ(alloc1_addr3, total_size - alloc_size_1K);

    // Deallocate first 1K in allocator 0
    // - Alloc0: |       free        |   1K    | 1K free |
    // - Alloc1: | free |     2K     | 1K free |   1K    |
    bank_manager.deallocate_buffer(alloc0_addr0, AllocatorID{0});
    EXPECT_EQ(bank_manager.lowest_occupied_address(bank_id, AllocatorID{0}), alloc0_addr1);
}

TEST(OverlappedAllocators, OverlappedAllocAndDeallocBottomUp) {
    // Two independent allocators (0 and 1); allocator 2 overlaps both 0 and 1
    BankManager::AllocatorDependencies deps{{{AllocatorID{0}, {AllocatorID{2}}}, {AllocatorID{1}, {AllocatorID{2}}}}};
    BankManager bank_manager = get_bank_manager_with_allocator_dependencies(1024 * 1024, 1024, deps);
    const uint32_t bank_id = 0;

    // Hard-coded allocation sizes
    const uint32_t alloc_size_1K = 1024;
    const uint32_t alloc_size_2K = 2048;

    // Allocate 1K in allocator 0
    // - Alloc0: | 1K | free |
    // - Alloc1: |   free    |
    // - Alloc2: |   free    |
    const auto alloc0_addr0 = bank_manager.allocate_buffer(
        alloc_size_1K,
        alloc_size_1K,
        /*bottom_up=*/true,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{0});
    EXPECT_EQ(alloc0_addr0, 0);

    // Allocate 2K in allocator 1
    // - Alloc0: | 1K |   free   |
    // - Alloc1: |   2K   | free |
    // - Alloc2: |     free      |
    const auto alloc1_addr0 = bank_manager.allocate_buffer(
        alloc_size_2K,
        alloc_size_2K,
        /*bottom_up=*/true,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{1});
    EXPECT_EQ(alloc1_addr0, 0);

    // Allocate 1K in allocator 1
    // - Alloc0: | 1K |     free      |
    // - Alloc1: |   2K   | 1K | free |
    // - Alloc2: |        free        |
    const auto alloc1_addr1 = bank_manager.allocate_buffer(
        alloc_size_1K,
        alloc_size_1K,
        /*bottom_up=*/true,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{1});
    EXPECT_EQ(alloc1_addr1, alloc1_addr0 + alloc_size_2K);

    // Allocate 1K in overlapped allocator 2 (should be placed after allocator 1's 1K)
    // - Alloc0: | 1K |        free        |
    // - Alloc1: |   2K   | 1K |   free    |
    // - Alloc2: |    free     | 1K | free |
    const auto alloc2_addr0 = bank_manager.allocate_buffer(
        alloc_size_1K,
        alloc_size_1K,
        /*bottom_up=*/true,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{2});
    EXPECT_EQ(alloc2_addr0, alloc1_addr1 + alloc_size_1K);

    // Allocate 1K in allocator 0 (should be placed after allocator 0's 1K and before allocator 2's 1K)
    // - Alloc0: | 1K | 1K |      free      |
    // - Alloc1: |   2K    | 1K |   free    |
    // - Alloc2: |     free     | 1K | free |
    const auto alloc0_addr1 = bank_manager.allocate_buffer(
        alloc_size_1K,
        alloc_size_1K,
        /*bottom_up=*/true,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{0});
    EXPECT_EQ(alloc0_addr1, alloc0_addr0 + alloc_size_1K);

    // Allocate 1K in allocator 1 (should be placed after allocator 2's 1K)
    // - Alloc0: | 1K | 1K |         free          |
    // - Alloc1: |   2K    | 1K | free | 1K | free |
    // - Alloc2: |     free     |  1K  |   free    |
    const auto alloc1_addr2 = bank_manager.allocate_buffer(
        alloc_size_1K,
        alloc_size_1K,
        /*bottom_up=*/true,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{1});
    EXPECT_EQ(alloc1_addr2, alloc2_addr0 + alloc_size_1K);

    /****************************************************************************
     * Deallocate from allocator 2 and allocate in 0 and 1, reusing freed space *
     ****************************************************************************/
    // Deallocate 1K from allocator 2
    // - Alloc0: | 1K | 1K |           free           |
    // - Alloc1: |   2K    | 1K |  free   | 1K | free |
    // - Alloc2: |     free     | 1K free |   free    |
    bank_manager.deallocate_buffer(alloc2_addr0, AllocatorID{2});
    // Lowest occupied address does not account for allocations in other allocators
    EXPECT_EQ(bank_manager.lowest_occupied_address(bank_id, AllocatorID{2}), std::nullopt);
    // Allocate 2K in allocator 0 (reusing freed space from allocator 2)
    // - Alloc0: | 1K | 1K |      2K      |   free    |
    // - Alloc1: |   2K    | 1K |  free   | 1K | free |
    // - Alloc2: |     free     | 1K free |   free    |
    const auto alloc0_addr2 = bank_manager.allocate_buffer(
        alloc_size_2K,
        alloc_size_2K,
        /*bottom_up=*/true,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{0});
    EXPECT_EQ(alloc0_addr2, alloc0_addr1 + alloc_size_1K);
    // Allocate 1K in allocator 1 (reusing freed space from allocator 2)
    // - Alloc0: | 1K | 1K |      2K      |   free    |
    // - Alloc1: |   2K    | 1K |   1K    | 1K | free |
    // - Alloc2: |     free     | 1K free |   free    |
    const auto alloc1_addr3 = bank_manager.allocate_buffer(
        alloc_size_1K,
        alloc_size_1K,
        /*bottom_up=*/true,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{1});
    EXPECT_EQ(alloc1_addr3, alloc2_addr0);

    /****************************************************************************
     * Deallocate from allocator 0 and 1 and allocate in 2, reusing freed space *
     ****************************************************************************/
    // Deallocate 1K from allocator 0 and 2K from allocator 1
    // - Alloc0: | 1K | 1K free |      2K      |   free    |
    // - Alloc1: |   2K free    | 1K |   1K    | 1K | free |
    // - Alloc2: |       free        | 1K free |   free    |
    bank_manager.deallocate_buffer(alloc0_addr1, AllocatorID{0});
    bank_manager.deallocate_buffer(alloc1_addr0, AllocatorID{1});
    // Allocate 1K in allocator 2 (reusing freed space)
    // - Alloc0: |  1K  | 1K free |       2K       |   free    |
    // - Alloc1: |    2K free     |  1K  |   1K    | 1K | free |
    // - Alloc2: | free |   1K    | free | 1K free |   free    |
    const auto alloc2_addr1 = bank_manager.allocate_buffer(
        alloc_size_1K,
        alloc_size_1K,
        /*bottom_up=*/true,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{2});
    EXPECT_EQ(alloc2_addr1, alloc0_addr1);
}

TEST(OverlappedAllocators, OverlappedAllocAndDeallocTopDown) {
    // Two independent allocators (0 and 1); allocator 2 overlaps both 0 and 1
    const uint64_t total_size = 1024 * 1024;
    BankManager::AllocatorDependencies deps{{{AllocatorID{0}, {AllocatorID{2}}}, {AllocatorID{1}, {AllocatorID{2}}}}};
    BankManager bank_manager = get_bank_manager_with_allocator_dependencies(total_size, 1024, deps);
    const uint32_t bank_id = 0;

    // Hard-coded allocation sizes
    const uint32_t alloc_size_1K = 1024;
    const uint32_t alloc_size_2K = 2048;

    // Allocate 1K in allocator 0
    // - Alloc0: | free | 1K |
    // - Alloc1: |   free    |
    // - Alloc2: |   free    |
    const auto alloc0_addr0 = bank_manager.allocate_buffer(
        alloc_size_1K,
        alloc_size_1K,
        /*bottom_up=*/false,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{0});
    EXPECT_EQ(alloc0_addr0, total_size - alloc_size_1K);

    // Allocate 2K in allocator 1
    // - Alloc0: |   free   | 1K |
    // - Alloc1: | free |   2K   |
    // - Alloc2: |     free      |
    const auto alloc1_addr0 = bank_manager.allocate_buffer(
        alloc_size_2K,
        alloc_size_2K,
        /*bottom_up=*/false,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{1});
    EXPECT_EQ(alloc1_addr0, total_size - alloc_size_2K);

    // Allocate 1K in allocator 1
    // - Alloc0: |     free      | 1K |
    // - Alloc1: | free | 1K |   2K   |
    // - Alloc2: |        free        |
    const auto alloc1_addr1 = bank_manager.allocate_buffer(
        alloc_size_1K,
        alloc_size_1K,
        /*bottom_up=*/false,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{1});
    EXPECT_EQ(alloc1_addr1, alloc1_addr0 - alloc_size_1K);

    // Allocate 1K in overlapped allocator 2 (should be placed before allocator 1's 1K)
    // - Alloc0: |        free        | 1K |
    // - Alloc1: |   free    | 1K |   2K   |
    // - Alloc2: | free | 1K |    free     |
    const auto alloc2_addr0 = bank_manager.allocate_buffer(
        alloc_size_1K,
        alloc_size_1K,
        /*bottom_up=*/false,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{2});
    EXPECT_EQ(alloc2_addr0, alloc1_addr1 - alloc_size_1K);

    // Allocate 1K in allocator 0 (should be placed before allocator 0's 1K and after allocator 2's 1K)
    // - Alloc0: |      free      | 1K | 1K |
    // - Alloc1: |   free    | 1K |   2K    |
    // - Alloc2: | free | 1K |     free     |
    const auto alloc0_addr1 = bank_manager.allocate_buffer(
        alloc_size_1K,
        alloc_size_1K,
        /*bottom_up=*/false,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{0});
    EXPECT_EQ(alloc0_addr1, alloc0_addr0 - alloc_size_1K);

    // Allocate 1K in allocator 1 (should be placed before allocator 2's 1K)
    // - Alloc0: |         free          | 1K | 1K |
    // - Alloc1: | free | 1K | free | 1K |   2K    |
    // - Alloc2: |   free    |  1K  |     free     |
    const auto alloc1_addr2 = bank_manager.allocate_buffer(
        alloc_size_1K,
        alloc_size_1K,
        /*bottom_up=*/false,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{1});
    EXPECT_EQ(alloc1_addr2, alloc2_addr0 - alloc_size_1K);

    /****************************************************************************
     * Deallocate from allocator 2 and allocate in 0 and 1, reusing freed space *
     ****************************************************************************/
    // Deallocate 1K from allocator 2
    // - Alloc0: |           free           | 1K | 1K |
    // - Alloc1: | free | 1K |  free   | 1K |   2K    |
    // - Alloc2: |   free    | 1K free |     free     |
    bank_manager.deallocate_buffer(alloc2_addr0, AllocatorID{2});
    // Lowest occupied address does not account for allocations in other allocators
    EXPECT_EQ(bank_manager.lowest_occupied_address(bank_id, AllocatorID{2}), std::nullopt);
    // Allocate 2K in allocator 0 (reusing freed space from allocator 2)
    // - Alloc0: |   free    |      2K      | 1K | 1K |
    // - Alloc1: | free | 1K |  free   | 1K |   2K    |
    // - Alloc2: |   free    | 1K free |     free     |
    const auto alloc0_addr2 = bank_manager.allocate_buffer(
        alloc_size_2K,
        alloc_size_2K,
        /*bottom_up=*/false,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{0});
    EXPECT_EQ(alloc0_addr2, alloc0_addr1 - alloc_size_2K);
    // Allocate 1K in allocator 1 (reusing freed space from allocator 2)
    // - Alloc0: |   free    |      2K      | 1K | 1K |
    // - Alloc1: | free | 1K |   1K    | 1K |   2K    |
    // - Alloc2: |   free    | 1K free |     free     |
    const auto alloc1_addr3 = bank_manager.allocate_buffer(
        alloc_size_1K,
        alloc_size_1K,
        /*bottom_up=*/false,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{1});
    EXPECT_EQ(alloc1_addr3, alloc2_addr0);

    /****************************************************************************
     * Deallocate from allocator 0 and 1 and allocate in 2, reusing freed space *
     ****************************************************************************/
    // Deallocate 1K from allocator 0 and 2K from allocator 1
    // - Alloc0: |   free    |      2K      | 1K free | 1K |
    // - Alloc1: | free | 1K |   1K    | 1K |   2K free    |
    // - Alloc2: |   free    | 1K free |       free        |
    bank_manager.deallocate_buffer(alloc0_addr1, AllocatorID{0});
    bank_manager.deallocate_buffer(alloc1_addr0, AllocatorID{1});
    // Allocate 1K in allocator 2 (reusing freed space)
    // - Alloc0: |   free    |       2K       | 1K free |  1K  |
    // - Alloc1: | free | 1K |   1K    |  1K  |    2K free     |
    // - Alloc2: |   free    | 1K free | free |   1K    | free |
    const auto alloc2_addr1 = bank_manager.allocate_buffer(
        alloc_size_1K,
        alloc_size_1K,
        /*bottom_up=*/false,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{2});
    EXPECT_EQ(alloc2_addr1, alloc0_addr1);
}

TEST(OverlappedAllocators, OverlappedAllocationsWithUnalignedSizesBottomUp) {
    // Two independent allocators (0 and 1); allocator 2 overlaps both 0 and 1
    const uint32_t alignment = 1024;
    BankManager::AllocatorDependencies deps{{{AllocatorID{0}, {AllocatorID{2}}}, {AllocatorID{1}, {AllocatorID{2}}}}};
    BankManager bank_manager = get_bank_manager_with_allocator_dependencies(1024 * 1024, alignment, deps);

    // Hard-coded allocation sizes
    const uint32_t alloc_size_unaligned = 512;
    const uint32_t alloc_size_aligned = (alloc_size_unaligned + alignment - 1) / alignment * alignment;
    ASSERT_EQ(alloc_size_aligned, 1024);
    const uint32_t alloc_size_2K = 2048;

    // Allocate unaligned size in allocator 0
    // - Alloc0: | 1K | free |
    // - Alloc1: |   free    |
    // - Alloc2: |   free    |
    const auto alloc0_addr0 = bank_manager.allocate_buffer(
        alloc_size_unaligned,
        alloc_size_unaligned,
        /*bottom_up=*/true,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{0});
    EXPECT_EQ(alloc0_addr0, 0);

    // Allocate unaligned size in allocator 2
    // - Alloc0: |  1K  |   free    |
    // - Alloc1: |       free       |
    // - Alloc2: | free | 1K | free |
    const auto alloc2_addr0 = bank_manager.allocate_buffer(
        alloc_size_unaligned,
        alloc_size_unaligned,
        /*bottom_up=*/true,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{2});
    EXPECT_EQ(alloc2_addr0, alloc0_addr0 + alloc_size_aligned);

    // Allocate unaligned size in allocator 1
    // - Alloc0: |  1K  |   free    |
    // - Alloc1: |  1K  |   free    |
    // - Alloc2: | free | 1K | free |
    const auto alloc1_addr0 = bank_manager.allocate_buffer(
        alloc_size_unaligned,
        alloc_size_unaligned,
        /*bottom_up=*/true,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{1});
    EXPECT_EQ(alloc1_addr0, 0);

    // Allocate another unaligned size in allocator 1 (should be placed after allocator 2's 1K)
    // - Alloc0: |  1K  |       free       |
    // - Alloc1: |  1K  | free | 1K | free |
    // - Alloc2: | free |  1K  |   free    |
    const auto alloc1_addr1 = bank_manager.allocate_buffer(
        alloc_size_unaligned,
        alloc_size_unaligned,
        /*bottom_up=*/true,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{1});
    EXPECT_EQ(alloc1_addr1, alloc2_addr0 + alloc_size_aligned);

    // Allocate 2K in allocator 0 (should be placed after allocator 2's 1K)
    // - Alloc0: |  1K  | free |   2K   | free |
    // - Alloc1: |  1K  | free | 1K |   free   |
    // - Alloc2: | free |  1K  |     free      |
    const auto alloc0_addr1 = bank_manager.allocate_buffer(
        alloc_size_2K,
        alloc_size_2K,
        /*bottom_up=*/true,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{0});
    EXPECT_EQ(alloc0_addr1, alloc2_addr0 + alloc_size_aligned);
}

TEST(OverlappedAllocators, OverlappedAllocationsWithUnalignedSizesTopDown) {
    // Two independent allocators (0 and 1); allocator 2 overlaps both 0 and 1
    const uint64_t total_size = 1024 * 1024;
    const uint32_t alignment = 1024;
    BankManager::AllocatorDependencies deps{{{AllocatorID{0}, {AllocatorID{2}}}, {AllocatorID{1}, {AllocatorID{2}}}}};
    BankManager bank_manager = get_bank_manager_with_allocator_dependencies(total_size, alignment, deps);

    // Hard-coded allocation sizes
    const uint32_t alloc_size_unaligned = 512;
    const uint32_t alloc_size_aligned = (alloc_size_unaligned + alignment - 1) / alignment * alignment;
    ASSERT_EQ(alloc_size_aligned, 1024);
    const uint32_t alloc_size_2K = 2048;

    // Allocate unaligned size in allocator 0
    // - Alloc0: | free | 1K |
    // - Alloc1: |   free    |
    // - Alloc2: |   free    |
    const auto alloc0_addr0 = bank_manager.allocate_buffer(
        alloc_size_unaligned,
        alloc_size_unaligned,
        /*bottom_up=*/false,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{0});
    EXPECT_EQ(alloc0_addr0, total_size - alloc_size_aligned);

    // Allocate unaligned size in allocator 2
    // - Alloc0: |   free    |  1K  |
    // - Alloc1: |       free       |
    // - Alloc2: | free | 1K | free |
    const auto alloc2_addr0 = bank_manager.allocate_buffer(
        alloc_size_unaligned,
        alloc_size_unaligned,
        /*bottom_up=*/false,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{2});
    EXPECT_EQ(alloc2_addr0, alloc0_addr0 - alloc_size_aligned);

    // Allocate unaligned size in allocator 1
    // - Alloc0: |   free    |  1K  |
    // - Alloc1: |   free    |  1K  |
    // - Alloc2: | free | 1K | free |
    const auto alloc1_addr0 = bank_manager.allocate_buffer(
        alloc_size_unaligned,
        alloc_size_unaligned,
        /*bottom_up=*/false,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{1});
    EXPECT_EQ(alloc1_addr0, total_size - alloc_size_aligned);

    // Allocate another unaligned size in allocator 1 (should be placed before allocator 2's 1K)
    // - Alloc0: |       free       |  1K  |
    // - Alloc1: | free | 1K | free |  1K  |
    // - Alloc2: |   free    |  1K  | free |
    const auto alloc1_addr1 = bank_manager.allocate_buffer(
        alloc_size_unaligned,
        alloc_size_unaligned,
        /*bottom_up=*/false,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{1});
    EXPECT_EQ(alloc1_addr1, alloc2_addr0 - alloc_size_aligned);

    // Allocate 2K in allocator 0 (should be placed before allocator 2's 1K)
    // - Alloc0: | free |   2K   | free |  1K  |
    // - Alloc1: |   free   | 1K | free |  1K  |
    // - Alloc2: |     free      |  1K  | free |
    const auto alloc0_addr1 = bank_manager.allocate_buffer(
        alloc_size_2K,
        alloc_size_2K,
        /*bottom_up=*/false,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{0});
    EXPECT_EQ(alloc0_addr1, alloc2_addr0 - alloc_size_2K);
}

TEST(OverlappedAllocators, OverlappedAllocationsFromBothSides) {
    // Two independent allocators (0 and 1); allocator 2 overlaps both 0 and 1
    const uint64_t total_size = 1024 * 1024;
    const uint32_t alignment = 1024;
    BankManager::AllocatorDependencies deps{{{AllocatorID{0}, {AllocatorID{2}}}, {AllocatorID{1}, {AllocatorID{2}}}}};
    BankManager bank_manager = get_bank_manager_with_allocator_dependencies(total_size, alignment, deps);

    // Hard-coded allocation sizes
    const uint32_t alloc_size_1K = 1024;
    const uint32_t alloc_size_2K = 2048;
    const uint32_t alloc_size_half_of_total_size = total_size / 2;
    const uint32_t alloc_size_quarter_of_total_size = total_size / 4;

    // Set up initial allocations
    const auto alloc0_addr0 = bank_manager.allocate_buffer(
        alloc_size_half_of_total_size,
        alloc_size_half_of_total_size,
        /*bottom_up=*/true,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{0});
    const auto alloc0_addr1 = bank_manager.allocate_buffer(
        alloc_size_1K,
        alloc_size_1K,
        /*bottom_up=*/true,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{0});
    const auto alloc1_addr0 = bank_manager.allocate_buffer(
        alloc_size_2K,
        alloc_size_2K,
        /*bottom_up=*/false,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{1});
    const auto alloc1_addr1 = bank_manager.allocate_buffer(
        alloc_size_half_of_total_size,
        alloc_size_half_of_total_size,
        /*bottom_up=*/false,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{1});

    // Initial allocator states after all allocations
    // Deallocate 512K from allocator 1
    // - Alloc0: |                     512k                      |  1K  |                  free                  |
    // - Alloc1: |                 free                 |                     512k                      |   2K   |
    // - Alloc2: |                                             free                                              |

    // Try to allocate in allocator 2 (overlapped) which should fail due to no available space
    EXPECT_THAT(
        [&]() {
            bank_manager.allocate_buffer(
                alloc_size_1K,
                alloc_size_1K,
                /*bottom_up=*/true,
                CoreRangeSet(std::vector<CoreRange>{}),
                std::nullopt,
                AllocatorID{2});
        },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Out of Memory: Not enough space after considering dependencies to allocate 1024 B "
                                 "DRAM across 1 banks (1024 B per bank)")));

    // Deallocate 512K from allocator 1
    // - Alloc0: |                     512k                      |  1K  |                  free                  |
    // - Alloc1: |                 free                 |                   512k free                   |   2K   |
    // - Alloc2: |                                             free                                              |
    bank_manager.deallocate_buffer(alloc1_addr1, AllocatorID{1});

    // Allocate 1K in allocator 2 bottom-up (should be placed after allocator 0's 1K)
    // - Alloc0: |                     512k                      |  1K  |                  free                  |
    // - Alloc1: |                 free                 |                   512k free                   |   2K   |
    // - Alloc2: |                         free                         | 1K |               free                |
    const auto alloc2_addr0 = bank_manager.allocate_buffer(
        alloc_size_1K,
        alloc_size_1K,
        /*bottom_up=*/true,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{2});
    EXPECT_EQ(alloc2_addr0, alloc0_addr1 + alloc_size_1K);

    // Allocate another 1K in allocator 2 top-down (should be placed before allocator 1's 2K)
    // - Alloc0: |                     512k                      |  1K  |                  free                  |
    // - Alloc1: |                 free                 |                   512k free                   |   2K   |
    // - Alloc2: |                     free                      | free | 1K |        free         | 1K |  free  |
    const auto alloc2_addr1 = bank_manager.allocate_buffer(
        alloc_size_1K,
        alloc_size_1K,
        /*bottom_up=*/false,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{2});
    EXPECT_EQ(alloc2_addr1, alloc1_addr0 - alloc_size_1K);

    // Deallocate 512K from allocator 0
    // - Alloc0: |                   512k free                   |  1K  |                  free                  |
    // - Alloc1: |                 free                 |                   512k free                   |   2K   |
    // - Alloc2: |                     free                      | free | 1K |        free         | 1K |  free  |
    bank_manager.deallocate_buffer(alloc0_addr0, AllocatorID{0});

    // Allocate 512K in allocator 2 top-down:
    // - Alloc0: |                   512k free                   |  1K  |                  free                  |
    // - Alloc1: |                 free                 |                   512k free                   |   2K   |
    // - Alloc2: |                     512K                      | free | 1K |        free         | 1K |  free  |
    const auto alloc2_addr2 = bank_manager.allocate_buffer(
        alloc_size_half_of_total_size,
        alloc_size_half_of_total_size,
        /*bottom_up=*/false,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{2});
    EXPECT_EQ(alloc2_addr2, alloc0_addr1 - alloc_size_half_of_total_size);
    EXPECT_EQ(alloc2_addr2, 0);  // Should be placed at the start of the bank

    // Allocate 256K in allocator 2 bottom-up
    // - Alloc0: |                   512k free                   |  1K  |                  free                  |
    // - Alloc1: |                 free                 |                   512k free                   |   2K   |
    // - Alloc2: |                     512K                      | free | 1K |   256K   |   free   | 1K |  free  |
    const auto alloc2_addr3 = bank_manager.allocate_buffer(
        alloc_size_quarter_of_total_size,
        alloc_size_quarter_of_total_size,
        /*bottom_up=*/true,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{2});
    EXPECT_EQ(alloc2_addr3, alloc2_addr0 + alloc_size_1K);
}

TEST(OverlappedAllocators, NonzeroAddressLimit) {
    // Tests the clamping logic in BankManager::allocate_buffer when address_limit is not 0

    // Two independent allocators (0 and 1); allocator 2 overlaps both 0 and 1
    const uint64_t total_size = 1024 * 1024;
    const uint32_t alignment = 1024;
    const DeviceAddr address_limit = 256 * 1024;  // 256KB - allocations must start from this address or later
    BankManager::AllocatorDependencies deps{{{AllocatorID{0}, {AllocatorID{2}}}, {AllocatorID{1}, {AllocatorID{2}}}}};

    // Use the constructor that takes interleaved_address_limit for L1 buffer type
    std::unordered_map<uint32_t, int64_t> bank_id_to_offset = {{0, 0}};
    BankManager bank_manager(
        BufferType::L1,
        bank_id_to_offset,
        total_size,
        address_limit,  // interleaved_address_limit
        alignment,
        0,      // alloc_offset
        false,  // disable_interleaved
        deps);

    const uint32_t alloc_size_1K = 1024;
    const uint32_t alloc_size_a_bit_more_than_half_of_total_size = (total_size / 2) + alloc_size_1K;
    const uint32_t alloc_size_same_as_address_limit = address_limit;

    // Allocate 1K in allocator 0 - should be placed at address_limit (256KB)
    // - Alloc0: |    256K addr_limit    | 1K |                               free                               |
    // - Alloc1: |    256K addr_limit    |                                 free                                  |
    // - Alloc2: |    256K addr_limit    |                                 free                                  |
    const auto alloc0_addr0 = bank_manager.allocate_buffer(
        alloc_size_1K,
        alloc_size_1K,
        /*bottom_up=*/true,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{0});
    EXPECT_EQ(alloc0_addr0, address_limit);  // Should start at 256KB, not 0

    // Allocate 2K in allocator 1 - should also start at address_limit since it's independent
    // - Alloc0: |    256K addr_limit    | 1K |                               free                               |
    // - Alloc1: |    256K addr_limit    |                      512K + 1K                       |      free      |
    // - Alloc2: |    256K addr_limit    |                                 free                                  |
    const auto alloc1_addr0 = bank_manager.allocate_buffer(
        alloc_size_a_bit_more_than_half_of_total_size,
        alloc_size_a_bit_more_than_half_of_total_size,
        /*bottom_up=*/true,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{1});
    EXPECT_EQ(alloc1_addr0, address_limit);  // Should start at 256KB, not 0

    EXPECT_THAT(
        [&]() {
            // address_limit is 256KB, which is a quarter of total size
            // In allocator 1, we allocated a bit over half of total size, so the only valid allocation of 256KB is
            // within the address_limit
            // - Alloc0: |    256K addr_limit    | 1K |                               free |
            // - Alloc1: |    256K addr_limit    |                      512K + 1K                       |      free |
            // - Alloc2: |    256K addr_limit    |                                 free |
            bank_manager.allocate_buffer(
                alloc_size_same_as_address_limit,
                alloc_size_same_as_address_limit,
                /*bottom_up=*/false,
                CoreRangeSet(std::vector<CoreRange>{}),
                std::nullopt,
                AllocatorID{2});
        },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Out of Memory: Not enough space after considering dependencies to allocate 262144 B "
                                 "L1 across 1 banks (262144 B per bank)")));

    // Allocate 1K in overlapped allocator 2 (should be placed after allocator 1's allocation)
    // - Alloc0: |    256K addr_limit    | 1K |                               free                               |
    // - Alloc1: |    256K addr_limit    |                      512K + 1K                       |      free      |
    // - Alloc2: |    256K addr_limit    |                      512K + 1K                       | 1K |   free    |
    const auto alloc2_addr0 = bank_manager.allocate_buffer(
        alloc_size_1K,
        alloc_size_1K,
        /*bottom_up=*/true,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{2});
    EXPECT_EQ(alloc2_addr0, address_limit + alloc_size_a_bit_more_than_half_of_total_size);

    // Deallocate all allocations and reallocate
    // This tests BankManager's destructor when:
    // - There are dependencies between allocators
    // - Address limit is not 0
    // - Reallocating after BankManager::deallocate_buffer() or BankManager::deallocate_all()
    // (@TT-BrianLiu) In the above scenario, I saw some error with memory freeing when using a custom destructor.
    // The custom destructor calls BankManager::deallocate_all(). There are two ways to fix this situation:
    //   1. Use default destructor
    //   2. Call bank_manager.clear() after deallocate_all()
    // I am switching back to default destructor because it doesn't look like there's a need for the custom destructor
    bank_manager.deallocate_all();

    // Reallocate 1K in allocator 0
    // - Alloc0: |    256K addr_limit    | 1K |                               free                               |
    // - Alloc1: |    256K addr_limit    |                                 free                                  |
    // - Alloc2: |    256K addr_limit    |                                 free                                  |
    const auto alloc0_addr0_realloc = bank_manager.allocate_buffer(
        alloc_size_1K,
        alloc_size_1K,
        /*bottom_up=*/true,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{0});
    EXPECT_EQ(alloc0_addr0_realloc, address_limit);  // Should still start at 256KB
}

TEST(OverlappedAllocators, NonzeroAllocOffset) {
    // Tests that BankManager::allocate_buffer properly accounts for allocator offsets
    // If there are dependencies, allocate_buffer will rely on available_addresses, allocated_addresses, and
    // allocate_at_address APIs. Returned addresses must be absolute addresses otherwise possible bugs:
    // - If top-down allocation and offset is larger than the allocation size, subsequent allocations will fail
    //   * First allocation will incorrectly allocate at local address - offset
    //   * Available addresses will incorrectly include top offset-sized gap as available address
    //   * Subsequent allocation will attempt to call allocate_at_address at same allocated address and fail
    // - If bottom-up allocation and offset is larger than the allocation size, subsequent allocations will fail but for
    //   a different reason. If you assume address_limit is same as alloc offset, this is what will happen:
    //   * First allocation will try to allocate at address_limit because of clamping logic so first allocation will be
    //   correct
    //   * Available addresses will include [alloc_size, ) as available address, which will still get clamped to
    //   address_limit
    //   * Subsequent allocation will attempt to call allocate_at_address at same allocated address and fail
    // TLDR: Support for address_limit, alloc_offset, and bottom_up/top_down allocation seems very fragile

    // Two independent allocators (0 and 1); allocator 2 overlaps both 0 and 1
    BankManager::AllocatorDependencies deps{{{AllocatorID{0}, {AllocatorID{2}}}, {AllocatorID{1}, {AllocatorID{2}}}}};

    // Create a BankManager mocking how L1BankingAllocator is setup
    std::unordered_map<uint32_t, int64_t> bank_id_to_offset = {{0, 0}};
    const uint64_t allocatable_l1_size = 1398720;
    const DeviceAddr interleaved_address_limit = 100432;  // address_limit
    const uint32_t l1_alignment = 16;
    const DeviceAddr l1_unreserved_base = 100416;  // offset
    const bool disable_interleaved = false;
    BankManager bank_manager(
        BufferType::L1,
        bank_id_to_offset,
        allocatable_l1_size,
        interleaved_address_limit,
        l1_alignment,
        l1_unreserved_base,
        disable_interleaved,
        deps);

    // Two consecutive top-down allocations
    const uint32_t alloc_size_4K = 4096;
    const auto alloc0_addr0 = bank_manager.allocate_buffer(
        alloc_size_4K,
        alloc_size_4K,
        /*bottom_up=*/false,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{0});
    EXPECT_EQ(alloc0_addr0, allocatable_l1_size - alloc_size_4K + l1_unreserved_base);

    const auto alloc0_addr1 = bank_manager.allocate_buffer(
        alloc_size_4K,
        alloc_size_4K,
        /*bottom_up=*/false,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{0});
    EXPECT_EQ(alloc0_addr1, alloc0_addr0 - alloc_size_4K);

    // Two consecutive bottom-up allocations
    const auto alloc0_addr2 = bank_manager.allocate_buffer(
        alloc_size_4K,
        alloc_size_4K,
        /*bottom_up=*/true,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{0});
    // With dependent allocators, the clamping logic will find an address above address limit
    // With independent allocators, regular allocate API will actually fail for this use case
    // - But in practice, we always allocate top-down so it's not an issue
    EXPECT_EQ(alloc0_addr2, interleaved_address_limit);

    const auto alloc0_addr3 = bank_manager.allocate_buffer(
        alloc_size_4K,
        alloc_size_4K,
        /*bottom_up=*/true,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        AllocatorID{0});
    EXPECT_EQ(alloc0_addr3, alloc0_addr2 + alloc_size_4K);
}
