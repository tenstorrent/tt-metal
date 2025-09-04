// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "gmock/gmock.h"
#include <cstddef>
#include <cstdint>
#include <tt-metalium/allocator.hpp>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/device.hpp>
#include "device_fixture.hpp"
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/metal_soc_descriptor.h>
#include "impl/context/metal_context.hpp"
#include "tt_metal/impl/allocator/bank_manager.hpp"
#include <unordered_map>
#include <algorithm>
#include <numeric>

namespace overlapped_bank_manager_tests {
struct StateDependenciesParam {
    std::unordered_map<
        tt::tt_metal::BankManager::StateDependencies::StateId,
        ttsl::SmallVector<tt::tt_metal::BankManager::StateDependencies::StateId>>
        input;
    // Expected dependencies per state (missing keys imply empty list)
    tt::tt_metal::BankManager::StateDependencies::AdjacencyList expected_dependencies;
};

tt::tt_metal::BankManager get_bank_manager_with_state_dependencies(
    const tt::tt_metal::BankManager::StateDependencies& state_dependencies) {
    std::vector<int64_t> bank_desc = {0};
    const uint64_t unreserved_base = 0;
    const uint32_t alignment = 1024;
    const uint64_t size = 1024 * 1024;
    return tt::tt_metal::BankManager(
        tt::tt_metal::BufferType::DRAM, bank_desc, size, alignment, unreserved_base, false, state_dependencies);
}

}  // namespace overlapped_bank_manager_tests

using namespace overlapped_bank_manager_tests;
using namespace tt::tt_metal;
using StateId = BankManager::StateDependencies::StateId;

// --- StateDependencies parameterized tests ---
TEST(StateDependencies, DefaultStateDependencies) {
    BankManager::StateDependencies state_dependencies;
    EXPECT_EQ(state_dependencies.dependencies, BankManager::StateDependencies::AdjacencyList{{}});
}

TEST(StateDependencies, DuplicateDependencies) {
    const std::unordered_map<StateId, ttsl::SmallVector<StateId>> dependencies_map = {
        {StateId{0}, ttsl::SmallVector<StateId>{StateId{1}, StateId{1}}}};

    EXPECT_THAT(
        [&]() { BankManager::StateDependencies state_dependencies{dependencies_map}; },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Duplicate dependency for state 0: 1 appears more than once!")));
}

TEST(StateDependencies, EquivalentDependenciesMaps) {
    const std::unordered_map<StateId, ttsl::SmallVector<StateId>> dependencies_map1 = {
        {StateId{0}, ttsl::SmallVector<StateId>{StateId{1}}}, {StateId{1}, ttsl::SmallVector<StateId>{StateId{0}}}};
    const std::unordered_map<StateId, ttsl::SmallVector<StateId>> dependencies_map2 = {
        {StateId{0}, ttsl::SmallVector<StateId>{StateId{1}}}};
    const std::unordered_map<StateId, ttsl::SmallVector<StateId>> dependencies_map3 = {
        {StateId{1}, ttsl::SmallVector<StateId>{StateId{0}}}};

    EXPECT_EQ(BankManager::StateDependencies(dependencies_map1), BankManager::StateDependencies(dependencies_map2));
    EXPECT_EQ(BankManager::StateDependencies(dependencies_map1), BankManager::StateDependencies(dependencies_map3));
}

class StateDependenciesParamTest : public ::testing::TestWithParam<StateDependenciesParam> {};

TEST_P(StateDependenciesParamTest, ValidateDependencies) {
    const auto& params = GetParam();

    BankManager::StateDependencies state_dependencies{params.input};

    // Validate dependencies
    // Stored dependencies are not sorted, so we need to sort them for comparison
    auto sort_nested_vector = [](BankManager::StateDependencies::AdjacencyList a) {
        for (auto& v : a) {
            std::sort(v.begin(), v.end());
        }
        return a;
    };
    EXPECT_EQ(sort_nested_vector(state_dependencies.dependencies), params.expected_dependencies);
}

INSTANTIATE_TEST_SUITE_P(
    StateDependencies,
    StateDependenciesParamTest,
    ::testing::Values(
        // Empty input
        StateDependenciesParam{
            /*input=*/{},
            /*expected_dependencies=*/{{}}},
        // Single state default behavior (explicit input)
        StateDependenciesParam{
            /*input=*/{{StateId{0}, {}}},
            /*expected_dependencies=*/{{}}},
        // Two-way dependency
        StateDependenciesParam{
            /*input=*/{{StateId{0}, {StateId{1}}}, {StateId{1}, {StateId{0}}}},
            /*expected_dependencies=*/{{StateId{1}}, {StateId{0}}}},
        // Sparse keys with one dependency specified
        StateDependenciesParam{
            /*input=*/{{StateId{3}, {StateId{0}, StateId{1}}}},
            /*expected_dependencies=*/{{StateId{3}}, {StateId{3}}, {}, {StateId{0}, StateId{1}}}},
        // Sparse keys with one dependency specified (value of dependents imply more states)
        StateDependenciesParam{
            /*input=*/{{StateId{1}, {StateId{3}}}},
            /*expected_dependencies=*/{{}, {StateId{3}}, {}, {StateId{1}}}},
        // Fan-in: 1,2,3 depend on 0
        StateDependenciesParam{
            /*input=*/{{StateId{1}, {StateId{0}}}, {StateId{2}, {StateId{0}}}, {StateId{3}, {StateId{0}}}},
            /*expected_dependencies=*/{{StateId{1}, StateId{2}, StateId{3}}, {StateId{0}}, {StateId{0}}, {StateId{0}}}},
        // Fan-out: 0 depends on 1,2,3
        StateDependenciesParam{
            /*input=*/{{StateId{0}, {StateId{1}, StateId{2}, StateId{3}}}},
            /*expected_dependencies=*/{{StateId{1}, StateId{2}, StateId{3}}, {StateId{0}}, {StateId{0}}, {StateId{0}}}},
        // Chain: 0->1->2->3
        StateDependenciesParam{
            /*input=*/{{StateId{0}, {StateId{1}}}, {StateId{1}, {StateId{2}}}, {StateId{2}, {StateId{3}}}},
            /*expected_dependencies=*/{{StateId{1}}, {StateId{0}, StateId{2}}, {StateId{1}, StateId{3}}, {StateId{2}}}},
        // Cycle: 0->1->2->0
        StateDependenciesParam{
            /*input=*/{{StateId{0}, {StateId{1}}}, {StateId{1}, {StateId{2}}}, {StateId{2}, {StateId{0}}}},
            /*expected_dependencies=*/{{StateId{1}, StateId{2}}, {StateId{0}, StateId{2}}, {StateId{0}, StateId{1}}}}));

// --- BankManager tests ---

TEST(OverlappedAllocator, InvalidState) {
    // Create bank manager with 2 states (0 and 1)
    BankManager::StateDependencies deps{{{StateId{0}, {}}, {StateId{1}, {}}}};
    BankManager bank_manager = get_bank_manager_with_state_dependencies(deps);

    // Test accessing invalid state that's greater than num_states
    StateId invalid_state{2};  // Should fail since we only have states 0 and 1

    // Test non-const overload of get_allocator_for_state
    EXPECT_THAT(
        [&]() {
            bank_manager.allocate_buffer(
                1024, 1024, true, CoreRangeSet(std::vector<CoreRange>{}), std::nullopt, invalid_state);
        },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("Invalid allocator state 2 (num_states=2)")));

    // Test const overload of get_allocator_for_state
    EXPECT_THAT(
        [&]() { bank_manager.lowest_occupied_address(0, invalid_state); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("Invalid allocator state 2 (num_states=2)")));
}

TEST(OverlappedAllocator, InvalidAPIsForOverlappedAllocators) {
    // Create bank manager with 2 states (0 and 1)
    BankManager::StateDependencies deps{{{StateId{0}, {}}, {StateId{1}, {}}}};
    BankManager bank_manager = get_bank_manager_with_state_dependencies(deps);

    // Test accessing an API that only works for single state
    EXPECT_THAT(
        [&]() { bank_manager.reset_size(StateId{0}); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("Expected single state allocator!")));
    EXPECT_THAT(
        [&]() { bank_manager.reset_size(StateId{1}); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("Expected single state allocator!")));
    EXPECT_THAT(
        [&]() { bank_manager.shrink_size(1024, true, StateId{0}); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("Expected single state allocator!")));
    EXPECT_THAT(
        [&]() { bank_manager.shrink_size(1024, true, StateId{1}); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("Expected single state allocator!")));
}

TEST(OverlappedAllocator, DeallocateAllAndClear) {
    // Two independent allocators (0 and 1); allocator 2 overlaps both 0 and 1
    BankManager::StateDependencies deps{{{StateId{0}, {StateId{2}}}, {StateId{1}, {StateId{2}}}}};
    BankManager bank_manager = get_bank_manager_with_state_dependencies(deps);
    const uint32_t bank_id = 0;

    uint32_t alloc_size_1K = 1024;
    uint32_t alloc_size_2K = 2048;

    // Verify no allocations exist initially
    EXPECT_EQ(bank_manager.lowest_occupied_address(bank_id, StateId{0}), std::nullopt);
    EXPECT_EQ(bank_manager.lowest_occupied_address(bank_id, StateId{1}), std::nullopt);
    EXPECT_EQ(bank_manager.lowest_occupied_address(bank_id, StateId{2}), std::nullopt);

    // Allocate in each state
    auto addr0 = bank_manager.allocate_buffer(
        alloc_size_1K, alloc_size_1K, true, CoreRangeSet(std::vector<CoreRange>{}), std::nullopt, StateId{0});
    auto addr1 = bank_manager.allocate_buffer(
        alloc_size_2K, alloc_size_2K, true, CoreRangeSet(std::vector<CoreRange>{}), std::nullopt, StateId{1});
    auto addr2 = bank_manager.allocate_buffer(
        alloc_size_1K, alloc_size_1K, true, CoreRangeSet(std::vector<CoreRange>{}), std::nullopt, StateId{2});

    // Verify allocations exist in all states
    EXPECT_EQ(bank_manager.lowest_occupied_address(bank_id, StateId{0}), 0);
    EXPECT_EQ(bank_manager.lowest_occupied_address(bank_id, StateId{1}), 0);
    EXPECT_EQ(bank_manager.lowest_occupied_address(bank_id, StateId{2}), addr1 + alloc_size_2K);

    // Clear all allocations
    bank_manager.deallocate_all();
    bank_manager.clear();

    // Verify all allocations are cleared in all states
    EXPECT_EQ(bank_manager.lowest_occupied_address(bank_id, StateId{0}), std::nullopt);
    EXPECT_EQ(bank_manager.lowest_occupied_address(bank_id, StateId{1}), std::nullopt);
    EXPECT_EQ(bank_manager.lowest_occupied_address(bank_id, StateId{2}), std::nullopt);

    // Verify we can allocate from the beginning again in all states
    auto new_addr0 = bank_manager.allocate_buffer(
        alloc_size_1K, alloc_size_1K, true, CoreRangeSet(std::vector<CoreRange>{}), std::nullopt, StateId{0});
    auto new_addr1 = bank_manager.allocate_buffer(
        alloc_size_2K, alloc_size_2K, true, CoreRangeSet(std::vector<CoreRange>{}), std::nullopt, StateId{1});
    auto new_addr2 = bank_manager.allocate_buffer(
        alloc_size_1K, alloc_size_1K, true, CoreRangeSet(std::vector<CoreRange>{}), std::nullopt, StateId{2});

    EXPECT_EQ(new_addr0, 0);
    EXPECT_EQ(new_addr1, 0);
    EXPECT_EQ(new_addr2, new_addr1 + alloc_size_2K);
}

TEST(OverlappedAllocator, IndependentStates) {
    // 2 independent allocators, no overlaps
    BankManager::StateDependencies deps{{{StateId{0}, {}}, {StateId{1}, {}}}};
    BankManager bank_manager = get_bank_manager_with_state_dependencies(deps);
    const uint32_t bank_id = 0;

    // Hard-coded allocation sizes
    uint32_t alloc_size_1K = 1024;
    uint32_t alloc_size_2K = 2048;

    // Allocate 1K in allocator 0:
    // - Alloc0: | 1K | free |
    // - Alloc1: |   free    |
    auto alloc0_addr0 = bank_manager.allocate_buffer(
        alloc_size_1K,
        alloc_size_1K,
        /*bottom_up=*/true,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        StateId{0});
    EXPECT_EQ(alloc0_addr0, 0);

    // Allocate 2K in allocator 1:
    // - Alloc0: | 1K |   free   |
    // - Alloc1: |   2K   | free |
    auto alloc1_addr0 = bank_manager.allocate_buffer(
        alloc_size_2K,
        alloc_size_2K,
        /*bottom_up=*/true,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        StateId{1});
    EXPECT_EQ(alloc1_addr0, 0);

    // Allocate another 1K in allocator 0:
    // - Alloc0: | 1K | 1K | free |
    // - Alloc1: |   2K    | free |
    auto alloc0_addr1 = bank_manager.allocate_buffer(
        alloc_size_1K,
        alloc_size_1K,
        /*bottom_up=*/true,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        StateId{0});
    EXPECT_EQ(alloc0_addr1, alloc0_addr0 + alloc_size_1K);

    // Allocate another 2K in allocator 1:
    // - Alloc0: | 1K | 1K |     free      |
    // - Alloc1: |   2K    |   2K   | free |
    auto alloc1_addr1 = bank_manager.allocate_buffer(
        alloc_size_2K,
        alloc_size_2K,
        /*bottom_up=*/true,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        StateId{1});
    EXPECT_EQ(alloc1_addr1, alloc1_addr0 + alloc_size_2K);

    bank_manager.deallocate_buffer(alloc1_addr0, StateId{1});
    EXPECT_EQ(bank_manager.lowest_occupied_address(bank_id, StateId{1}), alloc1_addr1);

    auto alloc1_addr3 = bank_manager.allocate_buffer(
        alloc_size_1K,
        alloc_size_1K,
        /*bottom_up=*/true,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        StateId{1});
    EXPECT_EQ(alloc1_addr3, 0);

    bank_manager.deallocate_buffer(alloc0_addr0, StateId{0});
    EXPECT_EQ(bank_manager.lowest_occupied_address(bank_id, StateId{0}), alloc0_addr1);
}

TEST(OverlappedAllocator, OverlappedStates) {
    // Two independent allocators (0 and 1); allocator 2 overlaps both 0 and 1
    BankManager::StateDependencies deps{{{StateId{0}, {StateId{2}}}, {StateId{1}, {StateId{2}}}}};
    BankManager bank_manager = get_bank_manager_with_state_dependencies(deps);
    const uint32_t bank_id = 0;

    // Hard-coded allocation sizes
    uint32_t alloc_size_1K = 1024;
    uint32_t alloc_size_2K = 2048;

    // Allocate 1K in allocator 0
    auto alloc0_addr0 = bank_manager.allocate_buffer(
        alloc_size_1K,
        alloc_size_1K,
        /*bottom_up=*/true,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        StateId{0});
    EXPECT_EQ(alloc0_addr0, 0);

    // Allocate 2K in allocator 1
    auto alloc1_addr0 = bank_manager.allocate_buffer(
        alloc_size_2K,
        alloc_size_2K,
        /*bottom_up=*/true,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        StateId{1});
    EXPECT_EQ(alloc1_addr0, 0);

    // Allocate 1K in allocator 1
    auto alloc1_addr1 = bank_manager.allocate_buffer(
        alloc_size_1K,
        alloc_size_1K,
        /*bottom_up=*/true,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        StateId{1});
    EXPECT_EQ(alloc1_addr1, alloc1_addr0 + alloc_size_2K);

    // Allocate 1K in overlapped allocator 2 (should be placed after allocator 1's 2K):
    auto alloc2_addr0 = bank_manager.allocate_buffer(
        alloc_size_1K,
        alloc_size_1K,
        /*bottom_up=*/true,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        StateId{2});
    EXPECT_EQ(alloc2_addr0, alloc1_addr1 + alloc_size_1K);

    // Allocate 1K in allocator 0 (should be placed after allocator 1's 1K and before allocator 2's 1K):
    auto alloc0_addr1 = bank_manager.allocate_buffer(
        alloc_size_1K,
        alloc_size_1K,
        /*bottom_up=*/true,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        StateId{0});
    EXPECT_EQ(alloc0_addr1, alloc0_addr0 + alloc_size_1K);

    // Allocate 1K in allocator 1 (should be placed after allocator 2's 1K):
    auto alloc1_addr2 = bank_manager.allocate_buffer(
        alloc_size_1K,
        alloc_size_1K,
        /*bottom_up=*/true,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        StateId{1});
    EXPECT_EQ(alloc1_addr2, alloc2_addr0 + alloc_size_1K);

    bank_manager.deallocate_buffer(alloc2_addr0, StateId{2});
    // Lowest occupied address does not account for allocations in other allocators
    EXPECT_EQ(bank_manager.lowest_occupied_address(bank_id, StateId{2}), std::nullopt);
    auto alloc0_addr2 = bank_manager.allocate_buffer(
        alloc_size_2K,
        alloc_size_2K,
        /*bottom_up=*/true,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        StateId{0});
    EXPECT_EQ(alloc0_addr2, alloc0_addr1 + alloc_size_1K);
    auto alloc1_addr3 = bank_manager.allocate_buffer(
        alloc_size_1K,
        alloc_size_1K,
        /*bottom_up=*/true,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        StateId{1});
    EXPECT_EQ(alloc1_addr3, alloc2_addr0);

    bank_manager.deallocate_buffer(alloc0_addr1, StateId{0});
    bank_manager.deallocate_buffer(alloc1_addr0, StateId{1});
    auto alloc2_addr1 = bank_manager.allocate_buffer(
        alloc_size_1K,
        alloc_size_1K,
        /*bottom_up=*/true,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        StateId{2});
    EXPECT_EQ(alloc2_addr1, alloc0_addr1);
}
