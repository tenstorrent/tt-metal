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

}  // namespace overlapped_bank_manager_tests

using namespace overlapped_bank_manager_tests;
using namespace tt::tt_metal;

// --- StateDependencies parameterized tests ---
using StateId = BankManager::StateDependencies::StateId;

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
TEST(OverlappedAllocator, IndependentStates) {
    // This test sets up a generic BankManager for DRAM/L1, mimicking Allocator initialization.

    // Single logical bank with offset 0 for simplicity
    std::vector<int64_t> bank_desc = {0};

    // Use per-channel DRAM size adjusted by allocator bases and alignment
    const uint64_t unreserved_base = 0;
    const uint32_t alignment = 1024;
    const uint64_t size = 1024 * 1024 * 1024;

    // Set up BankManager for DRAM with default single-state dependencies
    BankManager::StateDependencies deps{{{StateId{0}, {}}, {StateId{1}, {}}}};  // 2 states, no overlaps
    BankManager bank_manager(
        BufferType::DRAM,
        bank_desc,
        size,
        alignment,
        /*alloc_offset=*/unreserved_base,
        /*disable_interleaved=*/false,
        deps);

    // Allocate a buffer and check
    uint32_t alloc_size = 1024;
    uint32_t alloc_size2 = 2048;
    auto alloc1_addr1 = bank_manager.allocate_buffer(
        alloc_size,
        alloc_size,
        /*bottom_up=*/true,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        BankManager::StateDependencies::StateId{0});
    EXPECT_EQ(alloc1_addr1, 0);

    auto alloc2_addr1 = bank_manager.allocate_buffer(
        alloc_size2,
        alloc_size2,
        /*bottom_up=*/true,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        BankManager::StateDependencies::StateId{1});
    EXPECT_EQ(alloc2_addr1, 0);

    auto alloc1_addr2 = bank_manager.allocate_buffer(
        alloc_size,
        alloc_size,
        /*bottom_up=*/true,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        BankManager::StateDependencies::StateId{0});
    EXPECT_EQ(alloc1_addr2, alloc1_addr1 + alloc_size);

    auto alloc2_addr2 = bank_manager.allocate_buffer(
        alloc_size2,
        alloc_size2,
        /*bottom_up=*/true,
        CoreRangeSet(std::vector<CoreRange>{}),
        std::nullopt,
        BankManager::StateDependencies::StateId{1});
    EXPECT_EQ(alloc2_addr2, alloc2_addr1 + alloc_size2);
}

TEST(OverlappedAllocator, OverlappedStates) {
    // This test sets up a BankManager for DRAM and L1, mimicking Allocator initialization.

    // Single logical bank with offset 0 for simplicity
    std::vector<int64_t> bank_desc = {0};

    // DRAM setup
    {
        // Use per-channel DRAM size adjusted by allocator bases and alignment
        const uint64_t dram_unreserved_base = 0;
        const uint32_t dram_alignment = 1024;
        const uint64_t dram_trace_region_size = 0;
        const uint64_t dram_size = 1024 * 1024 * 1024;

        // Set up BankManager for DRAM with default single-state dependencies
        BankManager::StateDependencies deps{
            {{StateId{0}, {StateId{2}}},
             {StateId{1}, {StateId{2}}}}};  // 2 independent states; overlapped depends on both
        BankManager dram_bank_manager(
            BufferType::DRAM,
            bank_desc,
            dram_size,
            dram_alignment,
            /*alloc_offset=*/dram_unreserved_base,
            /*disable_interleaved=*/false,
            deps);

        // Allocate a buffer and check
        uint32_t alloc_size = 1024;
        uint32_t alloc_size2 = 2048;
        auto alloc1_addr1 = dram_bank_manager.allocate_buffer(
            alloc_size,
            alloc_size,
            /*bottom_up=*/true,
            CoreRangeSet(std::vector<CoreRange>{}),  // Not used for DRAM
            std::nullopt,
            BankManager::StateDependencies::StateId{0});
        EXPECT_EQ(alloc1_addr1, 0);

        auto alloc2_addr1 = dram_bank_manager.allocate_buffer(
            alloc_size2,
            alloc_size2,
            /*bottom_up=*/true,
            CoreRangeSet(std::vector<CoreRange>{}),  // Not used for DRAM
            std::nullopt,
            BankManager::StateDependencies::StateId{1});
        EXPECT_EQ(alloc2_addr1, 0);

        auto alloc3_addr1 = dram_bank_manager.allocate_buffer(
            alloc_size,
            alloc_size,
            /*bottom_up=*/true,
            CoreRangeSet(std::vector<CoreRange>{}),  // Not used for DRAM
            std::nullopt,
            BankManager::StateDependencies::StateId{2});
        EXPECT_EQ(alloc3_addr1, alloc2_addr1 + alloc_size2);
    }
}
