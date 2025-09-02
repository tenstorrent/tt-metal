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

namespace {
struct StateDependenciesParam {
    std::unordered_map<
        tt::tt_metal::BankManager::StateDependencies::StateId,
        ttsl::SmallVector<tt::tt_metal::BankManager::StateDependencies::StateId>>
        input;
    // Expected dependencies per state (missing keys imply empty list)
    tt::tt_metal::BankManager::StateDependencies::AdjacencyList expected_dependencies;
};

}  // namespace

namespace tt::tt_metal {

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

TEST_P(StateDependenciesParamTest, BuildsAdjacencyAndInfersAllowedStates) {
    const auto& params = GetParam();

    BankManager::StateDependencies state_dependencies{params.input};

    // Validate dependencies
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

TEST(OverlappedAllocator, IndependentStates) {
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
        BankManager::StateDependencies deps{{{StateId{0}, {}}, {StateId{1}, {}}}};  // 2 states, no overlaps
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

        auto alloc1_addr2 = dram_bank_manager.allocate_buffer(
            alloc_size,
            alloc_size,
            /*bottom_up=*/true,
            CoreRangeSet(std::vector<CoreRange>{}),  // Not used for DRAM
            std::nullopt,
            BankManager::StateDependencies::StateId{0});
        EXPECT_EQ(alloc1_addr2, alloc1_addr1 + alloc_size);

        auto alloc2_addr2 = dram_bank_manager.allocate_buffer(
            alloc_size2,
            alloc_size2,
            /*bottom_up=*/true,
            CoreRangeSet(std::vector<CoreRange>{}),  // Not used for DRAM
            std::nullopt,
            BankManager::StateDependencies::StateId{1});
        EXPECT_EQ(alloc2_addr2, alloc2_addr1 + alloc_size2);
    }
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
TEST_F(DeviceSingleCardBufferFixture, Overlay_MergeUnmerge_RG_into_B) {
    // States: 0=R, 1=G, 2=B
    using SD = BankManager::StateDependencies;
    std::unordered_map<SD::StateId, ttsl::SmallVector<SD::StateId>> deps_map;
    deps_map.emplace(SD::StateId{0}, ttsl::SmallVector<SD::StateId>{SD::StateId{2}});  // R depends on B
    deps_map.emplace(SD::StateId{1}, ttsl::SmallVector<SD::StateId>{SD::StateId{2}});  // G depends on B
    deps_map.emplace(
        SD::StateId{2}, ttsl::SmallVector<SD::StateId>{SD::StateId{0}, SD::StateId{1}});  // B depends on R,G
    SD deps{deps_map};

    // Single logical bank with offset 0 for simplicity
    std::vector<int64_t> bank_desc = {0};

    const uint64_t total_size = 512 * 1024;  // 512 KiB
    const uint32_t align = 1024;             // 1 KiB alignment

    BankManager bm(
        BufferType::DRAM,
        bank_desc,
        /*size_bytes=*/total_size,
        /*alignment_bytes=*/align,
        /*alloc_offset=*/0,
        /*disable_interleaved=*/true,
        deps);

    auto empty_grid = CoreRangeSet(std::vector<CoreRange>{});

    const uint64_t sz = 64 * 1024;  // 64 KiB per state allocation

    // t0: allocate in R bottom-up (expects near 0), and in G top-down (expects near top)
    auto r0 = bm.allocate_buffer(sz, sz, /*bottom_up=*/true, empty_grid, std::nullopt, /*state=*/SD::StateId{0});
    auto g0 = bm.allocate_buffer(sz, sz, /*bottom_up=*/false, empty_grid, std::nullopt, /*state=*/SD::StateId{1});

    ASSERT_NE(r0, 0u);
    ASSERT_NE(g0, 0u);
    ASSERT_NE(r0, g0);

    // t1: allocate in B top-down; must avoid union(R,G). With G at top, B should pick below g0.
    auto b0 = bm.allocate_buffer(sz, sz, /*bottom_up=*/false, empty_grid, std::nullopt, /*state=*/SD::StateId{2});
    ASSERT_NE(b0, 0u);
    EXPECT_LT(b0, g0);  // should not overlap the topmost region taken by G

    // t2: free G; now the top region becomes available again for B
    bm.deallocate_buffer(g0, /*state=*/SD::StateId{1});
    auto b1 = bm.allocate_buffer(sz, sz, /*bottom_up=*/false, empty_grid, std::nullopt, /*state=*/SD::StateId{2});
    ASSERT_NE(b1, 0u);
    EXPECT_GE(b1, g0);  // top region should now be usable again

    // cleanup
    bm.deallocate_buffer(b1, /*state=*/SD::StateId{2});
    bm.deallocate_buffer(b0, /*state=*/SD::StateId{2});
    bm.deallocate_buffer(r0, /*state=*/SD::StateId{0});
}

}  // namespace tt::tt_metal
