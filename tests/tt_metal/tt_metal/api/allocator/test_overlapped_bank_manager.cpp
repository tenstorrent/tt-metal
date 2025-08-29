// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
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
    EXPECT_EQ(state_dependencies.adjacency, BankManager::StateDependencies::AdjacencyList{{}});
}

class StateDependenciesParamTest : public ::testing::TestWithParam<StateDependenciesParam> {};

TEST_P(StateDependenciesParamTest, BuildsAdjacencyAndInfersAllowedStates) {
    const auto& params = GetParam();

    BankManager::StateDependencies state_dependencies{params.input};

    // Validate dependencies
    EXPECT_EQ(state_dependencies.adjacency, params.expected_dependencies);
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
            /*expected_dependencies=*/{{}, {}, {}, {StateId{0}, StateId{1}}}},
        // Sparse keys with one dependency specified (value of dependents imply more states)
        StateDependenciesParam{
            /*input=*/{{StateId{1}, {StateId{3}}}},
            /*expected_dependencies=*/{{}, {StateId{3}}, {}, {}}}));

TEST_F(DeviceSingleCardBufferFixture, TestOverlappedBankManager) {
    // This test sets up a BankManager for DRAM and L1, mimicking Allocator initialization.

    // Get device and soc descriptor
    IDevice* device = this->device_;
    const metal_SocDescriptor& soc_desc = MetalContext::instance().get_cluster().get_soc_desc(device->id());

    // DRAM setup
    {
        // Gather all DRAM bank descriptors (address offsets per channel, as in Allocator)
        const size_t num_channels = soc_desc.get_num_dram_views();
        std::vector<int64_t> dram_bank_descriptors(num_channels);
        for (size_t channel = 0; channel < num_channels; ++channel) {
            dram_bank_descriptors.at(channel) =
                static_cast<int64_t>(soc_desc.get_address_offset(static_cast<int>(channel)));
        }
        // Use per-channel DRAM size adjusted by allocator bases and alignment
        const uint64_t dram_unreserved_base = device->allocator()->get_base_allocator_addr(HalMemType::DRAM);
        const uint32_t dram_alignment = device->allocator()->get_alignment(BufferType::DRAM);
        const uint64_t dram_trace_region_size = device->allocator()->get_config().trace_region_size;
        const uint64_t dram_size =
            static_cast<uint64_t>(device->dram_size_per_channel()) - dram_unreserved_base - dram_trace_region_size;

        // Set up BankManager for DRAM with default single-state dependencies
        BankManager::StateDependencies deps;  // defaults to single state, no overlaps
        BankManager dram_bank_manager(
            BufferType::DRAM,
            dram_bank_descriptors,
            dram_size,
            dram_alignment,
            /*alloc_offset=*/dram_unreserved_base,
            /*disable_interleaved=*/false,
            deps);

        // Allocate a buffer and check
        uint32_t alloc_size = 64 * 1024;
        auto addr = dram_bank_manager.allocate_buffer(
            alloc_size,
            alloc_size,
            /*bottom_up=*/true,
            CoreRangeSet(std::vector<CoreRange>{}),  // Not used for DRAM
            std::nullopt,
            BankManager::StateDependencies::StateId{0});
        EXPECT_GT(addr, 0u);
        dram_bank_manager.deallocate_buffer(addr, BankManager::StateDependencies::StateId{0});
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
