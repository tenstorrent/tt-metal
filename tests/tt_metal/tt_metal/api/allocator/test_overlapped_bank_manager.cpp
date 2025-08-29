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
struct StateDepsCase {
    // Input as map of u32 -> vector<u32>
    std::unordered_map<uint32_t, std::vector<uint32_t>> input;
    // Expected allowed states
    std::vector<uint32_t> expected_states;
    // Expected dependencies per state (missing keys imply empty list)
    std::unordered_map<uint32_t, std::vector<uint32_t>> expected_edges;
};

std::vector<uint32_t> sorted(const std::vector<uint32_t>& v) {
    auto c = v;
    std::sort(c.begin(), c.end());
    return c;
}

}  // namespace

namespace tt::tt_metal {

// --- StateDependencies parameterized tests ---

class StateDependenciesParamTest : public ::testing::TestWithParam<StateDepsCase> {};

TEST_P(StateDependenciesParamTest, BuildsAdjacencyAndInfersAllowedStates) {
    const auto& params = GetParam();
    using StateId = BankManager::StateDependencies::StateId;

    // Convert input to StateId + SmallVector
    std::unordered_map<StateId, tt::stl::SmallVector<StateId>> deps{};
    for (const auto& kv : params.input) {
        tt::stl::SmallVector<StateId> neigh;
        for (auto d : kv.second) {
            neigh.push_back(StateId{d});
        }
        deps.emplace(StateId{kv.first}, std::move(neigh));
    }

    BankManager::StateDependencies sd{deps};

    // Validate allowed states: indices 0..N-1
    std::vector<uint32_t> got_states(sd.adjacency.size());
    std::iota(got_states.begin(), got_states.end(), 0);
    std::sort(got_states.begin(), got_states.end());
    EXPECT_EQ(got_states, sorted(params.expected_states));
    EXPECT_EQ(sd.num_states(), got_states.size());

    // Validate edges
    for (auto s : got_states) {
        const auto& neigh = sd.adjacency[s];
        std::vector<uint32_t> got;
        got.reserve(neigh.size());
        for (auto d : neigh) {
            got.push_back(d.value);
        }
        std::sort(got.begin(), got.end());

        auto exp_it = params.expected_edges.find(s);
        std::vector<uint32_t> exp = (exp_it == params.expected_edges.end()) ? std::vector<uint32_t>{} : exp_it->second;
        EXPECT_EQ(got, sorted(exp)) << "mismatch for state " << s;
    }
}

INSTANTIATE_TEST_SUITE_P(
    StateDeps,
    StateDependenciesParamTest,
    ::testing::Values(
        // Single state default behavior (explicit input)
        StateDepsCase{
            /*input=*/{{0, {}}},
            /*expected_states=*/{0},
            /*expected_edges=*/{{0, {}}}},
        // Two-way dependency
        StateDepsCase{
            /*input=*/{{0, {1}}, {1, {0}}},
            /*expected_states=*/{0, 1},
            /*expected_edges=*/{{0, {1}}, {1, {0}}}},
        // Sparse keys; values-only nodes included; indices are 0..7
        StateDepsCase{
            /*input=*/{{0, {2}}, {7, {3}}},
            /*expected_states=*/{0, 1, 2, 3, 4, 5, 6, 7},
            /*expected_edges=*/{{0, {2}}, {7, {3}}, {2, {}}, {3, {}}}},
        // Fan-out
        StateDepsCase{
            /*input=*/{{5, {1, 2, 3}}},
            /*expected_states=*/{0, 1, 2, 3, 4, 5},
            /*expected_edges=*/{{5, {1, 2, 3}}, {1, {}}, {2, {}}, {3, {}}}}));

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
    std::unordered_map<SD::StateId, tt::stl::SmallVector<SD::StateId>> deps_map;
    deps_map.emplace(SD::StateId{0}, tt::stl::SmallVector<SD::StateId>{SD::StateId{2}});  // R depends on B
    deps_map.emplace(SD::StateId{1}, tt::stl::SmallVector<SD::StateId>{SD::StateId{2}});  // G depends on B
    deps_map.emplace(
        SD::StateId{2}, tt::stl::SmallVector<SD::StateId>{SD::StateId{0}, SD::StateId{1}});  // B depends on R,G
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
