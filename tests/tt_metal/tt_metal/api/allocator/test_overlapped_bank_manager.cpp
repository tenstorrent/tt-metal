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

namespace unit_tests::test_overlapped_bank_manager {}  // namespace unit_tests::test_overlapped_bank_manager

namespace tt::tt_metal {

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
            0);
        EXPECT_GT(addr, 0u);
        dram_bank_manager.deallocate_buffer(addr, 0);
    }
}

// --- StateDependencies parameterized tests ---
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

class StateDependenciesParamTest : public ::testing::TestWithParam<StateDepsCase> {};

TEST_P(StateDependenciesParamTest, BuildsAdjacencyAndInfersAllowedStates) {
    const auto& tc = GetParam();
    using SD = BankManager::StateDependencies;

    // Convert input to StateId + SmallVector
    std::unordered_map<SD::StateId, tt::stl::SmallVector<SD::StateId>, SD::Hasher> deps{};
    for (const auto& kv : tc.input) {
        tt::stl::SmallVector<SD::StateId> neigh;
        for (auto d : kv.second) {
            neigh.push_back(SD::StateId{d});
        }
        deps.emplace(SD::StateId{kv.first}, std::move(neigh));
    }

    SD sd{deps};

    // Validate allowed states
    std::vector<uint32_t> got_states;
    got_states.reserve(sd.adjacency.size());
    for (const auto& kv : sd.adjacency) {
        got_states.push_back(kv.first.value);
    }
    std::sort(got_states.begin(), got_states.end());
    EXPECT_EQ(got_states, sorted(tc.expected_states));
    EXPECT_EQ(sd.num_states(), got_states.size());

    // Validate edges
    for (auto s : got_states) {
        auto it = sd.adjacency.find(SD::StateId{s});
        ASSERT_NE(it, sd.adjacency.end());
        std::vector<uint32_t> got;
        got.reserve(it->second.size());
        for (auto d : it->second) {
            got.push_back(d.value);
        }
        std::sort(got.begin(), got.end());

        auto exp_it = tc.expected_edges.find(s);
        std::vector<uint32_t> exp = (exp_it == tc.expected_edges.end()) ? std::vector<uint32_t>{} : exp_it->second;
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
        // Sparse keys; values-only nodes included
        StateDepsCase{
            /*input=*/{{0, {2}}, {7, {3}}},
            /*expected_states=*/{0, 2, 3, 7},
            /*expected_edges=*/{{0, {2}}, {7, {3}}, {2, {}}, {3, {}}}},
        // Fan-out
        StateDepsCase{
            /*input=*/{{5, {1, 2, 3}}},
            /*expected_states=*/{1, 2, 3, 5},
            /*expected_edges=*/{{5, {1, 2, 3}}, {1, {}}, {2, {}}, {3, {}}}}));

}  // namespace tt::tt_metal
