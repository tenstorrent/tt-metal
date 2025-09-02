// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <tt-metalium/control_plane.hpp>
#include "impl/context/metal_context.hpp"
#include <iomanip>
#include <sstream>
#include <set>

namespace tt::tt_fabric {
namespace intermesh_api_tests {

TEST(IntermeshAPIs, BasicQueries) {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    // Test that cluster supports intermesh links
    bool supports_intermesh = control_plane.system_has_intermesh_links();

    // Get all intermesh links
    auto all_links = control_plane.get_all_intermesh_eth_links();

    // Test per-chip queries
    for (const auto& chip_id : cluster.user_exposed_chip_ids()) {
        bool has_links = control_plane.has_intermesh_links(chip_id);
        auto chip_links = control_plane.get_intermesh_eth_links(chip_id);

        // Verify consistency
        if (has_links) {
            EXPECT_GT(chip_links.size(), 0) << "Chip " << chip_id << " reports having links but returned empty vector";
        } else {
            EXPECT_EQ(chip_links.size(), 0) << "Chip " << chip_id << " reports no links but returned non-empty vector";
        }

        // Test is_intermesh_eth_link for each link
        for (const auto& [eth_core, channel] : chip_links) {
            bool is_intermesh = control_plane.is_intermesh_eth_link(chip_id, eth_core);
            EXPECT_TRUE(is_intermesh) << "Eth core " << eth_core.str() << " on chip " << chip_id
                                     << " was returned by get_intermesh_eth_links but is_intermesh_eth_link returned false";
        }
    }

    // If cluster supports intermesh, at least one chip should have links
    if (supports_intermesh) {
        EXPECT_GT(all_links.size(), 0) << "Cluster reports supporting intermesh but no chips have links";
    }
}

TEST(IntermeshAPIs, ConsistencyChecks) {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    // Get all links via get_all_intermesh_eth_links
    auto all_links = control_plane.get_all_intermesh_eth_links();

    // Verify it matches individual chip queries
    for (const auto& [chip_id, expected_links] : all_links) {
        auto actual_links = control_plane.get_intermesh_eth_links(chip_id);

        EXPECT_EQ(expected_links.size(), actual_links.size())
            << "Mismatch in link count for chip " << chip_id;

        // Compare link contents (assuming order might differ)
        std::set<std::pair<CoreCoord, uint32_t>> expected_set(expected_links.begin(), expected_links.end());
        std::set<std::pair<CoreCoord, uint32_t>> actual_set(actual_links.begin(), actual_links.end());

        EXPECT_EQ(expected_set, actual_set)
            << "Link content mismatch for chip " << chip_id;
    }
}

}  // namespace intermesh_api_tests
}  // namespace tt::tt_fabric
