// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "impl/context/metal_context.hpp"
#include <iomanip>
#include <sstream>
#include <set>

namespace tt::tt_fabric {
namespace intermesh_api_tests {

TEST(IntermeshAPIs, BasicQueries) {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();

    // Test that cluster supports intermesh links
    bool supports_intermesh = cluster.contains_intermesh_links();

    // Get all intermesh links
    auto all_links = cluster.get_all_intermesh_eth_links();

    // Test per-chip queries
    for (const auto& chip_id : cluster.user_exposed_chip_ids()) {
        bool has_links = cluster.has_intermesh_links(chip_id);
        auto chip_links = cluster.get_intermesh_eth_links(chip_id);

        // Verify consistency
        if (has_links) {
            EXPECT_GT(chip_links.size(), 0) << "Chip " << chip_id << " reports having links but returned empty vector";
        } else {
            EXPECT_EQ(chip_links.size(), 0) << "Chip " << chip_id << " reports no links but returned non-empty vector";
        }

        // Test is_intermesh_eth_link for each link
        for (const auto& [eth_core, channel] : chip_links) {
            bool is_intermesh = cluster.is_intermesh_eth_link(chip_id, eth_core);
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

    // Get all links via get_all_intermesh_eth_links
    auto all_links = cluster.get_all_intermesh_eth_links();

    // Verify it matches individual chip queries
    for (const auto& [chip_id, expected_links] : all_links) {
        auto actual_links = cluster.get_intermesh_eth_links(chip_id);

        EXPECT_EQ(expected_links.size(), actual_links.size())
            << "Mismatch in link count for chip " << chip_id;

        // Compare link contents (assuming order might differ)
        std::set<std::pair<CoreCoord, uint32_t>> expected_set(expected_links.begin(), expected_links.end());
        std::set<std::pair<CoreCoord, uint32_t>> actual_set(actual_links.begin(), actual_links.end());

        EXPECT_EQ(expected_set, actual_set)
            << "Link content mismatch for chip " << chip_id;
    }
}

TEST(IntermeshAPIs, IntermeshLinksAreDistinctFromEthernetLinks) {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();

    if (!cluster.contains_intermesh_links()) {
        GTEST_SKIP() << "Cluster does not support intermesh links";
    }

    log_info(tt::LogTest, "=== Verifying Intermesh Links are Distinct from Ethernet Links ===");

    // This test documents that intermesh links are a distinct category from regular ethernet links
    auto all_intermesh_links = cluster.get_all_intermesh_eth_links();

    for (const auto& [chip_id, intermesh_links] : all_intermesh_links) {
        auto active_eth_cores = cluster.get_active_ethernet_cores(chip_id);
        auto inactive_eth_cores = cluster.get_inactive_ethernet_cores(chip_id);

        // Verify no overlap between intermesh links and active ethernet cores
        for (const auto& [eth_core, channel] : intermesh_links) {
            EXPECT_TRUE(active_eth_cores.find(eth_core) == active_eth_cores.end())
                << "Intermesh link at " << eth_core.str() << " on chip " << chip_id
                << " should not be in active ethernet cores";

            // Intermesh links appear in inactive ethernet cores
            EXPECT_TRUE(inactive_eth_cores.find(eth_core) != inactive_eth_cores.end())
                << "Intermesh link at " << eth_core.str() << " on chip " << chip_id
                << " should be in inactive ethernet cores";
        }

        log_info(tt::LogTest, "Chip {}: {} intermesh links are distinct from {} active ethernet cores",
                 chip_id, intermesh_links.size(), active_eth_cores.size());
    }
}

}  // namespace intermesh_api_tests
}  // namespace tt::tt_fabric
