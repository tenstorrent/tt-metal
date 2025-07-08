// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <tt-metalium/control_plane.hpp>
#include "impl/context/metal_context.hpp"
#include "tt_metal/fabric/serialization/intermesh_link_table.hpp"
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

TEST(IntermeshAPIs, LocalIntermeshLinkTable) {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    if (!control_plane.system_has_intermesh_links()) {
        GTEST_SKIP() << "Cluster does not support intermesh links";
    }

    log_info(tt::LogTest, "=== Verifying Local Intermesh Link Table Serialization ===");

    // Get the local intermesh link table
    const auto& intermesh_link_table = control_plane.get_local_intermesh_link_table();
    auto serialized_table = serialize_to_bytes(intermesh_link_table);
    IntermeshLinkTable deserialized_table = deserialize_from_bytes(serialized_table);
    EXPECT_EQ(intermesh_link_table.local_mesh_id, deserialized_table.local_mesh_id)
        << "Deserialized local mesh ID does not match original";

    for (const auto& [local_chan, remote_chan] : intermesh_link_table.intermesh_links) {
        EXPECT_TRUE(deserialized_table.intermesh_links.contains(local_chan))
            << "Deserialized table does not contain local channel Board " << local_chan.board_id << " Chan "
            << local_chan.chan_id;
        EXPECT_EQ(deserialized_table.intermesh_links.at(local_chan), remote_chan)
            << "Remote channel for Board " << local_chan.board_id << " Chan "
            << local_chan.chan_id << " does not match after deserialization";

        auto board_id = local_chan.board_id;
        bool chip_found = false;
        for (auto chip_id : cluster.user_exposed_chip_ids()) {
            if (control_plane.has_intermesh_links(chip_id) == false) {
                continue;
            }
            if (control_plane.get_asic_id(chip_id) == board_id) {
                EXPECT_TRUE(control_plane.is_intermesh_eth_link(chip_id, CoreCoord{0, local_chan.chan_id}))
                    << "Expected valid intermesh links in the local intermesh link table";
                chip_found = true;
                break;
            }
        }
        EXPECT_TRUE(chip_found) << "No chip found with ASIC ID " << board_id;
    }
}

}  // namespace intermesh_api_tests
}  // namespace tt::tt_fabric
