// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <map>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "fabric_fixture.hpp"
#include <tt-metalium/logger.hpp>
#include <tt-metalium/system_memory_manager.hpp>
#include "impl/context/metal_context.hpp"

namespace tt::tt_fabric {
namespace fabric_router_tests {

TEST_F(ControlPlaneFixture, TestUBBConnectivity) {
    const auto& eth_connections = tt::tt_metal::MetalContext::instance().get_cluster().get_ethernet_connections();
    EXPECT_EQ(eth_connections.size(), 32);
    for (const auto& [chip, connections] : eth_connections) {
        std::map<chip_id_t, int> num_connections_to_chip;
        for (const auto& [channel, remote_chip_and_channel] : connections) {
            log_debug(
                tt::LogTest,
                "Chip: {} Channel: {} Remote Chip: {} Remote Channel: {}",
                chip,
                channel,
                std::get<0>(remote_chip_and_channel),
                std::get<1>(remote_chip_and_channel));
            num_connections_to_chip[std::get<0>(remote_chip_and_channel)]++;
        }
        EXPECT_EQ(num_connections_to_chip.size(), 4);
        for (const auto& [other_chip, count] : num_connections_to_chip) {
            EXPECT_EQ(count, 4);
        }
    }
}


}  // namespace fabric_router_tests
}  // namespace tt::tt_fabric
