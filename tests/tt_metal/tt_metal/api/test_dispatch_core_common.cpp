// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <optional>

#include <tt-metalium/cluster.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_metal.hpp>

namespace tt::tt_metal {

TEST(DispatchCoreConfigTest, NamedFactoriesConstructorRules) {
    // Explicit ETH + COL is invalid regardless of flavor.
    EXPECT_ANY_THROW(
        DispatchCoreConfig::create_with_max_worker_availability(DispatchCoreType::ETH, DispatchCoreAxis::COL));

    // Max dispatch performance always defaults the type to WORKER.
    const auto col_axis_worker_config =
        DispatchCoreConfig::create_with_max_dispatch_performance(std::nullopt, DispatchCoreAxis::COL);
    EXPECT_EQ(col_axis_worker_config.get_dispatch_core_type(), DispatchCoreType::WORKER);
    EXPECT_EQ(col_axis_worker_config.get_dispatch_core_axis(), DispatchCoreAxis::COL);

    if (tt::tt_metal::hal::get_arch() == tt::ARCH::BLACKHOLE) {
        EXPECT_ANY_THROW(DispatchCoreConfig::create_with_max_worker_availability(
            std::nullopt, DispatchCoreAxis::ROW, tt::tt_fabric::FabricTensixConfig::DISABLED));

        const auto mux_config = DispatchCoreConfig::create_with_max_worker_availability(
            std::nullopt, std::nullopt, tt::tt_fabric::FabricTensixConfig::MUX);
        EXPECT_EQ(mux_config.get_dispatch_core_axis(), DispatchCoreAxis::ROW);
    }
}

TEST(DispatchCoreConfigTest, NamedFactoriesDefaultTypePolicies) {
    // Max dispatch performance prefers WORKER cores irrespective of cluster.
    EXPECT_EQ(
        DispatchCoreConfig::create_with_max_dispatch_performance().get_dispatch_core_type(), DispatchCoreType::WORKER);

    // Max worker availability resolves the type from the active cluster.
    const auto cluster_type = tt::tt_metal::GetClusterType();
    const bool eth_default_cluster = cluster_type == tt::tt_metal::ClusterType::N300 ||
                                     cluster_type == tt::tt_metal::ClusterType::T3K ||
                                     cluster_type == tt::tt_metal::ClusterType::N300_2x2;
    const auto expected_type = eth_default_cluster ? DispatchCoreType::ETH : DispatchCoreType::WORKER;
    EXPECT_EQ(DispatchCoreConfig::create_with_max_worker_availability().get_dispatch_core_type(), expected_type);
}

}  // namespace tt::tt_metal
