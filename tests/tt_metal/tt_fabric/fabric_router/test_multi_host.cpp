// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/mesh_graph.hpp>
#include <filesystem>
#include <memory>
#include <mpi.h>
#include <vector>

#include "fabric_fixture.hpp"
#include <tt-metalium/fabric_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include "impl/context/metal_context.hpp"
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/distributed_context.hpp>

namespace tt::tt_fabric {
namespace multi_host_tests {

TEST(MultiHost, TestBasicMPICluster) {
    tt::tt_metal::distributed::multihost::DistributedContext::create(0, nullptr);

    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& eth_connections = cluster.get_ethernet_connections();
    std::uint32_t num_expected_chips = 0;
    std::uint32_t num_connections_per_side = 0;
    auto cluster_type = cluster.get_cluster_type();
    if (cluster_type == tt::ClusterType::T3K) {
        num_expected_chips = 8;
        num_connections_per_side = 2;
    } else if (cluster_type == tt::ClusterType::GALAXY) {
        num_expected_chips = 32;
        // TODO: update to 4 when dual Galaxy system is fixed or replaced
        num_connections_per_side = 1;
    } else {
        GTEST_SKIP() << "Mesh check not supported for system type " << magic_enum::enum_name(cluster_type);
    }
    EXPECT_EQ(eth_connections.size(), num_expected_chips)
        << " Expected " << num_expected_chips << " in " << magic_enum::enum_name(cluster_type) << " cluster";
    for (const auto& [chip, connections] : eth_connections) {
        std::map<chip_id_t, int> num_connections_to_chip;
        for (const auto& [channel, remote_chip_and_channel] : connections) {
            num_connections_to_chip[std::get<0>(remote_chip_and_channel)]++;
        }
        for (const auto& [other_chip, count] : num_connections_to_chip) {
            EXPECT_TRUE(count >= num_connections_per_side)
                << "Chip " << chip << " has " << count << " connections to Chip " << other_chip << ", expected "
                << num_connections_per_side;
        }
    }
}

TEST(MultiHost, TestDualGalaxyControlPlaneInit) {
    // TODO: remove this when it's in the metal context
    const std::filesystem::path quanta_galaxy_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/dual_galaxy_mesh_graph_descriptor.yaml";
    auto control_plane = std::make_unique<GlobalControlPlane>(quanta_galaxy_mesh_graph_desc_path.string());
    // control_plane->configure_routing_tables_for_fabric_ethernet_channels();
}

}  // namespace multi_host_tests
}  // namespace tt::tt_fabric
