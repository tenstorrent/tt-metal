// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <typeinfo>

#include "umd/device/cluster.h"
#include "umd/device/tt_cluster_descriptor.h"
#include "llrt/rtoptions.hpp"
#include <tt-metalium/control_plane.hpp>
#include "impl/context/metal_context.hpp"

namespace tt::tt_fabric {

namespace mock_cluster_tests {

class ClusterFixture : public ::testing::Test {
protected:
    std::unique_ptr<tt_ClusterDescriptor> cluster_desc;

    void SetUp() override { printf("ClusterFixture SetUp\n"); }

    void TearDown() override { printf("ClusterFixture TearDown\n"); }
};


TEST_F(ClusterFixture, TestCustomCluster) {
    std::unique_ptr<tt_ClusterDescriptor> cluster_desc =
        tt_ClusterDescriptor::create_from_yaml("./t3k_cluster_desc.yaml");

    uint8_t num_routing_planes = std::numeric_limits<uint8_t>::max();
    tt::tt_metal::MetalContext::instance().get_cluster().configure_ethernet_cores_for_fabric_routers(
        tt::tt_fabric::FabricConfig::FABRIC_2D, num_routing_planes);

    const std::filesystem::path mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/custom_2x2_mesh_graph_descriptor.yaml";
        //"tt_metal/fabric/mesh_graph_descriptors/tg_mesh_graph_descriptor.yaml";

    auto control_plane = std::make_unique<tt::tt_fabric::ControlPlane>(mesh_graph_desc_path.string());

    control_plane->initialize_fabric_context(tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC);

    control_plane->configure_routing_tables_for_fabric_ethernet_channels(tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC, tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
}

}  // namespace mock_cluster_tests
}  // namespace tt::tt_fabric
