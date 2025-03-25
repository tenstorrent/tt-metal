// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include "fabric_fixture.hpp"
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/mesh_graph.hpp>
#include <tt-metalium/routing_table_generator.hpp>

namespace tt::tt_fabric {
namespace fabric_router_tests {

TEST_F(ControlPlaneFixture, TestTGMeshGraphInit) {
    const std::filesystem::path tg_mesh_graph_desc_path =
        std::filesystem::path(tt::llrt::RunTimeOptions::get_instance().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/tg_mesh_graph_descriptor.yaml";
    auto mesh_graph_desc = std::make_unique<MeshGraph>(tg_mesh_graph_desc_path.string());
}

TEST_F(ControlPlaneFixture, TestTGControlPlaneInit) {
    tt::tt_metal::detail::InitializeFabricConfig(tt::FabricConfig::FABRIC_2D);
    const std::filesystem::path tg_mesh_graph_desc_path =
        std::filesystem::path(tt::llrt::RunTimeOptions::get_instance().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/tg_mesh_graph_descriptor.yaml";
    auto control_plane = std::make_unique<ControlPlane>(tg_mesh_graph_desc_path.string());
    control_plane->configure_routing_tables_for_fabric_ethernet_channels();
}

TEST_F(ControlPlaneFixture, TestTGMeshAPIs) {
    const auto control_plane = tt::Cluster::instance().get_control_plane();
    auto user_meshes = control_plane->get_user_physical_mesh_ids();
    EXPECT_EQ(user_meshes.size(), 1);
    EXPECT_EQ(user_meshes[0], 4);
    EXPECT_EQ(control_plane->get_physical_mesh_shape(0), tt::tt_metal::distributed::MeshShape(1, 1));
    EXPECT_EQ(control_plane->get_physical_mesh_shape(1), tt::tt_metal::distributed::MeshShape(1, 1));
    EXPECT_EQ(control_plane->get_physical_mesh_shape(2), tt::tt_metal::distributed::MeshShape(1, 1));
    EXPECT_EQ(control_plane->get_physical_mesh_shape(3), tt::tt_metal::distributed::MeshShape(1, 1));
    EXPECT_EQ(control_plane->get_physical_mesh_shape(4), tt::tt_metal::distributed::MeshShape(4, 8));
}

TEST_F(ControlPlaneFixture, TestTGFabricRoutes) {
    tt::tt_metal::detail::InitializeFabricConfig(tt::FabricConfig::FABRIC_2D);
    const std::filesystem::path tg_mesh_graph_desc_path =
        std::filesystem::path(tt::llrt::RunTimeOptions::get_instance().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/tg_mesh_graph_descriptor.yaml";
    auto control_plane = std::make_unique<ControlPlane>(tg_mesh_graph_desc_path.string());
    control_plane->configure_routing_tables_for_fabric_ethernet_channels();
    auto valid_chans = control_plane->get_valid_eth_chans_on_routing_plane(0, 0, 3);
    for (auto chan : valid_chans) {
        auto path = control_plane->get_fabric_route(0, 0, 4, 31, chan);
    }
}

TEST_F(ControlPlaneFixture, TestT3kMeshGraphInit) {
    const std::filesystem::path t3k_mesh_graph_desc_path =
        std::filesystem::path(tt::llrt::RunTimeOptions::get_instance().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.yaml";
    auto mesh_graph_desc = std::make_unique<MeshGraph>(t3k_mesh_graph_desc_path.string());
}

TEST_F(ControlPlaneFixture, TestT3kControlPlaneInit) {
    tt::tt_metal::detail::InitializeFabricConfig(tt::FabricConfig::FABRIC_2D);
    const std::filesystem::path t3k_mesh_graph_desc_path =
        std::filesystem::path(tt::llrt::RunTimeOptions::get_instance().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.yaml";
    auto control_plane = std::make_unique<ControlPlane>(t3k_mesh_graph_desc_path.string());
    control_plane->configure_routing_tables_for_fabric_ethernet_channels();
}

TEST_F(ControlPlaneFixture, TestT3kFabricRoutes) {
    tt::tt_metal::detail::InitializeFabricConfig(tt::FabricConfig::FABRIC_2D);
    const std::filesystem::path t3k_mesh_graph_desc_path =
        std::filesystem::path(tt::llrt::RunTimeOptions::get_instance().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.yaml";
    auto control_plane = std::make_unique<ControlPlane>(t3k_mesh_graph_desc_path.string());
    control_plane->configure_routing_tables_for_fabric_ethernet_channels();
    auto valid_chans = control_plane->get_valid_eth_chans_on_routing_plane(0, 0, 0);
    for (auto chan : valid_chans) {
        auto path = control_plane->get_fabric_route(0, 0, 0, 7, chan);
    }
    valid_chans = control_plane->get_valid_eth_chans_on_routing_plane(0, 0, 1);
    for (auto chan : valid_chans) {
        auto path = control_plane->get_fabric_route(0, 0, 0, 7, chan);
    }
}

TEST_F(ControlPlaneFixture, TestQuantaGalaxyControlPlaneInit) {
    tt::tt_metal::detail::InitializeFabricConfig(tt::FabricConfig::FABRIC_2D);
    const std::filesystem::path quanta_galaxy_mesh_graph_desc_path =
        std::filesystem::path(tt::llrt::RunTimeOptions::get_instance().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/quanta_galaxy_mesh_graph_descriptor.yaml";
    auto control_plane = std::make_unique<ControlPlane>(quanta_galaxy_mesh_graph_desc_path.string());
    control_plane->configure_routing_tables_for_fabric_ethernet_channels();
}

TEST_F(ControlPlaneFixture, TestQuantaGalaxyMeshAPIs) {
    const auto control_plane = tt::Cluster::instance().get_control_plane();
    auto user_meshes = control_plane->get_user_physical_mesh_ids();
    EXPECT_EQ(user_meshes.size(), 1);
    EXPECT_EQ(user_meshes[0], 0);
    EXPECT_EQ(control_plane->get_physical_mesh_shape(0), tt::tt_metal::distributed::MeshShape(8, 4));
}


}  // namespace fabric_router_tests
}  // namespace tt::tt_fabric
