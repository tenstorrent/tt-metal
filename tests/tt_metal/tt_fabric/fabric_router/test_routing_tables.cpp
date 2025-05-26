// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/mesh_graph.hpp>
#include <filesystem>
#include <memory>
#include <vector>

#include "fabric_fixture.hpp"
#include "t3k_mesh_descriptor_chip_mappings.hpp"
#include "utils.hpp"
#include <tt-metalium/fabric_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include "impl/context/metal_context.hpp"
#include <tt-metalium/tt_metal.hpp>

namespace tt::tt_fabric {
namespace fabric_router_tests {

TEST_F(ControlPlaneFixture, TestTGMeshGraphInit) {
    const std::filesystem::path tg_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/tg_mesh_graph_descriptor.yaml";
    auto mesh_graph_desc = std::make_unique<MeshGraph>(tg_mesh_graph_desc_path.string());
}

TEST_F(ControlPlaneFixture, TestTGControlPlaneInit) {
    const std::filesystem::path tg_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/tg_mesh_graph_descriptor.yaml";
    auto control_plane = std::make_unique<ControlPlane>(tg_mesh_graph_desc_path.string());
    control_plane->configure_routing_tables_for_fabric_ethernet_channels();
}

TEST_F(ControlPlaneFixture, TestTGMeshAPIs) {
    const auto control_plane = tt::tt_metal::MetalContext::instance().get_cluster().get_control_plane();
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
    const std::filesystem::path tg_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/tg_mesh_graph_descriptor.yaml";
    auto control_plane = std::make_unique<ControlPlane>(tg_mesh_graph_desc_path.string());
    control_plane->configure_routing_tables_for_fabric_ethernet_channels();
    auto valid_chans = control_plane->get_valid_eth_chans_on_routing_plane(FabricNodeId(0, 0), 3);
    for (auto chan : valid_chans) {
        auto path = control_plane->get_fabric_route(FabricNodeId(0, 0), FabricNodeId(4, 31), chan);
        EXPECT_EQ(path.size() > 0, true);
    }
}

TEST_F(ControlPlaneFixture, TestT3kMeshGraphInit) {
    const std::filesystem::path t3k_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.yaml";
    auto mesh_graph_desc = std::make_unique<MeshGraph>(t3k_mesh_graph_desc_path.string());
}

TEST_F(ControlPlaneFixture, TestT3kControlPlaneInit) {
    const std::filesystem::path t3k_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.yaml";
    auto control_plane = std::make_unique<ControlPlane>(t3k_mesh_graph_desc_path.string());
    control_plane->configure_routing_tables_for_fabric_ethernet_channels();
}

TEST_F(ControlPlaneFixture, TestT3kFabricRoutes) {
    const std::filesystem::path t3k_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.yaml";
    auto control_plane = std::make_unique<ControlPlane>(t3k_mesh_graph_desc_path.string());
    control_plane->configure_routing_tables_for_fabric_ethernet_channels();
    auto valid_chans = control_plane->get_valid_eth_chans_on_routing_plane(FabricNodeId(0, 0), 0);
    for (auto chan : valid_chans) {
        auto path = control_plane->get_fabric_route(FabricNodeId(0, 0), FabricNodeId(0, 7), chan);
        EXPECT_EQ(path.size() > 0, true);
    }
    valid_chans = control_plane->get_valid_eth_chans_on_routing_plane(FabricNodeId(0, 0), 1);
    for (auto chan : valid_chans) {
        auto path = control_plane->get_fabric_route(FabricNodeId(0, 0), FabricNodeId(0, 7), chan);
        EXPECT_EQ(path.size() > 0, true);
    }
}

class T3kCustomMeshGraphControlPlaneFixture
    : public ControlPlaneFixture,
      public testing::WithParamInterface<std::tuple<std::string, std::vector<std::vector<eth_coord_t>>>> {};

TEST_P(T3kCustomMeshGraphControlPlaneFixture, TestT3kMeshGraphInit) {
    auto [mesh_graph_desc_path, _] = GetParam();
    const std::filesystem::path t3k_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) / mesh_graph_desc_path;
    auto mesh_graph_desc = std::make_unique<MeshGraph>(t3k_mesh_graph_desc_path.string());
}

TEST_P(T3kCustomMeshGraphControlPlaneFixture, TestT3kControlPlaneInit) {
    auto [mesh_graph_desc_path, mesh_graph_eth_coords] = GetParam();
    const std::filesystem::path t3k_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) / mesh_graph_desc_path;
    auto global_control_plane = std::make_unique<GlobalControlPlane>(
        t3k_mesh_graph_desc_path.string(), get_physical_chip_mapping_from_eth_coords_mapping(mesh_graph_eth_coords));
    auto control_plane = global_control_plane->get_local_node_control_plane();
    control_plane->configure_routing_tables_for_fabric_ethernet_channels();
}

TEST_P(T3kCustomMeshGraphControlPlaneFixture, TestT3kFabricRoutes) {
    auto [mesh_graph_desc_path, mesh_graph_eth_coords] = GetParam();
    const std::filesystem::path t3k_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) / mesh_graph_desc_path;
    auto global_control_plane = std::make_unique<GlobalControlPlane>(
        t3k_mesh_graph_desc_path.string(), get_physical_chip_mapping_from_eth_coords_mapping(mesh_graph_eth_coords));
    auto control_plane = global_control_plane->get_local_node_control_plane();
    control_plane->configure_routing_tables_for_fabric_ethernet_channels();
    // TODO: Query this
    constexpr uint32_t num_routing_planes = 2;
    for (const auto& src_mesh : control_plane->get_user_physical_mesh_ids()) {
        for (const auto& dst_mesh : control_plane->get_user_physical_mesh_ids()) {
            auto src_mesh_shape = control_plane->get_physical_mesh_shape(src_mesh);
            auto src_mesh_size = src_mesh_shape[0] * src_mesh_shape[1];
            auto dst_mesh_shape = control_plane->get_physical_mesh_shape(dst_mesh);
            auto dst_mesh_size = dst_mesh_shape[0] * dst_mesh_shape[1];
            auto valid_chans = control_plane->get_valid_eth_chans_on_routing_plane(
                FabricNodeId(src_mesh, std::rand() % src_mesh_size), std::rand() % num_routing_planes);
            for (auto chan : valid_chans) {
                auto path = control_plane->get_fabric_route(
                    FabricNodeId(src_mesh, std::rand() % src_mesh_size),
                    FabricNodeId(dst_mesh, std::rand() % dst_mesh_size),
                    chan);
                EXPECT_EQ(path.size() > 0, true);
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    T3kCustomMeshGraphControlPlaneTests,
    T3kCustomMeshGraphControlPlaneFixture,
    ::testing::ValuesIn(t3k_mesh_descriptor_chip_mappings));

TEST_F(ControlPlaneFixture, TestQuantaGalaxyControlPlaneInit) {
    const std::filesystem::path quanta_galaxy_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/quanta_galaxy_mesh_graph_descriptor.yaml";
    auto control_plane = std::make_unique<ControlPlane>(quanta_galaxy_mesh_graph_desc_path.string());
    control_plane->configure_routing_tables_for_fabric_ethernet_channels();
}

TEST_F(ControlPlaneFixture, TestQuantaGalaxyMeshAPIs) {
    const auto control_plane = tt::tt_metal::MetalContext::instance().get_cluster().get_control_plane();
    auto user_meshes = control_plane->get_user_physical_mesh_ids();
    EXPECT_EQ(user_meshes.size(), 1);
    EXPECT_EQ(user_meshes[0], 0);
    EXPECT_EQ(control_plane->get_physical_mesh_shape(0), tt::tt_metal::distributed::MeshShape(8, 4));
}


}  // namespace fabric_router_tests
}  // namespace tt::tt_fabric
