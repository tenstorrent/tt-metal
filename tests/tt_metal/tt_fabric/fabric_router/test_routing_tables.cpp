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
    auto valid_chans = control_plane->get_valid_eth_chans_on_routing_plane(0, 0, 3);
    for (auto chan : valid_chans) {
        auto path = control_plane->get_fabric_route(0, 0, 4, 31, chan);
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
    auto valid_chans = control_plane->get_valid_eth_chans_on_routing_plane(0, 0, 0);
    for (auto chan : valid_chans) {
        auto path = control_plane->get_fabric_route(0, 0, 0, 3, chan);
        EXPECT_EQ(path.size() > 0, true);
    }
    valid_chans = control_plane->get_valid_eth_chans_on_routing_plane(0, 1, 1);
    for (auto chan : valid_chans) {
        auto path = control_plane->get_fabric_route(0, 1, 1, 2, chan);
        EXPECT_EQ(path.size() > 0, true);
    }
    valid_chans = control_plane->get_valid_eth_chans_on_routing_plane(0, 0, 0);
    for (auto chan : valid_chans) {
        auto path = control_plane->get_fabric_route(0, 0, 1, 3, chan);
        EXPECT_EQ(path.size() > 0, true);
    }
    valid_chans = control_plane->get_valid_eth_chans_on_routing_plane(1, 2, 1);
    for (auto chan : valid_chans) {
        auto path = control_plane->get_fabric_route(1, 2, 0, 2, chan);
        EXPECT_EQ(path.size() > 0, true);
    }
}

std::vector<std::vector<chip_id_t>> get_physical_chip_mapping_from_eth_coords_mapping(
    const std::vector<std::vector<eth_coord_t>>& mesh_graph_eth_coords) {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    std::vector<std::vector<chip_id_t>> physical_chip_ids_mapping;
    physical_chip_ids_mapping.reserve(mesh_graph_eth_coords.size());
    for (const auto& mesh : mesh_graph_eth_coords) {
        std::vector<chip_id_t> physical_chip_ids;
        physical_chip_ids.reserve(mesh.size());
        for (const auto& eth_coord : mesh) {
            physical_chip_ids.push_back(cluster.get_physical_chip_id_from_eth_coord(eth_coord));
        }
        physical_chip_ids_mapping.push_back(physical_chip_ids);
    }
    return physical_chip_ids_mapping;
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
    auto control_plane = std::make_unique<ControlPlane>(
        t3k_mesh_graph_desc_path.string(), get_physical_chip_mapping_from_eth_coords_mapping(mesh_graph_eth_coords));
    control_plane->configure_routing_tables_for_fabric_ethernet_channels();
}

TEST_P(T3kCustomMeshGraphControlPlaneFixture, TestT3kFabricRoutes) {
    auto [mesh_graph_desc_path, mesh_graph_eth_coords] = GetParam();
    const std::filesystem::path t3k_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) / mesh_graph_desc_path;
    auto control_plane = std::make_unique<ControlPlane>(
        t3k_mesh_graph_desc_path.string(), get_physical_chip_mapping_from_eth_coords_mapping(mesh_graph_eth_coords));
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
                src_mesh, std::rand() % src_mesh_size, std::rand() % num_routing_planes);
            for (auto chan : valid_chans) {
                auto path = control_plane->get_fabric_route(
                    src_mesh, std::rand() % src_mesh_size, dst_mesh, std::rand() % dst_mesh_size, chan);
                EXPECT_EQ(path.size() > 0, true);
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    T3kCustomMeshGraphControlPlaneTests,
    T3kCustomMeshGraphControlPlaneFixture,
    ::testing::Values(
        std::tuple{
            "tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.yaml",
            std::vector<std::vector<eth_coord_t>>{
                {{0, 0, 0, 0, 0},
                 {0, 1, 0, 0, 0},
                 {0, 2, 0, 0, 0},
                 {0, 3, 0, 0, 0},
                 {0, 0, 1, 0, 0},
                 {0, 1, 1, 0, 0},
                 {0, 2, 1, 0, 0},
                 {0, 3, 1, 0, 0}}}},
        std::tuple{
            "tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_2x2_mesh_graph_descriptor.yaml",
            std::vector<std::vector<eth_coord_t>>{
                {{0, 0, 0, 0, 0}, {0, 1, 0, 0, 0}, {0, 0, 1, 0, 0}, {0, 1, 1, 0, 0}},
                {{0, 2, 0, 0, 0}, {0, 3, 0, 0, 0}, {0, 2, 1, 0, 0}, {0, 3, 1, 0, 0}}}},
        std::tuple{
            "tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_1x2_mesh_graph_descriptor.yaml",
            std::vector<std::vector<eth_coord_t>>{
                {{0, 0, 0, 0, 0}, {0, 1, 0, 0, 0}},
                {{0, 2, 0, 0, 0}, {0, 3, 0, 0, 0}},
                {{0, 0, 1, 0, 0}, {0, 1, 1, 0, 0}},
                {{0, 2, 1, 0, 0}, {0, 3, 1, 0, 0}}}},
        std::tuple{
            "tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_1x1_mesh_graph_descriptor.yaml",
            std::vector<std::vector<eth_coord_t>>{
                {{0, 0, 0, 0, 0}},
                {{0, 1, 0, 0, 0}},
                {{0, 2, 0, 0, 0}},
                {{0, 3, 0, 0, 0}},
                {{0, 0, 1, 0, 0}},
                {{0, 1, 1, 0, 0}},
                {{0, 2, 1, 0, 0}},
                {{0, 3, 1, 0, 0}}}},
        std::tuple{
            "tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_2x2_1x2_1x1_mesh_graph_descriptor.yaml",
            std::vector<std::vector<eth_coord_t>>{
                {{0, 0, 0, 0, 0}, {0, 1, 0, 0, 0}, {0, 0, 1, 0, 0}, {0, 1, 1, 0, 0}},
                {{0, 2, 0, 0, 0}, {0, 3, 0, 0, 0}},
                {{0, 2, 1, 0, 0}},
                {{0, 3, 1, 0, 0}}}}));

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
