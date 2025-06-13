// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>
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

namespace tt::tt_fabric::fabric_router_tests {

using ::testing::ElementsAre;

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
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto user_meshes = control_plane.get_user_physical_mesh_ids();
    EXPECT_EQ(user_meshes.size(), 1);
    EXPECT_EQ(user_meshes[0], MeshId{4});
    EXPECT_EQ(control_plane.get_physical_mesh_shape(MeshId{0}), tt::tt_metal::distributed::MeshShape(1, 1));
    EXPECT_EQ(control_plane.get_physical_mesh_shape(MeshId{1}), tt::tt_metal::distributed::MeshShape(1, 1));
    EXPECT_EQ(control_plane.get_physical_mesh_shape(MeshId{2}), tt::tt_metal::distributed::MeshShape(1, 1));
    EXPECT_EQ(control_plane.get_physical_mesh_shape(MeshId{3}), tt::tt_metal::distributed::MeshShape(1, 1));
    EXPECT_EQ(control_plane.get_physical_mesh_shape(MeshId{4}), tt::tt_metal::distributed::MeshShape(4, 8));
}

TEST_F(ControlPlaneFixture, TestTGFabricRoutes) {
    const std::filesystem::path tg_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/tg_mesh_graph_descriptor.yaml";
    auto control_plane = std::make_unique<ControlPlane>(tg_mesh_graph_desc_path.string());
    control_plane->configure_routing_tables_for_fabric_ethernet_channels();
    auto valid_chans = control_plane->get_valid_eth_chans_on_routing_plane(FabricNodeId(MeshId{0}, 0), 1);
    EXPECT_GT(valid_chans.size(), 0);
    for (auto chan : valid_chans) {
        auto path = control_plane->get_fabric_route(FabricNodeId(MeshId{0}, 0), FabricNodeId(MeshId{4}, 31), chan);
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
    auto valid_chans = control_plane->get_valid_eth_chans_on_routing_plane(FabricNodeId(MeshId{0}, 0), 0);
    EXPECT_GT(valid_chans.size(), 0);
    for (auto chan : valid_chans) {
        auto path = control_plane->get_fabric_route(FabricNodeId(MeshId{0}, 0), FabricNodeId(MeshId{0}, 7), chan);
        EXPECT_EQ(path.size() > 0, true);
    }
    valid_chans = control_plane->get_valid_eth_chans_on_routing_plane(FabricNodeId(MeshId{0}, 0), 1);
    EXPECT_GT(valid_chans.size(), 0);
    for (auto chan : valid_chans) {
        auto path = control_plane->get_fabric_route(FabricNodeId(MeshId{0}, 0), FabricNodeId(MeshId{0}, 7), chan);
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
    auto& control_plane = global_control_plane->get_local_node_control_plane();
    control_plane.configure_routing_tables_for_fabric_ethernet_channels();
}

TEST_P(T3kCustomMeshGraphControlPlaneFixture, TestT3kFabricRoutes) {
    std::srand(std::time(nullptr));  // Seed the RNG
    auto [mesh_graph_desc_path, mesh_graph_eth_coords] = GetParam();
    const std::filesystem::path t3k_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) / mesh_graph_desc_path;
    auto global_control_plane = std::make_unique<GlobalControlPlane>(
        t3k_mesh_graph_desc_path.string(), get_physical_chip_mapping_from_eth_coords_mapping(mesh_graph_eth_coords));
    auto& control_plane = global_control_plane->get_local_node_control_plane();
    control_plane.configure_routing_tables_for_fabric_ethernet_channels();
    for (const auto& src_mesh : control_plane.get_user_physical_mesh_ids()) {
        for (const auto& dst_mesh : control_plane.get_user_physical_mesh_ids()) {
            auto src_mesh_shape = control_plane.get_physical_mesh_shape(src_mesh);
            auto src_mesh_size = src_mesh_shape.mesh_size();
            auto dst_mesh_shape = control_plane.get_physical_mesh_shape(dst_mesh);
            auto dst_mesh_size = dst_mesh_shape.mesh_size();
            auto src_fabric_node_id = FabricNodeId(src_mesh, std::rand() % src_mesh_size);
            auto active_fabric_eth_channels = control_plane.get_active_fabric_eth_channels(src_fabric_node_id);
            EXPECT_GT(active_fabric_eth_channels.size(), 0);
            for (auto [chan, direction] : active_fabric_eth_channels) {
                auto dst_fabric_node_id = FabricNodeId(dst_mesh, std::rand() % dst_mesh_size);
                auto path = control_plane.get_fabric_route(src_fabric_node_id, dst_fabric_node_id, chan);
                EXPECT_EQ(src_fabric_node_id == dst_fabric_node_id ? path.size() == 0 : path.size() > 0, true);
            }
        }
    }
}

TEST_F(ControlPlaneFixture, TestT3kDisjointFabricRoutes) {
    auto [mesh_graph_desc_path, mesh_graph_eth_coords] = t3k_disjoint_mesh_descriptor_chip_mappings[0];
    const std::filesystem::path t3k_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) / mesh_graph_desc_path;
    auto global_control_plane = std::make_unique<GlobalControlPlane>(
        t3k_mesh_graph_desc_path.string(), get_physical_chip_mapping_from_eth_coords_mapping(mesh_graph_eth_coords));
    auto& control_plane = global_control_plane->get_local_node_control_plane();
    control_plane.configure_routing_tables_for_fabric_ethernet_channels();
    auto valid_chans = control_plane.get_valid_eth_chans_on_routing_plane(FabricNodeId(MeshId{0}, 0), 0);
    EXPECT_GT(valid_chans.size(), 0);
    for (auto chan : valid_chans) {
        auto path = control_plane.get_fabric_route(FabricNodeId(MeshId{0}, 0), FabricNodeId(MeshId{0}, 3), chan);
        EXPECT_EQ(path.size() > 0, true);
    }
    valid_chans = control_plane.get_valid_eth_chans_on_routing_plane(FabricNodeId(MeshId{1}, 0), 0);
    EXPECT_GT(valid_chans.size(), 0);
    for (auto chan : valid_chans) {
        auto path = control_plane.get_fabric_route(FabricNodeId(MeshId{1}, 0), FabricNodeId(MeshId{1}, 3), chan);
        EXPECT_EQ(path.size() > 0, true);
    }
    valid_chans = control_plane.get_valid_eth_chans_on_routing_plane(FabricNodeId(MeshId{0}, 0), 0);
    EXPECT_GT(valid_chans.size(), 0);
    for (auto chan : valid_chans) {
        auto path = control_plane.get_fabric_route(FabricNodeId(MeshId{0}, 0), FabricNodeId(MeshId{1}, 3), chan);
        EXPECT_EQ(path.size() == 0, true);
        auto direction = control_plane.get_forwarding_direction(FabricNodeId(MeshId{0}, 0), FabricNodeId(MeshId{1}, 3));
        EXPECT_EQ(direction.has_value(), false);
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
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto user_meshes = control_plane.get_user_physical_mesh_ids();
    EXPECT_EQ(user_meshes.size(), 1);
    EXPECT_EQ(user_meshes[0], MeshId{0});
    EXPECT_EQ(control_plane.get_physical_mesh_shape(MeshId{0}), tt::tt_metal::distributed::MeshShape(8, 4));
}

TEST(MeshGraphValidation, TestT3kDualHostMeshGraph) {
    const std::filesystem::path t3k_dual_host_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_dual_host_mesh_graph_descriptor.yaml";
    auto mesh_graph = std::make_unique<tt_fabric::MeshGraph>(t3k_dual_host_mesh_graph_desc_path.string());

    EXPECT_THAT(mesh_graph->get_mesh_ids(), ElementsAre(MeshId{0}));

    // Check host ranks by accessing the values vector
    const auto& host_ranks = mesh_graph->get_host_ranks(MeshId{0});
    EXPECT_EQ(host_ranks, MeshContainer<HostRankId>(MeshShape(1, 2), {HostRankId(0), HostRankId(1)}));

    EXPECT_EQ(mesh_graph->get_mesh_shape(MeshId{0}), MeshShape(2, 4));
    EXPECT_EQ(mesh_graph->get_mesh_shape(MeshId{0}, HostRankId(0)), MeshShape(2, 2));
    EXPECT_EQ(mesh_graph->get_mesh_shape(MeshId{0}, HostRankId(1)), MeshShape(2, 2));

    EXPECT_EQ(mesh_graph->get_coord_range(MeshId{0}), MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(1, 3)));
    EXPECT_EQ(
        mesh_graph->get_coord_range(MeshId{0}, HostRankId(0)),
        MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(1, 1)));
    EXPECT_EQ(
        mesh_graph->get_coord_range(MeshId{0}, HostRankId(1)),
        MeshCoordinateRange(MeshCoordinate(0, 2), MeshCoordinate(1, 3)));

    EXPECT_THAT(mesh_graph->get_mesh_ids(), ElementsAre(MeshId{0}));

    EXPECT_EQ(
        mesh_graph->get_chip_ids(MeshId{0}),
        MeshContainer<chip_id_t>(MeshShape(2, 4), std::vector<chip_id_t>{0, 1, 2, 3, 4, 5, 6, 7}));
    EXPECT_EQ(
        mesh_graph->get_chip_ids(MeshId{0}, HostRankId(0)),
        MeshContainer<chip_id_t>(MeshShape(2, 2), std::vector<chip_id_t>{0, 1, 4, 5}));
    EXPECT_EQ(
        mesh_graph->get_chip_ids(MeshId{0}, HostRankId(1)),
        MeshContainer<chip_id_t>(MeshShape(2, 2), std::vector<chip_id_t>{2, 3, 6, 7}));
}

TEST(MeshGraphValidation, TestT3k2x2MeshGraph) {
    const std::filesystem::path t3k_2x2_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_2x2_mesh_graph_descriptor.yaml";
    auto mesh_graph = std::make_unique<tt_fabric::MeshGraph>(t3k_2x2_mesh_graph_desc_path.string());

    // This configuration has two meshes (id 0 and id 1)
    EXPECT_THAT(mesh_graph->get_mesh_ids(), ElementsAre(MeshId{0}, MeshId{1}));

    // Check host ranks for mesh 0 - single host rank 0
    const auto& host_ranks_mesh0 = mesh_graph->get_host_ranks(MeshId{0});
    EXPECT_EQ(host_ranks_mesh0, MeshContainer<HostRankId>(MeshShape(1, 1), {HostRankId(0)}));

    // Check host ranks for mesh 1 - single host rank 0
    const auto& host_ranks_mesh1 = mesh_graph->get_host_ranks(MeshId{1});
    EXPECT_EQ(host_ranks_mesh1, MeshContainer<HostRankId>(MeshShape(1, 1), {HostRankId(0)}));

    // Each mesh has a 2x2 board topology
    EXPECT_EQ(mesh_graph->get_mesh_shape(MeshId{0}), MeshShape(2, 2));
    EXPECT_EQ(mesh_graph->get_mesh_shape(MeshId{1}), MeshShape(2, 2));

    // Since there's only one host rank per mesh, mesh shape should be same
    EXPECT_EQ(mesh_graph->get_mesh_shape(MeshId{0}, HostRankId(0)), MeshShape(2, 2));
    EXPECT_EQ(mesh_graph->get_mesh_shape(MeshId{1}, HostRankId(0)), MeshShape(2, 2));

    // Check coordinate ranges
    EXPECT_EQ(mesh_graph->get_coord_range(MeshId{0}), MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(1, 1)));
    EXPECT_EQ(mesh_graph->get_coord_range(MeshId{1}), MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(1, 1)));

    // Since each mesh has only one host rank, the coord range should be the same
    EXPECT_EQ(
        mesh_graph->get_coord_range(MeshId{0}, HostRankId(0)),
        MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(1, 1)));
    EXPECT_EQ(
        mesh_graph->get_coord_range(MeshId{1}, HostRankId(0)),
        MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(1, 1)));

    // Check chip IDs - each mesh has 4 chips (2x2)
    EXPECT_EQ(
        mesh_graph->get_chip_ids(MeshId{0}),
        MeshContainer<chip_id_t>(MeshShape(2, 2), std::vector<chip_id_t>{0, 1, 2, 3}));
    EXPECT_EQ(
        mesh_graph->get_chip_ids(MeshId{1}),
        MeshContainer<chip_id_t>(MeshShape(2, 2), std::vector<chip_id_t>{0, 1, 2, 3}));

    // Check chip IDs per host rank
    EXPECT_EQ(
        mesh_graph->get_chip_ids(MeshId{0}, HostRankId(0)),
        MeshContainer<chip_id_t>(MeshShape(2, 2), std::vector<chip_id_t>{0, 1, 2, 3}));
    EXPECT_EQ(
        mesh_graph->get_chip_ids(MeshId{1}, HostRankId(0)),
        MeshContainer<chip_id_t>(MeshShape(2, 2), std::vector<chip_id_t>{0, 1, 2, 3}));
}

TEST(MeshGraphValidation, TestGetHostRankForChip) {
    // Test with dual host T3K configuration
    const std::filesystem::path t3k_dual_host_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_dual_host_mesh_graph_descriptor.yaml";
    auto mesh_graph = std::make_unique<tt_fabric::MeshGraph>(t3k_dual_host_mesh_graph_desc_path.string());

    // Test valid chips for mesh 0
    // Based on the dual host configuration:
    // Host rank 0 controls chips 0, 1, 4, 5 (left board)
    // Host rank 1 controls chips 2, 3, 6, 7 (right board)
    EXPECT_EQ(mesh_graph->get_host_rank_for_chip(MeshId{0}, 0), HostRankId(0));
    EXPECT_EQ(mesh_graph->get_host_rank_for_chip(MeshId{0}, 1), HostRankId(0));
    EXPECT_EQ(mesh_graph->get_host_rank_for_chip(MeshId{0}, 4), HostRankId(0));
    EXPECT_EQ(mesh_graph->get_host_rank_for_chip(MeshId{0}, 5), HostRankId(0));

    EXPECT_EQ(mesh_graph->get_host_rank_for_chip(MeshId{0}, 2), HostRankId(1));
    EXPECT_EQ(mesh_graph->get_host_rank_for_chip(MeshId{0}, 3), HostRankId(1));
    EXPECT_EQ(mesh_graph->get_host_rank_for_chip(MeshId{0}, 6), HostRankId(1));
    EXPECT_EQ(mesh_graph->get_host_rank_for_chip(MeshId{0}, 7), HostRankId(1));

    // Test invalid chip IDs (out of range)
    EXPECT_EQ(mesh_graph->get_host_rank_for_chip(MeshId{0}, 8), std::nullopt);
    EXPECT_EQ(mesh_graph->get_host_rank_for_chip(MeshId{0}, 100), std::nullopt);

    // Test invalid mesh ID
    EXPECT_EQ(mesh_graph->get_host_rank_for_chip(MeshId{1}, 0), std::nullopt);

    // Test with single host T3K configuration
    const std::filesystem::path t3k_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.yaml";
    auto mesh_graph_single_host = std::make_unique<tt_fabric::MeshGraph>(t3k_mesh_graph_desc_path.string());

    // In single host configuration, all chips should belong to host rank 0
    for (chip_id_t chip_id = 0; chip_id < 8; chip_id++) {
        EXPECT_EQ(mesh_graph_single_host->get_host_rank_for_chip(MeshId{0}, chip_id), HostRankId(0));
    }

    // Test with 2x2 configuration (two separate meshes)
    const std::filesystem::path t3k_2x2_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_2x2_mesh_graph_descriptor.yaml";
    auto mesh_graph_2x2 = std::make_unique<tt_fabric::MeshGraph>(t3k_2x2_mesh_graph_desc_path.string());

    // Each mesh has only one host rank (0)
    for (chip_id_t chip_id = 0; chip_id < 4; chip_id++) {
        EXPECT_EQ(mesh_graph_2x2->get_host_rank_for_chip(MeshId{0}, chip_id), HostRankId(0));
        EXPECT_EQ(mesh_graph_2x2->get_host_rank_for_chip(MeshId{1}, chip_id), HostRankId(0));
    }

    // Test invalid chip IDs for 2x2 configuration
    EXPECT_EQ(mesh_graph_2x2->get_host_rank_for_chip(MeshId{0}, 4), std::nullopt);
    EXPECT_EQ(mesh_graph_2x2->get_host_rank_for_chip(MeshId{1}, 4), std::nullopt);
}

}  // namespace tt::tt_fabric::fabric_router_tests
