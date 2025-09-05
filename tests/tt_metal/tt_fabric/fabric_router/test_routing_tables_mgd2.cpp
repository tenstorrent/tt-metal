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

namespace {

constexpr auto k_FabricConfig = tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC;
constexpr auto k_ReliabilityMode = tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE;

std::unique_ptr<tt::tt_fabric::ControlPlane> make_control_plane(const std::filesystem::path& graph_desc) {
    auto control_plane = std::make_unique<tt::tt_fabric::ControlPlane>(graph_desc.string(), true);
    control_plane->initialize_fabric_context(k_FabricConfig);
    control_plane->configure_routing_tables_for_fabric_ethernet_channels(k_FabricConfig, k_ReliabilityMode);

    return control_plane;
}

std::unique_ptr<tt::tt_fabric::ControlPlane> make_control_plane(
    const std::filesystem::path& graph_desc,
    const std::map<tt::tt_fabric::FabricNodeId, chip_id_t>& logical_mesh_chip_id_to_physical_chip_id_mapping) {

    auto control_plane = std::make_unique<tt::tt_fabric::ControlPlane>(
        graph_desc.string(), logical_mesh_chip_id_to_physical_chip_id_mapping);
    control_plane->initialize_fabric_context(k_FabricConfig);
    control_plane->configure_routing_tables_for_fabric_ethernet_channels(k_FabricConfig, k_ReliabilityMode);

    return control_plane;
}

// Deep-compare helper for IntraMeshConnectivity
void expect_intra_mesh_connectivity_equal(
    const tt::tt_fabric::IntraMeshConnectivity &lhs, const tt::tt_fabric::IntraMeshConnectivity &rhs) {
    ASSERT_EQ(lhs.size(), rhs.size()) << "Number of meshes differ";
    for (std::size_t mesh_idx = 0; mesh_idx < lhs.size(); ++mesh_idx) {
        ASSERT_EQ(lhs[mesh_idx].size(), rhs[mesh_idx].size()) << "Number of chips differ at mesh index " << mesh_idx;
        for (std::size_t chip_idx = 0; chip_idx < lhs[mesh_idx].size(); ++chip_idx) {
            const auto &neighbors_lhs = lhs[mesh_idx][chip_idx];
            const auto &neighbors_rhs = rhs[mesh_idx][chip_idx];
            ASSERT_EQ(neighbors_lhs.size(), neighbors_rhs.size())
                << "Neighbor count differs at mesh " << mesh_idx << ", chip " << chip_idx;
            for (const auto &kv : neighbors_lhs) {
                const auto &neighbor_chip_id = kv.first;
                const tt::tt_fabric::RouterEdge &edge_lhs = kv.second;
                auto it = neighbors_rhs.find(neighbor_chip_id);
                ASSERT_NE(it, neighbors_rhs.end())
                    << "Mesh " << mesh_idx << ", chip " << chip_idx << " has no neighbor chip id " << neighbor_chip_id;
                const tt::tt_fabric::RouterEdge &edge_rhs = it->second;
                EXPECT_EQ(edge_lhs.port_direction, edge_rhs.port_direction)
                    << "Port direction differs at mesh " << mesh_idx << ", chip " << chip_idx;
                EXPECT_EQ(edge_lhs.weight, edge_rhs.weight)
                    << "Weight differs at mesh " << mesh_idx << ", chip " << chip_idx;
                EXPECT_EQ(edge_lhs.connected_chip_ids, edge_rhs.connected_chip_ids)
                    << "Connected chip IDs differ at mesh " << mesh_idx << ", chip " << chip_idx;
            }
        }
    }
}

void expect_inter_mesh_connectivity_equal(
    const tt::tt_fabric::InterMeshConnectivity &lhs, const tt::tt_fabric::InterMeshConnectivity &rhs) {
    ASSERT_EQ(lhs.size(), rhs.size()) << "Number of meshes differ";
    for (std::size_t mesh_idx = 0; mesh_idx < lhs.size(); ++mesh_idx) {
        ASSERT_EQ(lhs[mesh_idx].size(), rhs[mesh_idx].size()) << "Number of chips differ at mesh index " << mesh_idx;
        for (std::size_t chip_idx = 0; chip_idx < lhs[mesh_idx].size(); ++chip_idx) {
            const auto &neighbors_lhs = lhs[mesh_idx][chip_idx];
            const auto &neighbors_rhs = rhs[mesh_idx][chip_idx];
            ASSERT_EQ(neighbors_lhs.size(), neighbors_rhs.size())
                << "Neighbor count differs at mesh " << mesh_idx << ", chip " << chip_idx;
            for (const auto &kv : neighbors_lhs) {
                const auto &neighbor_mesh_id = kv.first;
                const tt::tt_fabric::RouterEdge &edge_lhs = kv.second;
                auto it = neighbors_rhs.find(neighbor_mesh_id);
                ASSERT_NE(it, neighbors_rhs.end())
                    << "Mesh " << mesh_idx << ", chip " << chip_idx << " has no neighbor mesh id " << neighbor_mesh_id.get();
                const tt::tt_fabric::RouterEdge &edge_rhs = it->second;
                EXPECT_EQ(edge_lhs.port_direction, edge_rhs.port_direction)
                    << "Port direction differs at mesh " << mesh_idx << ", chip " << chip_idx;
                EXPECT_EQ(edge_lhs.weight, edge_rhs.weight)
                    << "Weight differs at mesh " << mesh_idx << ", chip " << chip_idx;
                EXPECT_EQ(edge_lhs.connected_chip_ids, edge_rhs.connected_chip_ids)
                    << "Connected chip IDs differ at mesh " << mesh_idx << ", chip " << chip_idx;
            }
        }
    }
}

void expect_mesh_to_chip_ids_equal(
    const std::map<tt::tt_fabric::MeshId, tt::tt_fabric::MeshContainer<chip_id_t>>& lhs, const std::map<tt::tt_fabric::MeshId, tt::tt_fabric::MeshContainer<chip_id_t>>& rhs) {
    ASSERT_EQ(lhs.size(), rhs.size()) << "Number of meshes differ";
    for (const auto& [mesh_id, mesh_container] : lhs) {
        const auto& rhs_mesh_container = rhs.at(mesh_id);
        EXPECT_EQ(mesh_container.shape(), rhs_mesh_container.shape());
        EXPECT_EQ(mesh_container.size(), rhs_mesh_container.size());
        EXPECT_EQ(mesh_container, rhs_mesh_container);
    }
}

}  // namespace

namespace tt::tt_fabric::fabric_router_tests {

using ::testing::ElementsAre;

TEST(MeshGraphValidation, TestTGMeshGraphInitMGD2) {
    const std::filesystem::path tg_mesh_graph_desc_2_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/tg_mesh_graph_descriptor.textproto";
    auto mesh_graph_desc = std::make_unique<MeshGraph>(tg_mesh_graph_desc_2_path.string(), true);
    EXPECT_EQ(
        mesh_graph_desc->get_coord_range(MeshId{0}, MeshHostRankId(0)),
        MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(0, 0)));
    EXPECT_EQ(
        mesh_graph_desc->get_coord_range(MeshId{1}, MeshHostRankId(0)),
        MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(0, 0)));
    EXPECT_EQ(
        mesh_graph_desc->get_coord_range(MeshId{2}, MeshHostRankId(0)),
        MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(0, 0)));
    EXPECT_EQ(
        mesh_graph_desc->get_coord_range(MeshId{3}, MeshHostRankId(0)),
        MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(0, 0)));
    EXPECT_EQ(
        mesh_graph_desc->get_coord_range(MeshId{4}, MeshHostRankId(0)),
        MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(3, 7)));
}

TEST(MeshGraphValidation, TestTGMeshGraphInitConsistencyCheckMGD2) {
    // MGD 1.0 Path
    const std::filesystem::path tg_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/tg_mesh_graph_descriptor.textproto";
    auto mesh_graph = std::make_unique<MeshGraph>(tg_mesh_graph_desc_path.string());

    // MGD 2.0 Path
    const std::filesystem::path tg_mesh_graph_desc_2_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/tg_mesh_graph_descriptor.textproto";
    auto mesh_graph2 = std::make_unique<MeshGraph>(tg_mesh_graph_desc_2_path.string(), true);

    // Compare connectivity deeply
    expect_intra_mesh_connectivity_equal(
        mesh_graph->get_intra_mesh_connectivity(), mesh_graph2->get_intra_mesh_connectivity());
    expect_inter_mesh_connectivity_equal(
        mesh_graph->get_inter_mesh_connectivity(), mesh_graph2->get_inter_mesh_connectivity());
}

TEST_F(ControlPlaneFixture, TestTGControlPlaneInitMGD2) {
    const std::filesystem::path tg_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/tg_mesh_graph_descriptor.textproto";
    [[maybe_unused]] auto control_plane = make_control_plane(tg_mesh_graph_desc_path);
}

TEST_F(ControlPlaneFixture, TestTGMeshAPIsMGD2) {
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

TEST_F(ControlPlaneFixture, TestTGFabricRoutesMGD2) {
    const std::filesystem::path tg_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/tg_mesh_graph_descriptor.textproto";
    auto control_plane = make_control_plane(tg_mesh_graph_desc_path);
    auto valid_chans = control_plane->get_valid_eth_chans_on_routing_plane(FabricNodeId(MeshId{0}, 0), 1);
    EXPECT_GT(valid_chans.size(), 0);
    for (auto chan : valid_chans) {
        auto path = control_plane->get_fabric_route(FabricNodeId(MeshId{0}, 0), FabricNodeId(MeshId{4}, 31), chan);
        EXPECT_EQ(path.size() > 0, true);
    }
}

}  // namespace tt::tt_fabric::fabric_router_tests
