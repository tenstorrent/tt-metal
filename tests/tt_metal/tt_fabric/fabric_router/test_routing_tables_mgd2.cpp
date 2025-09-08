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
    auto control_plane = std::make_unique<tt::tt_fabric::ControlPlane>(graph_desc.string());
    control_plane->initialize_fabric_context(k_FabricConfig);
    control_plane->configure_routing_tables_for_fabric_ethernet_channels(k_FabricConfig, k_ReliabilityMode);

    return control_plane;
}

std::unique_ptr<tt::tt_fabric::ControlPlane> make_control_plane(
    const std::filesystem::path& graph_desc,
    const std::map<tt::tt_fabric::FabricNodeId, chip_id_t>& logical_mesh_chip_id_to_physical_chip_id_mapping) {

    auto control_plane = std::make_unique<tt::tt_fabric::ControlPlane>(
        graph_desc.string(), logical_mesh_chip_id_to_physical_chip_id_mapping, true);
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

void expect_get_host_ranks_equal(
    const tt::tt_fabric::MeshContainer<tt::tt_fabric::MeshHostRankId>& lhs, 
    const tt::tt_fabric::MeshContainer<tt::tt_fabric::MeshHostRankId>& rhs,
    int mesh_id) {
    // Test get_host_ranks function comparison for specified mesh ID
    EXPECT_EQ(lhs.size(), rhs.size()) 
        << "Mesh " << mesh_id << " host ranks count should be equal between MGD versions";
    EXPECT_EQ(lhs.shape(), rhs.shape()) 
        << "Mesh " << mesh_id << " host ranks shape should be equal between MGD versions";
    
    // Compare host rank values
    const auto& lhs_values = lhs.values();
    const auto& rhs_values = rhs.values();
    EXPECT_EQ(lhs_values.size(), rhs_values.size()) 
        << "Mesh " << mesh_id << " host ranks values count should be equal between MGD versions";
    
    for (size_t i = 0; i < lhs_values.size(); ++i) {
        EXPECT_EQ(*lhs_values[i], *rhs_values[i]) 
            << "Mesh " << mesh_id << " host rank " << i << " should be equal between MGD versions";
    }
}

void expect_get_chip_ids_equal(
    const tt::tt_fabric::MeshContainer<chip_id_t>& lhs, 
    const tt::tt_fabric::MeshContainer<chip_id_t>& rhs,
    int mesh_id) {
    // Test get_chip_ids function comparison for specified mesh ID
    EXPECT_EQ(lhs.size(), rhs.size()) 
        << "Mesh " << mesh_id << " chip count should be equal between MGD versions";
    EXPECT_EQ(lhs.shape(), rhs.shape()) 
        << "Mesh " << mesh_id << " chip shape should be equal between MGD versions";
    
    // Compare chip ID values
    const auto& lhs_values = lhs.values();
    const auto& rhs_values = rhs.values();
    EXPECT_EQ(lhs_values.size(), rhs_values.size()) 
        << "Mesh " << mesh_id << " chip values count should be equal between MGD versions";
    
    for (size_t i = 0; i < lhs_values.size(); ++i) {
        EXPECT_EQ(lhs_values[i], rhs_values[i]) 
            << "Mesh " << mesh_id << " chip ID " << i << " should be equal between MGD versions";
    }
}

void expect_get_chip_ids_submesh_equal(
    const tt::tt_fabric::MeshContainer<chip_id_t>& lhs, 
    const tt::tt_fabric::MeshContainer<chip_id_t>& rhs,
    int mesh_id) {
    // Test get_chip_ids function comparison for specified mesh ID submesh
    EXPECT_EQ(lhs.size(), rhs.size()) 
        << "Mesh " << mesh_id << " submesh chip count should be equal between MGD versions";
    EXPECT_EQ(lhs.shape(), rhs.shape()) 
        << "Mesh " << mesh_id << " submesh chip shape should be equal between MGD versions";
    
    // Compare chip ID values
    const auto& lhs_values = lhs.values();
    const auto& rhs_values = rhs.values();
    EXPECT_EQ(lhs_values.size(), rhs_values.size()) 
        << "Mesh " << mesh_id << " submesh chip values count should be equal between MGD versions";
    
    for (size_t i = 0; i < lhs_values.size(); ++i) {
        EXPECT_EQ(lhs_values[i], rhs_values[i]) 
            << "Mesh " << mesh_id << " submesh chip ID " << i << " should be equal between MGD versions";
    }
}

void expect_get_host_rank_for_chip_equal(
    const std::optional<tt::tt_fabric::MeshHostRankId>& lhs, 
    const std::optional<tt::tt_fabric::MeshHostRankId>& rhs,
    int mesh_id) {
    // Test get_host_rank_for_chip function comparison for specified mesh ID
    EXPECT_EQ(lhs.has_value(), rhs.has_value()) 
        << "Mesh " << mesh_id << " chip host rank presence should be consistent between MGD versions";
    
    if (lhs.has_value() && rhs.has_value()) {
        EXPECT_EQ(*lhs.value(), *rhs.value()) 
            << "Mesh " << mesh_id << " chip host rank should be equal between MGD versions";
    }
}

}  // namespace

namespace tt::tt_fabric::fabric_router_tests {

using ::testing::ElementsAre;

TEST(MeshGraphValidation, TestTGMeshGraphInitMGD2) {
    const std::filesystem::path tg_mesh_graph_desc_2_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/tg_mesh_graph_descriptor.textproto";
    auto mesh_graph_desc = std::make_unique<MeshGraph>(tg_mesh_graph_desc_2_path.string());
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
        "tt_metal/fabric/mesh_graph_descriptors/tg_mesh_graph_descriptor.yaml";
    auto mesh_graph = std::make_unique<MeshGraph>(tg_mesh_graph_desc_path.string());

    // MGD 2.0 Path
    const std::filesystem::path tg_mesh_graph_desc_2_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/tg_mesh_graph_descriptor.textproto";
    auto mesh_graph2 = std::make_unique<MeshGraph>(tg_mesh_graph_desc_2_path.string());

    // Compare connectivity deeply
    expect_intra_mesh_connectivity_equal(
        mesh_graph->get_intra_mesh_connectivity(), mesh_graph2->get_intra_mesh_connectivity());
    expect_inter_mesh_connectivity_equal(
        mesh_graph->get_inter_mesh_connectivity(), mesh_graph2->get_inter_mesh_connectivity());
    
    // Compare mesh graph functions between MGD 1.0 and MGD 2.0 for mesh IDs 0 and 4
    // Test get_host_ranks for mesh 0 and 4
    expect_get_host_ranks_equal(mesh_graph->get_host_ranks(tt::tt_fabric::MeshId{0}), mesh_graph2->get_host_ranks(tt::tt_fabric::MeshId{0}), 0);
    expect_get_host_ranks_equal(mesh_graph->get_host_ranks(tt::tt_fabric::MeshId{4}), mesh_graph2->get_host_ranks(tt::tt_fabric::MeshId{4}), 4);
    
    // Test get_chip_ids for mesh 0 and 4 (entire mesh)
    expect_get_chip_ids_equal(mesh_graph->get_chip_ids(tt::tt_fabric::MeshId{0}), mesh_graph2->get_chip_ids(tt::tt_fabric::MeshId{0}), 0);
    expect_get_chip_ids_equal(mesh_graph->get_chip_ids(tt::tt_fabric::MeshId{4}), mesh_graph2->get_chip_ids(tt::tt_fabric::MeshId{4}), 4);
    
    // Test get_chip_ids for mesh 0 and 4 (submesh with host rank 0)
    expect_get_chip_ids_submesh_equal(mesh_graph->get_chip_ids(tt::tt_fabric::MeshId{0}, tt::tt_fabric::MeshHostRankId{0}), mesh_graph2->get_chip_ids(tt::tt_fabric::MeshId{0}, tt::tt_fabric::MeshHostRankId{0}), 0);
    expect_get_chip_ids_submesh_equal(mesh_graph->get_chip_ids(tt::tt_fabric::MeshId{4}, tt::tt_fabric::MeshHostRankId{0}), mesh_graph2->get_chip_ids(tt::tt_fabric::MeshId{4}, tt::tt_fabric::MeshHostRankId{0}), 4);
    
    // Test get_host_rank_for_chip for mesh 0 and 4 (first chip in each mesh)
    expect_get_host_rank_for_chip_equal(mesh_graph->get_host_rank_for_chip(tt::tt_fabric::MeshId{0}, 0), mesh_graph2->get_host_rank_for_chip(tt::tt_fabric::MeshId{0}, 0), 0);
    expect_get_host_rank_for_chip_equal(mesh_graph->get_host_rank_for_chip(tt::tt_fabric::MeshId{4}, 0), mesh_graph2->get_host_rank_for_chip(tt::tt_fabric::MeshId{4}, 0), 4);
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

TEST(MeshGraphValidation, TestT3kMeshGraphInitMGD2) {
    const std::filesystem::path t3k_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.textproto";
    auto mesh_graph_desc = std::make_unique<MeshGraph>(t3k_mesh_graph_desc_path.string(), true);
    EXPECT_EQ(
        mesh_graph_desc->get_coord_range(MeshId{0}, MeshHostRankId(0)),
        MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(1, 3)));
}

TEST_F(ControlPlaneFixture, TestT3kControlPlaneInitMGD2) {
    const std::filesystem::path t3k_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.textproto";
    auto control_plane = make_control_plane(t3k_mesh_graph_desc_path);
}

TEST_F(ControlPlaneFixture, TestT3kFabricRoutesMGD2) {
    const std::filesystem::path t3k_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.textproto";
    auto control_plane = make_control_plane(t3k_mesh_graph_desc_path);

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

class T3kCustomMeshGraphControlPlaneFixtureMGD2
    : public ControlPlaneFixture,
      public testing::WithParamInterface<std::tuple<std::string, std::vector<std::vector<eth_coord_t>>>> {};

TEST_P(T3kCustomMeshGraphControlPlaneFixtureMGD2, TestT3kMeshGraphInitMGD2) {
    auto [mesh_graph_desc_path, _] = GetParam();
    const std::filesystem::path t3k_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) / mesh_graph_desc_path;
    auto mesh_graph_desc = std::make_unique<MeshGraph>(t3k_mesh_graph_desc_path.string(), true);
}

TEST_P(T3kCustomMeshGraphControlPlaneFixtureMGD2, TestT3kControlPlaneInitMGD2) {
    auto [mesh_graph_desc_path, mesh_graph_eth_coords] = GetParam();
    const std::filesystem::path t3k_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) / mesh_graph_desc_path;
    [[maybe_unused]] auto control_plane = make_control_plane(
        t3k_mesh_graph_desc_path.string(), get_physical_chip_mapping_from_eth_coords_mapping(mesh_graph_eth_coords));
}

TEST_P(T3kCustomMeshGraphControlPlaneFixtureMGD2, TestT3kFabricRoutesMGD2) {
    std::srand(std::time(nullptr));  // Seed the RNG
    auto [mesh_graph_desc_path, mesh_graph_eth_coords] = GetParam();
    const std::filesystem::path t3k_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) / mesh_graph_desc_path;
    auto control_plane = make_control_plane(
        t3k_mesh_graph_desc_path.string(), get_physical_chip_mapping_from_eth_coords_mapping(mesh_graph_eth_coords));

    for (const auto& src_mesh : control_plane->get_user_physical_mesh_ids()) {
        for (const auto& dst_mesh : control_plane->get_user_physical_mesh_ids()) {
            auto src_mesh_shape = control_plane->get_physical_mesh_shape(src_mesh);
            auto src_mesh_size = src_mesh_shape.mesh_size();
            auto dst_mesh_shape = control_plane->get_physical_mesh_shape(dst_mesh);
            auto dst_mesh_size = dst_mesh_shape.mesh_size();
            auto src_fabric_node_id = FabricNodeId(src_mesh, std::rand() % src_mesh_size);
            auto active_fabric_eth_channels = control_plane->get_active_fabric_eth_channels(src_fabric_node_id);
            EXPECT_GT(active_fabric_eth_channels.size(), 0);
            for (auto [chan, direction] : active_fabric_eth_channels) {
                auto dst_fabric_node_id = FabricNodeId(dst_mesh, std::rand() % dst_mesh_size);
                auto path = control_plane->get_fabric_route(src_fabric_node_id, dst_fabric_node_id, chan);
                EXPECT_EQ(src_fabric_node_id == dst_fabric_node_id ? path.size() == 0 : path.size() > 0, true);
            }
        }
    }
}

TEST_F(ControlPlaneFixture, TestT3kDisjointFabricRoutesMGD2) {
    auto [mesh_graph_desc_path, mesh_graph_eth_coords] = t3k_disjoint_mesh_descriptor_chip_mappings[0];
    const std::filesystem::path t3k_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) / mesh_graph_desc_path;
    auto control_plane = make_control_plane(
        t3k_mesh_graph_desc_path.string(), get_physical_chip_mapping_from_eth_coords_mapping(mesh_graph_eth_coords));

    auto valid_chans = control_plane->get_valid_eth_chans_on_routing_plane(FabricNodeId(MeshId{0}, 0), 0);
    EXPECT_GT(valid_chans.size(), 0);
    for (auto chan : valid_chans) {
        auto path = control_plane->get_fabric_route(FabricNodeId(MeshId{0}, 0), FabricNodeId(MeshId{0}, 3), chan);
        EXPECT_EQ(path.size() > 0, true);
    }
    valid_chans = control_plane->get_valid_eth_chans_on_routing_plane(FabricNodeId(MeshId{1}, 0), 0);
    EXPECT_GT(valid_chans.size(), 0);
    for (auto chan : valid_chans) {
        auto path = control_plane->get_fabric_route(FabricNodeId(MeshId{1}, 0), FabricNodeId(MeshId{1}, 3), chan);
        EXPECT_EQ(path.size() > 0, true);
    }
    valid_chans = control_plane->get_valid_eth_chans_on_routing_plane(FabricNodeId(MeshId{0}, 0), 0);
    EXPECT_GT(valid_chans.size(), 0);
    for (auto chan : valid_chans) {
        auto path = control_plane->get_fabric_route(FabricNodeId(MeshId{0}, 0), FabricNodeId(MeshId{1}, 3), chan);
        EXPECT_EQ(path.size() == 0, true);
        auto direction = control_plane->get_forwarding_direction(FabricNodeId(MeshId{0}, 0), FabricNodeId(MeshId{1}, 3));
        EXPECT_EQ(direction.has_value(), false);
    }
}

INSTANTIATE_TEST_SUITE_P(
    T3kCustomMeshGraphControlPlaneTests,
    T3kCustomMeshGraphControlPlaneFixtureMGD2,
    ::testing::ValuesIn(t3k_mesh_descriptor_chip_mappings_mgd2));


}  // namespace tt::tt_fabric::fabric_router_tests
