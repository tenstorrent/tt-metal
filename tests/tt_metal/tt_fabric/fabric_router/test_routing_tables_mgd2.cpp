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

constexpr auto kFabricConfig = tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC;
constexpr auto kReliabilityMode = tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE;

std::unique_ptr<tt::tt_fabric::ControlPlane> make_control_plane(const std::filesystem::path& graph_desc) {
    auto control_plane = std::make_unique<tt::tt_fabric::ControlPlane>(graph_desc.string());
    control_plane->initialize_fabric_context(kFabricConfig);
    control_plane->configure_routing_tables_for_fabric_ethernet_channels(kFabricConfig, kReliabilityMode);

    return control_plane;
}

std::unique_ptr<tt::tt_fabric::ControlPlane> make_control_plane(
    const std::filesystem::path& graph_desc,
    const std::map<tt::tt_fabric::FabricNodeId, chip_id_t>& logical_mesh_chip_id_to_physical_chip_id_mapping) {

    auto control_plane = std::make_unique<tt::tt_fabric::ControlPlane>(
        graph_desc.string(), logical_mesh_chip_id_to_physical_chip_id_mapping);
    control_plane->initialize_fabric_context(kFabricConfig);
    control_plane->configure_routing_tables_for_fabric_ethernet_channels(kFabricConfig, kReliabilityMode);

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
    EXPECT_EQ(lhs.size(), rhs.size()) << "Mesh " << mesh_id << " host ranks count should be equal between MGD versions";
    EXPECT_EQ(lhs.shape(), rhs.shape()) << "Mesh " << mesh_id
                                        << " host ranks shape should be equal between MGD versions";

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
    EXPECT_EQ(lhs.size(), rhs.size()) << "Mesh " << mesh_id << " chip count should be equal between MGD versions";
    EXPECT_EQ(lhs.shape(), rhs.shape()) << "Mesh " << mesh_id << " chip shape should be equal between MGD versions";

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
    EXPECT_EQ(lhs.size(), rhs.size()) << "Mesh " << mesh_id
                                      << " submesh chip count should be equal between MGD versions";
    EXPECT_EQ(lhs.shape(), rhs.shape()) << "Mesh " << mesh_id
                                        << " submesh chip shape should be equal between MGD versions";

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
    MeshGraph mesh_graph_desc(tg_mesh_graph_desc_2_path.string());
    EXPECT_EQ(
        mesh_graph_desc.get_coord_range(MeshId{0}, MeshHostRankId(0)),
        MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(0, 0)));
    EXPECT_EQ(
        mesh_graph_desc.get_coord_range(MeshId{1}, MeshHostRankId(0)),
        MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(0, 0)));
    EXPECT_EQ(
        mesh_graph_desc.get_coord_range(MeshId{2}, MeshHostRankId(0)),
        MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(0, 0)));
    EXPECT_EQ(
        mesh_graph_desc.get_coord_range(MeshId{3}, MeshHostRankId(0)),
        MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(0, 0)));
    EXPECT_EQ(
        mesh_graph_desc.get_coord_range(MeshId{4}, MeshHostRankId(0)),
        MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(3, 7)));
}

TEST(MeshGraphValidation, TestTGMeshGraphInitConsistencyCheckMGD2) {
    // MGD 1.0 Path
    const std::filesystem::path tg_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/tg_mesh_graph_descriptor.yaml";
    MeshGraph mesh_graph(tg_mesh_graph_desc_path.string());

    // MGD 2.0 Path
    const std::filesystem::path tg_mesh_graph_desc_2_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/tg_mesh_graph_descriptor.textproto";
    MeshGraph mesh_graph2(tg_mesh_graph_desc_2_path.string());

    // Compare connectivity deeply
    expect_intra_mesh_connectivity_equal(
        mesh_graph.get_intra_mesh_connectivity(), mesh_graph2.get_intra_mesh_connectivity());
    expect_inter_mesh_connectivity_equal(
        mesh_graph.get_inter_mesh_connectivity(), mesh_graph2.get_inter_mesh_connectivity());

    // Compare mesh graph functions between MGD 1.0 and MGD 2.0 for mesh IDs 0 and 4
    // Test get_host_ranks for mesh 0 and 4
    expect_get_host_ranks_equal(
        mesh_graph.get_host_ranks(tt::tt_fabric::MeshId{0}), mesh_graph2.get_host_ranks(tt::tt_fabric::MeshId{0}), 0);
    expect_get_host_ranks_equal(
        mesh_graph.get_host_ranks(tt::tt_fabric::MeshId{4}), mesh_graph2.get_host_ranks(tt::tt_fabric::MeshId{4}), 4);

    // Test get_chip_ids for mesh 0 and 4 (entire mesh)
    expect_get_chip_ids_equal(
        mesh_graph.get_chip_ids(tt::tt_fabric::MeshId{0}), mesh_graph2.get_chip_ids(tt::tt_fabric::MeshId{0}), 0);
    expect_get_chip_ids_equal(
        mesh_graph.get_chip_ids(tt::tt_fabric::MeshId{4}), mesh_graph2.get_chip_ids(tt::tt_fabric::MeshId{4}), 4);

    // Test get_chip_ids for mesh 0 and 4 (submesh with host rank 0)
    expect_get_chip_ids_submesh_equal(
        mesh_graph.get_chip_ids(tt::tt_fabric::MeshId{0}, tt::tt_fabric::MeshHostRankId{0}),
        mesh_graph2.get_chip_ids(tt::tt_fabric::MeshId{0}, tt::tt_fabric::MeshHostRankId{0}),
        0);
    expect_get_chip_ids_submesh_equal(
        mesh_graph.get_chip_ids(tt::tt_fabric::MeshId{4}, tt::tt_fabric::MeshHostRankId{0}),
        mesh_graph2.get_chip_ids(tt::tt_fabric::MeshId{4}, tt::tt_fabric::MeshHostRankId{0}),
        4);

    // Test get_host_rank_for_chip for mesh 0 and 4 (first chip in each mesh)
    expect_get_host_rank_for_chip_equal(
        mesh_graph.get_host_rank_for_chip(tt::tt_fabric::MeshId{0}, 0),
        mesh_graph2.get_host_rank_for_chip(tt::tt_fabric::MeshId{0}, 0),
        0);
    expect_get_host_rank_for_chip_equal(
        mesh_graph.get_host_rank_for_chip(tt::tt_fabric::MeshId{4}, 0),
        mesh_graph2.get_host_rank_for_chip(tt::tt_fabric::MeshId{4}, 0),
        4);
}

TEST_F(ControlPlaneFixture, TestTGControlPlaneInitMGD2) {
    const std::filesystem::path tg_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/tg_mesh_graph_descriptor.textproto";
    [[maybe_unused]] auto control_plane = make_control_plane(tg_mesh_graph_desc_path);
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
        EXPECT_FALSE(path.empty());
    }
}

TEST(MeshGraphValidation, TestT3kMeshGraphInitMGD2) {
    const std::filesystem::path t3k_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.textproto";
    MeshGraph mesh_graph_desc(t3k_mesh_graph_desc_path.string());
    EXPECT_EQ(
        mesh_graph_desc.get_coord_range(MeshId{0}, MeshHostRankId(0)),
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

TEST_F(ControlPlaneFixture, TestSingleGalaxyControlPlaneInitMGD2) {
    const std::filesystem::path single_galaxy_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/single_galaxy_mesh_graph_descriptor.textproto";
    [[maybe_unused]] auto control_plane = make_control_plane(single_galaxy_mesh_graph_desc_path.string());
}

TEST(MeshGraphValidation, TestT3kDualHostMeshGraphMGD2) {
    const std::filesystem::path t3k_dual_host_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_dual_host_mesh_graph_descriptor.textproto";
    tt_fabric::MeshGraph mesh_graph(t3k_dual_host_mesh_graph_desc_path.string());

    EXPECT_THAT(mesh_graph.get_mesh_ids(), ElementsAre(MeshId{0}));

    // Check host ranks by accessing the values vector
    const auto& host_ranks = mesh_graph.get_host_ranks(MeshId{0});
    EXPECT_EQ(host_ranks, MeshContainer<MeshHostRankId>(MeshShape(1, 2), {MeshHostRankId(0), MeshHostRankId(1)}));

    EXPECT_EQ(mesh_graph.get_mesh_shape(MeshId{0}), MeshShape(2, 4));
    EXPECT_EQ(mesh_graph.get_mesh_shape(MeshId{0}, MeshHostRankId(0)), MeshShape(2, 2));
    EXPECT_EQ(mesh_graph.get_mesh_shape(MeshId{0}, MeshHostRankId(1)), MeshShape(2, 2));

    EXPECT_EQ(mesh_graph.get_coord_range(MeshId{0}), MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(1, 3)));
    EXPECT_EQ(
        mesh_graph.get_coord_range(MeshId{0}, MeshHostRankId(0)),
        MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(1, 1)));
    EXPECT_EQ(
        mesh_graph.get_coord_range(MeshId{0}, MeshHostRankId(1)),
        MeshCoordinateRange(MeshCoordinate(0, 2), MeshCoordinate(1, 3)));

    EXPECT_THAT(mesh_graph.get_mesh_ids(), ElementsAre(MeshId{0}));

    EXPECT_EQ(
        mesh_graph.get_chip_ids(MeshId{0}),
        MeshContainer<chip_id_t>(MeshShape(2, 4), std::vector<chip_id_t>{0, 1, 2, 3, 4, 5, 6, 7}));
    EXPECT_EQ(
        mesh_graph.get_chip_ids(MeshId{0}, MeshHostRankId(0)),
        MeshContainer<chip_id_t>(MeshShape(2, 2), std::vector<chip_id_t>{0, 1, 4, 5}));
    EXPECT_EQ(
        mesh_graph.get_chip_ids(MeshId{0}, MeshHostRankId(1)),
        MeshContainer<chip_id_t>(MeshShape(2, 2), std::vector<chip_id_t>{2, 3, 6, 7}));
}

TEST(MeshGraphValidation, TestT3k2x2MeshGraphMGD2) {
    const std::filesystem::path t3k_2x2_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_2x2_mesh_graph_descriptor.textproto";
    tt_fabric::MeshGraph mesh_graph(t3k_2x2_mesh_graph_desc_path.string());

    // This configuration has two meshes (id 0 and id 1)
    EXPECT_THAT(mesh_graph.get_mesh_ids(), ElementsAre(MeshId{0}, MeshId{1}));

    // Check host ranks for mesh 0 - single host rank 0
    const auto& host_ranks_mesh0 = mesh_graph.get_host_ranks(MeshId{0});
    EXPECT_EQ(host_ranks_mesh0, MeshContainer<MeshHostRankId>(MeshShape(1, 1), {MeshHostRankId(0)}));

    // Check host ranks for mesh 1 - single host rank 0
    const auto& host_ranks_mesh1 = mesh_graph.get_host_ranks(MeshId{1});
    EXPECT_EQ(host_ranks_mesh1, MeshContainer<MeshHostRankId>(MeshShape(1, 1), {MeshHostRankId(0)}));

    // Each mesh has a 2x2 board topology
    EXPECT_EQ(mesh_graph.get_mesh_shape(MeshId{0}), MeshShape(2, 2));
    EXPECT_EQ(mesh_graph.get_mesh_shape(MeshId{1}), MeshShape(2, 2));

    // Since there's only one host rank per mesh, mesh shape should be same
    EXPECT_EQ(mesh_graph.get_mesh_shape(MeshId{0}, MeshHostRankId(0)), MeshShape(2, 2));
    EXPECT_EQ(mesh_graph.get_mesh_shape(MeshId{1}, MeshHostRankId(0)), MeshShape(2, 2));

    // Check coordinate ranges
    EXPECT_EQ(mesh_graph.get_coord_range(MeshId{0}), MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(1, 1)));
    EXPECT_EQ(mesh_graph.get_coord_range(MeshId{1}), MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(1, 1)));

    // Since each mesh has only one host rank, the coord range should be the same
    EXPECT_EQ(
        mesh_graph.get_coord_range(MeshId{0}, MeshHostRankId(0)),
        MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(1, 1)));
    EXPECT_EQ(
        mesh_graph.get_coord_range(MeshId{1}, MeshHostRankId(0)),
        MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(1, 1)));

    // Check chip IDs - each mesh has 4 chips (2x2)
    EXPECT_EQ(
        mesh_graph.get_chip_ids(MeshId{0}),
        MeshContainer<chip_id_t>(MeshShape(2, 2), std::vector<chip_id_t>{0, 1, 2, 3}));
    EXPECT_EQ(
        mesh_graph.get_chip_ids(MeshId{1}),
        MeshContainer<chip_id_t>(MeshShape(2, 2), std::vector<chip_id_t>{0, 1, 2, 3}));

    // Check chip IDs per host rank
    EXPECT_EQ(
        mesh_graph.get_chip_ids(MeshId{0}, MeshHostRankId(0)),
        MeshContainer<chip_id_t>(MeshShape(2, 2), std::vector<chip_id_t>{0, 1, 2, 3}));
    EXPECT_EQ(
        mesh_graph.get_chip_ids(MeshId{1}, MeshHostRankId(0)),
        MeshContainer<chip_id_t>(MeshShape(2, 2), std::vector<chip_id_t>{0, 1, 2, 3}));
}

TEST(MeshGraphValidation, TestGetHostRankForChipMGD2) {
    // Test with dual host T3K configuration
    const std::filesystem::path t3k_dual_host_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_dual_host_mesh_graph_descriptor.textproto";
    tt_fabric::MeshGraph mesh_graph(t3k_dual_host_mesh_graph_desc_path.string());

    // Test valid chips for mesh 0
    // Based on the dual host configuration:
    // Host rank 0 controls chips 0, 1, 4, 5 (left board)
    // Host rank 1 controls chips 2, 3, 6, 7 (right board)
    EXPECT_EQ(mesh_graph.get_host_rank_for_chip(MeshId{0}, 0), MeshHostRankId(0));
    EXPECT_EQ(mesh_graph.get_host_rank_for_chip(MeshId{0}, 1), MeshHostRankId(0));
    EXPECT_EQ(mesh_graph.get_host_rank_for_chip(MeshId{0}, 4), MeshHostRankId(0));
    EXPECT_EQ(mesh_graph.get_host_rank_for_chip(MeshId{0}, 5), MeshHostRankId(0));

    EXPECT_EQ(mesh_graph.get_host_rank_for_chip(MeshId{0}, 2), MeshHostRankId(1));
    EXPECT_EQ(mesh_graph.get_host_rank_for_chip(MeshId{0}, 3), MeshHostRankId(1));
    EXPECT_EQ(mesh_graph.get_host_rank_for_chip(MeshId{0}, 6), MeshHostRankId(1));
    EXPECT_EQ(mesh_graph.get_host_rank_for_chip(MeshId{0}, 7), MeshHostRankId(1));

    // Test invalid chip IDs (out of range)
    EXPECT_EQ(mesh_graph.get_host_rank_for_chip(MeshId{0}, 8), std::nullopt);
    EXPECT_EQ(mesh_graph.get_host_rank_for_chip(MeshId{0}, 100), std::nullopt);

    // Test invalid mesh ID
    EXPECT_EQ(mesh_graph.get_host_rank_for_chip(MeshId{1}, 0), std::nullopt);

    // Test with single host T3K configuration
    const std::filesystem::path t3k_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.textproto";
    auto mesh_graph_single_host = tt_fabric::MeshGraph(t3k_mesh_graph_desc_path.string());

    // In single host configuration, all chips should belong to host rank 0
    for (chip_id_t chip_id = 0; chip_id < 8; chip_id++) {
        EXPECT_EQ(mesh_graph_single_host.get_host_rank_for_chip(MeshId{0}, chip_id), MeshHostRankId(0));
    }

    // Test with 2x2 configuration (two separate meshes)
    const std::filesystem::path t3k_2x2_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_2x2_mesh_graph_descriptor.textproto";
    auto mesh_graph_2x2 = tt_fabric::MeshGraph(t3k_2x2_mesh_graph_desc_path.string());

    // Each mesh has only one host rank (0)
    for (chip_id_t chip_id = 0; chip_id < 4; chip_id++) {
        EXPECT_EQ(mesh_graph_2x2.get_host_rank_for_chip(MeshId{0}, chip_id), MeshHostRankId(0));
        EXPECT_EQ(mesh_graph_2x2.get_host_rank_for_chip(MeshId{1}, chip_id), MeshHostRankId(0));
    }

    // Test invalid chip IDs for 2x2 configuration
    EXPECT_EQ(mesh_graph_2x2.get_host_rank_for_chip(MeshId{0}, 4), std::nullopt);
    EXPECT_EQ(mesh_graph_2x2.get_host_rank_for_chip(MeshId{1}, 4), std::nullopt);
}

namespace single_galaxy_constants {
constexpr std::uint32_t mesh_size = 32;  // 8x4 mesh
constexpr std::uint32_t mesh_row_size = 4;
constexpr std::uint32_t num_ports_per_side = 4;
constexpr std::uint32_t nw_fabric_id = 0;
constexpr std::uint32_t ne_fabric_id = 3;
constexpr std::uint32_t sw_fabric_id = 28;
constexpr std::uint32_t se_fabric_id = 31;
}  // namespace single_galaxy_constants

TEST(MeshGraphValidation, TestSingleGalaxyMeshMGD2) {
    using namespace single_galaxy_constants;
    const std::filesystem::path mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/single_galaxy_mesh_graph_descriptor.textproto";
    MeshGraph mesh_graph(mesh_graph_desc_path.string());
    const auto& intra_mesh_connectivity = mesh_graph.get_intra_mesh_connectivity();

    EXPECT_EQ(intra_mesh_connectivity.size(), 1);

    EXPECT_EQ(intra_mesh_connectivity[0].size(), mesh_size);
    for (std::uint32_t i = 0; i < mesh_size; ++i) {
        int N = i - mesh_row_size;  // North neighbor
        int E = i + 1;              // East neighbor
        int S = i + mesh_row_size;  // South neighbor
        int W = i - 1;              // West neighbor

        auto row = i / mesh_row_size;
        auto col = i % mesh_row_size;
        int N_wrap = (i - mesh_row_size + mesh_size) % mesh_size;
        int E_wrap = row * mesh_row_size + (col + 1) % mesh_row_size;
        int S_wrap = (i + mesh_row_size) % mesh_size;
        int W_wrap = row * mesh_row_size + (col - 1 + mesh_row_size) % mesh_row_size;

        // _wrap represents the wrapped neighbor indices
        // if X == X_wrap, it means that the neighbor is within the mesh and should be connected
        // if not, the neighbour represents a wrap-around connection and should not be present in a MESH
        if (N == N_wrap) {
            EXPECT_EQ(intra_mesh_connectivity[0][i].count(N_wrap), 1);
            EXPECT_EQ(intra_mesh_connectivity[0][i].at(N_wrap).port_direction, RoutingDirection::N);
            EXPECT_EQ(
                intra_mesh_connectivity[0][i].at(N_wrap).connected_chip_ids,
                std::vector<chip_id_t>(num_ports_per_side, N_wrap));
        } else {
            EXPECT_EQ(intra_mesh_connectivity[0][i].count(N_wrap), 0);
        }
        if (E == E_wrap) {
            EXPECT_EQ(intra_mesh_connectivity[0][i].count(E_wrap), 1);
            EXPECT_EQ(intra_mesh_connectivity[0][i].at(E_wrap).port_direction, RoutingDirection::E);
            EXPECT_EQ(
                intra_mesh_connectivity[0][i].at(E_wrap).connected_chip_ids,
                std::vector<chip_id_t>(num_ports_per_side, E_wrap));
        } else {
            EXPECT_EQ(intra_mesh_connectivity[0][i].count(E_wrap), 0);
        }
        if (S == S_wrap) {
            EXPECT_EQ(intra_mesh_connectivity[0][i].count(S_wrap), 1);
            EXPECT_EQ(intra_mesh_connectivity[0][i].at(S_wrap).port_direction, RoutingDirection::S);
            EXPECT_EQ(
                intra_mesh_connectivity[0][i].at(S_wrap).connected_chip_ids,
                std::vector<chip_id_t>(num_ports_per_side, S_wrap));
        } else {
            EXPECT_EQ(intra_mesh_connectivity[0][i].count(S_wrap), 0);
        }
        if (W == W_wrap) {
            EXPECT_EQ(intra_mesh_connectivity[0][i].count(W_wrap), 1);
            EXPECT_EQ(intra_mesh_connectivity[0][i].at(W_wrap).port_direction, RoutingDirection::W);
            EXPECT_EQ(
                intra_mesh_connectivity[0][i].at(W_wrap).connected_chip_ids,
                std::vector<chip_id_t>(num_ports_per_side, W_wrap));
        } else {
            EXPECT_EQ(intra_mesh_connectivity[0][i].count(W_wrap), 0);
        }
    }
}

TEST(RoutingTableValidation, TestSingleGalaxyMeshMGD2) {
    using namespace single_galaxy_constants;
    // Testing XY dimension order routing, if algorithm changes we can remove this test
    const std::filesystem::path mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/single_galaxy_mesh_graph_descriptor.textproto";
    RoutingTableGenerator routing_table_generator(mesh_graph_desc_path.string());
    const auto& intra_mesh_routing_table = routing_table_generator.get_intra_mesh_table();

    EXPECT_EQ(intra_mesh_routing_table[0][nw_fabric_id][nw_fabric_id], RoutingDirection::C);
    EXPECT_EQ(intra_mesh_routing_table[0][nw_fabric_id][ne_fabric_id], RoutingDirection::E);
    EXPECT_EQ(intra_mesh_routing_table[0][nw_fabric_id][sw_fabric_id], RoutingDirection::S);
    EXPECT_EQ(intra_mesh_routing_table[0][nw_fabric_id][se_fabric_id], RoutingDirection::S);

    EXPECT_EQ(intra_mesh_routing_table[0][ne_fabric_id][nw_fabric_id], RoutingDirection::W);
    EXPECT_EQ(intra_mesh_routing_table[0][ne_fabric_id][ne_fabric_id], RoutingDirection::C);
    EXPECT_EQ(intra_mesh_routing_table[0][ne_fabric_id][sw_fabric_id], RoutingDirection::S);
    EXPECT_EQ(intra_mesh_routing_table[0][ne_fabric_id][se_fabric_id], RoutingDirection::S);

    EXPECT_EQ(intra_mesh_routing_table[0][sw_fabric_id][nw_fabric_id], RoutingDirection::N);
    EXPECT_EQ(intra_mesh_routing_table[0][sw_fabric_id][ne_fabric_id], RoutingDirection::N);
    EXPECT_EQ(intra_mesh_routing_table[0][sw_fabric_id][sw_fabric_id], RoutingDirection::C);
    EXPECT_EQ(intra_mesh_routing_table[0][sw_fabric_id][se_fabric_id], RoutingDirection::E);

    EXPECT_EQ(intra_mesh_routing_table[0][se_fabric_id][nw_fabric_id], RoutingDirection::N);
    EXPECT_EQ(intra_mesh_routing_table[0][se_fabric_id][ne_fabric_id], RoutingDirection::N);
    EXPECT_EQ(intra_mesh_routing_table[0][se_fabric_id][sw_fabric_id], RoutingDirection::W);
    EXPECT_EQ(intra_mesh_routing_table[0][se_fabric_id][se_fabric_id], RoutingDirection::C);
}

TEST(MeshGraphValidation, TestSingleGalaxyTorusXYMGD2) {
    using namespace single_galaxy_constants;
    const std::filesystem::path mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/single_galaxy_torus_xy_graph_descriptor.textproto";
    MeshGraph mesh_graph(mesh_graph_desc_path.string());
    const auto& intra_mesh_connectivity = mesh_graph.get_intra_mesh_connectivity();

    EXPECT_EQ(intra_mesh_connectivity.size(), 1);

    EXPECT_EQ(intra_mesh_connectivity[0].size(), mesh_size);
    for (std::uint32_t i = 0; i < mesh_size; ++i) {
        auto row = i / mesh_row_size;
        auto col = i % mesh_row_size;
        int N_wrap = (i - mesh_row_size + mesh_size) % mesh_size;
        int E_wrap = row * mesh_row_size + (col + 1) % mesh_row_size;
        int S_wrap = (i + mesh_row_size) % mesh_size;
        int W_wrap = row * mesh_row_size + (col - 1 + mesh_row_size) % mesh_row_size;

        // _wrap represents the wrapped neighbor indices
        // check all neighbors including wrap-around connections are present in TORUS_XY
        EXPECT_EQ(intra_mesh_connectivity[0][i].count(N_wrap), 1);
        EXPECT_EQ(intra_mesh_connectivity[0][i].at(N_wrap).port_direction, RoutingDirection::N);
        EXPECT_EQ(
            intra_mesh_connectivity[0][i].at(N_wrap).connected_chip_ids,
            std::vector<chip_id_t>(num_ports_per_side, N_wrap));
        EXPECT_EQ(intra_mesh_connectivity[0][i].count(E_wrap), 1);
        EXPECT_EQ(intra_mesh_connectivity[0][i].at(E_wrap).port_direction, RoutingDirection::E);
        EXPECT_EQ(
            intra_mesh_connectivity[0][i].at(E_wrap).connected_chip_ids,
            std::vector<chip_id_t>(num_ports_per_side, E_wrap));
        EXPECT_EQ(intra_mesh_connectivity[0][i].count(S_wrap), 1);
        EXPECT_EQ(intra_mesh_connectivity[0][i].at(S_wrap).port_direction, RoutingDirection::S);
        EXPECT_EQ(
            intra_mesh_connectivity[0][i].at(S_wrap).connected_chip_ids,
            std::vector<chip_id_t>(num_ports_per_side, S_wrap));
        EXPECT_EQ(intra_mesh_connectivity[0][i].count(W_wrap), 1);
        EXPECT_EQ(intra_mesh_connectivity[0][i].at(W_wrap).port_direction, RoutingDirection::W);
        EXPECT_EQ(
            intra_mesh_connectivity[0][i].at(W_wrap).connected_chip_ids,
            std::vector<chip_id_t>(num_ports_per_side, W_wrap));
    }
}

TEST(RoutingTableValidation, TestSingleGalaxyTorusXYMGD2) {
    using namespace single_galaxy_constants;
    // Testing XY dimension order routing, if algorithm changes we can remove this test
    const std::filesystem::path mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/single_galaxy_torus_xy_graph_descriptor.textproto";
    RoutingTableGenerator routing_table_generator(mesh_graph_desc_path.string());
    const auto& intra_mesh_routing_table = routing_table_generator.get_intra_mesh_table();

    EXPECT_EQ(intra_mesh_routing_table[0][nw_fabric_id][nw_fabric_id], RoutingDirection::C);
    EXPECT_EQ(intra_mesh_routing_table[0][nw_fabric_id][ne_fabric_id], RoutingDirection::W);
    EXPECT_EQ(intra_mesh_routing_table[0][nw_fabric_id][sw_fabric_id], RoutingDirection::N);
    EXPECT_EQ(intra_mesh_routing_table[0][nw_fabric_id][se_fabric_id], RoutingDirection::N);

    EXPECT_EQ(intra_mesh_routing_table[0][ne_fabric_id][nw_fabric_id], RoutingDirection::E);
    EXPECT_EQ(intra_mesh_routing_table[0][ne_fabric_id][ne_fabric_id], RoutingDirection::C);
    EXPECT_EQ(intra_mesh_routing_table[0][ne_fabric_id][sw_fabric_id], RoutingDirection::N);
    EXPECT_EQ(intra_mesh_routing_table[0][ne_fabric_id][se_fabric_id], RoutingDirection::N);

    EXPECT_EQ(intra_mesh_routing_table[0][sw_fabric_id][nw_fabric_id], RoutingDirection::S);
    EXPECT_EQ(intra_mesh_routing_table[0][sw_fabric_id][ne_fabric_id], RoutingDirection::S);
    EXPECT_EQ(intra_mesh_routing_table[0][sw_fabric_id][sw_fabric_id], RoutingDirection::C);
    EXPECT_EQ(intra_mesh_routing_table[0][sw_fabric_id][se_fabric_id], RoutingDirection::W);

    EXPECT_EQ(intra_mesh_routing_table[0][se_fabric_id][nw_fabric_id], RoutingDirection::S);
    EXPECT_EQ(intra_mesh_routing_table[0][se_fabric_id][ne_fabric_id], RoutingDirection::S);
    EXPECT_EQ(intra_mesh_routing_table[0][se_fabric_id][sw_fabric_id], RoutingDirection::E);
    EXPECT_EQ(intra_mesh_routing_table[0][se_fabric_id][se_fabric_id], RoutingDirection::C);
}

TEST(MeshGraphValidation, TestSingleGalaxyTorusXMGD2) {
    using namespace single_galaxy_constants;
    const std::filesystem::path mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/single_galaxy_torus_x_graph_descriptor.textproto";
    MeshGraph mesh_graph(mesh_graph_desc_path.string());
    const auto& intra_mesh_connectivity = mesh_graph.get_intra_mesh_connectivity();

    EXPECT_EQ(intra_mesh_connectivity.size(), 1);

    EXPECT_EQ(intra_mesh_connectivity[0].size(), mesh_size);
    for (std::uint32_t i = 0; i < mesh_size; ++i) {
        int N = i - mesh_row_size;  // North neighbor
        int S = i + mesh_row_size;  // South neighbor
        auto row = i / mesh_row_size;
        auto col = i % mesh_row_size;
        int N_wrap = (i - mesh_row_size + mesh_size) % mesh_size;
        int E_wrap = row * mesh_row_size + (col + 1) % mesh_row_size;
        int S_wrap = (i + mesh_row_size) % mesh_size;
        int W_wrap = row * mesh_row_size + (col - 1 + mesh_row_size) % mesh_row_size;

        // _wrap represents the wrapped neighbor indices
        // if X == X_wrap, it means that the neighbor is within the mesh and should be connected
        // if not, the neighbour represents a wrap-around connection and should not be present
        // in a TORUS_X configuration, we expect wrap around for E/W directions
        EXPECT_EQ(intra_mesh_connectivity[0][i].count(E_wrap), 1);
        EXPECT_EQ(intra_mesh_connectivity[0][i].at(E_wrap).port_direction, RoutingDirection::E);
        EXPECT_EQ(
            intra_mesh_connectivity[0][i].at(E_wrap).connected_chip_ids,
            std::vector<chip_id_t>(num_ports_per_side, E_wrap));
        EXPECT_EQ(intra_mesh_connectivity[0][i].count(W_wrap), 1);
        EXPECT_EQ(intra_mesh_connectivity[0][i].at(W_wrap).port_direction, RoutingDirection::W);
        EXPECT_EQ(
            intra_mesh_connectivity[0][i].at(W_wrap).connected_chip_ids,
            std::vector<chip_id_t>(num_ports_per_side, W_wrap));
        if (N == N_wrap) {
            EXPECT_EQ(intra_mesh_connectivity[0][i].count(N_wrap), 1);
            EXPECT_EQ(intra_mesh_connectivity[0][i].at(N_wrap).port_direction, RoutingDirection::N);
            EXPECT_EQ(
                intra_mesh_connectivity[0][i].at(N_wrap).connected_chip_ids,
                std::vector<chip_id_t>(num_ports_per_side, N_wrap));
        } else {
            EXPECT_EQ(intra_mesh_connectivity[0][i].count(N_wrap), 0);
        }
        if (S == S_wrap) {
            EXPECT_EQ(intra_mesh_connectivity[0][i].count(S_wrap), 1);
            EXPECT_EQ(intra_mesh_connectivity[0][i].at(S_wrap).port_direction, RoutingDirection::S);
            EXPECT_EQ(
                intra_mesh_connectivity[0][i].at(S_wrap).connected_chip_ids,
                std::vector<chip_id_t>(num_ports_per_side, S_wrap));
        } else {
            EXPECT_EQ(intra_mesh_connectivity[0][i].count(S_wrap), 0);
        }
    }
}

TEST(RoutingTableValidation, TestSingleGalaxyTorusXMGD2) {
    using namespace single_galaxy_constants;
    // Testing XY dimension order routing, if algorithm changes we can remove this test
    const std::filesystem::path mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/single_galaxy_torus_x_graph_descriptor.textproto";
    RoutingTableGenerator routing_table_generator(mesh_graph_desc_path.string());
    const auto& intra_mesh_routing_table = routing_table_generator.get_intra_mesh_table();

    EXPECT_EQ(intra_mesh_routing_table[0][nw_fabric_id][nw_fabric_id], RoutingDirection::C);
    EXPECT_EQ(intra_mesh_routing_table[0][nw_fabric_id][ne_fabric_id], RoutingDirection::W);
    EXPECT_EQ(intra_mesh_routing_table[0][nw_fabric_id][sw_fabric_id], RoutingDirection::S);
    EXPECT_EQ(intra_mesh_routing_table[0][nw_fabric_id][se_fabric_id], RoutingDirection::S);

    EXPECT_EQ(intra_mesh_routing_table[0][ne_fabric_id][nw_fabric_id], RoutingDirection::E);
    EXPECT_EQ(intra_mesh_routing_table[0][ne_fabric_id][ne_fabric_id], RoutingDirection::C);
    EXPECT_EQ(intra_mesh_routing_table[0][ne_fabric_id][sw_fabric_id], RoutingDirection::S);
    EXPECT_EQ(intra_mesh_routing_table[0][ne_fabric_id][se_fabric_id], RoutingDirection::S);

    EXPECT_EQ(intra_mesh_routing_table[0][sw_fabric_id][nw_fabric_id], RoutingDirection::N);
    EXPECT_EQ(intra_mesh_routing_table[0][sw_fabric_id][ne_fabric_id], RoutingDirection::N);
    EXPECT_EQ(intra_mesh_routing_table[0][sw_fabric_id][sw_fabric_id], RoutingDirection::C);
    EXPECT_EQ(intra_mesh_routing_table[0][sw_fabric_id][se_fabric_id], RoutingDirection::W);

    EXPECT_EQ(intra_mesh_routing_table[0][se_fabric_id][nw_fabric_id], RoutingDirection::N);
    EXPECT_EQ(intra_mesh_routing_table[0][se_fabric_id][ne_fabric_id], RoutingDirection::N);
    EXPECT_EQ(intra_mesh_routing_table[0][se_fabric_id][sw_fabric_id], RoutingDirection::E);
    EXPECT_EQ(intra_mesh_routing_table[0][se_fabric_id][se_fabric_id], RoutingDirection::C);
}

TEST(MeshGraphValidation, TestSingleGalaxyTorusYMGD2) {
    using namespace single_galaxy_constants;
    const std::filesystem::path mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/single_galaxy_torus_y_graph_descriptor.textproto";
    MeshGraph mesh_graph(mesh_graph_desc_path.string());
    const auto& intra_mesh_connectivity = mesh_graph.get_intra_mesh_connectivity();

    EXPECT_EQ(intra_mesh_connectivity.size(), 1);

    EXPECT_EQ(intra_mesh_connectivity[0].size(), mesh_size);
    for (std::uint32_t i = 0; i < mesh_size; ++i) {
        int E = i + 1;  // East neighbor
        int W = i - 1;  // West neighbor
        auto row = i / mesh_row_size;
        auto col = i % mesh_row_size;
        int N_wrap = (i - mesh_row_size + mesh_size) % mesh_size;
        int E_wrap = row * mesh_row_size + (col + 1) % mesh_row_size;
        int S_wrap = (i + mesh_row_size) % mesh_size;
        int W_wrap = row * mesh_row_size + (col - 1 + mesh_row_size) % mesh_row_size;

        // _wrap represents the wrapped neighbor indices
        // if X == X_wrap, it means that the neighbor is within the mesh and should be connected
        // if not, the neighbour represents a wrap-around connection and should not be present
        // in a TORUS_Y configuration, we expect wrap around for N/S directions
        EXPECT_EQ(intra_mesh_connectivity[0][i].count(N_wrap), 1);
        EXPECT_EQ(intra_mesh_connectivity[0][i].at(N_wrap).port_direction, RoutingDirection::N);
        EXPECT_EQ(
            intra_mesh_connectivity[0][i].at(N_wrap).connected_chip_ids,
            std::vector<chip_id_t>(num_ports_per_side, N_wrap));

        EXPECT_EQ(intra_mesh_connectivity[0][i].count(S_wrap), 1);
        EXPECT_EQ(intra_mesh_connectivity[0][i].at(S_wrap).port_direction, RoutingDirection::S);
        EXPECT_EQ(
            intra_mesh_connectivity[0][i].at(S_wrap).connected_chip_ids,
            std::vector<chip_id_t>(num_ports_per_side, S_wrap));
        if (E == E_wrap) {
            EXPECT_EQ(intra_mesh_connectivity[0][i].count(E_wrap), 1);
            EXPECT_EQ(intra_mesh_connectivity[0][i].at(E_wrap).port_direction, RoutingDirection::E);
            EXPECT_EQ(
                intra_mesh_connectivity[0][i].at(E_wrap).connected_chip_ids,
                std::vector<chip_id_t>(num_ports_per_side, E_wrap));
        } else {
            EXPECT_EQ(intra_mesh_connectivity[0][i].count(E_wrap), 0);
        }
        if (W == W_wrap) {
            EXPECT_EQ(intra_mesh_connectivity[0][i].count(W_wrap), 1);
            EXPECT_EQ(intra_mesh_connectivity[0][i].at(W_wrap).port_direction, RoutingDirection::W);
            EXPECT_EQ(
                intra_mesh_connectivity[0][i].at(W_wrap).connected_chip_ids,
                std::vector<chip_id_t>(num_ports_per_side, W_wrap));
        } else {
            EXPECT_EQ(intra_mesh_connectivity[0][i].count(W_wrap), 0);
        }
    }
}

TEST(RoutingTableValidation, TestSingleGalaxyTorusYMGD2) {
    using namespace single_galaxy_constants;
    // Testing XY dimension order routing, if algorithm changes we can remove this test
    const std::filesystem::path mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/single_galaxy_torus_y_graph_descriptor.textproto";
    RoutingTableGenerator routing_table_generator(mesh_graph_desc_path.string());
    const auto& intra_mesh_routing_table = routing_table_generator.get_intra_mesh_table();

    EXPECT_EQ(intra_mesh_routing_table[0][nw_fabric_id][nw_fabric_id], RoutingDirection::C);
    EXPECT_EQ(intra_mesh_routing_table[0][nw_fabric_id][ne_fabric_id], RoutingDirection::E);
    EXPECT_EQ(intra_mesh_routing_table[0][nw_fabric_id][sw_fabric_id], RoutingDirection::N);
    EXPECT_EQ(intra_mesh_routing_table[0][nw_fabric_id][se_fabric_id], RoutingDirection::N);

    EXPECT_EQ(intra_mesh_routing_table[0][ne_fabric_id][nw_fabric_id], RoutingDirection::W);
    EXPECT_EQ(intra_mesh_routing_table[0][ne_fabric_id][ne_fabric_id], RoutingDirection::C);
    EXPECT_EQ(intra_mesh_routing_table[0][ne_fabric_id][sw_fabric_id], RoutingDirection::N);
    EXPECT_EQ(intra_mesh_routing_table[0][ne_fabric_id][se_fabric_id], RoutingDirection::N);

    EXPECT_EQ(intra_mesh_routing_table[0][sw_fabric_id][nw_fabric_id], RoutingDirection::S);
    EXPECT_EQ(intra_mesh_routing_table[0][sw_fabric_id][ne_fabric_id], RoutingDirection::S);
    EXPECT_EQ(intra_mesh_routing_table[0][sw_fabric_id][sw_fabric_id], RoutingDirection::C);
    EXPECT_EQ(intra_mesh_routing_table[0][sw_fabric_id][se_fabric_id], RoutingDirection::E);

    EXPECT_EQ(intra_mesh_routing_table[0][se_fabric_id][nw_fabric_id], RoutingDirection::S);
    EXPECT_EQ(intra_mesh_routing_table[0][se_fabric_id][ne_fabric_id], RoutingDirection::S);
    EXPECT_EQ(intra_mesh_routing_table[0][se_fabric_id][sw_fabric_id], RoutingDirection::W);
    EXPECT_EQ(intra_mesh_routing_table[0][se_fabric_id][se_fabric_id], RoutingDirection::C);
}

TEST(MeshGraphValidation, TestDualGalaxyMeshGraphMGD2) {
    const std::filesystem::path mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/dual_galaxy_mesh_graph_descriptor.textproto";
    MeshGraph mesh_graph_desc(mesh_graph_desc_path.string());
    EXPECT_EQ(
        mesh_graph_desc.get_coord_range(MeshId{0}, MeshHostRankId(0)),
        MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(7, 3)));
    EXPECT_EQ(
        mesh_graph_desc.get_coord_range(MeshId{0}, MeshHostRankId(1)),
        MeshCoordinateRange(MeshCoordinate(0, 4), MeshCoordinate(7, 7)));
}

// Black hole tests for p150, p100, p150 x8 - MGD2
TEST(MeshGraphValidation, TestP150BlackHoleMeshGraphMGD2) {
    const std::filesystem::path p150_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto";
    MeshGraph mesh_graph(p150_mesh_graph_desc_path.string());

    EXPECT_THAT(mesh_graph.get_mesh_ids(), ElementsAre(MeshId{0}));
    EXPECT_EQ(mesh_graph.get_mesh_shape(MeshId{0}), MeshShape(1, 1));
    EXPECT_EQ(mesh_graph.get_coord_range(MeshId{0}), MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(0, 0)));

    // Check chip IDs - single chip
    EXPECT_EQ(mesh_graph.get_chip_ids(MeshId{0}), MeshContainer<chip_id_t>(MeshShape(1, 1), std::vector<chip_id_t>{0}));
}

TEST_F(ControlPlaneFixture, TestP150BlackHoleControlPlaneInitMGD2) {
    const std::filesystem::path p150_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto";
    [[maybe_unused]] auto control_plane = make_control_plane(p150_mesh_graph_desc_path);
}

TEST(MeshGraphValidation, TestP100BlackHoleMeshGraphMGD2) {
    const std::filesystem::path p100_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/p100_mesh_graph_descriptor.textproto";
    MeshGraph mesh_graph(p100_mesh_graph_desc_path.string());

    EXPECT_THAT(mesh_graph.get_mesh_ids(), ElementsAre(MeshId{0}));
    EXPECT_EQ(mesh_graph.get_mesh_shape(MeshId{0}), MeshShape(1, 1));
    EXPECT_EQ(mesh_graph.get_coord_range(MeshId{0}), MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(0, 0)));

    // Check chip IDs - single chip
    EXPECT_EQ(mesh_graph.get_chip_ids(MeshId{0}), MeshContainer<chip_id_t>(MeshShape(1, 1), std::vector<chip_id_t>{0}));
}

TEST_F(ControlPlaneFixture, TestP100BlackHoleControlPlaneInitMGD2) {
    const std::filesystem::path p100_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/p100_mesh_graph_descriptor.textproto";
    [[maybe_unused]] auto control_plane = make_control_plane(p100_mesh_graph_desc_path);
}

TEST(MeshGraphValidation, TestP150X8BlackHoleMeshGraphMGD2) {
    const std::filesystem::path p150_x8_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/p150_x8_mesh_graph_descriptor.textproto";
    MeshGraph mesh_graph(p150_x8_mesh_graph_desc_path.string());

    EXPECT_THAT(mesh_graph.get_mesh_ids(), ElementsAre(MeshId{0}));
    EXPECT_EQ(mesh_graph.get_mesh_shape(MeshId{0}), MeshShape(2, 4));
    EXPECT_EQ(mesh_graph.get_coord_range(MeshId{0}), MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(1, 3)));

    // Check chip IDs - 8 chips in 2x4 configuration
    EXPECT_EQ(
        mesh_graph.get_chip_ids(MeshId{0}),
        MeshContainer<chip_id_t>(MeshShape(2, 4), std::vector<chip_id_t>{0, 1, 2, 3, 4, 5, 6, 7}));
}

TEST_F(ControlPlaneFixture, TestP150X8BlackHoleControlPlaneInitMGD2) {
    const std::filesystem::path p150_x8_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/p150_x8_mesh_graph_descriptor.textproto";
    [[maybe_unused]] auto control_plane = make_control_plane(p150_x8_mesh_graph_desc_path);
}

TEST_F(ControlPlaneFixture, TestP150X8BlackHoleFabricRoutesMGD2) {
    const std::filesystem::path p150_x8_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/p150_x8_mesh_graph_descriptor.textproto";
    auto control_plane = make_control_plane(p150_x8_mesh_graph_desc_path);

    // Test routing between different chips in the 2x4 mesh
    auto valid_chans = control_plane->get_valid_eth_chans_on_routing_plane(FabricNodeId(MeshId{0}, 0), 0);
    EXPECT_GT(valid_chans.size(), 0);
    for (auto chan : valid_chans) {
        auto path = control_plane->get_fabric_route(FabricNodeId(MeshId{0}, 0), FabricNodeId(MeshId{0}, 7), chan);
        EXPECT_EQ(path.size() > 0, true);
    }
}

}  // namespace tt::tt_fabric::fabric_router_tests
