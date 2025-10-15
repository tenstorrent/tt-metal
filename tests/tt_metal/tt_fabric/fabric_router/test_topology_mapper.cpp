// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <tt-metalium/mesh_graph.hpp>
#include <tt-metalium/mesh_graph_descriptor.hpp>
#include <tt-metalium/control_plane.hpp>
#include "impl/context/metal_context.hpp"
#include <memory>

#include "tt_metal/fabric/physical_system_descriptor.hpp"
#include "tt_metal/fabric/topology_mapper.hpp"

namespace tt::tt_fabric {

class TopologyMapperTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto distributed_context = tt::tt_metal::MetalContext::instance().get_distributed_context_ptr();
        const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
        const auto& hal = tt::tt_metal::MetalContext::instance().hal();
        const auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
        constexpr bool run_discovery = true;

        physical_system_descriptor_ = std::make_unique<tt::tt_metal::PhysicalSystemDescriptor>(
            cluster.get_driver(), distributed_context, &hal, rtoptions, run_discovery);
    }

    void TearDown() override { physical_system_descriptor_.reset(); }

    std::unique_ptr<tt::tt_metal::PhysicalSystemDescriptor> physical_system_descriptor_;
};

bool contains(const std::vector<tt::tt_metal::AsicID>& asic_ids, const tt::tt_metal::AsicID& asic_id) {
    return std::find(asic_ids.begin(), asic_ids.end(), asic_id) != asic_ids.end();
}

TEST_F(TopologyMapperTest, T3kMeshGraphTest) {
    const std::filesystem::path t3k_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.yaml";

    auto mesh_graph = MeshGraph(t3k_mesh_graph_desc_path.string());

    // Create a local mesh binding for testing
    LocalMeshBinding local_mesh_binding;
    local_mesh_binding.mesh_ids = {MeshId{0}};
    local_mesh_binding.host_rank = MeshHostRankId{0};

    // Test that TopologyMapper can be constructed with valid parameters
    // This is a basic smoke test
    auto topology_mapper = TopologyMapper(mesh_graph, *physical_system_descriptor_, local_mesh_binding);

    // Fabric Node ID layout:
    // 0 1 2 3
    // 4 5 6 7

    auto asic_id_0 = topology_mapper.get_asic_id_from_fabric_node_id(FabricNodeId(MeshId{0}, 0));
    auto asic_id_1 = topology_mapper.get_asic_id_from_fabric_node_id(FabricNodeId(MeshId{0}, 1));
    auto asic_id_3 = topology_mapper.get_asic_id_from_fabric_node_id(FabricNodeId(MeshId{0}, 3));
    auto asic_id_4 = topology_mapper.get_asic_id_from_fabric_node_id(FabricNodeId(MeshId{0}, 4));
    auto asic_id_7 = topology_mapper.get_asic_id_from_fabric_node_id(FabricNodeId(MeshId{0}, 7));
    auto asic_id_6 = topology_mapper.get_asic_id_from_fabric_node_id(FabricNodeId(MeshId{0}, 6));

    // Check fabric node id 0 and 1 are adjacent for mesh 0
    EXPECT_TRUE(contains(physical_system_descriptor_->get_asic_neighbors(asic_id_0), asic_id_1));
    EXPECT_TRUE(contains(physical_system_descriptor_->get_asic_neighbors(asic_id_1), asic_id_0));

    // Check 0 and 4 are adjacent for mesh 0
    EXPECT_TRUE(contains(physical_system_descriptor_->get_asic_neighbors(asic_id_0), asic_id_4));
    EXPECT_TRUE(contains(physical_system_descriptor_->get_asic_neighbors(asic_id_4), asic_id_0));

    // 3 and 7 are adjacent for mesh 0
    EXPECT_TRUE(contains(physical_system_descriptor_->get_asic_neighbors(asic_id_3), asic_id_7));
    EXPECT_TRUE(contains(physical_system_descriptor_->get_asic_neighbors(asic_id_7), asic_id_3));

    // 6 and 7 are adjacent for mesh 0
    EXPECT_TRUE(contains(physical_system_descriptor_->get_asic_neighbors(asic_id_6), asic_id_7));
    EXPECT_TRUE(contains(physical_system_descriptor_->get_asic_neighbors(asic_id_7), asic_id_6));

    // Validate only one host rank for mesh 0
    const MeshId mesh_id{0};
    const auto& host_ranks = topology_mapper.get_host_ranks(mesh_id);
    EXPECT_EQ(host_ranks.size(), 1u);

    // Validate that the full shape and sub shape is 2x4
    MeshShape full_shape = mesh_graph.get_mesh_shape(mesh_id);
    EXPECT_EQ(full_shape, MeshShape(2, 4));
    EXPECT_EQ(topology_mapper.get_mesh_shape(mesh_id), full_shape);

    // Validate that the host rank is 0
    EXPECT_EQ(host_ranks.values(), std::vector<MeshHostRankId>({MeshHostRankId(0)}));
}

TEST_F(TopologyMapperTest, DualGalaxyBigMeshTest) {
    const std::filesystem::path dual_galaxy_big_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/dual_galaxy_mesh_graph_descriptor.yaml";

    auto mesh_graph = MeshGraph(dual_galaxy_big_mesh_graph_desc_path.string());

    // Create a local mesh binding for testing
    LocalMeshBinding local_mesh_binding;
    if (*tt::tt_metal::MetalContext::instance().global_distributed_context().rank() == 0) {
        local_mesh_binding.mesh_ids = {MeshId{0}};
        local_mesh_binding.host_rank = MeshHostRankId{0};
    } else {
        local_mesh_binding.mesh_ids = {MeshId{0}};
        local_mesh_binding.host_rank = MeshHostRankId{0};
    }

    auto topology_mapper = TopologyMapper(mesh_graph, *physical_system_descriptor_, local_mesh_binding);

    // Physical System Descriptor: 8x8
    //  0  1  2  3  4  5  6  7
    //  8  9 10 11 12 13 14 15
    // 16 17 18 19 20 21 22 23  H0
    // 24 25 26 27 28 29 30 31
    // -----------------------
    // 32 33 34 35 36 37 38 39
    // 40 41 42 43 44 45 46 47
    // 48 49 50 51 52 53 54 55 H1
    // 56 57 58 59 60 61 62 63

    // Validate host ranks for selected fabric node IDs
    // Pairs: 56 48, 8 1, 39 31, 35 27
    auto asic_id_56 = topology_mapper.get_asic_id_from_fabric_node_id(FabricNodeId(MeshId{0}, 56));
    auto asic_id_48 = topology_mapper.get_asic_id_from_fabric_node_id(FabricNodeId(MeshId{0}, 48));
    auto asic_id_8 = topology_mapper.get_asic_id_from_fabric_node_id(FabricNodeId(MeshId{0}, 8));
    auto asic_id_0 = topology_mapper.get_asic_id_from_fabric_node_id(FabricNodeId(MeshId{0}, 0));
    auto asic_id_39 = topology_mapper.get_asic_id_from_fabric_node_id(FabricNodeId(MeshId{0}, 39));
    auto asic_id_31 = topology_mapper.get_asic_id_from_fabric_node_id(FabricNodeId(MeshId{0}, 31));
    auto asic_id_35 = topology_mapper.get_asic_id_from_fabric_node_id(FabricNodeId(MeshId{0}, 35));
    auto asic_id_27 = topology_mapper.get_asic_id_from_fabric_node_id(FabricNodeId(MeshId{0}, 27));

    // Check they belong to the right host rank per ASCII map (rows 0-31 => H0, 32-63 => H1)
    EXPECT_EQ(
        1,
        physical_system_descriptor_->get_rank_for_hostname(
            physical_system_descriptor_->get_host_name_for_asic(asic_id_56)));
    EXPECT_EQ(
        1,
        physical_system_descriptor_->get_rank_for_hostname(
            physical_system_descriptor_->get_host_name_for_asic(asic_id_48)));
    EXPECT_EQ(
        0,
        physical_system_descriptor_->get_rank_for_hostname(
            physical_system_descriptor_->get_host_name_for_asic(asic_id_8)));
    EXPECT_EQ(
        0,
        physical_system_descriptor_->get_rank_for_hostname(
            physical_system_descriptor_->get_host_name_for_asic(asic_id_0)));
    EXPECT_EQ(
        1,
        physical_system_descriptor_->get_rank_for_hostname(
            physical_system_descriptor_->get_host_name_for_asic(asic_id_39)));
    EXPECT_EQ(
        0,
        physical_system_descriptor_->get_rank_for_hostname(
            physical_system_descriptor_->get_host_name_for_asic(asic_id_31)));
    EXPECT_EQ(
        1,
        physical_system_descriptor_->get_rank_for_hostname(
            physical_system_descriptor_->get_host_name_for_asic(asic_id_35)));
    EXPECT_EQ(
        0,
        physical_system_descriptor_->get_rank_for_hostname(
            physical_system_descriptor_->get_host_name_for_asic(asic_id_27)));

    // Check for adjacency
    EXPECT_TRUE(contains(physical_system_descriptor_->get_asic_neighbors(asic_id_56), asic_id_48));
    EXPECT_TRUE(contains(physical_system_descriptor_->get_asic_neighbors(asic_id_48), asic_id_56));
    EXPECT_TRUE(contains(physical_system_descriptor_->get_asic_neighbors(asic_id_8), asic_id_0));
    EXPECT_TRUE(contains(physical_system_descriptor_->get_asic_neighbors(asic_id_0), asic_id_8));
    EXPECT_TRUE(contains(physical_system_descriptor_->get_asic_neighbors(asic_id_39), asic_id_31));
    EXPECT_TRUE(contains(physical_system_descriptor_->get_asic_neighbors(asic_id_31), asic_id_39));
    EXPECT_TRUE(contains(physical_system_descriptor_->get_asic_neighbors(asic_id_35), asic_id_27));
    EXPECT_TRUE(contains(physical_system_descriptor_->get_asic_neighbors(asic_id_27), asic_id_35));

    // Check the host ranks are right
    const MeshId mesh_id{0};
    const auto& host_ranks = topology_mapper.get_host_ranks(mesh_id);
    EXPECT_EQ(host_ranks.size(), 2u);

    // Check the full shape and sub shape are right
    MeshShape full_shape = mesh_graph.get_mesh_shape(mesh_id);
    EXPECT_EQ(full_shape, MeshShape(8, 8));
    EXPECT_EQ(topology_mapper.get_mesh_shape(mesh_id), full_shape);

    // Check coord range for host rank 0 is 0,0 4, 7
    EXPECT_EQ(
        topology_mapper.get_coord_range(mesh_id, MeshHostRankId(0)),
        MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(3, 7)));
    // Check coord range for host rank 1 is 4,0 8, 7
    EXPECT_EQ(
        topology_mapper.get_coord_range(mesh_id, MeshHostRankId(1)),
        MeshCoordinateRange(MeshCoordinate(4, 0), MeshCoordinate(7, 7)));

    // Check the host rank for chip 56 is 0
    EXPECT_EQ(topology_mapper.get_host_rank_for_chip(mesh_id, 56), MeshHostRankId(1));
    // Check the host rank for chip 48 is 1
    EXPECT_EQ(topology_mapper.get_host_rank_for_chip(mesh_id, 48), MeshHostRankId(1));
    // Check the host rank for chip 8 is 0
    EXPECT_EQ(topology_mapper.get_host_rank_for_chip(mesh_id, 8), MeshHostRankId(0));
    // Check the host rank for chip 0 is 0
    EXPECT_EQ(topology_mapper.get_host_rank_for_chip(mesh_id, 0), MeshHostRankId(0));
}

TEST_F(TopologyMapperTest, N300MeshGraphTest) {
    const std::filesystem::path n300_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/n300_mesh_graph_descriptor.yaml";

    auto mesh_graph = MeshGraph(n300_mesh_graph_desc_path.string());

    // Create a local mesh binding for testing
    LocalMeshBinding local_mesh_binding;
    local_mesh_binding.mesh_ids = {MeshId{0}};
    local_mesh_binding.host_rank = MeshHostRankId{0};

    auto topology_mapper = TopologyMapper(mesh_graph, *physical_system_descriptor_, local_mesh_binding);

    // Physical System Descriptor: 1x2
    // 0 1

    auto asic_id_0 = topology_mapper.get_asic_id_from_fabric_node_id(FabricNodeId(MeshId{0}, 0));
    auto asic_id_1 = topology_mapper.get_asic_id_from_fabric_node_id(FabricNodeId(MeshId{0}, 1));

    // Check for adjacency
    EXPECT_TRUE(contains(physical_system_descriptor_->get_asic_neighbors(asic_id_0), asic_id_1));
    EXPECT_TRUE(contains(physical_system_descriptor_->get_asic_neighbors(asic_id_1), asic_id_0));

    // Check the host ranks are right
    const MeshId mesh_id{0};
    const auto& host_ranks = topology_mapper.get_host_ranks(mesh_id);
    EXPECT_EQ(host_ranks.size(), 1u);

    // Check the full shape and sub shape are right
    MeshShape full_shape = mesh_graph.get_mesh_shape(mesh_id);
    EXPECT_EQ(full_shape, MeshShape(1, 2));
    EXPECT_EQ(topology_mapper.get_mesh_shape(mesh_id), full_shape);

    // Check coord range for host rank 0 is 0,0 1, 1
    EXPECT_EQ(
        topology_mapper.get_coord_range(mesh_id, MeshHostRankId(0)),
        MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(0, 1)));
}

TEST_F(TopologyMapperTest, T3kMultiMeshTest) {
    // TODO: This test is currently disabled due to lack of support for multi-mesh-per-host systems
    GTEST_SKIP();

    const std::filesystem::path t3k_multimesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_2x2_1x2_1x1_mesh_graph_descriptor.textproto";

    auto mesh_graph = MeshGraph(t3k_multimesh_graph_desc_path.string());

    // Create a local mesh binding for testing
    LocalMeshBinding local_mesh_binding;
    local_mesh_binding.mesh_ids = {MeshId{0}, MeshId{1}, MeshId{2}};
    local_mesh_binding.host_rank = MeshHostRankId{0};

    auto topology_mapper = TopologyMapper(mesh_graph, *physical_system_descriptor_, local_mesh_binding);
}

}  // namespace tt::tt_fabric
