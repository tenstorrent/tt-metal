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
        setenv("TT_METAL_OPERATION_TIMEOUT_SECONDS", "10", 1);

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
        local_mesh_binding.host_rank = MeshHostRankId{1};
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

TEST_F(TopologyMapperTest, P100MeshGraphTest) {
    const std::filesystem::path p100_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/p100_mesh_graph_descriptor.yaml";

    auto mesh_graph = MeshGraph(p100_mesh_graph_desc_path.string());

    // Create a local mesh binding for testing
    LocalMeshBinding local_mesh_binding;
    local_mesh_binding.mesh_ids = {MeshId{0}};
    local_mesh_binding.host_rank = MeshHostRankId{0};

    auto topology_mapper = TopologyMapper(mesh_graph, *physical_system_descriptor_, local_mesh_binding);

    // Single-chip mesh: 1x1 with chip id 0
    const MeshId mesh_id{0};
    EXPECT_EQ(mesh_graph.get_mesh_shape(mesh_id), MeshShape(1, 1));
    EXPECT_EQ(topology_mapper.get_mesh_shape(mesh_id), MeshShape(1, 1));

    const auto& host_ranks = topology_mapper.get_host_ranks(mesh_id);
    EXPECT_EQ(host_ranks.size(), 1u);
    EXPECT_EQ(host_ranks.values(), std::vector<MeshHostRankId>({MeshHostRankId(0)}));

    // Mapping checks for the single fabric node
    auto fabric_node = FabricNodeId(mesh_id, 0);
    auto asic_id = topology_mapper.get_asic_id_from_fabric_node_id(fabric_node);

    // Neighbors should be empty for a single-chip system
    EXPECT_EQ(physical_system_descriptor_->get_asic_neighbors(asic_id).size(), 0u);

    // Bidirectional mappings
    auto phys_chip_id = topology_mapper.get_physical_chip_id_from_fabric_node_id(fabric_node);
    EXPECT_EQ(phys_chip_id, 0);
    EXPECT_EQ(topology_mapper.get_fabric_node_id_from_physical_chip_id(0), fabric_node);
    EXPECT_EQ(topology_mapper.get_physical_chip_id_from_asic_id(asic_id), 0);
    EXPECT_EQ(topology_mapper.get_fabric_node_id_from_asic_id(asic_id), fabric_node);

    // Host rank for chip 0 should be rank 0
    EXPECT_EQ(topology_mapper.get_host_rank_for_chip(mesh_id, 0), MeshHostRankId(0));

    // Chip IDs list
    EXPECT_EQ(topology_mapper.get_chip_ids(mesh_id), MeshContainer<ChipId>(MeshShape(1, 1), std::vector<ChipId>{0}));
}

TEST_F(TopologyMapperTest, BHQB4x4MeshGraphTest) {
    const std::filesystem::path bh_qb_4x4_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/bh_qb_4x4_mesh_graph_descriptor.textproto";

    auto mesh_graph = MeshGraph(bh_qb_4x4_mesh_graph_desc_path.string());

    // Create a local mesh binding for testing
    LocalMeshBinding local_mesh_binding;
    if (*tt::tt_metal::MetalContext::instance().global_distributed_context().rank() == 0) {
        local_mesh_binding.mesh_ids = {MeshId{0}};
        local_mesh_binding.host_rank = MeshHostRankId{0};
    } else if (*tt::tt_metal::MetalContext::instance().global_distributed_context().rank() == 1) {
        local_mesh_binding.mesh_ids = {MeshId{0}};
        local_mesh_binding.host_rank = MeshHostRankId{1};
    } else if (*tt::tt_metal::MetalContext::instance().global_distributed_context().rank() == 2) {
        local_mesh_binding.mesh_ids = {MeshId{0}};
        local_mesh_binding.host_rank = MeshHostRankId{2};
    } else if (*tt::tt_metal::MetalContext::instance().global_distributed_context().rank() == 3) {
        local_mesh_binding.mesh_ids = {MeshId{0}};
        local_mesh_binding.host_rank = MeshHostRankId{3};
    }


    auto topology_mapper = TopologyMapper(mesh_graph, *physical_system_descriptor_, local_mesh_binding);

    // Physical System Descriptor: 4x4 Blackhole mesh
    // 0  1  | 2  3
    // 4  5  | 6  7
    // ------+-------
    // 8  9  | 10 11
    // 12 13 | 14 15

    auto asic_id_0 = topology_mapper.get_asic_id_from_fabric_node_id(FabricNodeId(MeshId{0}, 0));
    auto asic_id_1 = topology_mapper.get_asic_id_from_fabric_node_id(FabricNodeId(MeshId{0}, 1));
    auto asic_id_7 = topology_mapper.get_asic_id_from_fabric_node_id(FabricNodeId(MeshId{0}, 7));
    auto asic_id_4 = topology_mapper.get_asic_id_from_fabric_node_id(FabricNodeId(MeshId{0}, 4));
    auto asic_id_13 = topology_mapper.get_asic_id_from_fabric_node_id(FabricNodeId(MeshId{0}, 13));
    auto asic_id_14 = topology_mapper.get_asic_id_from_fabric_node_id(FabricNodeId(MeshId{0}, 14));
    auto asic_id_15 = topology_mapper.get_asic_id_from_fabric_node_id(FabricNodeId(MeshId{0}, 15));
    auto asic_id_3 = topology_mapper.get_asic_id_from_fabric_node_id(FabricNodeId(MeshId{0}, 3));

    // Check for adjacency
    EXPECT_TRUE(contains(physical_system_descriptor_->get_asic_neighbors(asic_id_0), asic_id_1));
    EXPECT_TRUE(contains(physical_system_descriptor_->get_asic_neighbors(asic_id_1), asic_id_0));
    EXPECT_TRUE(contains(physical_system_descriptor_->get_asic_neighbors(asic_id_7), asic_id_4));
    EXPECT_TRUE(contains(physical_system_descriptor_->get_asic_neighbors(asic_id_4), asic_id_7));
    EXPECT_TRUE(contains(physical_system_descriptor_->get_asic_neighbors(asic_id_13), asic_id_14));
    EXPECT_TRUE(contains(physical_system_descriptor_->get_asic_neighbors(asic_id_14), asic_id_13));
    EXPECT_TRUE(contains(physical_system_descriptor_->get_asic_neighbors(asic_id_15), asic_id_3));
    EXPECT_TRUE(contains(physical_system_descriptor_->get_asic_neighbors(asic_id_3), asic_id_15));

    // Check the host ranks are right
    const MeshId mesh_id{0};
    const auto& host_ranks = topology_mapper.get_host_ranks(mesh_id);
    EXPECT_EQ(host_ranks.size(), 4u);

    // Check the full shape and sub shape are right
    MeshShape full_shape = mesh_graph.get_mesh_shape(mesh_id);
    EXPECT_EQ(full_shape, MeshShape(4, 4));
    EXPECT_EQ(topology_mapper.get_mesh_shape(mesh_id), full_shape);
}

TEST_F(TopologyMapperTest, BHQB4x4StrictReducedMeshGraphTest) {
    const std::filesystem::path bh_qb_4x4_strict_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/bh_qb_4x4_strict_reduced_mesh_graph_descriptor.textproto";

    auto mesh_graph = MeshGraph(bh_qb_4x4_strict_mesh_graph_desc_path.string());

    // Create a local mesh binding for testing
    LocalMeshBinding local_mesh_binding;
    if (*tt::tt_metal::MetalContext::instance().global_distributed_context().rank() == 0) {
        local_mesh_binding.mesh_ids = {MeshId{0}};
        local_mesh_binding.host_rank = MeshHostRankId{0};
    } else if (*tt::tt_metal::MetalContext::instance().global_distributed_context().rank() == 1) {
        local_mesh_binding.mesh_ids = {MeshId{0}};
        local_mesh_binding.host_rank = MeshHostRankId{1};
    } else if (*tt::tt_metal::MetalContext::instance().global_distributed_context().rank() == 2) {
        local_mesh_binding.mesh_ids = {MeshId{0}};
        local_mesh_binding.host_rank = MeshHostRankId{2};
    } else if (*tt::tt_metal::MetalContext::instance().global_distributed_context().rank() == 3) {
        local_mesh_binding.mesh_ids = {MeshId{0}};
        local_mesh_binding.host_rank = MeshHostRankId{3};
    }

    auto topology_mapper = TopologyMapper(mesh_graph, *physical_system_descriptor_, local_mesh_binding);

    // Physical System Descriptor: 4x4 Blackhole mesh
    // 0  1  | 2  3
    // 4  5  | 6  7
    // ------+-------
    // 8  9  | 10 11
    // 12 13 | 14 15

    auto asic_id_0 = topology_mapper.get_asic_id_from_fabric_node_id(FabricNodeId(MeshId{0}, 0));
    auto asic_id_1 = topology_mapper.get_asic_id_from_fabric_node_id(FabricNodeId(MeshId{0}, 1));
    auto asic_id_7 = topology_mapper.get_asic_id_from_fabric_node_id(FabricNodeId(MeshId{0}, 7));
    auto asic_id_4 = topology_mapper.get_asic_id_from_fabric_node_id(FabricNodeId(MeshId{0}, 4));
    auto asic_id_13 = topology_mapper.get_asic_id_from_fabric_node_id(FabricNodeId(MeshId{0}, 13));
    auto asic_id_14 = topology_mapper.get_asic_id_from_fabric_node_id(FabricNodeId(MeshId{0}, 14));
    auto asic_id_15 = topology_mapper.get_asic_id_from_fabric_node_id(FabricNodeId(MeshId{0}, 15));
    auto asic_id_3 = topology_mapper.get_asic_id_from_fabric_node_id(FabricNodeId(MeshId{0}, 3));

    // Check for adjacency
    EXPECT_TRUE(contains(physical_system_descriptor_->get_asic_neighbors(asic_id_0), asic_id_1));
    EXPECT_TRUE(contains(physical_system_descriptor_->get_asic_neighbors(asic_id_1), asic_id_0));
    EXPECT_TRUE(contains(physical_system_descriptor_->get_asic_neighbors(asic_id_7), asic_id_4));
    EXPECT_TRUE(contains(physical_system_descriptor_->get_asic_neighbors(asic_id_4), asic_id_7));
    EXPECT_TRUE(contains(physical_system_descriptor_->get_asic_neighbors(asic_id_13), asic_id_14));
    EXPECT_TRUE(contains(physical_system_descriptor_->get_asic_neighbors(asic_id_14), asic_id_13));
    EXPECT_TRUE(contains(physical_system_descriptor_->get_asic_neighbors(asic_id_15), asic_id_3));
    EXPECT_TRUE(contains(physical_system_descriptor_->get_asic_neighbors(asic_id_3), asic_id_15));

    // Check the host ranks are right
    const MeshId mesh_id{0};
    const auto& host_ranks = topology_mapper.get_host_ranks(mesh_id);
    EXPECT_EQ(host_ranks.size(), 4u);

    // Check the full shape and sub shape are right
    MeshShape full_shape = mesh_graph.get_mesh_shape(mesh_id);
    EXPECT_EQ(full_shape, MeshShape(4, 4));
    EXPECT_EQ(topology_mapper.get_mesh_shape(mesh_id), full_shape);
}

TEST_F(TopologyMapperTest, BHQB4x4RelaxedMeshGraphTest) {
    const std::filesystem::path bh_qb_4x4_relaxed_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/bh_qb_4x4_relaxed_mesh_graph_descriptor.textproto";

    auto mesh_graph = MeshGraph(bh_qb_4x4_relaxed_mesh_graph_desc_path.string());

    // Create a local mesh binding for testing
    LocalMeshBinding local_mesh_binding;
    if (*tt::tt_metal::MetalContext::instance().global_distributed_context().rank() == 0) {
        local_mesh_binding.mesh_ids = {MeshId{0}};
        local_mesh_binding.host_rank = MeshHostRankId{0};
    } else if (*tt::tt_metal::MetalContext::instance().global_distributed_context().rank() == 1) {
        local_mesh_binding.mesh_ids = {MeshId{0}};
        local_mesh_binding.host_rank = MeshHostRankId{1};
    } else if (*tt::tt_metal::MetalContext::instance().global_distributed_context().rank() == 2) {
        local_mesh_binding.mesh_ids = {MeshId{0}};
        local_mesh_binding.host_rank = MeshHostRankId{2};
    } else if (*tt::tt_metal::MetalContext::instance().global_distributed_context().rank() == 3) {
        local_mesh_binding.mesh_ids = {MeshId{0}};
        local_mesh_binding.host_rank = MeshHostRankId{3};
    }

    auto topology_mapper = TopologyMapper(mesh_graph, *physical_system_descriptor_, local_mesh_binding);

    // Physical System Descriptor: 4x4 Blackhole mesh
    // 0  1  | 2  3
    // 4  5  | 6  7
    // ------+-------
    // 8  9  | 10 11
    // 12 13 | 14 15

    auto asic_id_0 = topology_mapper.get_asic_id_from_fabric_node_id(FabricNodeId(MeshId{0}, 0));
    auto asic_id_1 = topology_mapper.get_asic_id_from_fabric_node_id(FabricNodeId(MeshId{0}, 1));
    auto asic_id_7 = topology_mapper.get_asic_id_from_fabric_node_id(FabricNodeId(MeshId{0}, 7));
    auto asic_id_4 = topology_mapper.get_asic_id_from_fabric_node_id(FabricNodeId(MeshId{0}, 4));
    auto asic_id_13 = topology_mapper.get_asic_id_from_fabric_node_id(FabricNodeId(MeshId{0}, 13));
    auto asic_id_14 = topology_mapper.get_asic_id_from_fabric_node_id(FabricNodeId(MeshId{0}, 14));
    auto asic_id_15 = topology_mapper.get_asic_id_from_fabric_node_id(FabricNodeId(MeshId{0}, 15));
    auto asic_id_3 = topology_mapper.get_asic_id_from_fabric_node_id(FabricNodeId(MeshId{0}, 3));

    // Check for adjacency
    EXPECT_TRUE(contains(physical_system_descriptor_->get_asic_neighbors(asic_id_0), asic_id_1));
    EXPECT_TRUE(contains(physical_system_descriptor_->get_asic_neighbors(asic_id_1), asic_id_0));
    EXPECT_TRUE(contains(physical_system_descriptor_->get_asic_neighbors(asic_id_7), asic_id_4));
    EXPECT_TRUE(contains(physical_system_descriptor_->get_asic_neighbors(asic_id_4), asic_id_7));
    EXPECT_TRUE(contains(physical_system_descriptor_->get_asic_neighbors(asic_id_13), asic_id_14));
    EXPECT_TRUE(contains(physical_system_descriptor_->get_asic_neighbors(asic_id_14), asic_id_13));
    EXPECT_TRUE(contains(physical_system_descriptor_->get_asic_neighbors(asic_id_15), asic_id_3));
    EXPECT_TRUE(contains(physical_system_descriptor_->get_asic_neighbors(asic_id_3), asic_id_15));

    // Check the host ranks are right
    const MeshId mesh_id{0};
    const auto& host_ranks = topology_mapper.get_host_ranks(mesh_id);
    EXPECT_EQ(host_ranks.size(), 4u);

    // Check the full shape and sub shape are right
    MeshShape full_shape = mesh_graph.get_mesh_shape(mesh_id);
    EXPECT_EQ(full_shape, MeshShape(4, 4));
    EXPECT_EQ(topology_mapper.get_mesh_shape(mesh_id), full_shape);
}

TEST_F(TopologyMapperTest, BHQB4x4StrictInvalidMeshGraphTest) {
    const std::filesystem::path bh_qb_4x4_strict_invalid_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/bh_qb_4x4_strict_invalid_mesh_graph_descriptor.textproto";

    auto mesh_graph = MeshGraph(bh_qb_4x4_strict_invalid_mesh_graph_desc_path.string());

    // Create a local mesh binding for testing
    LocalMeshBinding local_mesh_binding;
    if (*tt::tt_metal::MetalContext::instance().global_distributed_context().rank() == 0) {
        local_mesh_binding.mesh_ids = {MeshId{0}};
        local_mesh_binding.host_rank = MeshHostRankId{0};
    } else if (*tt::tt_metal::MetalContext::instance().global_distributed_context().rank() == 1) {
        local_mesh_binding.mesh_ids = {MeshId{0}};
        local_mesh_binding.host_rank = MeshHostRankId{1};
    } else if (*tt::tt_metal::MetalContext::instance().global_distributed_context().rank() == 2) {
        local_mesh_binding.mesh_ids = {MeshId{0}};
        local_mesh_binding.host_rank = MeshHostRankId{2};
    } else if (*tt::tt_metal::MetalContext::instance().global_distributed_context().rank() == 3) {
        local_mesh_binding.mesh_ids = {MeshId{0}};
        local_mesh_binding.host_rank = MeshHostRankId{3};
    }

    EXPECT_THROW(TopologyMapper(mesh_graph, *physical_system_descriptor_, local_mesh_binding), std::exception);
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
