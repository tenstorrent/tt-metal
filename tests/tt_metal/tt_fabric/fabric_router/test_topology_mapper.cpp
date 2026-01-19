// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph_descriptor.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include "impl/context/metal_context.hpp"
#include <memory>
#include <unordered_set>

#include "tt_metal/fabric/physical_system_descriptor.hpp"
#include <tt-metalium/experimental/fabric/topology_mapper.hpp>
#include "t3k_mesh_descriptor_chip_mappings.hpp"
#include "utils.hpp"

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
        "tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.textproto";

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
        "tt_metal/fabric/mesh_graph_descriptors/dual_galaxy_mesh_graph_descriptor.textproto";

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
        "tt_metal/fabric/mesh_graph_descriptors/n300_mesh_graph_descriptor.textproto";

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
        "tt_metal/fabric/mesh_graph_descriptors/p100_mesh_graph_descriptor.textproto";

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

    const MeshId mesh_id{0};

    // Check that the rank bindings line up
    auto host_ranks = mesh_graph.get_host_ranks(mesh_id);
    const auto& tp_host_ranks = topology_mapper.get_host_ranks(mesh_id);
    EXPECT_EQ(host_ranks, tp_host_ranks);

    // Check coord range of host ranks are right
    for (const auto& [_, host_rank] : host_ranks) {
        auto coord_range = mesh_graph.get_coord_range(mesh_id, host_rank);
        auto tp_coord_range = topology_mapper.get_coord_range(mesh_id, host_rank);

        EXPECT_EQ(coord_range, tp_coord_range);
    }

    // Check chip ids of host ranks are right
    for (const auto& [_, host_rank] : host_ranks) {
        auto chip_ids = mesh_graph.get_chip_ids(mesh_id, host_rank);
        auto tp_chip_ids = topology_mapper.get_chip_ids(mesh_id, host_rank);
        EXPECT_EQ(chip_ids, tp_chip_ids);
    }

    // Check mesh shape of host ranks are right
    for (const auto& [_, host_rank] : host_ranks) {
        auto mesh_shape = mesh_graph.get_mesh_shape(mesh_id, host_rank);
        auto tp_mesh_shape = topology_mapper.get_mesh_shape(mesh_id, host_rank);
        EXPECT_EQ(mesh_shape, tp_mesh_shape);
    }

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

TEST_F(TopologyMapperTest, ClosetBox3PodTTSwitchHostnameAPIs) {
    const std::filesystem::path mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/wh_closetbox_3pod_ttswitch_mgd.textproto";

    auto mesh_graph = MeshGraph(mesh_graph_desc_path.string());

    // Create local mesh binding (for testing, bind all meshes including switch)
    LocalMeshBinding local_mesh_binding;
    if (*tt::tt_metal::MetalContext::instance().global_distributed_context().rank() == 0) {
        local_mesh_binding.mesh_ids = {MeshId{0}};
    } else if (*tt::tt_metal::MetalContext::instance().global_distributed_context().rank() == 1) {
        local_mesh_binding.mesh_ids = {MeshId{1}};
    } else if (*tt::tt_metal::MetalContext::instance().global_distributed_context().rank() == 2) {
        local_mesh_binding.mesh_ids = {MeshId{2}};
    } else if (*tt::tt_metal::MetalContext::instance().global_distributed_context().rank() == 3) {
        local_mesh_binding.mesh_ids = {MeshId{3}};
    }

    auto topology_mapper = TopologyMapper(mesh_graph, *physical_system_descriptor_, local_mesh_binding);

    // Get the current hostname from the physical system descriptor
    const auto& current_hostname = physical_system_descriptor_->my_host_name();

    // ========== Test get_hostname_for_switch() ==========
    const auto& switch_ids = mesh_graph.get_switch_ids();
    ASSERT_EQ(switch_ids.size(), 1) << "Should have exactly 1 switch";

    SwitchId switch_id = switch_ids[0];
    EXPECT_EQ(*switch_id, 3) << "Switch ID should be 3";

    HostName switch_hostname = topology_mapper.get_hostname_for_switch(switch_id);
    EXPECT_FALSE(switch_hostname.empty()) << "Switch hostname should not be empty";

    // Verify switch hostname matches one of the hostnames in the system
    // (could be current hostname or another hostname in the system)
    auto all_hostnames = physical_system_descriptor_->get_all_hostnames();
    bool found_valid_hostname = false;
    for (const auto& hostname : all_hostnames) {
        if (switch_hostname == hostname) {
            found_valid_hostname = true;
            break;
        }
    }
    EXPECT_TRUE(found_valid_hostname) << "Switch hostname should be one of the system hostnames";

    // ========== Test get_hostname_for_mesh() ==========
    // Test hostname for each mesh (meshes 0, 1, 2)
    const auto& connected_meshes = mesh_graph.get_meshes_connected_to_switch(switch_id);
    EXPECT_EQ(connected_meshes.size(), 3) << "Switch should be connected to 3 meshes";

    for (const auto& mesh_id : connected_meshes) {
        HostName mesh_hostname = topology_mapper.get_hostname_for_mesh(mesh_id);
        EXPECT_FALSE(mesh_hostname.empty()) << "Mesh hostname should not be empty for mesh " << *mesh_id;

        // Verify mesh hostname matches one of the hostnames in the system
        found_valid_hostname = false;
        for (const auto& hostname : all_hostnames) {
            if (mesh_hostname == hostname) {
                found_valid_hostname = true;
                break;
            }
        }
        EXPECT_TRUE(found_valid_hostname)
            << "Mesh hostname should be one of the system hostnames for mesh " << *mesh_id;
    }

    // Test hostname for switch mesh_id as well
    HostName switch_mesh_hostname = topology_mapper.get_hostname_for_switch(switch_id);
    EXPECT_FALSE(switch_mesh_hostname.empty()) << "Switch mesh hostname should not be empty";
    EXPECT_EQ(switch_mesh_hostname, switch_hostname) << "Switch mesh hostname should match switch hostname";

    // ========== Test get_hostname_for_fabric_node_id() ==========
    // Test hostname for various fabric node IDs

    // Test fabric node IDs from a regular mesh
    MeshId test_mesh_id = *connected_meshes.begin();
    const auto& chip_ids = mesh_graph.get_chip_ids(test_mesh_id);
    ASSERT_GT(chip_ids.size(), 0) << "Test mesh should have at least one chip";

    FabricNodeId test_fabric_node_id(test_mesh_id, chip_ids.values()[0]);
    HostName fabric_node_hostname = topology_mapper.get_hostname_for_fabric_node_id(test_fabric_node_id);
    EXPECT_FALSE(fabric_node_hostname.empty()) << "Fabric node hostname should not be empty";

    // Verify fabric node hostname matches one of the hostnames in the system
    found_valid_hostname = false;
    for (const auto& hostname : all_hostnames) {
        if (fabric_node_hostname == hostname) {
            found_valid_hostname = true;
            break;
        }
    }
    EXPECT_TRUE(found_valid_hostname) << "Fabric node hostname should be one of the system hostnames";

    // Verify fabric node hostname matches mesh hostname for the same mesh
    HostName mesh_hostname_from_fabric_node = topology_mapper.get_hostname_for_mesh(test_mesh_id);
    EXPECT_EQ(fabric_node_hostname, mesh_hostname_from_fabric_node)
        << "Fabric node hostname should match mesh hostname for the same mesh";

    // Test fabric node ID from switch
    const auto& switch_chip_ids = mesh_graph.get_chip_ids(MeshId(*switch_id));
    ASSERT_GT(switch_chip_ids.size(), 0) << "Switch should have at least one chip";

    FabricNodeId switch_fabric_node_id(MeshId(*switch_id), switch_chip_ids.values()[0]);
    HostName switch_fabric_node_hostname = topology_mapper.get_hostname_for_fabric_node_id(switch_fabric_node_id);
    EXPECT_FALSE(switch_fabric_node_hostname.empty()) << "Switch fabric node hostname should not be empty";
    EXPECT_EQ(switch_fabric_node_hostname, switch_hostname)
        << "Switch fabric node hostname should match switch hostname";

    // Verify consistency: all chips in the same mesh should have the same hostname (for single-host meshes)
    // or at least valid hostnames
    for (const auto& chip_id : chip_ids.values()) {
        FabricNodeId fabric_node_id(test_mesh_id, chip_id);
        HostName chip_hostname = topology_mapper.get_hostname_for_fabric_node_id(fabric_node_id);
        EXPECT_FALSE(chip_hostname.empty()) << "Chip hostname should not be empty for chip " << chip_id;

        // Verify it's a valid hostname
        found_valid_hostname = false;
        for (const auto& hostname : all_hostnames) {
            if (chip_hostname == hostname) {
                found_valid_hostname = true;
                break;
            }
        }
        EXPECT_TRUE(found_valid_hostname) << "Chip hostname should be one of the system hostnames for chip " << chip_id;
    }
}

TEST_F(TopologyMapperTest, PinningHonorsFixedAsicPositionOnDualGalaxyMesh_1pin) {
    const std::filesystem::path galaxy_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/dual_galaxy_mesh_graph_descriptor.textproto";

    auto mesh_graph = MeshGraph(galaxy_mesh_graph_desc_path.string());

    // Local mesh binding for single-host
    LocalMeshBinding local_mesh_binding;
    if (*tt::tt_metal::MetalContext::instance().global_distributed_context().rank() == 0) {
        local_mesh_binding.mesh_ids = {MeshId{0}};
        local_mesh_binding.host_rank = MeshHostRankId{0};
    } else {
        local_mesh_binding.mesh_ids = {MeshId{0}};
        local_mesh_binding.host_rank = MeshHostRankId{1};
    }

    // Choose a real ASIC on this host and pin its (tray, location) to logical node (mesh 0, chip 0)
    const auto my_host = physical_system_descriptor_->my_host_name();
    auto pinned_asic = AsicPosition{1, 1};

    std::vector<std::pair<AsicPosition, FabricNodeId>> pins = {
        {pinned_asic, FabricNodeId(MeshId{0}, 0)},
    };

    TopologyMapper topology_mapper_with_pins(mesh_graph, *physical_system_descriptor_, local_mesh_binding, pins);

    tt::tt_metal::AsicID mapped_asic;
    for (const auto& asics : physical_system_descriptor_->get_asics_connected_to_host(my_host)) {
        auto tray = physical_system_descriptor_->get_tray_id(asics);
        auto loc = physical_system_descriptor_->get_asic_location(asics);
        if (tray == pinned_asic.first && loc == pinned_asic.second) {
            mapped_asic = topology_mapper_with_pins.get_asic_id_from_fabric_node_id(FabricNodeId(MeshId{0}, 0));
            break;
        }
    }

    // Verify that the mapping for FabricNodeId(0,0) resolves to the pinned ASIC
    auto mapped_asic_for_node0 = topology_mapper_with_pins.get_asic_id_from_fabric_node_id(FabricNodeId(MeshId{0}, 0));
    EXPECT_EQ(mapped_asic_for_node0, mapped_asic);
}

TEST_F(TopologyMapperTest, PinningHonorsFixedAsicPositionOnDualGalaxyMesh_2pins) {
    const std::filesystem::path galaxy_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/dual_galaxy_mesh_graph_descriptor.textproto";

    auto mesh_graph = MeshGraph(galaxy_mesh_graph_desc_path.string());

    // Local mesh binding for single-host
    LocalMeshBinding local_mesh_binding;
    if (*tt::tt_metal::MetalContext::instance().global_distributed_context().rank() == 0) {
        local_mesh_binding.mesh_ids = {MeshId{0}};
        local_mesh_binding.host_rank = MeshHostRankId{0};
    } else {
        local_mesh_binding.mesh_ids = {MeshId{0}};
        local_mesh_binding.host_rank = MeshHostRankId{1};
    }

    // Choose a real ASIC on this host and pin its (tray, location) to logical node (mesh 0, chip 0)
    auto pinned_asic = AsicPosition{1, 1};
    auto pinned_asic2 = AsicPosition{1, 5};

    std::vector<std::pair<AsicPosition, FabricNodeId>> pins = {
        {pinned_asic, FabricNodeId(MeshId{0}, 0)},
        {pinned_asic2, FabricNodeId(MeshId{0}, 1)},
    };

    TopologyMapper topology_mapper_with_pins(mesh_graph, *physical_system_descriptor_, local_mesh_binding, pins);

    // Check that the potential mapped ASICs are correctly for the pinned ASICs
    std::vector<tt::tt_metal::AsicID> potential_mapped_asics;
    std::vector<tt::tt_metal::AsicID> potential_mapped_asics2;
    for (const auto& [asic_id, _] : physical_system_descriptor_->get_asic_descriptors()) {
        auto tray = physical_system_descriptor_->get_tray_id(asic_id);
        auto loc = physical_system_descriptor_->get_asic_location(asic_id);
        if (tray == pinned_asic.first && loc == pinned_asic.second) {
            potential_mapped_asics.push_back(asic_id);
        }
        if (tray == pinned_asic2.first && loc == pinned_asic2.second) {
            potential_mapped_asics2.push_back(asic_id);
        }
    }

    // Verify that the mapping for FabricNodeId(0,0) resolves to the pinned ASIC
    auto mapped_asic_for_node0 = topology_mapper_with_pins.get_asic_id_from_fabric_node_id(FabricNodeId(MeshId{0}, 0));
    auto mapped_asic_for_node1 = topology_mapper_with_pins.get_asic_id_from_fabric_node_id(FabricNodeId(MeshId{0}, 1));
    EXPECT_TRUE(contains(potential_mapped_asics, mapped_asic_for_node0));
    EXPECT_TRUE(contains(potential_mapped_asics2, mapped_asic_for_node1));
}

TEST_F(TopologyMapperTest, PinningThrowsOnBadAsicPositionGalaxyMesh) {
    const std::filesystem::path galaxy_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/dual_galaxy_mesh_graph_descriptor.textproto";

    auto mesh_graph = MeshGraph(galaxy_mesh_graph_desc_path.string());

    LocalMeshBinding local_mesh_binding;
    if (*tt::tt_metal::MetalContext::instance().global_distributed_context().rank() == 0) {
        local_mesh_binding.mesh_ids = {MeshId{0}};
        local_mesh_binding.host_rank = MeshHostRankId{0};
    } else {
        local_mesh_binding.mesh_ids = {MeshId{0}};
        local_mesh_binding.host_rank = MeshHostRankId{1};
    }

    // Use an ASIC position that does not exist in this environment
    std::vector<std::pair<AsicPosition, FabricNodeId>> pins_missing = {
        {AsicPosition{tt::tt_metal::TrayID{1}, tt::tt_metal::ASICLocation{3}}, FabricNodeId(MeshId{0}, 0)},
    };

    // Expect a throw due to missing ASIC position in the local mesh physical topology
    EXPECT_THROW(
        TopologyMapper(mesh_graph, *physical_system_descriptor_, local_mesh_binding, pins_missing), std::exception);
}

// Parameterized test fixture for testing TopologyMapper with custom mappings
class T3kTopologyMapperWithCustomMappingFixture
    : public TopologyMapperTest,
      public testing::WithParamInterface<std::tuple<std::string, std::vector<std::vector<EthCoord>>>> {};

TEST_P(T3kTopologyMapperWithCustomMappingFixture, T3kMeshGraphWithCustomMapping) {
    auto [mesh_graph_desc_path, mesh_graph_eth_coords] = GetParam();
    const std::filesystem::path t3k_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) / mesh_graph_desc_path;

    auto mesh_graph = MeshGraph(t3k_mesh_graph_desc_path.string());

    // Create logical to physical chip mapping from eth_coords
    auto logical_mesh_chip_id_to_physical_chip_id_mapping =
        fabric_router_tests::get_physical_chip_mapping_from_eth_coords_mapping(mesh_graph_eth_coords);

    // Determine which meshes are in the mapping and create local mesh binding
    std::unordered_set<MeshId> mesh_ids_in_mapping;
    for (const auto& [fabric_node_id, _] : logical_mesh_chip_id_to_physical_chip_id_mapping) {
        mesh_ids_in_mapping.insert(fabric_node_id.mesh_id);
    }

    // Create local mesh binding - use the first mesh ID found, or all meshes if multi-mesh
    LocalMeshBinding local_mesh_binding;
    if (mesh_ids_in_mapping.size() == 1) {
        local_mesh_binding.mesh_ids = {*mesh_ids_in_mapping.begin()};
    } else {
        // For multi-mesh cases, bind to all meshes
        local_mesh_binding.mesh_ids.assign(mesh_ids_in_mapping.begin(), mesh_ids_in_mapping.end());
    }
    local_mesh_binding.host_rank = MeshHostRankId{0};

    // Create TopologyMapper using the new constructor that skips discovery
    auto topology_mapper_with_mapping = TopologyMapper(
        mesh_graph, *physical_system_descriptor_, local_mesh_binding, logical_mesh_chip_id_to_physical_chip_id_mapping);

    // Verify that the mapper correctly uses the provided mapping for each mesh
    for (const auto& mesh_id : mesh_ids_in_mapping) {
        const auto mesh_shape = mesh_graph.get_mesh_shape(mesh_id);
        const auto mesh_size = mesh_shape.mesh_size();

        // Verify all fabric nodes in this mesh
        for (ChipId chip_id = 0; chip_id < mesh_size; ++chip_id) {
            FabricNodeId fabric_node_id(mesh_id, chip_id);

            // Skip if this fabric node is not in the provided mapping (for multi-mesh cases)
            if (!logical_mesh_chip_id_to_physical_chip_id_mapping.contains(fabric_node_id)) {
                continue;
            }

            // Verify that the physical chip ID matches what we provided
            auto phys_chip_id_from_mapper =
                topology_mapper_with_mapping.get_physical_chip_id_from_fabric_node_id(fabric_node_id);
            auto expected_phys_chip_id = logical_mesh_chip_id_to_physical_chip_id_mapping.at(fabric_node_id);
            EXPECT_EQ(phys_chip_id_from_mapper, expected_phys_chip_id)
                << "Physical chip ID mismatch for fabric node " << fabric_node_id << " (mesh_id=" << mesh_id.get()
                << ", chip_id=" << chip_id << ")";

            // Verify bidirectional mappings work correctly
            auto asic_id = topology_mapper_with_mapping.get_asic_id_from_fabric_node_id(fabric_node_id);
            EXPECT_EQ(topology_mapper_with_mapping.get_fabric_node_id_from_asic_id(asic_id), fabric_node_id);
            EXPECT_EQ(
                topology_mapper_with_mapping.get_fabric_node_id_from_physical_chip_id(phys_chip_id_from_mapper),
                fabric_node_id);

            // Verify physical chip ID to ASIC ID conversion
            auto phys_chip_id_from_asic = topology_mapper_with_mapping.get_physical_chip_id_from_asic_id(asic_id);
            EXPECT_EQ(phys_chip_id_from_asic, expected_phys_chip_id);
        }

        // Verify host rank structures are built correctly
        const auto& host_ranks = topology_mapper_with_mapping.get_host_ranks(mesh_id);
        EXPECT_GT(host_ranks.size(), 0u) << "No host ranks found for mesh " << mesh_id.get();

        // Verify mesh shape matches mesh graph
        auto mapper_mesh_shape = topology_mapper_with_mapping.get_mesh_shape(mesh_id);
        EXPECT_EQ(mapper_mesh_shape, mesh_shape) << "Mesh shape mismatch for mesh " << mesh_id.get();

        // Verify coordinate range
        auto coord_range = topology_mapper_with_mapping.get_coord_range(mesh_id);
        auto expected_coord_range = mesh_graph.get_coord_range(mesh_id);
        EXPECT_EQ(coord_range.start_coord(), expected_coord_range.start_coord())
            << "Coordinate range start mismatch for mesh " << mesh_id.get();
        EXPECT_EQ(coord_range.end_coord(), expected_coord_range.end_coord())
            << "Coordinate range end mismatch for mesh " << mesh_id.get();

        // Verify chip IDs
        auto chip_ids = topology_mapper_with_mapping.get_chip_ids(mesh_id);
        EXPECT_EQ(chip_ids.size(), mesh_size) << "Chip IDs size mismatch for mesh " << mesh_id.get();
    }

    // Verify local logical mesh chip id to physical chip id mapping
    auto local_mapping = topology_mapper_with_mapping.get_local_logical_mesh_chip_id_to_physical_chip_id_mapping();
    const auto& my_host = physical_system_descriptor_->my_host_name();

    // Count how many chips from the provided mapping are on this host
    size_t expected_local_mapping_size = 0;
    for (const auto& [fabric_node_id, physical_chip_id] : logical_mesh_chip_id_to_physical_chip_id_mapping) {
        auto asic_id = topology_mapper_with_mapping.get_asic_id_from_fabric_node_id(fabric_node_id);
        if (physical_system_descriptor_->get_host_name_for_asic(asic_id) == my_host) {
            expected_local_mapping_size++;
            EXPECT_EQ(local_mapping.at(fabric_node_id), physical_chip_id)
                << "Local mapping mismatch for fabric node " << fabric_node_id;
        }
    }
    EXPECT_EQ(local_mapping.size(), expected_local_mapping_size)
        << "Local mapping size mismatch. Expected " << expected_local_mapping_size << " but got "
        << local_mapping.size();
}

// Instantiate the parameterized test with all t3k mesh descriptor chip mappings
INSTANTIATE_TEST_SUITE_P(
    T3kTopologyMapperCustomMapping,
    T3kTopologyMapperWithCustomMappingFixture,
    ::testing::ValuesIn(fabric_router_tests::t3k_mesh_descriptor_chip_mappings));

TEST_F(TopologyMapperTest, T3kMeshGraphTestFromPhysicalSystemDescriptor) {
    // Test that TopologyMapper::generate_mesh_graph_from_physical_system_descriptor uses map_mesh_to_physical
    // to find a valid mesh shape that can be mapped to the physical topology
    FabricConfig fabric_config = FabricConfig::FABRIC_2D;

    // Generate mesh graph from physical system descriptor
    // This should internally use map_mesh_to_physical to find a valid mapping
    MeshGraph mesh_graph = TopologyMapper::generate_mesh_graph_from_physical_system_descriptor(
        *physical_system_descriptor_, fabric_config);

    // Verify that the mesh graph was generated successfully
    const MeshId mesh_id{0};
    EXPECT_TRUE(!mesh_graph.get_mesh_ids().empty()) << "Mesh graph should have at least one mesh";

    // Verify that the mesh graph has chips
    auto chip_ids = mesh_graph.get_chip_ids(mesh_id);
    EXPECT_EQ(chip_ids.values().size(), physical_system_descriptor_->get_asic_descriptors().size())
        << "Mesh graph should have same number of chips as the physical system descriptor";

    // Check that the mesh shape is either 2x4 or 4x2
    auto mesh_shape = mesh_graph.get_mesh_shape(mesh_id);
    EXPECT_TRUE(mesh_shape == MeshShape(2, 4) || mesh_shape == MeshShape(4, 2))
        << "Mesh shape should be either 2x4 or 4x2";

    // Verify that host ranks are set up correctly
    for (const auto& chip_id : chip_ids.values()) {
        auto host_rank = mesh_graph.get_host_rank_for_chip(mesh_id, chip_id);
        EXPECT_TRUE(host_rank.has_value()) << "Host rank should be set for chip " << chip_id;
    }

    // Verify that the mesh graph can be used with TopologyMapper
    LocalMeshBinding local_mesh_binding;
    local_mesh_binding.mesh_ids = {mesh_id};
    local_mesh_binding.host_rank = MeshHostRankId{0};

    // This should work without throwing since the mesh graph was generated
    // to match the physical topology
    EXPECT_NO_THROW({
        TopologyMapper topology_mapper(mesh_graph, *physical_system_descriptor_, local_mesh_binding);
        // Verify that mappings exist
        for (const auto& chip_id : chip_ids.values()) {
            FabricNodeId fabric_node_id(mesh_id, chip_id);
            auto asic_id = topology_mapper.get_asic_id_from_fabric_node_id(fabric_node_id);
            EXPECT_NE(asic_id.get(), 0u) << "ASIC ID should be valid for fabric node " << fabric_node_id;
        }
    });
}

}  // namespace tt::tt_fabric
