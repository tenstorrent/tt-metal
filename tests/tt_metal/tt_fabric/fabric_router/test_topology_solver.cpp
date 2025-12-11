// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <gtest/gtest.h>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/experimental/fabric/topology_solver.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include "impl/context/metal_context.hpp"
#include "tt_metal/fabric/physical_system_descriptor.hpp"
#include "tt_metal/fabric/serialization/physical_system_descriptor_serialization.hpp"

namespace tt::tt_fabric {

class TopologySolverTest : public ::testing::Test {
protected:
    void SetUp() override { setenv("TT_METAL_OPERATION_TIMEOUT_SECONDS", "10", 1); }

    void TearDown() override {}
};

TEST_F(TopologySolverTest, BuildAdjacencyMapLogical) {
    // Use 2x2 T3K multiprocess MGD (has 2 meshes: mesh_id 0 and 1)
    const std::filesystem::path mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_2x2_mesh_graph_descriptor.textproto";

    // Create mesh graph from descriptor
    auto mesh_graph = MeshGraph(mesh_graph_desc_path.string());

    // Build adjacency map logical
    auto adjacency_map = build_adjacency_map_logical(mesh_graph);

    // Verify that we have adjacency graphs for each mesh
    EXPECT_GT(adjacency_map.size(), 0u);

    // Verify each mesh has a valid adjacency graph
    for (const auto& [mesh_id, adj_graph] : adjacency_map) {
        const auto& nodes = adj_graph.get_nodes();
        EXPECT_GT(nodes.size(), 0u) << "Mesh " << mesh_id.get() << " should have nodes";

        // Verify that nodes belong to the correct mesh
        for (const auto& node : nodes) {
            EXPECT_EQ(node.mesh_id, mesh_id) << "Node should belong to mesh " << mesh_id.get();

            // Verify we can query neighbors
            const auto& neighbors = adj_graph.get_neighbors(node);
            for (const auto& neighbor : neighbors) {
                EXPECT_EQ(neighbor.mesh_id, mesh_id) << "Neighbor should belong to the same mesh " << mesh_id.get();
            }
        }
    }

    // For T3K 2x2 multiprocess, we expect 2 meshes
    EXPECT_EQ(adjacency_map.size(), 2u);
}

TEST_F(TopologySolverTest, BuildAdjacencyMapPhysical) {
    // Load PSD from pre-written test file
    const std::filesystem::path root_dir =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir());
    const std::filesystem::path psd_file_path =
        root_dir / "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_4asic_2mesh.textproto";

    // Verify the file exists
    ASSERT_TRUE(std::filesystem::exists(psd_file_path)) << "PSD test file not found: " << psd_file_path.string();

    // Load PhysicalSystemDescriptor from file
    tt::tt_metal::PhysicalSystemDescriptor physical_system_descriptor(psd_file_path.string());

    // Hand-craft the asic_id_to_mesh_rank mapping
    // Mesh 0: ASICs 100, 101 (connected)
    // Mesh 1: ASICs 102, 103 (connected)
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> asic_id_to_mesh_rank;
    asic_id_to_mesh_rank[MeshId{0}][tt::tt_metal::AsicID{100}] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{0}][tt::tt_metal::AsicID{101}] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{1}][tt::tt_metal::AsicID{102}] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{1}][tt::tt_metal::AsicID{103}] = MeshHostRankId{0};

    // Build adjacency map physical
    auto adjacency_map = build_adjacency_map_physical(physical_system_descriptor, asic_id_to_mesh_rank);

    // Verify that we have adjacency graphs for each mesh
    EXPECT_EQ(adjacency_map.size(), 2u) << "Should have 2 meshes";

    // Verify mesh 0
    auto it0 = adjacency_map.find(MeshId{0});
    ASSERT_NE(it0, adjacency_map.end()) << "Mesh 0 should exist";
    const auto& graph0 = it0->second;
    const auto& nodes0 = graph0.get_nodes();
    EXPECT_EQ(nodes0.size(), 2u) << "Mesh 0 should have 2 ASICs";

    // Check that ASIC 100 has ASIC 101 as neighbor (2 connections = 2 entries)
    auto asic100_it = std::find(nodes0.begin(), nodes0.end(), tt::tt_metal::AsicID{100});
    ASSERT_NE(asic100_it, nodes0.end()) << "Mesh 0 should contain ASIC 100";
    const auto& neighbors100 = graph0.get_neighbors(tt::tt_metal::AsicID{100});
    EXPECT_EQ(neighbors100.size(), 2u) << "ASIC 100 should have 2 neighbor entries (2 eth connections)";
    EXPECT_EQ(neighbors100[0], tt::tt_metal::AsicID{101}) << "ASIC 100 should be connected to ASIC 101";
    EXPECT_EQ(neighbors100[1], tt::tt_metal::AsicID{101}) << "ASIC 100 should have 2 connections to ASIC 101";

    // Verify mesh 1
    auto it1 = adjacency_map.find(MeshId{1});
    ASSERT_NE(it1, adjacency_map.end()) << "Mesh 1 should exist";
    const auto& graph1 = it1->second;
    const auto& nodes1 = graph1.get_nodes();
    EXPECT_EQ(nodes1.size(), 2u) << "Mesh 1 should have 2 ASICs";

    // Check that ASIC 102 has ASIC 103 as neighbor
    auto asic102_it = std::find(nodes1.begin(), nodes1.end(), tt::tt_metal::AsicID{102});
    ASSERT_NE(asic102_it, nodes1.end()) << "Mesh 1 should contain ASIC 102";
    const auto& neighbors102 = graph1.get_neighbors(tt::tt_metal::AsicID{102});
    EXPECT_EQ(neighbors102.size(), 2u) << "ASIC 102 should have 2 neighbor entries (2 eth connections)";
    EXPECT_EQ(neighbors102[0], tt::tt_metal::AsicID{103}) << "ASIC 102 should be connected to ASIC 103";
    EXPECT_EQ(neighbors102[1], tt::tt_metal::AsicID{103}) << "ASIC 102 should have 2 connections to ASIC 103";
}

}  // namespace tt::tt_fabric
