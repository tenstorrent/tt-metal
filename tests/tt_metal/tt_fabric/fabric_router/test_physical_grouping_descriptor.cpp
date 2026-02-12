// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>
#include <set>

#include <tt-metalium/experimental/fabric/physical_grouping_descriptor.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph_descriptor.hpp>
#include <tt-metalium/experimental/fabric/topology_solver.hpp>
#include "tt_metal/fabric/physical_system_descriptor.hpp"
#include "tt_metal/fabric/serialization/physical_system_descriptor_serialization.hpp"

using namespace tt::tt_fabric;

namespace tt::tt_fabric::fabric_router_tests {

// Helper to check that a node's neighbors match expected (order-independent)
static void expect_neighbors(
    const AdjacencyGraph<uint32_t>& graph, uint32_t node_id, const std::vector<uint32_t>& expected) {
    const auto& neighbors = graph.get_neighbors(node_id);
    std::set<uint32_t> actual_set(neighbors.begin(), neighbors.end());
    std::set<uint32_t> expected_set(expected.begin(), expected.end());
    EXPECT_EQ(actual_set, expected_set) << "Node " << node_id << " has wrong neighbors";
}

// ============================================================================
// ADJACENCY GRAPH TESTS
// ============================================================================

TEST(PhysicalGroupingDescriptorTests, AdjacencyGraph_AllToAll_ThreeNodes) {
    const std::string text_proto = R"proto(
        groupings {
          name: "meshes_1"
          custom_type: "meshes"
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }
            , { id: 1 asic_location: ASIC_LOCATION_2 }
            , { id: 2 asic_location: ASIC_LOCATION_3 }]
        }
        groupings {
          name: "pods_1"
          custom_type: "pods"
          instances:
          [ {
            id: 10
            grouping_ref { custom_type: "meshes" }
          }
            , {
              id: 20
              grouping_ref { custom_type: "meshes" }
            }
            , {
              id: 30
              grouping_ref { custom_type: "meshes" }
            }]
          all_to_all { num_connections: 2 }
        }
    )proto";

    PhysicalGroupingDescriptor desc(text_proto);
    auto pods = desc.get_groupings_by_name("pods");
    ASSERT_EQ(pods.size(), 1);

    const auto& adj = pods[0].adjacency_graph;
    const auto& nodes = adj.get_nodes();
    ASSERT_EQ(nodes.size(), 3u);

    // All-to-all: each node connects to every other node
    expect_neighbors(adj, 10, {20, 30});
    expect_neighbors(adj, 20, {10, 30});
    expect_neighbors(adj, 30, {10, 20});
}

TEST(PhysicalGroupingDescriptorTests, AdjacencyGraph_RowMajorMesh_2x2_LineLine) {
    const std::string text_proto = R"proto(
        groupings {
          name: "meshes_2"
          custom_type: "meshes"
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }
            , { id: 1 asic_location: ASIC_LOCATION_2 }]
        }
        groupings {
          name: "grid_1"
          custom_type: "grid"
          instances:
          [ {
            id: 100
            grouping_ref { custom_type: "meshes" }
          }
            , {
              id: 101
              grouping_ref { custom_type: "meshes" }
            }
            , {
              id: 102
              grouping_ref { custom_type: "meshes" }
            }
            , {
              id: 103
              grouping_ref { custom_type: "meshes" }
            }]
          row_major_mesh {
            dims: [ 2, 2 ]
            dim_types: [ LINE, LINE ]
            num_connections: 2
          }
        }
    )proto";

    PhysicalGroupingDescriptor desc(text_proto);
    auto grids = desc.get_groupings_by_name("grid");
    ASSERT_EQ(grids.size(), 1);

    const auto& adj = grids[0].adjacency_graph;
    // 2x2 LINE,LINE grid: row-major order
    // idx 0 (0,0): neighbors (1,0)=idx1, (0,1)=idx2
    // idx 1 (1,0): neighbors (0,0)=idx0, (1,1)=idx3
    // idx 2 (0,1): neighbors (0,0)=idx0, (1,1)=idx3
    // idx 3 (1,1): neighbors (1,0)=idx1, (0,1)=idx2
    expect_neighbors(adj, 100, {101, 102});
    expect_neighbors(adj, 101, {100, 103});
    expect_neighbors(adj, 102, {100, 103});
    expect_neighbors(adj, 103, {101, 102});
}

TEST(PhysicalGroupingDescriptorTests, AdjacencyGraph_RowMajorMesh_1x4_Ring) {
    const std::string text_proto = R"proto(
        groupings {
          name: "meshes_3"
          custom_type: "meshes"
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "ring_1"
          custom_type: "ring"
          instances:
          [ {
            id: 5
            grouping_ref { custom_type: "meshes" }
          }
            , {
              id: 6
              grouping_ref { custom_type: "meshes" }
            }
            , {
              id: 7
              grouping_ref { custom_type: "meshes" }
            }
            , {
              id: 8
              grouping_ref { custom_type: "meshes" }
            }]
          row_major_mesh {
            dims: [ 1, 4 ]
            dim_types: [ LINE, RING ]
            num_connections: 2
          }
        }
    )proto";

    PhysicalGroupingDescriptor desc(text_proto);
    auto rings = desc.get_groupings_by_name("ring");
    ASSERT_EQ(rings.size(), 1);

    const auto& adj = rings[0].adjacency_graph;
    // 1x4 RING: 5-6-7-8-5 (wrap around)
    expect_neighbors(adj, 5, {6, 8});
    expect_neighbors(adj, 6, {5, 7});
    expect_neighbors(adj, 7, {6, 8});
    expect_neighbors(adj, 8, {5, 7});
}

TEST(PhysicalGroupingDescriptorTests, AdjacencyGraph_CustomConnections) {
    const std::string text_proto = R"proto(
        groupings {
          name: "meshes_4"
          custom_type: "meshes"
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "custom_topology_1"
          custom_type: "custom_topology"
          instances:
          [ {
            id: 1
            grouping_ref { custom_type: "meshes" }
          }
            , {
              id: 2
              grouping_ref { custom_type: "meshes" }
            }
            , {
              id: 3
              grouping_ref { custom_type: "meshes" }
            }]
          custom {
            connections:
            [ { src_instance: 0 dst_instance: 1 num_connections: 2 }
              , { src_instance: 0 dst_instance: 2 num_connections: 2 }
              , { src_instance: 1 dst_instance: 2 num_connections: 2 }]
          }
        }
    )proto";

    PhysicalGroupingDescriptor desc(text_proto);
    auto custom = desc.get_groupings_by_name("custom_topology");
    ASSERT_EQ(custom.size(), 1);

    const auto& adj = custom[0].adjacency_graph;
    // Custom connections use 0-based instance index; instance ids are 1,2,3 (from id field)
    // index 0 -> id 1, index 1 -> id 2, index 2 -> id 3
    // edges: 0-1, 0-2, 1-2  =>  id 1-2, id 1-3, id 2-3
    expect_neighbors(adj, 1, {2, 3});
    expect_neighbors(adj, 2, {1, 3});
    expect_neighbors(adj, 3, {1, 2});
}

TEST(PhysicalGroupingDescriptorTests, AdjacencyGraph_EmptyWhenNoConnection) {
    const std::string text_proto = R"proto(
        groupings {
          name: "meshes_5"
          custom_type: "meshes"
          instances:
          [ {
            id: 0
            grouping_ref { custom_type: "trays" }
          }]
        }
        groupings {
          name: "trays_1"
          custom_type: "trays"
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }
            , { id: 1 asic_location: ASIC_LOCATION_2 }]
        }
    )proto";

    PhysicalGroupingDescriptor desc(text_proto);
    auto meshes = desc.get_groupings_by_name("meshes");
    ASSERT_EQ(meshes.size(), 1);

    const auto& adj = meshes[0].adjacency_graph;
    const auto& nodes = adj.get_nodes();
    // No connection specified -> empty graph (no edges)
    EXPECT_TRUE(nodes.empty() || adj.get_neighbors(nodes[0]).empty());
}

// ============================================================================
// ADJACENCY GRAPH - GROUPING CONSTRAINT TESTS
// ============================================================================

// All-to-all: every node must have degree (n-1) and graph must be symmetric
TEST(PhysicalGroupingDescriptorTests, AdjacencyGraph_AllToAll_DegreeConstraint) {
    const std::string text_proto = R"proto(
        groupings {
          name: "meshes_6"
          custom_type: "meshes"
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "clique_1"
          custom_type: "clique"
          instances:
          [ {
            id: 0
            grouping_ref { custom_type: "meshes" }
          }
            , {
              id: 1
              grouping_ref { custom_type: "meshes" }
            }
            , {
              id: 2
              grouping_ref { custom_type: "meshes" }
            }
            , {
              id: 3
              grouping_ref { custom_type: "meshes" }
            }]
          all_to_all { num_connections: 2 }
        }
    )proto";

    PhysicalGroupingDescriptor desc(text_proto);
    auto clique = desc.get_groupings_by_name("clique");
    ASSERT_EQ(clique.size(), 1);

    const auto& adj = clique[0].adjacency_graph;
    const auto& nodes = adj.get_nodes();
    ASSERT_EQ(nodes.size(), 4u);

    // Constraint: all-to-all with n nodes -> each node has degree (n-1) = 3
    for (uint32_t node : nodes) {
        const auto& neighbors = adj.get_neighbors(node);
        EXPECT_EQ(neighbors.size(), 3u) << "All-to-all node " << node << " should have degree 3";
    }

    // Constraint: symmetry - if A neighbors B, then B neighbors A
    for (uint32_t node : nodes) {
        for (uint32_t neigh : adj.get_neighbors(node)) {
            const auto& rev = adj.get_neighbors(neigh);
            EXPECT_NE(std::find(rev.begin(), rev.end(), node), rev.end())
                << "Graph must be symmetric: " << node << "->" << neigh;
        }
    }
}

// Row-major mesh 2x4 LINE,LINE: corners=2, edges=3, interior has expected degree
TEST(PhysicalGroupingDescriptorTests, AdjacencyGraph_RowMajorMesh_2x4_DegreeConstraints) {
    const std::string text_proto = R"proto(
        groupings {
          name: "meshes_7"
          custom_type: "meshes"
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "grid_2x4_1"
          custom_type: "grid_2x4"
          instances:
          [ {
            id: 0
            grouping_ref { custom_type: "meshes" }
          }
            , {
              id: 1
              grouping_ref { custom_type: "meshes" }
            }
            , {
              id: 2
              grouping_ref { custom_type: "meshes" }
            }
            , {
              id: 3
              grouping_ref { custom_type: "meshes" }
            }
            , {
              id: 4
              grouping_ref { custom_type: "meshes" }
            }
            , {
              id: 5
              grouping_ref { custom_type: "meshes" }
            }
            , {
              id: 6
              grouping_ref { custom_type: "meshes" }
            }
            , {
              id: 7
              grouping_ref { custom_type: "meshes" }
            }]
          row_major_mesh {
            dims: [ 2, 4 ]
            dim_types: [ LINE, LINE ]
            num_connections: 2
          }
        }
    )proto";

    PhysicalGroupingDescriptor desc(text_proto);
    auto grids = desc.get_groupings_by_name("grid_2x4");
    ASSERT_EQ(grids.size(), 1);

    const auto& adj = grids[0].adjacency_graph;
    // 2x4 LINE,LINE: corners (0,3,4,7) have degree 2, edges have degree 3
    // idx: 0(0,0) 1(0,1) 2(0,2) 3(0,3) | 4(1,0) 5(1,1) 6(1,2) 7(1,3)
    // corners: 0,3,4,7 -> degree 2
    // edges: 1,2,5,6 -> degree 3
    EXPECT_EQ(adj.get_neighbors(0).size(), 2u);  // corner
    EXPECT_EQ(adj.get_neighbors(3).size(), 2u);  // corner
    EXPECT_EQ(adj.get_neighbors(4).size(), 2u);  // corner
    EXPECT_EQ(adj.get_neighbors(7).size(), 2u);  // corner
    EXPECT_EQ(adj.get_neighbors(1).size(), 3u);  // edge
    EXPECT_EQ(adj.get_neighbors(2).size(), 3u);  // edge
    EXPECT_EQ(adj.get_neighbors(5).size(), 3u);  // edge
    EXPECT_EQ(adj.get_neighbors(6).size(), 3u);  // edge
}

// RING dimension: all nodes in that dimension have degree 2 (for 1D ring)
TEST(PhysicalGroupingDescriptorTests, AdjacencyGraph_Ring_UniformDegree) {
    const std::string text_proto = R"proto(
        groupings {
          name: "meshes_8"
          custom_type: "meshes"
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "ring5_1"
          custom_type: "ring5"
          instances:
          [ {
            id: 0
            grouping_ref { custom_type: "meshes" }
          }
            , {
              id: 1
              grouping_ref { custom_type: "meshes" }
            }
            , {
              id: 2
              grouping_ref { custom_type: "meshes" }
            }
            , {
              id: 3
              grouping_ref { custom_type: "meshes" }
            }
            , {
              id: 4
              grouping_ref { custom_type: "meshes" }
            }]
          row_major_mesh {
            dims: [ 5 ]
            dim_types: [ RING ]
            num_connections: 2
          }
        }
    )proto";

    PhysicalGroupingDescriptor desc(text_proto);
    auto ring = desc.get_groupings_by_name("ring5");
    ASSERT_EQ(ring.size(), 1);

    const auto& adj = ring[0].adjacency_graph;
    // 1D ring of 5: every node has exactly 2 neighbors
    for (const auto& node : adj.get_nodes()) {
        EXPECT_EQ(adj.get_neighbors(node).size(), 2u) << "Ring node " << node << " must have degree 2";
    }
}

// ============================================================================
// ADJACENCY GRAPH - SPECIAL CASE TESTS
// ============================================================================

TEST(PhysicalGroupingDescriptorTests, AdjacencyGraph_SpecialCase_TwoNodes_AllToAll) {
    const std::string text_proto = R"proto(
        groupings {
          name: "meshes_9"
          custom_type: "meshes"
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "pair_1"
          custom_type: "pair"
          instances:
          [ {
            id: 10
            grouping_ref { custom_type: "meshes" }
          }
            , {
              id: 20
              grouping_ref { custom_type: "meshes" }
            }]
          all_to_all { num_connections: 2 }
        }
    )proto";

    PhysicalGroupingDescriptor desc(text_proto);
    auto pair = desc.get_groupings_by_name("pair");
    ASSERT_EQ(pair.size(), 1);

    const auto& adj = pair[0].adjacency_graph;
    expect_neighbors(adj, 10, {20});
    expect_neighbors(adj, 20, {10});
}

TEST(PhysicalGroupingDescriptorTests, AdjacencyGraph_SpecialCase_TwoNodes_1x2_Line) {
    const std::string text_proto = R"proto(
        groupings {
          name: "meshes_10"
          custom_type: "meshes"
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "chain2_1"
          custom_type: "chain2"
          instances:
          [ {
            id: 100
            grouping_ref { custom_type: "meshes" }
          }
            , {
              id: 200
              grouping_ref { custom_type: "meshes" }
            }]
          row_major_mesh {
            dims: [ 1, 2 ]
            dim_types: [ LINE, LINE ]
            num_connections: 2
          }
        }
    )proto";

    PhysicalGroupingDescriptor desc(text_proto);
    auto chain = desc.get_groupings_by_name("chain2");
    ASSERT_EQ(chain.size(), 1);

    const auto& adj = chain[0].adjacency_graph;
    expect_neighbors(adj, 100, {200});
    expect_neighbors(adj, 200, {100});
}

TEST(PhysicalGroupingDescriptorTests, AdjacencyGraph_SpecialCase_TwoNodes_1x2_Ring) {
    const std::string text_proto = R"proto(
        groupings {
          name: "meshes_11"
          custom_type: "meshes"
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "ring2_1"
          custom_type: "ring2"
          instances:
          [ {
            id: 7
            grouping_ref { custom_type: "meshes" }
          }
            , {
              id: 8
              grouping_ref { custom_type: "meshes" }
            }]
          row_major_mesh {
            dims: [ 1, 2 ]
            dim_types: [ LINE, RING ]
            num_connections: 2
          }
        }
    )proto";

    PhysicalGroupingDescriptor desc(text_proto);
    auto ring2 = desc.get_groupings_by_name("ring2");
    ASSERT_EQ(ring2.size(), 1);

    const auto& adj = ring2[0].adjacency_graph;
    // Ring of 2: each node's only neighbor is the other
    expect_neighbors(adj, 7, {8});
    expect_neighbors(adj, 8, {7});
}

TEST(PhysicalGroupingDescriptorTests, AdjacencyGraph_SpecialCase_NonSequentialInstanceIds) {
    const std::string text_proto = R"proto(
        groupings {
          name: "meshes_12"
          custom_type: "meshes"
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "sparse_1"
          custom_type: "sparse"
          instances:
          [ {
            id: 1000
            grouping_ref { custom_type: "meshes" }
          }
            , {
              id: 2000
              grouping_ref { custom_type: "meshes" }
            }
            , {
              id: 3000
              grouping_ref { custom_type: "meshes" }
            }]
          all_to_all { num_connections: 2 }
        }
    )proto";

    PhysicalGroupingDescriptor desc(text_proto);
    auto sparse = desc.get_groupings_by_name("sparse");
    ASSERT_EQ(sparse.size(), 1);

    const auto& adj = sparse[0].adjacency_graph;
    expect_neighbors(adj, 1000, {2000, 3000});
    expect_neighbors(adj, 2000, {1000, 3000});
    expect_neighbors(adj, 3000, {1000, 2000});
}

TEST(PhysicalGroupingDescriptorTests, AdjacencyGraph_SpecialCase_Custom_LinearChain) {
    const std::string text_proto = R"proto(
        groupings {
          name: "meshes_13"
          custom_type: "meshes"
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "linear_1"
          custom_type: "linear"
          instances:
          [ {
            id: 1
            grouping_ref { custom_type: "meshes" }
          }
            , {
              id: 2
              grouping_ref { custom_type: "meshes" }
            }
            , {
              id: 3
              grouping_ref { custom_type: "meshes" }
            }]
          custom {
            connections:
            [ { src_instance: 0 dst_instance: 1 num_connections: 1 }
              , { src_instance: 1 dst_instance: 2 num_connections: 1 }]
          }
        }
    )proto";

    PhysicalGroupingDescriptor desc(text_proto);
    auto linear = desc.get_groupings_by_name("linear");
    ASSERT_EQ(linear.size(), 1);

    const auto& adj = linear[0].adjacency_graph;
    // Chain: 1-2-3 (index 0-1-2)
    expect_neighbors(adj, 1, {2});
    expect_neighbors(adj, 2, {1, 3});
    expect_neighbors(adj, 3, {2});
}

TEST(PhysicalGroupingDescriptorTests, AdjacencyGraph_SpecialCase_Custom_Disconnected) {
    const std::string text_proto = R"proto(
        groupings {
          name: "meshes_14"
          custom_type: "meshes"
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "disconnected_1"
          custom_type: "disconnected"
          instances:
          [ {
            id: 1
            grouping_ref { custom_type: "meshes" }
          }
            , {
              id: 2
              grouping_ref { custom_type: "meshes" }
            }
            , {
              id: 3
              grouping_ref { custom_type: "meshes" }
            }
            , {
              id: 4
              grouping_ref { custom_type: "meshes" }
            }]
          custom {
            connections:
            [ { src_instance: 0 dst_instance: 1 num_connections: 1 }
              , { src_instance: 2 dst_instance: 3 num_connections: 1 }]
          }
        }
    )proto";

    PhysicalGroupingDescriptor desc(text_proto);
    auto disc = desc.get_groupings_by_name("disconnected");
    ASSERT_EQ(disc.size(), 1);

    const auto& adj = disc[0].adjacency_graph;
    // Two separate edges: 1-2 and 3-4
    expect_neighbors(adj, 1, {2});
    expect_neighbors(adj, 2, {1});
    expect_neighbors(adj, 3, {4});
    expect_neighbors(adj, 4, {3});
}

TEST(PhysicalGroupingDescriptorTests, AdjacencyGraph_SpecialCase_Custom_SingleEdge) {
    const std::string text_proto = R"proto(
        groupings {
          name: "meshes_15"
          custom_type: "meshes"
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "single_edge_1"
          custom_type: "single_edge"
          instances:
          [ {
            id: 42
            grouping_ref { custom_type: "meshes" }
          }
            , {
              id: 99
              grouping_ref { custom_type: "meshes" }
            }]
          custom {
            connections:
            [ { src_instance: 0 dst_instance: 1 num_connections: 4 }]
          }
        }
    )proto";

    PhysicalGroupingDescriptor desc(text_proto);
    auto se = desc.get_groupings_by_name("single_edge");
    ASSERT_EQ(se.size(), 1);

    const auto& adj = se[0].adjacency_graph;
    expect_neighbors(adj, 42, {99});
    expect_neighbors(adj, 99, {42});
}

TEST(PhysicalGroupingDescriptorTests, AdjacencyGraph_SpecialCase_1x6_Line_Chain) {
    const std::string text_proto = R"proto(
        groupings {
          name: "meshes_16"
          custom_type: "meshes"
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "chain6_1"
          custom_type: "chain6"
          instances:
          [ {
            id: 0
            grouping_ref { custom_type: "meshes" }
          }
            , {
              id: 1
              grouping_ref { custom_type: "meshes" }
            }
            , {
              id: 2
              grouping_ref { custom_type: "meshes" }
            }
            , {
              id: 3
              grouping_ref { custom_type: "meshes" }
            }
            , {
              id: 4
              grouping_ref { custom_type: "meshes" }
            }
            , {
              id: 5
              grouping_ref { custom_type: "meshes" }
            }]
          row_major_mesh {
            dims: [ 1, 6 ]
            dim_types: [ LINE, LINE ]
            num_connections: 2
          }
        }
    )proto";

    PhysicalGroupingDescriptor desc(text_proto);
    auto chain = desc.get_groupings_by_name("chain6");
    ASSERT_EQ(chain.size(), 1);

    const auto& adj = chain[0].adjacency_graph;
    // Linear chain 0-1-2-3-4-5: endpoints degree 1, interior degree 2
    EXPECT_EQ(adj.get_neighbors(0).size(), 1u);
    EXPECT_EQ(adj.get_neighbors(5).size(), 1u);
    for (uint32_t i = 1; i <= 4; ++i) {
        EXPECT_EQ(adj.get_neighbors(i).size(), 2u) << "Interior node " << i;
    }
    expect_neighbors(adj, 0, {1});
    expect_neighbors(adj, 5, {4});
}

// ============================================================================
// VALID CONFIGURATION TESTS
// ============================================================================

TEST(PhysicalGroupingDescriptorTests, ParsesValidBasicConfiguration) {
    const std::string text_proto = R"proto(
        groupings {
          name: "trays_1"
          custom_type: "trays"
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }
            , { id: 1 asic_location: ASIC_LOCATION_2 }
            , { id: 2 asic_location: ASIC_LOCATION_3 }
            , { id: 3 asic_location: ASIC_LOCATION_4 }
            , { id: 4 asic_location: ASIC_LOCATION_5 }
            , { id: 5 asic_location: ASIC_LOCATION_6 }
            , { id: 6 asic_location: ASIC_LOCATION_7 }
            , { id: 7 asic_location: ASIC_LOCATION_8 }]
          row_major_mesh {
            dims: [ 2, 4 ]
            dim_types: [ LINE, LINE ]
            num_connections: 2
          }
        }
        groupings {
          name: "hosts_1"
          custom_type: "hosts"
          instances:
          [ {
            id: 0
            grouping_ref { custom_type: "trays" }
          }
            , {
              id: 1
              grouping_ref { custom_type: "trays" }
            }
            , {
              id: 2
              grouping_ref { custom_type: "trays" }
            }
            , {
              id: 3
              grouping_ref { custom_type: "trays" }
            }]
          row_major_mesh {
            dims: [ 1, 4 ]
            dim_types: [ LINE, RING ]
            num_connections: 2
          }
        }
        groupings {
          name: "meshes_17"
          custom_type: "meshes"
          instances:
          [ {
            id: 0
            grouping_ref { custom_type: "hosts" }
          }]
        }
    )proto";

    EXPECT_NO_THROW({
        PhysicalGroupingDescriptor desc(text_proto);
        EXPECT_TRUE(desc.has_grouping("meshes"));
        EXPECT_TRUE(desc.has_grouping("hosts"));
        EXPECT_TRUE(desc.has_grouping("trays"));
    });
}

TEST(PhysicalGroupingDescriptorTests, ParsesValidConfigurationWithPods) {
    // Test that preset names can be referenced without being defined
    const std::string text_proto = R"proto(
        groupings {
          name: "pods_2"
          custom_type: "pods"
          instances:
          [ {
            id: 0
            grouping_ref { preset_type: TRAY_1 }
          }
            , {
              id: 1
              grouping_ref { preset_type: TRAY_2 }
            }
            , {
              id: 2
              grouping_ref { preset_type: TRAY_3 }
            }
            , {
              id: 3
              grouping_ref { preset_type: TRAY_4 }
            }]
          row_major_mesh {
            dims: [ 1, 4 ]
            dim_types: [ LINE, RING ]
            num_connections: 2
          }
        }
        groupings {
          name: "clusters_1"
          custom_type: "clusters"
          instances:
          [ {
            id: 0
            grouping_ref { preset_type: HOSTS }
          }
            , {
              id: 1
              grouping_ref { preset_type: HOSTS }
            }]
          all_to_all { num_connections: 2 }
        }
    )proto";

    // Should not throw - preset names are optional and skipped during dependency resolution
    EXPECT_NO_THROW({
        PhysicalGroupingDescriptor desc(text_proto);
        auto pods = desc.get_groupings_by_name("pods");
        EXPECT_EQ(pods.size(), 1);
        // Note: ASIC count will be 0 because preset names don't exist yet
        EXPECT_EQ(pods[0].asic_count, 0u);
        auto clusters = desc.get_groupings_by_name("clusters");
        EXPECT_EQ(clusters.size(), 1);
        EXPECT_EQ(clusters[0].asic_count, 0u);
    });
}

TEST(PhysicalGroupingDescriptorTests, ParsesValidConfigurationWithMultipleDefinitions) {
    const std::string text_proto = R"proto(
        groupings {
          name: "halftray_1"
          custom_type: "halftray"
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }
            , { id: 1 asic_location: ASIC_LOCATION_2 }
            , { id: 2 asic_location: ASIC_LOCATION_3 }
            , { id: 3 asic_location: ASIC_LOCATION_4 }]
          row_major_mesh {
            dims: [ 1, 4 ]
            dim_types: [ LINE, LINE ]
            num_connections: 2
          }
        }
        groupings {
          name: "halftray_2"
          custom_type: "halftray"
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_5 }
            , { id: 1 asic_location: ASIC_LOCATION_6 }
            , { id: 2 asic_location: ASIC_LOCATION_7 }
            , { id: 3 asic_location: ASIC_LOCATION_8 }]
          row_major_mesh {
            dims: [ 1, 4 ]
            dim_types: [ LINE, LINE ]
            num_connections: 2
          }
        }
        groupings {
          name: "meshes_18"
          custom_type: "meshes"
          instances:
          [ {
            id: 0
            grouping_ref { custom_type: "halftray" }
          }]
        }
    )proto";

    EXPECT_NO_THROW({
        PhysicalGroupingDescriptor desc(text_proto);
        auto halftrays = desc.get_groupings_by_name("halftray");
        EXPECT_EQ(halftrays.size(), 2);
    });
}

TEST(PhysicalGroupingDescriptorTests, ParsesFromTriple16x8QuadBhGalaxyFile) {
    const std::filesystem::path text_proto_file_path =
        "tests/tt_metal/tt_fabric/physical_groupings/triple_16x8_quad_bh_galaxy_physical_groupings.textproto";
    EXPECT_NO_THROW({ PhysicalGroupingDescriptor desc(text_proto_file_path); });
}

// ============================================================================
// VALIDATION TESTS
// ============================================================================

TEST(PhysicalGroupingDescriptorTests, ValidationPassesWhenMeshesMissing) {
    // Meshes grouping is no longer required - it can be auto-populated from PhysicalSystemDescriptor
    // Also test that preset names (TRAY_1, TRAY_2, HOSTS, MESH) can be referenced without being defined
    const std::string text_proto = R"proto(
        groupings {
          name: "pods_3"
          custom_type: "pods"
          instances:
          [ {
            id: 0
            grouping_ref { preset_type: TRAY_1 }
          }
            , {
              id: 1
              grouping_ref { preset_type: TRAY_2 }
            }
            , {
              id: 2
              grouping_ref { preset_type: TRAY_3 }
            }
            , {
              id: 3
              grouping_ref { preset_type: TRAY_4 }
            }]
          row_major_mesh {
            dims: [ 1, 4 ]
            dim_types: [ LINE, RING ]
            num_connections: 2
          }
        }
        groupings {
          name: "clusters_2"
          custom_type: "clusters"
          instances:
          [ {
            id: 0
            grouping_ref { preset_type: HOSTS }
          }
            , {
              id: 1
              grouping_ref { preset_type: HOSTS }
            }]
          all_to_all { num_connections: 2 }
        }
        groupings {
          name: "meshes_19"
          custom_type: "meshes"
          instances:
          [ {
            id: 0
            grouping_ref { preset_type: MESH }
          }]
        }
    )proto";

    // Should not throw - preset names and meshes are optional and skipped during dependency resolution
    EXPECT_NO_THROW({ PhysicalGroupingDescriptor desc(text_proto); });
}

TEST(PhysicalGroupingDescriptorTests, ValidationFailsWhenReferencingNonExistentGrouping) {
    // Test that custom names must exist, but preset names don't need to
    const std::string text_proto = R"proto(
        groupings {
          name: "meshes_20"
          custom_type: "meshes"
          instances:
          [ {
            id: 0
            grouping_ref { custom_type: "nonexistent" }
          }]
        }
    )proto";

    EXPECT_THAT(
        ([&]() { PhysicalGroupingDescriptor desc(text_proto); }),
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("references non-existent grouping")));

    // But preset names should pass validation and population even if not defined
    const std::string text_proto_preset = R"proto(
        groupings {
          name: "meshes_21"
          custom_type: "meshes"
          instances:
          [ {
            id: 0
            grouping_ref { preset_type: TRAY_1 }
          }]
        }
    )proto";

    // Should not throw - preset names are skipped during dependency resolution
    EXPECT_NO_THROW({ PhysicalGroupingDescriptor desc(text_proto_preset); });
}

TEST(PhysicalGroupingDescriptorTests, ValidationFailsWhenGroupingHasNoInstances) {
    const std::string text_proto = R"proto(
        groupings { name: "meshes_22" custom_type: "meshes" }
    )proto";

    EXPECT_THAT(
        ([&]() { PhysicalGroupingDescriptor desc(text_proto); }),
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("must have at least one instance")));
}

TEST(PhysicalGroupingDescriptorTests, ValidationFailsWhenNonMeshesHasOneInstance) {
    const std::string text_proto = R"proto(
        groupings {
          name: "meshes_23"
          custom_type: "meshes"
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "pods_4"
          custom_type: "pods"
          instances:
          [ {
            id: 0
            grouping_ref { custom_type: "meshes" }
          }]
          all_to_all { num_connections: 2 }
        }
    )proto";

    EXPECT_THAT(
        ([&]() { PhysicalGroupingDescriptor desc(text_proto); }),
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("groupings other than meshes must have at least 2 instances")));
}

TEST(PhysicalGroupingDescriptorTests, DuplicateNamesAreUniquified) {
    const std::string text_proto = R"proto(
        groupings {
          name: "duplicate_name"
          custom_type: "meshes"
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "duplicate_name"
          custom_type: "pods"
          instances:
          [ {
            id: 0
            grouping_ref { custom_type: "meshes" }
          }
            , {
              id: 1
              grouping_ref { custom_type: "meshes" }
            }]
          all_to_all { num_connections: 2 }
        }
    )proto";

    // Duplicate names should be automatically uniquified, not cause an error
    EXPECT_NO_THROW({
        PhysicalGroupingDescriptor desc(text_proto);

        // Verify that names were uniquified
        auto meshes_groupings = desc.get_groupings_by_name("meshes");
        auto pods_groupings = desc.get_groupings_by_name("pods");

        EXPECT_EQ(meshes_groupings.size(), 1u);
        EXPECT_EQ(pods_groupings.size(), 1u);

        // First occurrence keeps original name, second gets uniquified
        EXPECT_EQ(meshes_groupings[0].name, "duplicate_name");
        EXPECT_EQ(pods_groupings[0].name, "duplicate_name_1");

        // Verify all names are unique
        std::set<std::string> all_names;
        auto all_groupings = desc.get_all_groupings();
        for (const auto& grouping : all_groupings) {
            EXPECT_TRUE(all_names.find(grouping.name) == all_names.end()) << "Duplicate name found: " << grouping.name;
            all_names.insert(grouping.name);
        }
    });
}

// ============================================================================
// API TESTS
// ============================================================================

TEST(PhysicalGroupingDescriptorTests, HasGroupingReturnsTrueForExistingGrouping) {
    const std::string text_proto = R"proto(
        groupings {
          name: "meshes_24"
          custom_type: "meshes"
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "pods_5"
          custom_type: "pods"
          instances:
          [ {
            id: 0
            grouping_ref { custom_type: "meshes" }
          }
            , {
              id: 1
              grouping_ref { custom_type: "meshes" }
            }]
          all_to_all { num_connections: 2 }
        }
    )proto";

    PhysicalGroupingDescriptor desc(text_proto);
    EXPECT_TRUE(desc.has_grouping("meshes"));
    EXPECT_TRUE(desc.has_grouping("pods"));
    EXPECT_FALSE(desc.has_grouping("nonexistent"));
}

TEST(PhysicalGroupingDescriptorTests, GetGroupingsByNameReturnsAllDefinitions) {
    const std::string text_proto = R"proto(
        groupings {
          name: "halftray_3"
          custom_type: "halftray"
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }
            , { id: 1 asic_location: ASIC_LOCATION_2 }]
          row_major_mesh {
            dims: [ 1, 2 ]
            dim_types: [ LINE, LINE ]
            num_connections: 2
          }
        }
        groupings {
          name: "halftray_4"
          custom_type: "halftray"
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_3 }
            , { id: 1 asic_location: ASIC_LOCATION_4 }]
          row_major_mesh {
            dims: [ 1, 2 ]
            dim_types: [ LINE, LINE ]
            num_connections: 2
          }
        }
        groupings {
          name: "meshes_25"
          custom_type: "meshes"
          instances:
          [ {
            id: 0
            grouping_ref { custom_type: "halftray" }
          }]
        }
    )proto";

    PhysicalGroupingDescriptor desc(text_proto);
    auto halftrays = desc.get_groupings_by_name("halftray");
    EXPECT_EQ(halftrays.size(), 2);
    EXPECT_EQ(halftrays[0].type, "halftray");
    EXPECT_EQ(halftrays[0].items.size(), 2);
    EXPECT_EQ(halftrays[1].items.size(), 2);

    auto meshes = desc.get_groupings_by_name("meshes");
    EXPECT_EQ(meshes.size(), 1);
    EXPECT_EQ(meshes[0].items[0].type, GroupingItemInfo::ItemType::GROUPING_REF);
    EXPECT_EQ(meshes[0].items[0].grouping_name, "halftray");
}

TEST(PhysicalGroupingDescriptorTests, GetGroupingCountReturnsCorrectCount) {
    const std::string text_proto = R"proto(
        groupings {
          name: "trays_2"
          custom_type: "trays"
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }
            , { id: 1 asic_location: ASIC_LOCATION_2 }]
          row_major_mesh {
            dims: [ 1, 2 ]
            dim_types: [ LINE, LINE ]
            num_connections: 2
          }
        }
        groupings {
          name: "meshes_26"
          custom_type: "meshes"
          instances:
          [ {
            id: 0
            grouping_ref { custom_type: "trays" }
          }]
        }
        groupings {
          name: "pods_6"
          custom_type: "pods"
          instances:
          [ {
            id: 0
            grouping_ref { custom_type: "meshes" }
          }
            , {
              id: 1
              grouping_ref { custom_type: "meshes" }
            }]
          all_to_all { num_connections: 2 }
        }
    )proto";

    PhysicalGroupingDescriptor desc(text_proto);
    EXPECT_EQ(desc.get_grouping_count(), 3);
}

TEST(PhysicalGroupingDescriptorTests, ValidateAndPopulatePreformedGroupsFromTripleClusterPSD) {
    // Load the triple cluster PSD
    const std::filesystem::path psd_file_path =
        "tests/tt_metal/tt_fabric/custom_mock_PSDs/triple_8x16_cluster_psd.textproto";
    auto physical_system_desc =
        tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(psd_file_path.string());

    // Create a PhysicalGroupingDescriptor that references preset names but doesn't define them
    const std::string text_proto = R"proto(
        groupings {
          name: "meshes_27"
          custom_type: "meshes"
          instances:
          [ {
            id: 0
            grouping_ref { preset_type: HOSTS }
          }]
        }
    )proto";

    PhysicalGroupingDescriptor pgd(text_proto);

    // Call validate_and_populate_preformed_groups_from_physical_system
    pgd.validate_and_populate_preformed_groups_from_physical_system(physical_system_desc);

    // Verify that TRAY_1, TRAY_2, TRAY_3, TRAY_4, and HOSTS were auto-populated
    auto tray1_groupings = pgd.get_groupings_by_name("TRAY_1");
    EXPECT_EQ(tray1_groupings.size(), 1u) << "TRAY_1 should be auto-populated";
    EXPECT_EQ(tray1_groupings[0].asic_count, 8u) << "TRAY_1 should have 8 ASICs";
    EXPECT_EQ(tray1_groupings[0].items.size(), 8u) << "TRAY_1 should have 8 ASIC_LOCATION items";

    auto tray2_groupings = pgd.get_groupings_by_name("TRAY_2");
    EXPECT_EQ(tray2_groupings.size(), 1u) << "TRAY_2 should be auto-populated";
    EXPECT_EQ(tray2_groupings[0].asic_count, 8u) << "TRAY_2 should have 8 ASICs";

    auto tray3_groupings = pgd.get_groupings_by_name("TRAY_3");
    EXPECT_EQ(tray3_groupings.size(), 1u) << "TRAY_3 should be auto-populated";
    EXPECT_EQ(tray3_groupings[0].asic_count, 8u) << "TRAY_3 should have 8 ASICs";

    auto tray4_groupings = pgd.get_groupings_by_name("TRAY_4");
    EXPECT_EQ(tray4_groupings.size(), 1u) << "TRAY_4 should be auto-populated";
    EXPECT_EQ(tray4_groupings[0].asic_count, 8u) << "TRAY_4 should have 8 ASICs";

    auto hosts_groupings = pgd.get_groupings_by_name("HOSTS");
    EXPECT_EQ(hosts_groupings.size(), 1u) << "HOSTS should be auto-populated";

    // Verify that auto-generated names are unique and have the correct format
    EXPECT_TRUE(tray1_groupings[0].name.find("TRAY_1_") == 0 || tray1_groupings[0].name == "TRAY_1_0")
        << "Auto-generated TRAY_1 name should have format TRAY_1_<id>, got: " << tray1_groupings[0].name;
    EXPECT_TRUE(hosts_groupings[0].name.find("HOSTS_") == 0 || hosts_groupings[0].name == "HOSTS_0")
        << "Auto-generated HOSTS name should have format HOSTS_<id>, got: " << hosts_groupings[0].name;

    // Verify all names are unique
    std::set<std::string> all_names;
    for (const auto& tray : tray1_groupings) {
        all_names.insert(tray.name);
    }
    for (const auto& tray : tray2_groupings) {
        all_names.insert(tray.name);
    }
    for (const auto& tray : tray3_groupings) {
        all_names.insert(tray.name);
    }
    for (const auto& tray : tray4_groupings) {
        all_names.insert(tray.name);
    }
    for (const auto& host : hosts_groupings) {
        all_names.insert(host.name);
    }

    EXPECT_EQ(all_names.size(), 5u) << "All auto-generated names should be unique";
    EXPECT_EQ(hosts_groupings[0].asic_count, 32u) << "HOSTS should have 32 ASICs (4 trays x 8 ASICs)";
    EXPECT_EQ(hosts_groupings[0].items.size(), 4u) << "HOSTS should have 4 GROUPING_REF items";

    // Verify that TRAY groupings have ASIC_LOCATION items (not GROUPING_REF)
    for (const auto& item : tray1_groupings[0].items) {
        EXPECT_EQ(item.type, GroupingItemInfo::ItemType::ASIC_LOCATION) << "TRAY_1 items should be ASIC_LOCATION";
    }
}

TEST(PhysicalGroupingDescriptorTests, AutoGeneratedNamesAvoidExistingNames) {
    // Load the triple cluster PSD
    const std::filesystem::path psd_file_path =
        "tests/tt_metal/tt_fabric/custom_mock_PSDs/triple_8x16_cluster_psd.textproto";
    auto physical_system_desc =
        tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(psd_file_path.string());

    // Create a PhysicalGroupingDescriptor that already has some groupings with names that would conflict
    const std::string text_proto = R"proto(
        groupings {
          name: "TRAY_1_0"
          preset_type: TRAY_1
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }
            , { id: 1 asic_location: ASIC_LOCATION_2 }
            , { id: 2 asic_location: ASIC_LOCATION_3 }
            , { id: 3 asic_location: ASIC_LOCATION_4 }
            , { id: 4 asic_location: ASIC_LOCATION_5 }
            , { id: 5 asic_location: ASIC_LOCATION_6 }
            , { id: 6 asic_location: ASIC_LOCATION_7 }
            , { id: 7 asic_location: ASIC_LOCATION_8 }]
        }
        groupings {
          name: "HOSTS_0"
          preset_type: HOSTS
          instances:
          [ {
            id: 0
            grouping_ref { preset_type: TRAY_1 }
          }
            , {
              id: 1
              grouping_ref { preset_type: TRAY_2 }
            }
            , {
              id: 2
              grouping_ref { preset_type: TRAY_3 }
            }
            , {
              id: 3
              grouping_ref { preset_type: TRAY_4 }
            }]
        }
        groupings {
          name: "meshes_28"
          custom_type: "meshes"
          instances:
          [ {
            id: 0
            grouping_ref { preset_type: HOSTS }
          }]
        }
    )proto";

    PhysicalGroupingDescriptor pgd(text_proto);

    // Call validate_and_populate_preformed_groups_from_physical_system
    // This should auto-generate TRAY_2, TRAY_3, TRAY_4, and HOSTS
    // Since TRAY_1_0 and HOSTS_0 already exist, new ones should use TRAY_1_1, HOSTS_1, etc.
    pgd.validate_and_populate_preformed_groups_from_physical_system(physical_system_desc);

    // Verify that TRAY_1_0 already exists and new trays use different IDs
    auto tray1_groupings = pgd.get_groupings_by_name("TRAY_1");
    EXPECT_GE(tray1_groupings.size(), 1u) << "TRAY_1 should exist";

    // Find the auto-generated TRAY_1 (should be TRAY_1_1 since TRAY_1_0 already exists)
    bool found_tray1_0 = false;
    bool found_tray1_1 = false;
    for (const auto& tray : tray1_groupings) {
        if (tray.name == "TRAY_1_0") {
            found_tray1_0 = true;
        } else if (tray.name == "TRAY_1_1") {
            found_tray1_1 = true;
        }
    }
    EXPECT_TRUE(found_tray1_0) << "TRAY_1_0 should exist (predefined)";

    // Verify all names are unique across all groupings
    std::set<std::string> all_names;
    auto all_groupings = pgd.get_all_groupings();
    for (const auto& grouping : all_groupings) {
        EXPECT_TRUE(all_names.find(grouping.name) == all_names.end()) << "Duplicate name found: " << grouping.name;
        all_names.insert(grouping.name);
    }

    // Verify that if TRAY_1_0 already existed, new TRAY_1 uses TRAY_1_1
    if (found_tray1_0 && tray1_groupings.size() > 1) {
        EXPECT_TRUE(found_tray1_1) << "When TRAY_1_0 exists, auto-generated TRAY_1 should use TRAY_1_1";
    }

    // Verify HOSTS names are unique - HOSTS_0 already exists, so new one should be HOSTS_1
    auto hosts_groupings = pgd.get_groupings_by_name("HOSTS");
    EXPECT_GE(hosts_groupings.size(), 1u) << "HOSTS should exist";

    bool found_hosts_0 = false;
    bool found_hosts_1 = false;
    for (const auto& host : hosts_groupings) {
        if (host.name == "HOSTS_0") {
            found_hosts_0 = true;
        } else if (host.name == "HOSTS_1") {
            found_hosts_1 = true;
        }
    }
    EXPECT_TRUE(found_hosts_0) << "HOSTS_0 should exist (predefined)";

    // If there are multiple HOSTS, verify the new one uses HOSTS_1
    if (hosts_groupings.size() > 1) {
        EXPECT_TRUE(found_hosts_1) << "When HOSTS_0 exists, auto-generated HOSTS should use HOSTS_1";
    }
}

TEST(PhysicalGroupingDescriptorTests, ValidatePreformedGroupsFromTriple16x8Groupings) {
    // Load the triple cluster PSD
    const std::filesystem::path psd_file_path =
        "tests/tt_metal/tt_fabric/custom_mock_PSDs/triple_8x16_cluster_psd.textproto";
    auto physical_system_desc =
        tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(psd_file_path.string());

    // Load the triple 16x8 groupings file which already defines TRAY_1-TRAY_4 and HOSTS
    const std::filesystem::path pgd_file_path =
        "tests/tt_metal/tt_fabric/physical_groupings/triple_16x8_quad_bh_galaxy_physical_groupings.textproto";
    PhysicalGroupingDescriptor pgd(pgd_file_path);

    // Verify that the predefined groupings exist before validation
    auto tray1_before = pgd.get_groupings_by_name("TRAY_1");
    auto tray2_before = pgd.get_groupings_by_name("TRAY_2");
    auto tray3_before = pgd.get_groupings_by_name("TRAY_3");
    auto tray4_before = pgd.get_groupings_by_name("TRAY_4");
    auto hosts_before = pgd.get_groupings_by_name("HOSTS");

    EXPECT_EQ(tray1_before.size(), 1u) << "TRAY_1 should exist before validation";
    EXPECT_EQ(tray2_before.size(), 1u) << "TRAY_2 should exist before validation";
    EXPECT_EQ(tray3_before.size(), 1u) << "TRAY_3 should exist before validation";
    EXPECT_EQ(tray4_before.size(), 1u) << "TRAY_4 should exist before validation";
    EXPECT_EQ(hosts_before.size(), 1u) << "HOSTS should exist before validation";

    // Call validate_and_populate_preformed_groups_from_physical_system
    // This should validate the existing groupings match the PSD structure
    // Since we fixed the groupings to match, validation should pass
    EXPECT_NO_THROW({ pgd.validate_and_populate_preformed_groups_from_physical_system(physical_system_desc); })
        << "Validation should pass when predefined groupings match PSD structure";

    // Verify that groupings still exist after validation (they were validated, not replaced)
    auto tray1_after = pgd.get_groupings_by_name("TRAY_1");
    auto tray2_after = pgd.get_groupings_by_name("TRAY_2");
    auto tray3_after = pgd.get_groupings_by_name("TRAY_3");
    auto tray4_after = pgd.get_groupings_by_name("TRAY_4");
    auto hosts_after = pgd.get_groupings_by_name("HOSTS");

    EXPECT_EQ(tray1_after.size(), 1u) << "TRAY_1 should exist after validation";
    EXPECT_EQ(tray2_after.size(), 1u) << "TRAY_2 should exist after validation";
    EXPECT_EQ(tray3_after.size(), 1u) << "TRAY_3 should exist after validation";
    EXPECT_EQ(tray4_after.size(), 1u) << "TRAY_4 should exist after validation";
    EXPECT_EQ(hosts_after.size(), 1u) << "HOSTS should exist after validation";

    // Verify the structure of the predefined groupings
    EXPECT_EQ(tray1_after[0].asic_count, 8u) << "TRAY_1 should have 8 ASICs";
    EXPECT_EQ(tray1_after[0].items.size(), 8u) << "TRAY_1 should have 8 ASIC_LOCATION items";
    EXPECT_EQ(tray2_after[0].asic_count, 8u) << "TRAY_2 should have 8 ASICs";
    EXPECT_EQ(tray3_after[0].asic_count, 8u) << "TRAY_3 should have 8 ASICs";
    EXPECT_EQ(tray4_after[0].asic_count, 8u) << "TRAY_4 should have 8 ASICs";
    EXPECT_EQ(hosts_after[0].asic_count, 32u) << "HOSTS should have 32 ASICs (4 trays x 8 ASICs)";
    EXPECT_EQ(hosts_after[0].items.size(), 4u) << "HOSTS should have 4 GROUPING_REF items";

    // Verify that TRAY groupings have ASIC_LOCATION items (not GROUPING_REF)
    for (const auto& item : tray1_after[0].items) {
        EXPECT_EQ(item.type, GroupingItemInfo::ItemType::ASIC_LOCATION) << "TRAY_1 items should be ASIC_LOCATION";
    }

    // Verify that HOSTS grouping has GROUPING_REF items
    for (const auto& item : hosts_after[0].items) {
        EXPECT_EQ(item.type, GroupingItemInfo::ItemType::GROUPING_REF) << "HOSTS items should be GROUPING_REF";
        // Verify that HOSTS references TRAY_1, TRAY_2, TRAY_3, TRAY_4
        EXPECT_TRUE(
            item.grouping_name == "TRAY_1" || item.grouping_name == "TRAY_2" || item.grouping_name == "TRAY_3" ||
            item.grouping_name == "TRAY_4")
            << "HOSTS should reference TRAY_1, TRAY_2, TRAY_3, or TRAY_4";
    }
}

// ============================================================================
// ASIC COUNT TESTS
// ============================================================================

TEST(PhysicalGroupingDescriptorTests, AsicCountCalculation_BaseGrouping) {
    const std::string text_proto = R"proto(
        groupings {
          name: "meshes_28"
          custom_type: "meshes"
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }
            , { id: 1 asic_location: ASIC_LOCATION_2 }
            , { id: 2 asic_location: ASIC_LOCATION_3 }
            , { id: 3 asic_location: ASIC_LOCATION_4 }]
          row_major_mesh {
            dims: [ 2, 2 ]
            dim_types: [ LINE, LINE ]
            num_connections: 2
          }
        }
    )proto";

    PhysicalGroupingDescriptor desc(text_proto);
    auto meshes = desc.get_groupings_by_name("meshes");
    ASSERT_EQ(meshes.size(), 1);
    EXPECT_EQ(meshes[0].asic_count, 4u);
}

TEST(PhysicalGroupingDescriptorTests, AsicCountCalculation_NestedGroupings) {
    const std::string text_proto = R"proto(
        groupings {
          name: "trays_3"
          custom_type: "trays"
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }
            , { id: 1 asic_location: ASIC_LOCATION_2 }
            , { id: 2 asic_location: ASIC_LOCATION_3 }
            , { id: 3 asic_location: ASIC_LOCATION_4 }
            , { id: 4 asic_location: ASIC_LOCATION_5 }
            , { id: 5 asic_location: ASIC_LOCATION_6 }
            , { id: 6 asic_location: ASIC_LOCATION_7 }
            , { id: 7 asic_location: ASIC_LOCATION_8 }]
          row_major_mesh {
            dims: [ 2, 4 ]
            dim_types: [ LINE, LINE ]
            num_connections: 2
          }
        }
        groupings {
          name: "hosts_1"
          custom_type: "hosts"
          instances:
          [ {
            id: 0
            grouping_ref { custom_type: "trays" }
          }
            , {
              id: 1
              grouping_ref { custom_type: "trays" }
            }
            , {
              id: 2
              grouping_ref { custom_type: "trays" }
            }
            , {
              id: 3
              grouping_ref { custom_type: "trays" }
            }]
          row_major_mesh {
            dims: [ 1, 4 ]
            dim_types: [ LINE, RING ]
            num_connections: 2
          }
        }
        groupings {
          name: "meshes_29"
          custom_type: "meshes"
          instances:
          [ {
            id: 0
            grouping_ref { custom_type: "hosts" }
          }]
        }
    )proto";

    PhysicalGroupingDescriptor desc(text_proto);
    auto trays = desc.get_groupings_by_name("trays");
    auto hosts = desc.get_groupings_by_name("hosts");
    auto meshes = desc.get_groupings_by_name("meshes");
    ASSERT_EQ(trays.size(), 1);
    ASSERT_EQ(hosts.size(), 1);
    ASSERT_EQ(meshes.size(), 1);
    EXPECT_EQ(trays[0].asic_count, 8u);
    EXPECT_EQ(hosts[0].asic_count, 32u);   // 4 * 8
    EXPECT_EQ(meshes[0].asic_count, 32u);  // 1 * 32
}

// ============================================================================
// GET_VALID_GROUPINGS_FOR_MGD TESTS (unchanged - use file-based configs)
// ============================================================================

TEST(PhysicalGroupingDescriptorTests, GetValidGroupingsForMGD_32x4Quad) {
    const std::filesystem::path pgd_file_path =
        "tests/tt_metal/tt_fabric/physical_groupings/triple_16x8_quad_bh_galaxy_physical_groupings.textproto";
    PhysicalGroupingDescriptor pgd(pgd_file_path);

    const std::filesystem::path mgd_file_path =
        "tt_metal/fabric/mesh_graph_descriptors/32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto";
    MeshGraphDescriptor mgd(mgd_file_path);

    auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd);
    EXPECT_FALSE(valid_groupings.empty());

    // Verify that mesh type is matched
    EXPECT_TRUE(valid_groupings.find("MESH") != valid_groupings.end());
    if (valid_groupings.find("MESH") != valid_groupings.end()) {
        const auto& mesh_groupings = valid_groupings.at("MESH");
        EXPECT_TRUE(mesh_groupings.find("M0") != mesh_groupings.end());
        if (mesh_groupings.find("M0") != mesh_groupings.end()) {
            const auto& assigned_grouping = mesh_groupings.at("M0");
            // 32x4 = 128 ASICs with host_topology [4, 1] (1x4 linear), should match "4x32_Mesh" (dims [1, 4])
            EXPECT_EQ(assigned_grouping.asic_count, 128u) << "32x4 mesh should match 128 ASIC grouping";
            EXPECT_EQ(assigned_grouping.name, "4x32_Mesh")
                << "32x4 mesh with host_topology [4, 1] should be assigned to 4x32_Mesh (dims [1, 4]), got: "
                << assigned_grouping.name;
            // Verify connections: 4x32_Mesh has 4 instances with row_major_mesh connections (dims [1, 4])
            EXPECT_GE(assigned_grouping.items.size(), 4u)
                << "Grouping " << assigned_grouping.name << " should have at least 4 items (instances) for connections";
        }
    }
}

TEST(PhysicalGroupingDescriptorTests, GetValidGroupingsForMGD_SingleBHGalaxy) {
    const std::filesystem::path pgd_file_path =
        "tests/tt_metal/tt_fabric/physical_groupings/triple_16x8_quad_bh_galaxy_physical_groupings.textproto";
    PhysicalGroupingDescriptor pgd(pgd_file_path);

    const std::filesystem::path mgd_file_path =
        "tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto";
    MeshGraphDescriptor mgd(mgd_file_path);

    auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd);
    EXPECT_FALSE(valid_groupings.empty());

    // Verify that mesh type is matched
    EXPECT_TRUE(valid_groupings.find("MESH") != valid_groupings.end());
    if (valid_groupings.find("MESH") != valid_groupings.end()) {
        const auto& mesh_groupings = valid_groupings.at("MESH");
        EXPECT_TRUE(mesh_groupings.find("M0") != mesh_groupings.end());
        if (mesh_groupings.find("M0") != mesh_groupings.end()) {
            const auto& assigned_grouping = mesh_groupings.at("M0");
            // 8x4 = 32 ASICs with host_topology [1, 1] (single host), should match "4x8_Mesh" (1 instance)
            EXPECT_EQ(assigned_grouping.asic_count, 32u) << "8x4 mesh should match 32 ASIC grouping";
            EXPECT_EQ(assigned_grouping.name, "4x8_Mesh")
                << "8x4 mesh with host_topology [1, 1] should be assigned to 4x8_Mesh (single instance), got: "
                << assigned_grouping.name;
            // Verify connections: 4x8_Mesh has 1 instance, so adjacency graph should be empty or have 1 node
            EXPECT_LE(assigned_grouping.adjacency_graph.get_nodes().size(), 1u)
                << "Grouping 4x8_Mesh should have at most 1 node in adjacency graph (single instance)";
        }
    }
}

TEST(PhysicalGroupingDescriptorTests, GetValidGroupingsForMGD_16x8QuadBHGalaxy) {
    const std::filesystem::path pgd_file_path =
        "tests/tt_metal/tt_fabric/physical_groupings/triple_16x8_quad_bh_galaxy_physical_groupings.textproto";
    PhysicalGroupingDescriptor pgd(pgd_file_path);

    const std::filesystem::path mgd_file_path =
        "tt_metal/fabric/mesh_graph_descriptors/16x8_quad_bh_galaxy_torus_xy_graph_descriptor.textproto";
    MeshGraphDescriptor mgd(mgd_file_path);

    auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd);
    EXPECT_FALSE(valid_groupings.empty());

    // Verify that mesh type is matched
    EXPECT_TRUE(valid_groupings.find("MESH") != valid_groupings.end());
    if (valid_groupings.find("MESH") != valid_groupings.end()) {
        const auto& mesh_groupings = valid_groupings.at("MESH");
        EXPECT_TRUE(mesh_groupings.find("M0") != mesh_groupings.end());
        if (mesh_groupings.find("M0") != mesh_groupings.end()) {
            const auto& assigned_grouping = mesh_groupings.at("M0");
            // 16x8 = 128 ASICs with host_topology [2, 2] (2x2 grid), should match "8x16_Mesh" (dims [2, 2])
            EXPECT_EQ(assigned_grouping.asic_count, 128u) << "16x8 mesh should match 128 ASIC grouping";
            EXPECT_EQ(assigned_grouping.name, "8x16_Mesh")
                << "16x8 mesh with host_topology [2, 2] should be assigned to 8x16_Mesh (dims [2, 2]), got: "
                << assigned_grouping.name;
            // Verify connections: 8x16_Mesh has 4 instances with row_major_mesh connections (dims [2, 2])
            EXPECT_EQ(assigned_grouping.adjacency_graph.get_nodes().size(), 4u)
                << "Grouping 8x16_Mesh should have 4 nodes in adjacency graph";
            // Verify that connections exist by checking that at least one node has neighbors
            bool has_connections = false;
            for (const auto& node : assigned_grouping.adjacency_graph.get_nodes()) {
                if (!assigned_grouping.adjacency_graph.get_neighbors(node).empty()) {
                    has_connections = true;
                    break;
                }
            }
            EXPECT_TRUE(has_connections)
                << "Grouping 8x16_Mesh should have connections (4 instances with row_major_mesh)";
        }
    }
}

TEST(PhysicalGroupingDescriptorTests, GetValidGroupingsForMGD_Triple16x8QuadGalaxy) {
    const std::filesystem::path pgd_file_path =
        "tests/tt_metal/tt_fabric/physical_groupings/triple_16x8_quad_bh_galaxy_physical_groupings.textproto";
    PhysicalGroupingDescriptor pgd(pgd_file_path);

    const std::filesystem::path mgd_file_path =
        "tt_metal/fabric/mesh_graph_descriptors/triple_pod_16x8_quad_bh_galaxy_torus_xy_graph_descriptor.textproto";
    MeshGraphDescriptor mgd(mgd_file_path);

    auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd);
    EXPECT_FALSE(valid_groupings.empty());

    // Verify that mesh type is matched
    EXPECT_TRUE(valid_groupings.find("MESH") != valid_groupings.end());
    if (valid_groupings.find("MESH") != valid_groupings.end()) {
        const auto& mesh_groupings = valid_groupings.at("MESH");
        EXPECT_TRUE(mesh_groupings.find("M0") != mesh_groupings.end());
        if (mesh_groupings.find("M0") != mesh_groupings.end()) {
            const auto& assigned_grouping = mesh_groupings.at("M0");
            // 16x8 = 128 ASICs with host_topology [2, 2] (2x2 grid), should match "8x16_Mesh" (dims [2, 2])
            EXPECT_EQ(assigned_grouping.asic_count, 128u) << "16x8 mesh should match 128 ASIC grouping";
            EXPECT_EQ(assigned_grouping.name, "8x16_Mesh")
                << "16x8 mesh with host_topology [2, 2] should be assigned to 8x16_Mesh (dims [2, 2]), got: "
                << assigned_grouping.name;
            // Verify connections: 8x16_Mesh has 4 instances with row_major_mesh connections (dims [2, 2])
            EXPECT_EQ(assigned_grouping.adjacency_graph.get_nodes().size(), 4u)
                << "Grouping 8x16_Mesh should have 4 nodes in adjacency graph";
            // Verify that connections exist by checking that at least one node has neighbors
            bool has_connections = false;
            for (const auto& node : assigned_grouping.adjacency_graph.get_nodes()) {
                if (!assigned_grouping.adjacency_graph.get_neighbors(node).empty()) {
                    has_connections = true;
                    break;
                }
            }
            EXPECT_TRUE(has_connections)
                << "Grouping 8x16_Mesh should have connections (4 instances with row_major_mesh)";
        }
    }

    // Verify that graph instances are matched (if any)
    for (const auto& [type, name_map] : valid_groupings) {
        if (type != "MESH") {
            EXPECT_FALSE(name_map.empty()) << "Graph type " << type << " should have matched groupings";
        }
    }
}

TEST(PhysicalGroupingDescriptorTests, GetValidGroupingsForMGD_BHGalaxyPipeline) {
    const std::filesystem::path pgd_file_path =
        "tests/tt_metal/tt_fabric/physical_groupings/triple_16x8_quad_bh_galaxy_physical_groupings.textproto";
    PhysicalGroupingDescriptor pgd(pgd_file_path);

    const std::filesystem::path mgd_file_path =
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/bh_galaxy_2x4_pipeline.textproto";
    MeshGraphDescriptor mgd(mgd_file_path);

    auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd);
    EXPECT_FALSE(valid_groupings.empty());

    // Verify that mesh type is matched
    EXPECT_TRUE(valid_groupings.find("MESH") != valid_groupings.end());
    if (valid_groupings.find("MESH") != valid_groupings.end()) {
        const auto& mesh_groupings = valid_groupings.at("MESH");
        EXPECT_TRUE(mesh_groupings.find("MESH") != mesh_groupings.end());
        if (mesh_groupings.find("MESH") != mesh_groupings.end()) {
            const auto& assigned_grouping = mesh_groupings.at("MESH");
            // 2x4 = 8 ASICs with host_topology [1, 1] (single host), should match "2x4_Mesh" (single instance)
            EXPECT_EQ(assigned_grouping.asic_count, 8u) << "2x4 mesh should match 8 ASIC grouping";
            EXPECT_EQ(assigned_grouping.name, "2x4_Mesh")
                << "2x4 mesh with host_topology [1, 1] should be assigned to 2x4_Mesh (single instance), got: "
                << assigned_grouping.name;
            // Verify connections: 2x4_Mesh has 1 instance, so adjacency graph should be empty or have 1 node
            EXPECT_LE(assigned_grouping.adjacency_graph.get_nodes().size(), 1u)
                << "Grouping 2x4_Mesh should have at most 1 node in adjacency graph (single instance)";
        }
    }
}

TEST(PhysicalGroupingDescriptorTests, GetValidGroupingsForMGD_4x4) {
    const std::filesystem::path pgd_file_path =
        "tests/tt_metal/tt_fabric/physical_groupings/triple_16x8_quad_bh_galaxy_physical_groupings.textproto";
    PhysicalGroupingDescriptor pgd(pgd_file_path);

    const std::filesystem::path mgd_file_path =
        "tt_metal/fabric/mesh_graph_descriptors/bh_qb_4x4_mesh_graph_descriptor.textproto";
    MeshGraphDescriptor mgd(mgd_file_path);

    auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd);
    EXPECT_FALSE(valid_groupings.empty());

    // Verify that mesh type is matched
    EXPECT_TRUE(valid_groupings.find("MESH") != valid_groupings.end());
    if (valid_groupings.find("MESH") != valid_groupings.end()) {
        const auto& mesh_groupings = valid_groupings.at("MESH");
        EXPECT_TRUE(mesh_groupings.find("M0") != mesh_groupings.end());
        if (mesh_groupings.find("M0") != mesh_groupings.end()) {
            const auto& assigned_grouping = mesh_groupings.at("M0");
            // 4x4 = 16 ASICs, should match "4x4_Mesh", "2x8_Mesh" (first), or "2x8_Mesh" (second) - all are 16 ASICs
            EXPECT_EQ(assigned_grouping.asic_count, 16u) << "4x4 mesh should match 16 ASIC grouping";
            EXPECT_TRUE(assigned_grouping.name == "4x4_Mesh" || assigned_grouping.name == "2x8_Mesh")
                << "4x4 mesh should be assigned to 4x4_Mesh or 2x8_Mesh, got: " << assigned_grouping.name;
            // Verify connections: all these groupings have 2 instances with row_major_mesh connections
            // Connections are defined in textproto: row_major_mesh topology connecting the 2 instances
            // The adjacency graph represents the topology between the 2 instances
            EXPECT_GE(assigned_grouping.items.size(), 2u)
                << "Grouping " << assigned_grouping.name << " should have at least 2 items (instances) for connections";
        }
    }
}

TEST(PhysicalGroupingDescriptorTests, GetValidGroupingsForMGD_8x2) {
    const std::filesystem::path pgd_file_path =
        "tests/tt_metal/tt_fabric/physical_groupings/triple_16x8_quad_bh_galaxy_physical_groupings.textproto";
    PhysicalGroupingDescriptor pgd(pgd_file_path);

    const std::filesystem::path mgd_file_path =
        "tt_metal/fabric/mesh_graph_descriptors/bh_8x2_mesh_graph_descriptor.textproto";
    MeshGraphDescriptor mgd(mgd_file_path);

    auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd);
    EXPECT_FALSE(valid_groupings.empty());

    // Verify that mesh type is matched
    EXPECT_TRUE(valid_groupings.find("MESH") != valid_groupings.end());
    if (valid_groupings.find("MESH") != valid_groupings.end()) {
        const auto& mesh_groupings = valid_groupings.at("MESH");
        EXPECT_TRUE(mesh_groupings.find("M0") != mesh_groupings.end());
        if (mesh_groupings.find("M0") != mesh_groupings.end()) {
            const auto& assigned_grouping = mesh_groupings.at("M0");
            // 8x2 = 16 ASICs, should match "4x4_Mesh", "2x8_Mesh" (first), or "2x8_Mesh" (second) - all are 16 ASICs
            EXPECT_EQ(assigned_grouping.asic_count, 16u) << "8x2 mesh should match 16 ASIC grouping";
            EXPECT_TRUE(assigned_grouping.name == "4x4_Mesh" || assigned_grouping.name == "2x8_Mesh")
                << "8x2 mesh should be assigned to 4x4_Mesh or 2x8_Mesh, got: " << assigned_grouping.name;
            // Verify connections: all these groupings have 2 instances with row_major_mesh connections
            // Connections are defined in textproto: row_major_mesh topology connecting the 2 instances
            // The adjacency graph represents the topology between the 2 instances
            EXPECT_GE(assigned_grouping.items.size(), 2u)
                << "Grouping " << assigned_grouping.name << " should have at least 2 items (instances) for connections";
        }
    }
}
}  // namespace tt::tt_fabric::fabric_router_tests
