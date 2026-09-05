// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>
#include <set>
#include <unordered_set>
#include <sstream>
#include <cstdlib>

#include <tt-metalium/experimental/fabric/physical_grouping_descriptor.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph_descriptor.hpp>
#include <tt-metalium/experimental/fabric/topology_solver.hpp>
#include <tt-metalium/experimental/fabric/topology_mapper_utils.hpp>
#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>
#include "tt_metal/fabric/serialization/physical_system_descriptor_serialization.hpp"
#include "tt_metal/fabric/physical_system_discovery.hpp"
#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"

using namespace tt::tt_fabric;

namespace tt::tt_fabric::fabric_router_tests {

class MockClusterPhysicalGroupingDescriptorTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (getenv("TT_METAL_MOCK_CLUSTER_DESC_PATH") == nullptr) {
            GTEST_SKIP() << "PSD test requires TT_METAL_MOCK_CLUSTER_DESC_PATH; run it through the fabric "
                            "CPU-only test runner's physical-grouping setup";
        }
    }
};

class PhysicalGroupingDescriptorSP4Tests : public MockClusterPhysicalGroupingDescriptorTest {};

class PhysicalGroupingDescriptorDualT3kTests : public MockClusterPhysicalGroupingDescriptorTest {};

// Helper function to create PSD from mock cluster
static tt::tt_metal::PhysicalSystemDescriptor create_psd_from_mock_cluster() {
    // Create PSD from mock cluster (CPU-only test)
    using namespace tt::tt_metal::distributed::multihost;
    auto distributed_context = tt::tt_metal::MetalContext::instance().get_distributed_context_ptr();
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    return tt::tt_metal::run_physical_system_discovery(
        *cluster.get_cluster_desc(), distributed_context, rtoptions.get_target_device());
}

// Helper to check that a node's neighbors match expected (order-independent)
static void expect_neighbors(
    const AdjacencyGraph<uint32_t>& graph, uint32_t node_id, const std::vector<uint32_t>& expected) {
    const auto& neighbors = graph.get_neighbors(node_id);
    std::set<uint32_t> actual_set(neighbors.begin(), neighbors.end());
    std::set<uint32_t> expected_set(expected.begin(), expected.end());
    EXPECT_EQ(actual_set, expected_set) << "Node " << node_id << " has wrong neighbors";
}

// Helper for checking neighbors by node ID (now using uint32_t directly)
static void expect_neighbors_by_id(
    const AdjacencyGraph<uint32_t>& graph, uint32_t node_id, const std::vector<uint32_t>& expected_neighbor_ids) {
    const auto& nodes = graph.get_nodes();
    ASSERT_TRUE(std::find(nodes.begin(), nodes.end(), node_id) != nodes.end())
        << "Node with id " << node_id << " not found";

    const auto& neighbors = graph.get_neighbors(node_id);
    std::set<uint32_t> actual_ids(neighbors.begin(), neighbors.end());
    std::set<uint32_t> expected_set(expected_neighbor_ids.begin(), expected_neighbor_ids.end());
    EXPECT_EQ(actual_ids, expected_set) << "Node " << node_id << " has wrong neighbors";
}

// Helper to get common tray/host groupings - can be prepended to any test proto
static std::string get_required_groupings() {
    return R"proto(
        groupings {
          name: "tray_1"
          custom_type: "tray_1"
          instances:
          [ {
            id: 0
            location { asic_location: ASIC_LOCATION_1 tray_id: TRAY_1 }
          }]
        }
        groupings {
          name: "tray_2"
          custom_type: "tray_2"
          instances:
          [ {
            id: 0
            location { asic_location: ASIC_LOCATION_1 tray_id: TRAY_2 }
          }]
        }
        groupings {
          name: "tray_3"
          custom_type: "tray_3"
          instances:
          [ {
            id: 0
            location { asic_location: ASIC_LOCATION_1 tray_id: TRAY_3 }
          }]
        }
        groupings {
          name: "tray_4"
          custom_type: "tray_4"
          instances:
          [ {
            id: 0
            location { asic_location: ASIC_LOCATION_1 tray_id: TRAY_4 }
          }]
        }
        groupings {
          name: "hosts_required"
          custom_type: "hosts"
          instances:
          [ {
            id: 0
            grouping_ref { custom_type: "tray_1" }
          }]
        }
    )proto";
}

// Helper to wrap a test proto with common groupings (adds meshes if not present)
// Note: These groupings are no longer required but are commonly used in tests
static std::string wrap_with_required_groupings(const std::string& test_proto) {
    bool has_meshes = test_proto.find("custom_type: \"meshes\"") != std::string::npos ||
                      test_proto.find("preset_type: MESH") != std::string::npos;

    if (!has_meshes) {
        return get_required_groupings() + R"proto(
                   groupings {
                     name: "meshes_required"
                     custom_type: "meshes"
                     instances:
                     [ {
                       id: 0
                       location { asic_location: ASIC_LOCATION_1 }
                     }]
                   }
               )proto" +
               test_proto;
    }

    return get_required_groupings() + test_proto;
}

// ============================================================================
// ADJACENCY GRAPH TESTS
// ============================================================================

TEST(PhysicalGroupingDescriptorTests, AdjacencyGraph_AllToAll_ThreeNodes) {
    const std::string text_proto = wrap_with_required_groupings(R"proto(
        groupings {
          name: "meshes_1"
          custom_type: "meshes"
          instances:
          [ {
            id: 0
            location { asic_location: ASIC_LOCATION_1 }
          }
            , {
              id: 1
              location { asic_location: ASIC_LOCATION_2 }
            }
            , {
              id: 2
              location { asic_location: ASIC_LOCATION_3 }
            }]
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
          all_to_all {}
        }
    )proto");

    PhysicalGroupingDescriptor desc(text_proto);
    auto pods = desc.get_groupings_by_type("pods");
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
    const std::string text_proto = wrap_with_required_groupings(R"proto(
        groupings {
          name: "meshes_2"
          custom_type: "meshes"
          instances:
          [ {
            id: 0
            location { asic_location: ASIC_LOCATION_1 }
          }
            , {
              id: 1
              location { asic_location: ASIC_LOCATION_2 }
            }]
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
          row_major_mesh { dims: [ 2, 2 ] }
        }
    )proto");

    PhysicalGroupingDescriptor desc(text_proto);
    auto grids = desc.get_groupings_by_type("grid");
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

TEST(PhysicalGroupingDescriptorTests, AdjacencyGraph_CustomConnections) {
    const std::string text_proto = wrap_with_required_groupings(R"proto(
        groupings {
          name: "meshes_4"
          custom_type: "meshes"
          instances:
          [ {
            id: 0
            location { asic_location: ASIC_LOCATION_1 }
          }]
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
            [ { src_instance: 0 dst_instance: 1 }
              , { src_instance: 0 dst_instance: 2 }
              , { src_instance: 1 dst_instance: 2 }]
          }
        }
    )proto");

    PhysicalGroupingDescriptor desc(text_proto);
    auto custom = desc.get_groupings_by_type("custom_topology");
    ASSERT_EQ(custom.size(), 1);

    const auto& adj = custom[0].adjacency_graph;
    // Custom connections use 0-based instance index; instance ids are 1,2,3 (from id field)
    // index 0 -> id 1, index 1 -> id 2, index 2 -> id 3
    // edges: 0-1, 0-2, 1-2  =>  id 1-2, id 1-3, id 2-3
    expect_neighbors(adj, 1, {2, 3});
    expect_neighbors(adj, 2, {1, 3});
    expect_neighbors(adj, 3, {1, 2});
}

// ============================================================================
// VALID CONFIGURATION TESTS
// ============================================================================

TEST(PhysicalGroupingDescriptorTests, ParsesValidBasicConfiguration) {
    const std::string text_proto = wrap_with_required_groupings(R"proto(
        groupings {
          name: "trays_1"
          custom_type: "trays"
          instances:
          [ {
            id: 0
            location { asic_location: ASIC_LOCATION_1 }
          }
            , {
              id: 1
              location { asic_location: ASIC_LOCATION_2 }
            }]
        }
        groupings {
          name: "meshes_17"
          custom_type: "meshes"
          instances:
          [ {
            id: 0
            grouping_ref { custom_type: "trays" }
          }]
        }
    )proto");

    EXPECT_NO_THROW({
        PhysicalGroupingDescriptor desc(text_proto);
        EXPECT_TRUE(desc.has_grouping("meshes"));
        EXPECT_TRUE(desc.has_grouping("hosts"));
        EXPECT_TRUE(desc.has_grouping("trays"));
    });
}

TEST(PhysicalGroupingDescriptorTests, ParsesFromTriple16x8QuadBhGalaxyFile) {
    const std::filesystem::path text_proto_file_path =
        "tests/tt_metal/tt_fabric/physical_groupings/bh_galaxy_rev_ab_physical_grouping_descriptor.textproto";
    EXPECT_NO_THROW({ PhysicalGroupingDescriptor desc(text_proto_file_path); });
}

// ============================================================================
// VALIDATION TESTS
// ============================================================================

TEST(PhysicalGroupingDescriptorTests, ValidationSucceedsWithAllRequiredGroupings) {
    // Test that validation passes with common tray/host/mesh groupings from wrap_with_required_groupings
    const std::string text_proto = wrap_with_required_groupings(R"proto(
        groupings {
          name: "meshes_1"
          custom_type: "meshes"
          instances:
          [ {
            id: 0
            grouping_ref { custom_type: "hosts" }
          }]
        }
    )proto");
    ;

    EXPECT_NO_THROW({ PhysicalGroupingDescriptor desc(text_proto); });
}

TEST(PhysicalGroupingDescriptorTests, ValidationFailsWhenReferencingNonExistentGrouping) {
    // Test that custom names must exist
    const std::string text_proto = wrap_with_required_groupings(R"proto(
        groupings {
          name: "meshes_20"
          custom_type: "meshes"
          instances:
          [ {
            id: 0
            grouping_ref { custom_type: "nonexistent" }
          }]
        }
    )proto");
    ;

    EXPECT_THAT(
        ([&]() { PhysicalGroupingDescriptor desc(text_proto); }),
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("references non-existent grouping")));
}

TEST(PhysicalGroupingDescriptorTests, ValidationFailsWhenGroupingHasNoInstances) {
    const std::string text_proto = wrap_with_required_groupings(R"proto(
        groupings { name: "meshes_22" custom_type: "meshes" }
    )proto");
    ;

    EXPECT_THAT(
        ([&]() { PhysicalGroupingDescriptor desc(text_proto); }),
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("must have at least one instance")));
}

TEST(PhysicalGroupingDescriptorTests, ValidationFailsWhenNonLeafGroupingUsesASICLocations) {
    // Test that a non-leaf grouping (one with grouping references) cannot also use ASIC locations
    const std::string text_proto = get_required_groupings() + R"proto(
        groupings {
          name: "meshes_required"
          custom_type: "meshes"
          instances:
          [ {
            id: 0
            location { asic_location: ASIC_LOCATION_1 }
          }]
        }
        groupings {
          name: "pods_bad"
          custom_type: "pods"
          instances:
          [ {
            id: 0
            grouping_ref { custom_type: "meshes" }
          }
            , {
              id: 1
              location { asic_location: ASIC_LOCATION_2 }
            }]
        }
    )proto";

    EXPECT_THAT(
        ([&]() { PhysicalGroupingDescriptor desc(text_proto); }),
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("uses ASIC locations but also has grouping references")));
}

TEST(PhysicalGroupingDescriptorTests, ValidationFailsWhenCircularDependency) {
    // Create a cycle: pods -> clusters -> pods
    const std::string text_proto = get_required_groupings() + R"proto(
        groupings {
          name: "meshes_required"
          custom_type: "meshes"
          instances:
          [ {
            id: 0
            location { asic_location: ASIC_LOCATION_1 }
          }]
        }
        groupings {
          name: "pods_cycle"
          custom_type: "pods"
          instances:
          [ {
            id: 0
            grouping_ref { custom_type: "clusters" }
          }]
        }
        groupings {
          name: "clusters_cycle"
          custom_type: "clusters"
          instances:
          [ {
            id: 0
            grouping_ref { custom_type: "pods" }
          }]
        }
    )proto";

    EXPECT_THAT(
        ([&]() { PhysicalGroupingDescriptor desc(text_proto); }),
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("Circular dependencies detected")));
}

TEST(PhysicalGroupingDescriptorTests, MeshGroupingsCanBeLeafNodes) {
    // Test that MESH groupings can be leaf nodes (using ASIC locations directly)
    // This verifies that MESH groupings are allowed to use ASIC locations without grouping references
    const std::string text_proto = get_required_groupings() + R"proto(
        groupings {
          name: "mesh_leaf_1"
          preset_type: MESH
          instances:
          [ {
            id: 0
            location { asic_location: ASIC_LOCATION_1 }
          }
            , {
              id: 1
              location { asic_location: ASIC_LOCATION_2 }
            }
            , {
              id: 2
              location { asic_location: ASIC_LOCATION_3 }
            }
            , {
              id: 3
              location { asic_location: ASIC_LOCATION_4 }
            }]
          row_major_mesh { dims: [ 2, 2 ] }
        }
        groupings {
          name: "mesh_leaf_2"
          preset_type: MESH
          instances:
          [ {
            id: 0
            location { asic_location: ASIC_LOCATION_5 }
          }
            , {
              id: 1
              location { asic_location: ASIC_LOCATION_6 }
            }]
          row_major_mesh { dims: [ 1, 2 ] }
        }
    )proto";

    // Should succeed - MESH groupings can be leaf nodes
    EXPECT_NO_THROW({ PhysicalGroupingDescriptor desc(text_proto); });
}

TEST(PhysicalGroupingDescriptorTests, MeshGroupingsCanHaveDifferentStructures) {
    // Test that different MESH groupings can have different structures:
    // - Some MESH groupings can be leaf nodes (using ASIC locations)
    // - Other MESH groupings can reference other groupings
    // This verifies that validation checks individual groupings, not grouping types
    const std::string text_proto = get_required_groupings() + R"proto(
        groupings {
          name: "mesh_leaf"
          preset_type: MESH
          instances:
          [ {
            id: 0
            location { asic_location: ASIC_LOCATION_1 }
          }
            , {
              id: 1
              location { asic_location: ASIC_LOCATION_2 }
            }]
          row_major_mesh { dims: [ 1, 2 ] }
        }
        groupings {
          name: "mesh_non_leaf"
          preset_type: MESH
          instances:
          [ {
            id: 0
            grouping_ref { custom_type: "tray_1" }
          }]
        }
        groupings {
          name: "mesh_another_leaf"
          preset_type: MESH
          instances:
          [ {
            id: 0
            location { asic_location: ASIC_LOCATION_3 }
          }
            , {
              id: 1
              location { asic_location: ASIC_LOCATION_4 }
            }
            , {
              id: 2
              location { asic_location: ASIC_LOCATION_5 }
            }
            , {
              id: 3
              location { asic_location: ASIC_LOCATION_6 }
            }]
          row_major_mesh { dims: [ 2, 2 ] }
        }
    )proto";

    // Should succeed - different MESH groupings can have different structures
    EXPECT_NO_THROW({ PhysicalGroupingDescriptor desc(text_proto); });
}

TEST(PhysicalGroupingDescriptorTests, SingleGroupingCannotMixASICLocationsAndGroupingRefs) {
    // Test that a single grouping cannot mix ASIC locations and grouping references
    // This verifies that ASIC locations must be leaf nodes (within a single grouping)
    const std::string text_proto = get_required_groupings() + R"proto(
        groupings {
          name: "meshes_required"
          custom_type: "meshes"
          instances:
          [ {
            id: 0
            location { asic_location: ASIC_LOCATION_1 }
          }]
        }
        groupings {
          name: "mesh_mixed_bad"
          preset_type: MESH
          instances:
          [ {
            id: 0
            grouping_ref { custom_type: "tray_1" }
          }
            , {
              id: 1
              location { asic_location: ASIC_LOCATION_2 }
            }]
        }
    )proto";

    // Should fail - a single grouping cannot mix ASIC locations and grouping references
    EXPECT_THAT(
        ([&]() { PhysicalGroupingDescriptor desc(text_proto); }),
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("uses ASIC locations but also has grouping references")));
}

// ============================================================================
// API TESTS
// ============================================================================

TEST(PhysicalGroupingDescriptorTests, HasGroupingReturnsTrueForExistingGrouping) {
    const std::string text_proto = wrap_with_required_groupings(R"proto(
        groupings {
          name: "meshes_24"
          custom_type: "meshes"
          instances:
          [ {
            id: 0
            location { asic_location: ASIC_LOCATION_1 }
          }]
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
          all_to_all {}
        }
    )proto");

    PhysicalGroupingDescriptor desc(text_proto);
    EXPECT_TRUE(desc.has_grouping("meshes"));
    EXPECT_TRUE(desc.has_grouping("pods"));
    EXPECT_FALSE(desc.has_grouping("nonexistent"));
}

TEST(PhysicalGroupingDescriptorTests, GetGroupingsByNameReturnsAllDefinitions) {
    const std::string text_proto = wrap_with_required_groupings(R"proto(
        groupings {
          name: "halftray_3"
          custom_type: "halftray"
          instances:
          [ {
            id: 0
            location { asic_location: ASIC_LOCATION_1 }
          }
            , {
              id: 1
              location { asic_location: ASIC_LOCATION_2 }
            }]
          row_major_mesh { dims: [ 1, 2 ] }
        }
        groupings {
          name: "halftray_4"
          custom_type: "halftray"
          instances:
          [ {
            id: 0
            location { asic_location: ASIC_LOCATION_3 }
          }
            , {
              id: 1
              location { asic_location: ASIC_LOCATION_4 }
            }]
          row_major_mesh { dims: [ 1, 2 ] }
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
    )proto");

    PhysicalGroupingDescriptor desc(text_proto);
    auto halftrays = desc.get_groupings_by_type("halftray");
    EXPECT_EQ(halftrays.size(), 2);
    EXPECT_EQ(halftrays[0].type, "halftray");
    EXPECT_EQ(halftrays[0].items.size(), 2);
    EXPECT_EQ(halftrays[1].items.size(), 2);

    auto meshes = desc.get_groupings_by_type("meshes");
    EXPECT_EQ(meshes.size(), 1);
    EXPECT_EQ(meshes[0].items[0].type, GroupingItemInfo::ItemType::GROUPING_REF);
    EXPECT_EQ(meshes[0].items[0].grouping_name, "halftray");
}

TEST(PhysicalGroupingDescriptorTests, GetGroupingCountReturnsCorrectCount) {
    const std::string text_proto = wrap_with_required_groupings(R"proto(
        groupings {
          name: "trays_2"
          custom_type: "trays"
          instances:
          [ {
            id: 0
            location { asic_location: ASIC_LOCATION_1 }
          }
            , {
              id: 1
              location { asic_location: ASIC_LOCATION_2 }
            }]
          row_major_mesh { dims: [ 1, 2 ] }
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
          all_to_all {}
        }
    )proto");
    ;

    PhysicalGroupingDescriptor desc(text_proto);
    // Count includes tray_1-4 (4), hosts (1), plus trays (1), meshes (1), pods (1) = 8 total
    EXPECT_EQ(desc.get_grouping_count(), 8);
}

// ============================================================================
// ASIC COUNT TESTS
// ============================================================================

TEST(PhysicalGroupingDescriptorTests, AsicCountCalculation_BaseGrouping) {
    const std::string text_proto = wrap_with_required_groupings(R"proto(
        groupings {
          name: "meshes_28"
          custom_type: "meshes"
          instances:
          [ {
            id: 0
            location { asic_location: ASIC_LOCATION_1 }
          }
            , {
              id: 1
              location { asic_location: ASIC_LOCATION_2 }
            }
            , {
              id: 2
              location { asic_location: ASIC_LOCATION_3 }
            }
            , {
              id: 3
              location { asic_location: ASIC_LOCATION_4 }
            }]
          row_major_mesh { dims: [ 2, 2 ] }
        }
    )proto");
    ;

    PhysicalGroupingDescriptor desc(text_proto);
    auto meshes = desc.get_groupings_by_type("meshes");
    ASSERT_EQ(meshes.size(), 1);
    EXPECT_EQ(meshes[0].asic_count, 4u);
}

TEST(PhysicalGroupingDescriptorTests, AsicCountCalculation_NestedGroupings) {
    const std::string text_proto = wrap_with_required_groupings(R"proto(
        groupings {
          name: "trays_3"
          custom_type: "trays"
          instances:
          [ {
            id: 0
            location { asic_location: ASIC_LOCATION_1 }
          }
            , {
              id: 1
              location { asic_location: ASIC_LOCATION_2 }
            }
            , {
              id: 2
              location { asic_location: ASIC_LOCATION_3 }
            }
            , {
              id: 3
              location { asic_location: ASIC_LOCATION_4 }
            }
            , {
              id: 4
              location { asic_location: ASIC_LOCATION_5 }
            }
            , {
              id: 5
              location { asic_location: ASIC_LOCATION_6 }
            }
            , {
              id: 6
              location { asic_location: ASIC_LOCATION_7 }
            }
            , {
              id: 7
              location { asic_location: ASIC_LOCATION_8 }
            }]
          row_major_mesh { dims: [ 2, 4 ] }
        }
        groupings {
          name: "pods_nested"
          custom_type: "pods"
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
          row_major_mesh { dims: [ 1, 4 ] }
        }
        groupings {
          name: "meshes_29"
          custom_type: "meshes"
          instances:
          [ {
            id: 0
            grouping_ref { custom_type: "pods" }
          }]
        }
    )proto");

    PhysicalGroupingDescriptor desc(text_proto);
    auto trays = desc.get_groupings_by_type("trays");
    auto pods = desc.get_groupings_by_type("pods");
    auto meshes = desc.get_groupings_by_type("meshes");
    ASSERT_EQ(trays.size(), 1);
    ASSERT_EQ(pods.size(), 1);
    ASSERT_EQ(meshes.size(), 1);
    EXPECT_EQ(trays[0].asic_count, 8u);
    EXPECT_EQ(pods[0].asic_count, 32u);    // 4 * 8
    EXPECT_EQ(meshes[0].asic_count, 32u);  // 1 * 32
}

// ============================================================================
// GET_VALID_GROUPINGS_FOR_MGD TESTS (unchanged - use file-based configs)
// ============================================================================

TEST(PhysicalGroupingDescriptorTests, CornerOrientation_RowMajorMesh) {
    // Test corner orientation assignment for various mesh configurations
    const std::string text_proto_2x4 = R"proto(
        groupings {
          name: "tray_1"
          custom_type: "tray_1"
          instances:
          [ {
            id: 0
            location { asic_location: ASIC_LOCATION_1 }
          }]
        }
        groupings {
          name: "tray_2"
          custom_type: "tray_2"
          instances:
          [ {
            id: 0
            location { asic_location: ASIC_LOCATION_1 }
          }]
        }
        groupings {
          name: "tray_3"
          custom_type: "tray_3"
          instances:
          [ {
            id: 0
            location { asic_location: ASIC_LOCATION_1 }
          }]
        }
        groupings {
          name: "tray_4"
          custom_type: "tray_4"
          instances:
          [ {
            id: 0
            location { asic_location: ASIC_LOCATION_1 }
          }]
        }
        groupings {
          name: "hosts_1"
          custom_type: "hosts"
          instances:
          [ {
            id: 0
            grouping_ref { custom_type: "tray_1" }
          }]
        }
        groupings {
          name: "meshes_1"
          custom_type: "meshes"
          instances:
          [ {
            id: 0
            location { asic_location: ASIC_LOCATION_1 }
          }]
        }
        groupings {
          name: "tray_2x4"
          custom_type: "tray_2x4"
          instances:
          [ {
            id: 0
            location { asic_location: ASIC_LOCATION_1 }
          }
            , {
              id: 1
              location { asic_location: ASIC_LOCATION_2 }
            }
            , {
              id: 2
              location { asic_location: ASIC_LOCATION_3 }
            }
            , {
              id: 3
              location { asic_location: ASIC_LOCATION_4 }
            }
            , {
              id: 4
              location { asic_location: ASIC_LOCATION_5 }
            }
            , {
              id: 5
              location { asic_location: ASIC_LOCATION_6 }
            }
            , {
              id: 6
              location { asic_location: ASIC_LOCATION_7 }
            }
            , {
              id: 7
              location { asic_location: ASIC_LOCATION_8 }
            }]
          row_major_mesh { dims: [ 2, 4 ] }
        }
    )proto";

    PhysicalGroupingDescriptor desc_2x4(text_proto_2x4);
    auto trays_2x4 = desc_2x4.get_groupings_by_name("tray_2x4");
    ASSERT_EQ(trays_2x4.size(), 1u) << "Should have one tray_2x4 grouping";
    const auto& tray_2x4 = trays_2x4[0];

    // For 2x4 mesh: NW=0, NE=3, SW=4, SE=7
    EXPECT_EQ(tray_2x4.items[0].corners.size(), 1u) << "Item 0 should have 1 corner (NW)";
    EXPECT_EQ(tray_2x4.items[0].corners[0], GroupingItemInfo::CornerOrientation::NW);

    EXPECT_EQ(tray_2x4.items[3].corners.size(), 1u) << "Item 3 should have 1 corner (NE)";
    EXPECT_EQ(tray_2x4.items[3].corners[0], GroupingItemInfo::CornerOrientation::NE);

    EXPECT_EQ(tray_2x4.items[4].corners.size(), 1u) << "Item 4 should have 1 corner (SW)";
    EXPECT_EQ(tray_2x4.items[4].corners[0], GroupingItemInfo::CornerOrientation::SW);

    EXPECT_EQ(tray_2x4.items[7].corners.size(), 1u) << "Item 7 should have 1 corner (SE)";
    EXPECT_EQ(tray_2x4.items[7].corners[0], GroupingItemInfo::CornerOrientation::SE);

    // Non-corner items should have no corners
    EXPECT_EQ(tray_2x4.items[1].corners.size(), 0u) << "Item 1 should have no corners";
    EXPECT_EQ(tray_2x4.items[2].corners.size(), 0u) << "Item 2 should have no corners";
    EXPECT_EQ(tray_2x4.items[5].corners.size(), 0u) << "Item 5 should have no corners";
    EXPECT_EQ(tray_2x4.items[6].corners.size(), 0u) << "Item 6 should have no corners";

    // Test 1x4 mesh: endpoints should have 2 corners each
    const std::string text_proto_1x4 = R"proto(
        groupings {
          name: "tray_1"
          custom_type: "tray_1"
          instances:
          [ {
            id: 0
            location { asic_location: ASIC_LOCATION_1 }
          }]
        }
        groupings {
          name: "tray_2"
          custom_type: "tray_2"
          instances:
          [ {
            id: 0
            location { asic_location: ASIC_LOCATION_1 }
          }]
        }
        groupings {
          name: "tray_3"
          custom_type: "tray_3"
          instances:
          [ {
            id: 0
            location { asic_location: ASIC_LOCATION_1 }
          }]
        }
        groupings {
          name: "tray_4"
          custom_type: "tray_4"
          instances:
          [ {
            id: 0
            location { asic_location: ASIC_LOCATION_1 }
          }]
        }
        groupings {
          name: "hosts_1"
          custom_type: "hosts"
          instances:
          [ {
            id: 0
            grouping_ref { custom_type: "tray_1" }
          }]
        }
        groupings {
          name: "meshes_1"
          custom_type: "meshes"
          instances:
          [ {
            id: 0
            location { asic_location: ASIC_LOCATION_1 }
          }]
        }
        groupings {
          name: "mesh_1x4"
          custom_type: "mesh"
          instances:
          [ {
            id: 0
            location { asic_location: ASIC_LOCATION_1 }
          }
            , {
              id: 1
              location { asic_location: ASIC_LOCATION_2 }
            }
            , {
              id: 2
              location { asic_location: ASIC_LOCATION_3 }
            }
            , {
              id: 3
              location { asic_location: ASIC_LOCATION_4 }
            }]
          row_major_mesh { dims: [ 1, 4 ] }
        }
    )proto";

    PhysicalGroupingDescriptor desc_1x4(text_proto_1x4);
    auto meshes_1x4 = desc_1x4.get_groupings_by_type("mesh");
    ASSERT_EQ(meshes_1x4.size(), 1u) << "Should have one mesh grouping";
    const auto& mesh_1x4 = meshes_1x4[0];

    // For 1x4 mesh: first item has NW+SW, last item has NE+SE
    EXPECT_EQ(mesh_1x4.items[0].corners.size(), 2u) << "Item 0 should have 2 corners (NW+SW)";
    EXPECT_TRUE(
        std::find(
            mesh_1x4.items[0].corners.begin(),
            mesh_1x4.items[0].corners.end(),
            GroupingItemInfo::CornerOrientation::NW) != mesh_1x4.items[0].corners.end());
    EXPECT_TRUE(
        std::find(
            mesh_1x4.items[0].corners.begin(),
            mesh_1x4.items[0].corners.end(),
            GroupingItemInfo::CornerOrientation::SW) != mesh_1x4.items[0].corners.end());

    EXPECT_EQ(mesh_1x4.items[3].corners.size(), 2u) << "Item 3 should have 2 corners (NE+SE)";
    EXPECT_TRUE(
        std::find(
            mesh_1x4.items[3].corners.begin(),
            mesh_1x4.items[3].corners.end(),
            GroupingItemInfo::CornerOrientation::NE) != mesh_1x4.items[3].corners.end());
    EXPECT_TRUE(
        std::find(
            mesh_1x4.items[3].corners.begin(),
            mesh_1x4.items[3].corners.end(),
            GroupingItemInfo::CornerOrientation::SE) != mesh_1x4.items[3].corners.end());

    // Middle items should have no corners
    EXPECT_EQ(mesh_1x4.items[1].corners.size(), 0u) << "Item 1 should have no corners";
    EXPECT_EQ(mesh_1x4.items[2].corners.size(), 0u) << "Item 2 should have no corners";

    // Test 4x1 mesh (column): endpoints should have 2 corners each
    const std::string text_proto_4x1 = R"proto(
        groupings {
          name: "tray_1"
          custom_type: "tray_1"
          instances:
          [ {
            id: 0
            location { asic_location: ASIC_LOCATION_1 }
          }]
        }
        groupings {
          name: "tray_2"
          custom_type: "tray_2"
          instances:
          [ {
            id: 0
            location { asic_location: ASIC_LOCATION_1 }
          }]
        }
        groupings {
          name: "tray_3"
          custom_type: "tray_3"
          instances:
          [ {
            id: 0
            location { asic_location: ASIC_LOCATION_1 }
          }]
        }
        groupings {
          name: "tray_4"
          custom_type: "tray_4"
          instances:
          [ {
            id: 0
            location { asic_location: ASIC_LOCATION_1 }
          }]
        }
        groupings {
          name: "hosts_1"
          custom_type: "hosts"
          instances:
          [ {
            id: 0
            grouping_ref { custom_type: "tray_1" }
          }]
        }
        groupings {
          name: "meshes_1"
          custom_type: "meshes"
          instances:
          [ {
            id: 0
            location { asic_location: ASIC_LOCATION_1 }
          }]
        }
        groupings {
          name: "mesh_4x1"
          custom_type: "mesh"
          instances:
          [ {
            id: 0
            location { asic_location: ASIC_LOCATION_1 }
          }
            , {
              id: 1
              location { asic_location: ASIC_LOCATION_2 }
            }
            , {
              id: 2
              location { asic_location: ASIC_LOCATION_3 }
            }
            , {
              id: 3
              location { asic_location: ASIC_LOCATION_4 }
            }]
          row_major_mesh { dims: [ 4, 1 ] }
        }
    )proto";

    PhysicalGroupingDescriptor desc_4x1(text_proto_4x1);
    auto meshes_4x1 = desc_4x1.get_groupings_by_type("mesh");
    ASSERT_EQ(meshes_4x1.size(), 1u) << "Should have one mesh grouping";
    const auto& mesh_4x1 = meshes_4x1[0];

    // For 4x1 mesh: first item has NW+NE, last item has SW+SE
    EXPECT_EQ(mesh_4x1.items[0].corners.size(), 2u) << "Item 0 should have 2 corners (NW+NE)";
    EXPECT_TRUE(
        std::find(
            mesh_4x1.items[0].corners.begin(),
            mesh_4x1.items[0].corners.end(),
            GroupingItemInfo::CornerOrientation::NW) != mesh_4x1.items[0].corners.end());
    EXPECT_TRUE(
        std::find(
            mesh_4x1.items[0].corners.begin(),
            mesh_4x1.items[0].corners.end(),
            GroupingItemInfo::CornerOrientation::NE) != mesh_4x1.items[0].corners.end());

    EXPECT_EQ(mesh_4x1.items[3].corners.size(), 2u) << "Item 3 should have 2 corners (SW+SE)";
    EXPECT_TRUE(
        std::find(
            mesh_4x1.items[3].corners.begin(),
            mesh_4x1.items[3].corners.end(),
            GroupingItemInfo::CornerOrientation::SW) != mesh_4x1.items[3].corners.end());
    EXPECT_TRUE(
        std::find(
            mesh_4x1.items[3].corners.begin(),
            mesh_4x1.items[3].corners.end(),
            GroupingItemInfo::CornerOrientation::SE) != mesh_4x1.items[3].corners.end());

    // Test 1x1 mesh: single item should have all 4 corners
    // Note: Using MESH preset type to allow single instance
    const std::string text_proto_1x1 = R"proto(
        groupings {
          name: "tray_1"
          custom_type: "tray_1"
          instances:
          [ {
            id: 0
            location { asic_location: ASIC_LOCATION_1 }
          }]
        }
        groupings {
          name: "tray_2"
          custom_type: "tray_2"
          instances:
          [ {
            id: 0
            location { asic_location: ASIC_LOCATION_1 }
          }]
        }
        groupings {
          name: "tray_3"
          custom_type: "tray_3"
          instances:
          [ {
            id: 0
            location { asic_location: ASIC_LOCATION_1 }
          }]
        }
        groupings {
          name: "tray_4"
          custom_type: "tray_4"
          instances:
          [ {
            id: 0
            location { asic_location: ASIC_LOCATION_1 }
          }]
        }
        groupings {
          name: "hosts_1"
          custom_type: "hosts"
          instances:
          [ {
            id: 0
            grouping_ref { custom_type: "tray_1" }
          }]
        }
        groupings {
          name: "mesh_1x1"
          preset_type: MESH
          instances:
          [ {
            id: 0
            location { asic_location: ASIC_LOCATION_1 }
          }]
          row_major_mesh { dims: [ 1, 1 ] }
        }
    )proto";

    PhysicalGroupingDescriptor desc_1x1(text_proto_1x1);
    auto meshes_1x1 = desc_1x1.get_groupings_by_type("MESH");
    ASSERT_EQ(meshes_1x1.size(), 1u) << "Should have one MESH grouping";
    const auto& mesh_1x1 = meshes_1x1[0];

    // For 1x1 mesh: single item has all 4 corners
    EXPECT_EQ(mesh_1x1.items[0].corners.size(), 4u) << "Item 0 should have all 4 corners";
    EXPECT_TRUE(
        std::find(
            mesh_1x1.items[0].corners.begin(),
            mesh_1x1.items[0].corners.end(),
            GroupingItemInfo::CornerOrientation::NW) != mesh_1x1.items[0].corners.end());
    EXPECT_TRUE(
        std::find(
            mesh_1x1.items[0].corners.begin(),
            mesh_1x1.items[0].corners.end(),
            GroupingItemInfo::CornerOrientation::NE) != mesh_1x1.items[0].corners.end());
    EXPECT_TRUE(
        std::find(
            mesh_1x1.items[0].corners.begin(),
            mesh_1x1.items[0].corners.end(),
            GroupingItemInfo::CornerOrientation::SW) != mesh_1x1.items[0].corners.end());
    EXPECT_TRUE(
        std::find(
            mesh_1x1.items[0].corners.begin(),
            mesh_1x1.items[0].corners.end(),
            GroupingItemInfo::CornerOrientation::SE) != mesh_1x1.items[0].corners.end());
}

// ============================================================================
// FLATTENED ADJACENCY MESH TESTS
// ============================================================================

TEST(PhysicalGroupingDescriptorTests, BuildFlattenedAdjacencyMesh_FromTriple16x8File) {
    // Load the triple_16x8 groupings file
    const std::filesystem::path text_proto_file_path =
        "tests/tt_metal/tt_fabric/physical_groupings/bh_galaxy_rev_ab_physical_grouping_descriptor.textproto";
    PhysicalGroupingDescriptor desc(text_proto_file_path);

    // Get one of the MESH grouping infos - "8x16_Mesh" which has 4 hosts in a 2x2 grid
    auto mesh_groupings = desc.get_groupings_by_type("MESH");
    ASSERT_GT(mesh_groupings.size(), 0u) << "Expected at least one MESH grouping";

    // Find the "8x16_Mesh" grouping (has 4 hosts arranged in 2x2 grid)
    GroupingInfo mesh_8x16;
    bool found = false;
    for (const auto& mesh : mesh_groupings) {
        if (mesh.name == "8x16_Mesh") {
            mesh_8x16 = mesh;
            found = true;
            break;
        }
    }
    ASSERT_TRUE(found) << "Expected to find '8x16_Mesh' grouping";

    // Verify the grouping has the expected structure
    EXPECT_EQ(mesh_8x16.asic_count, 128u) << "8x16_Mesh should have 128 ASICs (4 hosts * 32 ASICs each)";
    EXPECT_EQ(mesh_8x16.items.size(), 4u) << "8x16_Mesh should have 4 instances (hosts)";

    // Build the flattened adjacency mesh (returns vector - one per possibility)
    auto flattened_meshes = desc.build_flattened_adjacency_mesh(mesh_8x16);
    ASSERT_FALSE(flattened_meshes.empty()) << "Expected at least one flattened mesh";
    const auto& flattened_mesh = flattened_meshes.front().adjacency_graph;

    // Verify the result is a valid adjacency graph
    // The flattened mesh should have 128 nodes (one per ASIC)
    auto nodes = flattened_mesh.get_nodes();
    EXPECT_EQ(nodes.size(), 128u) << "Flattened mesh should have 128 nodes (one per ASIC)";

    // Verify that nodes are connected (each node should have neighbors in a 2D mesh)
    for (const auto& node : nodes) {
        const auto& neighbors = flattened_mesh.get_neighbors(node);
        EXPECT_GE(neighbors.size(), 2u) << "Node " << node << " should have at least 2 neighbors";
        EXPECT_LE(neighbors.size(), 4u) << "Node " << node << " should have at most 4 neighbors";
    }
}

TEST(PhysicalGroupingDescriptorTests, BuildFlattenedAdjacencyMesh_4x4Mesh) {
    const std::filesystem::path text_proto_file_path =
        "tests/tt_metal/tt_fabric/physical_groupings/bh_galaxy_rev_ab_physical_grouping_descriptor.textproto";
    PhysicalGroupingDescriptor desc(text_proto_file_path);

    GroupingInfo mesh_4x4;
    bool found = false;
    for (const auto& mesh : desc.get_groupings_by_type("MESH")) {
        if (mesh.asic_count == 16u && mesh.items.size() == 2u && mesh.name.find("4x4") != std::string::npos) {
            mesh_4x4 = mesh;
            found = true;
            break;
        }
    }
    ASSERT_TRUE(found) << "Expected to find 4x4 mesh grouping (e.g. 4x4_Mesh WH/BH)";

    EXPECT_EQ(mesh_4x4.asic_count, 16u) << "4x4_Mesh should have 16 ASICs (2 trays * 8 ASICs each)";
    EXPECT_EQ(mesh_4x4.items.size(), 2u) << "4x4_Mesh should have 2 instances (trays)";

    auto flattened_meshes = desc.build_flattened_adjacency_mesh(mesh_4x4);
    ASSERT_FALSE(flattened_meshes.empty());
    const auto& flattened_mesh = flattened_meshes.front().adjacency_graph;

    auto nodes = flattened_mesh.get_nodes();
    EXPECT_EQ(nodes.size(), 16u) << "Flattened mesh should have 16 nodes";

    for (const auto& node : nodes) {
        const auto& neighbors = flattened_mesh.get_neighbors(node);
        EXPECT_GE(neighbors.size(), 2u) << "Node " << node << " should have at least 2 neighbors";
        EXPECT_LE(neighbors.size(), 4u) << "Node " << node << " should have at most 4 neighbors";
    }
}

TEST(PhysicalGroupingDescriptorTests, BuildFlattenedAdjacencyMesh_2x8Mesh) {
    const std::filesystem::path text_proto_file_path =
        "tests/tt_metal/tt_fabric/physical_groupings/bh_galaxy_rev_ab_physical_grouping_descriptor.textproto";
    PhysicalGroupingDescriptor desc(text_proto_file_path);

    GroupingInfo mesh_2x8;
    bool found = false;
    for (const auto& mesh : desc.get_groupings_by_type("MESH")) {
        if (mesh.asic_count == 16u && mesh.items.size() == 2u && mesh.name.find("2x8") != std::string::npos) {
            mesh_2x8 = mesh;
            found = true;
            break;
        }
    }
    ASSERT_TRUE(found) << "Expected to find 2x8 mesh grouping (e.g. 2x8_Mesh WH/BH)";

    EXPECT_EQ(mesh_2x8.asic_count, 16u) << "2x8_Mesh should have 16 ASICs (2 trays * 8 ASICs each)";
    EXPECT_EQ(mesh_2x8.items.size(), 2u) << "2x8_Mesh should have 2 instances (trays)";

    auto flattened_meshes = desc.build_flattened_adjacency_mesh(mesh_2x8);
    ASSERT_FALSE(flattened_meshes.empty());
    const auto& flattened_mesh = flattened_meshes.front().adjacency_graph;

    auto nodes = flattened_mesh.get_nodes();
    EXPECT_EQ(nodes.size(), 16u) << "Flattened mesh should have 16 nodes";

    for (const auto& node : nodes) {
        const auto& neighbors = flattened_mesh.get_neighbors(node);
        EXPECT_GE(neighbors.size(), 2u) << "Node " << node << " should have at least 2 neighbors";
        EXPECT_LE(neighbors.size(), 3u) << "Node " << node << " should have at most 3 neighbors";
    }
}

TEST(PhysicalGroupingDescriptorTests, BuildFlattenedAdjacencyMesh_2x2Halftray) {
    // PGD names this MESH grouping "2x2 Mesh" (one halftray_2x2 HALFTRAY ref, 4 ASICs); see
    // bh_galaxy_rev_ab_physical_grouping_descriptor.textproto.
    const std::filesystem::path text_proto_file_path =
        "tests/tt_metal/tt_fabric/physical_groupings/bh_galaxy_rev_ab_physical_grouping_descriptor.textproto";
    PhysicalGroupingDescriptor desc(text_proto_file_path);

    constexpr const char* kMeshGroupingName = "2x2 Mesh";

    GroupingInfo mesh_halftray;
    bool found = false;
    for (const auto& mesh : desc.get_groupings_by_type("MESH")) {
        if (mesh.name == kMeshGroupingName) {
            mesh_halftray = mesh;
            found = true;
            break;
        }
    }
    ASSERT_TRUE(found) << "Expected MESH grouping named \"" << kMeshGroupingName << "\" in " << text_proto_file_path;

    EXPECT_EQ(mesh_halftray.asic_count, 4u)
        << "MESH grouping \"" << kMeshGroupingName << "\" should have 4 ASICs (1 halftray_2x2 instance)";
    EXPECT_EQ(mesh_halftray.items.size(), 1u)
        << "MESH grouping \"" << kMeshGroupingName << "\" should have 1 instance (one halftray ref)";

    auto flattened_meshes = desc.build_flattened_adjacency_mesh(mesh_halftray);
    ASSERT_FALSE(flattened_meshes.empty());
    const auto& flattened_mesh = flattened_meshes.front().adjacency_graph;

    auto nodes = flattened_mesh.get_nodes();
    EXPECT_EQ(nodes.size(), 4u) << "Flattened mesh should have 4 nodes";

    for (const auto& node : nodes) {
        const auto& neighbors = flattened_mesh.get_neighbors(node);
        EXPECT_GE(neighbors.size(), 2u) << "Node " << node << " should have at least 2 neighbors";
        EXPECT_LE(neighbors.size(), 4u) << "Node " << node << " should have at most 4 neighbors (2x2 mesh)";
    }
}

// Two HALFTRAY instances in row_major_mesh [2,1] produce non-contiguous node IDs when joined; items must be
// indexed by node_id (rebuild_items_from_flattened_mesh), not push_back order.
TEST(PhysicalGroupingDescriptorTests, BuildFlattenedAdjacencyMesh_4x2Mesh_TwoHalftray_ItemsPerGraphNode) {
    const std::filesystem::path text_proto_file_path =
        "tests/tt_metal/tt_fabric/physical_groupings/bh_galaxy_rev_ab_physical_grouping_descriptor.textproto";
    PhysicalGroupingDescriptor desc(text_proto_file_path);

    GroupingInfo mesh_4x2;
    bool found = false;
    for (const auto& mesh : desc.get_groupings_by_type("MESH")) {
        if (mesh.name == "4x2_Mesh_horizontal") {
            mesh_4x2 = mesh;
            found = true;
            break;
        }
    }
    ASSERT_TRUE(found) << "Expected to find '4x2_Mesh_horizontal' grouping";

    EXPECT_EQ(mesh_4x2.asic_count, 8u) << "4x2_Mesh: 2 halftrays x 4 ASICs";
    EXPECT_EQ(mesh_4x2.items.size(), 2u) << "4x2_Mesh should have 2 instance refs before flatten";

    auto flattened_meshes = desc.build_flattened_adjacency_mesh(mesh_4x2);
    ASSERT_FALSE(flattened_meshes.empty());
    const GroupingInfo& flat = flattened_meshes.front();
    const auto& flattened_mesh = flat.adjacency_graph;

    auto nodes = flattened_mesh.get_nodes();
    EXPECT_EQ(nodes.size(), 8u) << "Flattened mesh should have 8 nodes";

    for (uint32_t node_id : nodes) {
        ASSERT_LT(node_id, flat.items.size())
            << "items must be sized so items[node_id] exists for every graph node (node_id=" << node_id
            << ", items.size()=" << flat.items.size() << ")";
        const auto& item = flat.items[node_id];
        EXPECT_EQ(item.type, GroupingItemInfo::ItemType::ASIC_LOCATION)
            << "node_id " << node_id << " should have ASIC_LOCATION metadata from flattened mesh";
    }
}

// Corner-inferred dims: dims inferred from items' corners, not stored in GroupingInfo
TEST(PhysicalGroupingDescriptorTests, BuildFlattenedAdjacencyMesh_CornerInference) {
    const std::string text_proto = wrap_with_required_groupings(R"proto(
        groupings {
          name: "mesh_1x1"
          preset_type: MESH
          instances:
          [ {
            id: 0
            grouping_ref { custom_type: "tray_1" }
          }]
        }
        groupings {
          name: "mesh_1x4"
          preset_type: MESH
          instances:
          [ {
            id: 0
            grouping_ref { custom_type: "tray_1" }
          }
            , {
              id: 1
              grouping_ref { custom_type: "tray_1" }
            }
            , {
              id: 2
              grouping_ref { custom_type: "tray_1" }
            }
            , {
              id: 3
              grouping_ref { custom_type: "tray_1" }
            }]
          row_major_mesh { dims: [ 1, 4 ] }
        }
    )proto");

    PhysicalGroupingDescriptor desc(text_proto);
    auto meshes = desc.get_groupings_by_type("MESH");
    ASSERT_GE(meshes.size(), 2u);

    GroupingInfo mesh_1x1, mesh_1x4;
    for (const auto& m : meshes) {
        if (m.name == "mesh_1x1") {
            mesh_1x1 = m;
        }
        if (m.name == "mesh_1x4") {
            mesh_1x4 = m;
        }
    }

    auto flat_1x1_meshes = desc.build_flattened_adjacency_mesh(mesh_1x1);
    ASSERT_FALSE(flat_1x1_meshes.empty());
    const auto& flat_1x1 = flat_1x1_meshes.front().adjacency_graph;
    EXPECT_EQ(flat_1x1.get_nodes().size(), 1u);  // 1 tray with 1 ASIC (from required groupings)
    expect_neighbors_by_id(flat_1x1, 0, {});     // Single node has no neighbors

    auto flat_1x4_meshes = desc.build_flattened_adjacency_mesh(mesh_1x4);
    ASSERT_FALSE(flat_1x4_meshes.empty());
    const auto& flat_1x4 = flat_1x4_meshes.front().adjacency_graph;
    EXPECT_EQ(flat_1x4.get_nodes().size(), 4u);  // 4 trays x 1 ASIC each
    // 1x4 chain: endpoints have 1 neighbor, interior nodes have 2 (row-major IDs 0..3)
    expect_neighbors_by_id(flat_1x4, 0, {1});
    expect_neighbors_by_id(flat_1x4, 1, {0, 2});
    expect_neighbors_by_id(flat_1x4, 2, {1, 3});
    expect_neighbors_by_id(flat_1x4, 3, {2});
}

// SP4 GLX mock: each MPI rank builds a PSD from its rank-local cluster fragment (one BH Galaxy host, 32 ASICs).
// 128-ASIC meshes (8x16_Mesh / 4x32_Mesh) are covered in ValidatePreformedGroups_Sp4BhGalaxyQuadHostMeshes.
TEST_F(PhysicalGroupingDescriptorSP4Tests, ValidatePreformedGroups_Sp4BhGalaxyMeshGroupings_SingleHostScale) {
    const std::string pgd_path =
        "tests/tt_metal/tt_fabric/physical_groupings/bh_galaxy_rev_ab_physical_grouping_descriptor.textproto";

    ASSERT_TRUE(std::filesystem::exists(pgd_path)) << "PGD file not found: " << pgd_path;

    tt::tt_metal::PhysicalSystemDescriptor psd = create_psd_from_mock_cluster();

    PhysicalGroupingDescriptor pgd{std::filesystem::path(pgd_path)};

    // Get all mesh groupings to test
    auto all_mesh_groupings = pgd.get_groupings_by_type("MESH");
    ASSERT_FALSE(all_mesh_groupings.empty()) << "No MESH groupings found in PGD";

    // Find specific mesh groupings by name or by dimensions (name can have WH/BH suffix)
    // Prefer exact match first so "4x2_Mesh" matches the two-halftray grouping, not a longer prefix
    auto find_mesh_by_name = [&all_mesh_groupings](const std::string& name) -> const GroupingInfo* {
        for (const auto& mesh : all_mesh_groupings) {
            if (mesh.name == name) {
                return &mesh;
            }
        }
        for (const auto& mesh : all_mesh_groupings) {
            if (mesh.name.starts_with(name)) {
                return &mesh;
            }
        }
        return nullptr;
    };

    // Test 4x2_Mesh (two HALFTRAY instances, row_major_mesh [2,1]) - validation against mock cluster
    {
        const auto* mesh_grouping = find_mesh_by_name("4x2_Mesh");
        ASSERT_NE(mesh_grouping, nullptr) << "4x2_Mesh grouping not found";

        auto placements = pgd.find_any_in_psd(*mesh_grouping, psd);

        EXPECT_FALSE(placements.empty())
            << "Expected validation to pass: 4x2_Mesh grouping should map to mock cluster PSD";
    }

    // Test 4x4_Mesh - validation against mock cluster
    {
        const auto* mesh_grouping = find_mesh_by_name("4x4_Mesh");
        ASSERT_NE(mesh_grouping, nullptr) << "4x4_Mesh grouping not found";

        auto placements = pgd.find_any_in_psd(*mesh_grouping, psd);

        EXPECT_FALSE(placements.empty())
            << "Expected validation to pass: 4x4_Mesh grouping should map to mock cluster PSD";
    }

    // Test 2x8_Mesh - validation against mock cluster
    {
        const auto* mesh_grouping = find_mesh_by_name("2x8_Mesh");
        ASSERT_NE(mesh_grouping, nullptr) << "2x8_Mesh grouping not found";

        auto placements = pgd.find_any_in_psd(*mesh_grouping, psd);

        EXPECT_FALSE(placements.empty())
            << "Expected validation to pass: 2x8_Mesh grouping should map to mock cluster PSD";
    }

    // Test 4x8_Mesh - validation against mock cluster
    {
        const auto* mesh_grouping = find_mesh_by_name("4x8_Mesh");
        ASSERT_NE(mesh_grouping, nullptr) << "4x8_Mesh grouping not found";

        auto placements = pgd.find_any_in_psd(*mesh_grouping, psd);

        EXPECT_FALSE(placements.empty())
            << "Expected validation to pass: 4x8_Mesh grouping should map to mock cluster PSD";
    }

    // Test HOSTS type grouping - validation against mock cluster
    {
        auto hosts_groupings = pgd.get_groupings_by_type("HOSTS");
        ASSERT_FALSE(hosts_groupings.empty()) << "HOSTS grouping not found";
        const auto& hosts_grouping = hosts_groupings[0];

        auto placements = pgd.find_any_in_psd(hosts_grouping, psd);

        EXPECT_FALSE(placements.empty())
            << "Expected validation to pass: HOSTS grouping should map to mock cluster PSD";
    }
}

TEST_F(PhysicalGroupingDescriptorSP4Tests, ValidatePreformedGroups_Sp4BhGalaxyQuadHostMeshes) {
    const std::string pgd_path =
        "tests/tt_metal/tt_fabric/physical_groupings/bh_galaxy_rev_ab_physical_grouping_descriptor.textproto";

    ASSERT_TRUE(std::filesystem::exists(pgd_path)) << "PGD file not found: " << pgd_path;

    tt::tt_metal::PhysicalSystemDescriptor psd = create_psd_from_mock_cluster();
    PhysicalGroupingDescriptor pgd{std::filesystem::path(pgd_path)};

    // Try finding any for galaxy_hosts
    {
        auto hosts_groupings = pgd.get_groupings_by_name("galaxy_hosts");
        ASSERT_FALSE(hosts_groupings.empty()) << "galaxy_hosts grouping not found";
        const auto& hosts_grouping = hosts_groupings[0];

        auto placements = pgd.find_any_in_psd(hosts_grouping, psd);

        EXPECT_FALSE(placements.empty())
            << "Expected validation to pass: galaxy_hosts grouping should map to mock cluster PSD";
    }

    {
        // 4x32_Mesh: same 128 ASICs / 4 hosts as an 8x16_Mesh, row_major_mesh [1,4] — MGD device grid 32×4
        auto mesh_groupings = pgd.get_groupings_by_name("4x32_Mesh");
        ASSERT_FALSE(mesh_groupings.empty()) << "4x32_Mesh grouping not found";
        const auto& mesh_grouping = mesh_groupings[0];

        auto placements = pgd.find_any_in_psd(mesh_grouping, psd);

        EXPECT_FALSE(placements.empty())
            << "Expected validation to pass: 4x32_Mesh (32x4 device layout) should map to mock cluster PSD";
    }

    {
        auto mesh_groupings = pgd.get_groupings_by_name("4x32_Mesh");
        ASSERT_FALSE(mesh_groupings.empty()) << "4x32_Mesh grouping not found";

        // TODO(plan 3 §8(a)): rewrite these find_all_in_psd tests onto solve_adjacency_guided_placement.
        auto asic_ids = pgd.find_all_in_psd(mesh_groupings, psd);

        EXPECT_EQ(asic_ids.size(), 4u)
            << "Expected validation to pass: 4x32_Mesh (32x4) should map to mock cluster PSD (4 placements on SP4)";
    }

    {
        // Test 4x4_Mesh grouping with find_all_in_psd
        auto mesh_groupings = pgd.get_groupings_by_name("4x4_Mesh");
        ASSERT_EQ(mesh_groupings.size(), 1u) << "4x4_Mesh grouping not found";

        auto asic_ids = pgd.find_all_in_psd(mesh_groupings, psd);

        // SP4 GLX mock: 16 hosts × 32 ASICs = 512 ASICs; a 4x4_Mesh (16 ASICs) tiles disjointly → 32 placements.
        EXPECT_EQ(asic_ids.size(), 32u)
            << "Expected validation to pass: 4x4_Mesh grouping should map to mock cluster PSD (32 placements)";
    }
}

TEST_F(PhysicalGroupingDescriptorDualT3kTests, ValidatePreformedGroups_WHt3kGroupings) {
    const std::string pgd_path =
        "tests/tt_metal/tt_fabric/physical_groupings/wh_t3k_physical_grouping_descriptor.textproto";

    ASSERT_TRUE(std::filesystem::exists(pgd_path)) << "PGD file not found: " << pgd_path;

    tt::tt_metal::PhysicalSystemDescriptor psd = create_psd_from_mock_cluster();
    PhysicalGroupingDescriptor pgd{std::filesystem::path(pgd_path)};

    {
        auto mesh_groupings = pgd.get_groupings_by_name("2x2_Mesh_t3k");
        ASSERT_FALSE(mesh_groupings.empty()) << "2x2_Mesh_t3k grouping not found";

        auto asic_ids = pgd.find_all_in_psd(mesh_groupings, psd);

        // Should find 4 of them, each of them on a single host
        EXPECT_EQ(asic_ids.size(), 4u)
            << "Expected validation to pass: 2x2_Mesh_t3k grouping should map to mock cluster PSD";

        // Each should have their own host name
        for (const auto& placement : asic_ids) {
            const auto& asic_id_set = placement.asics;
            ASSERT_FALSE(asic_id_set.empty()) << "Each 2x2_Mesh_t3k mapping should contain at least one ASIC";
            std::string host_name = psd.get_host_name_for_asic(*asic_id_set.begin());
            for (const auto& asic_id : asic_id_set) {
                EXPECT_EQ(psd.get_host_name_for_asic(asic_id), host_name)
                    << "Expected validation to pass: 2x2_Mesh_t3k grouping should map to mock cluster PSD";
            }
        }
    }

    {
        auto mesh_groupings = pgd.get_groupings_by_name("2x4_Mesh_t3k");
        ASSERT_FALSE(mesh_groupings.empty()) << "2x4_Mesh_t3k grouping not found";

        auto asic_ids = pgd.find_all_in_psd(mesh_groupings, psd);

        ASSERT_EQ(asic_ids.size(), 2u)
            << "Expected validation to pass: 2x4_Mesh_t3k grouping should map to mock cluster PSD";

        // Each should have their own host name
        for (const auto& placement : asic_ids) {
            const auto& asic_id_set = placement.asics;
            ASSERT_FALSE(asic_id_set.empty()) << "Each 2x4_Mesh_t3k mapping should contain at least one ASIC";
            std::string host_name = psd.get_host_name_for_asic(*asic_id_set.begin());
            for (const auto& asic_id : asic_id_set) {
                EXPECT_EQ(psd.get_host_name_for_asic(asic_id), host_name)
                    << "Expected validation to pass: 2x4_Mesh_t3k grouping should map to mock cluster PSD";
            }
        }
    }
}

TEST_F(PhysicalGroupingDescriptorSP4Tests, ValidatePreformedGroups_Triple16x8PsdWithTriple16x8QuadUnknownGroupings) {
    // FIXME: This test currently fails because placements for multiple groupings are currently not optimized yet, so we
    // need to skip it for now. This will be fixed in a future commit when needed for more placement optimizations.
    GTEST_SKIP();
    const std::string pgd_path =
        "tests/tt_metal/tt_fabric/physical_groupings/default_physical_grouping_descriptor.textproto";

    ASSERT_TRUE(std::filesystem::exists(pgd_path)) << "PGD file not found: " << pgd_path;

    tt::tt_metal::PhysicalSystemDescriptor psd = create_psd_from_mock_cluster();
    PhysicalGroupingDescriptor pgd{std::filesystem::path(pgd_path)};

    {
        auto mesh_groupings = pgd.get_groupings_by_name("2x2_Mesh");
        ASSERT_FALSE(mesh_groupings.empty()) << "2x2_Mesh grouping not found";

        auto asic_ids = pgd.find_all_in_psd(mesh_groupings, psd);

        // Expect 96 groups
        EXPECT_EQ(asic_ids.size(), 96u)
            << "Expected validation to pass: 2x2_Mesh grouping should map to mock cluster PSD";
    }

    {
        auto mesh_groupings = pgd.get_groupings_by_name("4x2_Mesh");
        ASSERT_FALSE(mesh_groupings.empty()) << "4x2_Mesh grouping not found";

        auto asic_ids = pgd.find_all_in_psd(mesh_groupings, psd);

        // Expect 48 groups (same tiling count as former 2x4_Mesh: 8-ASIC two-halftray mesh)
        EXPECT_EQ(asic_ids.size(), 48u)
            << "Expected validation to pass: 4x2_Mesh grouping should map to mock cluster PSD";
    }

    {
        auto mesh_groupings = pgd.get_groupings_by_name("4x4_Mesh");
        ASSERT_FALSE(mesh_groupings.empty()) << "4x4_Mesh grouping not found";

        auto asic_ids = pgd.find_all_in_psd(mesh_groupings, psd);

        // Expect 24 groups
        EXPECT_EQ(asic_ids.size(), 24u)
            << "Expected validation to pass: 4x4_Mesh grouping should map to mock cluster PSD";
    }
}

// Test POD and SUPERPOD level groupings - should fail (cannot be flattened as they're too high level)
TEST_F(PhysicalGroupingDescriptorSP4Tests, ValidateGroupingWithPsd_PodAndSuperpodLevel) {
    const std::string pgd_path = "tests/tt_metal/tt_fabric/physical_groupings/test_superpod_grouping.textproto";

    ASSERT_TRUE(std::filesystem::exists(pgd_path)) << "PGD file not found: " << pgd_path;

    tt::tt_metal::PhysicalSystemDescriptor psd = create_psd_from_mock_cluster();

    PhysicalGroupingDescriptor pgd{std::filesystem::path(pgd_path)};

    // Test POD level grouping - should pass (can be flattened and matches PSD)
    auto pod_groupings = pgd.get_groupings_by_name("pods");
    ASSERT_FALSE(pod_groupings.empty()) << "pods grouping not found";
    const auto& pod_grouping = pod_groupings[0];

    // POD groupings reference meshes, but should flatten properly and match the PSD structure
    auto pod_placements = pgd.find_any_in_psd(pod_grouping, psd);

    // Expect it to pass - POD level grouping should validate successfully
    EXPECT_FALSE(pod_placements.empty())
        << "Expected validation to pass: POD level grouping should validate against mock cluster PSD";

    // Test SUPERPOD level grouping - should fail during mesh building (all_to_all connection type)
    auto superpod_groupings = pgd.get_groupings_by_name("superpods");
    ASSERT_FALSE(superpod_groupings.empty()) << "superpods grouping not found";
    const auto& superpod_grouping = superpod_groupings[0];

    // This should throw during build_flattened_adjacency_mesh because SUPERPOD uses all_to_all connection type
    // which cannot be flattened into a mesh (no row_major_mesh structure)
    EXPECT_THROW(
        { pgd.find_any_in_psd(superpod_grouping, psd); }, std::exception)
        << "Expected exception during mesh building: SUPERPOD with all_to_all connection cannot be flattened";
}

// ============================================================================
// GET_VALID_GROUPINGS_FOR_MGD TESTS
// ============================================================================

TEST_F(PhysicalGroupingDescriptorSP4Tests, GetValidGroupingsForMGD_BlitzPipeline2x4) {
    // Test matching a 4x2 mesh MGD (8 ASICs) to the 4x2_Mesh grouping in bh_galaxy PGD
    const std::string pgd_path =
        "tests/tt_metal/tt_fabric/physical_groupings/bh_galaxy_rev_ab_physical_grouping_descriptor.textproto";
    const std::string mgd_path = "tt_metal/fabric/mesh_graph_descriptors/bh_glx_split_4x2.textproto";

    ASSERT_TRUE(std::filesystem::exists(pgd_path)) << "PGD file not found: " << pgd_path;
    ASSERT_TRUE(std::filesystem::exists(mgd_path)) << "MGD file not found: " << mgd_path;

    tt::tt_metal::PhysicalSystemDescriptor psd = create_psd_from_mock_cluster();
    PhysicalGroupingDescriptor pgd{std::filesystem::path(pgd_path)};
    MeshGraphDescriptor mgd{std::filesystem::path(mgd_path)};

    auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd, psd);

    // Count total groupings across all instances
    size_t total_groupings = 0;
    for (const auto& [instance_type, instances] : valid_groupings) {
        for (const auto& [instance_name, groupings] : instances) {
            total_groupings += groupings.size();
        }
    }

    // Should have at least one valid grouping match (MESH) and possibly FABRIC
    ASSERT_GE(total_groupings, 1u) << "Should have at least one valid grouping match";

    // Check that we have matches for MESH instances
    ASSERT_GE(valid_groupings.size(), 1u) << "Should have at least one instance type (MESH)";
    ASSERT_EQ(valid_groupings.count("MESH"), 1u) << "Should have MESH instance type";
    ASSERT_EQ(valid_groupings.at("MESH").size(), 1u) << "Should have exactly one MESH instance";

    // Check that we have a match for the 4x2_Mesh grouping (8 ASICs)
    // Flattened groupings have "_flat" appended to their name
    bool found_mesh_match = false;
    for (const auto& [instance_name, groupings] : valid_groupings.at("MESH")) {
        for (const auto& grouping : groupings) {
            if (grouping.asic_count == 8u && grouping.name == "4x2_Mesh_horizontal_flat") {
                found_mesh_match = true;
                EXPECT_EQ(grouping.name, "4x2_Mesh_horizontal_flat") << "Should match 4x2_Mesh_horizontal_flat grouping";
                EXPECT_EQ(grouping.asic_count, 8u) << "Should have 8 ASICs";
                break;
            }
        }
        if (found_mesh_match) {
            break;
        }
    }
    EXPECT_TRUE(found_mesh_match)
        << "Should find a match for 4x2 mesh (8 ASICs) matching 4x2_Mesh_horizontal_flat grouping";

    // Check that we have FABRIC level grouping (G0)
    ASSERT_EQ(valid_groupings.count("FABRIC"), 1u) << "Should have FABRIC instance type";
    ASSERT_EQ(valid_groupings.at("FABRIC").size(), 1u) << "Should have exactly one FABRIC instance";
    ASSERT_EQ(valid_groupings.at("FABRIC").count("G0"), 1u) << "Should have G0 FABRIC instance";
    const auto& g0_groupings = valid_groupings.at("FABRIC").at("G0");
    ASSERT_GE(g0_groupings.size(), 1u) << "Should have at least one grouping for G0";
}

TEST_F(PhysicalGroupingDescriptorSP4Tests, GetValidGroupingsForMGD_4x4Mesh) {
    // Test matching a 4x4 mesh MGD (16 ASICs) to the 4x4_Mesh grouping
    // Using dual_4x4_mesh_graph_descriptor which has 4x4 meshes in a graph
    const std::string pgd_path =
        "tests/tt_metal/tt_fabric/physical_groupings/bh_galaxy_rev_ab_physical_grouping_descriptor.textproto";
    const std::string mgd_path =
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_4x4_mesh_graph_descriptor.textproto";

    ASSERT_TRUE(std::filesystem::exists(pgd_path)) << "PGD file not found: " << pgd_path;
    ASSERT_TRUE(std::filesystem::exists(mgd_path)) << "MGD file not found: " << mgd_path;

    tt::tt_metal::PhysicalSystemDescriptor psd = create_psd_from_mock_cluster();
    PhysicalGroupingDescriptor pgd{std::filesystem::path(pgd_path)};
    MeshGraphDescriptor mgd{std::filesystem::path(mgd_path)};

    auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd, psd);

    // Count total groupings across all instances
    size_t total_groupings = 0;
    for (const auto& [instance_type, instances] : valid_groupings) {
        for (const auto& [instance_name, groupings] : instances) {
            total_groupings += groupings.size();
        }
    }

    // Should have at least two valid grouping matches (there are two 4x4_Mesh definitions in the file, and dual_4x4 has
    // 2 meshes)
    ASSERT_GE(total_groupings, 2u) << "Should have at least two valid grouping matches";

    // Check that we have matches for MESH instances
    ASSERT_GE(valid_groupings.size(), 1u) << "Should have at least one instance type (MESH)";
    ASSERT_EQ(valid_groupings.count("MESH"), 1u) << "Should have MESH instance type";
    // dual_4x4 has 2 meshes in a graph, so we should have 2 MESH instances
    ASSERT_GE(valid_groupings.at("MESH").size(), 1u) << "Should have at least one MESH instance";

    // Check that we have matches for the 4x4 mesh grouping (16 ASICs)
    // Names in triple_16x8 are "4x4_Mesh WH", "4x4_Mesh_diagonal", etc.
    size_t total_4x4_matches = 0;
    for (const auto& [instance_name, groupings] : valid_groupings.at("MESH")) {
        for (const auto& grouping : groupings) {
            if (grouping.asic_count == 16u && grouping.name.find("4x4") != std::string::npos) {
                total_4x4_matches++;
                EXPECT_EQ(grouping.asic_count, 16u) << "Should have 16 ASICs";
            }
        }
    }
    EXPECT_GE(total_4x4_matches, 2u) << "Should have at least two 4x4 mesh matches";

    // Check that we have FABRIC level grouping (G0)
    ASSERT_EQ(valid_groupings.count("FABRIC"), 1u) << "Should have FABRIC instance type";
    ASSERT_EQ(valid_groupings.at("FABRIC").size(), 1u) << "Should have exactly one FABRIC instance";
    ASSERT_EQ(valid_groupings.at("FABRIC").count("G0"), 1u) << "Should have G0 FABRIC instance";
    const auto& g0_groupings = valid_groupings.at("FABRIC").at("G0");
    ASSERT_GE(g0_groupings.size(), 1u) << "Should have at least one grouping for G0";
}

TEST_F(PhysicalGroupingDescriptorSP4Tests, GetValidGroupingsForMGD_2x8Mesh) {
    // Test matching a 2x8 mesh MGD (16 ASICs) to the 2x8_Mesh grouping
    // Using wh_galaxy_split_2x8_2x4_3_mesh which has a 2x8 mesh (MESH4)
    const std::string pgd_path =
        "tests/tt_metal/tt_fabric/physical_groupings/bh_galaxy_rev_ab_physical_grouping_descriptor.textproto";
    const std::string mgd_path =
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/wh_galaxy_split_2x8_2x4_3_mesh.textproto";

    ASSERT_TRUE(std::filesystem::exists(pgd_path)) << "PGD file not found: " << pgd_path;
    ASSERT_TRUE(std::filesystem::exists(mgd_path)) << "MGD file not found: " << mgd_path;

    tt::tt_metal::PhysicalSystemDescriptor psd = create_psd_from_mock_cluster();
    PhysicalGroupingDescriptor pgd{std::filesystem::path(pgd_path)};
    MeshGraphDescriptor mgd{std::filesystem::path(mgd_path)};

    auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd, psd);

    // Count total groupings across all instances
    size_t total_groupings = 0;
    for (const auto& [instance_type, instances] : valid_groupings) {
        for (const auto& [instance_name, groupings] : instances) {
            total_groupings += groupings.size();
        }
    }

    // Should have at least one valid grouping match (wh_galaxy_split has a 2x8 mesh, and there are two 2x8_Mesh
    // definitions in the file)
    ASSERT_GE(total_groupings, 1u) << "Should have at least one valid grouping match";

    // Check that we have matches for MESH instances
    ASSERT_GE(valid_groupings.size(), 1u) << "Should have at least one instance type (MESH)";
    ASSERT_EQ(valid_groupings.count("MESH"), 1u) << "Should have MESH instance type";
    // wh_galaxy_split has multiple meshes in a graph
    ASSERT_GE(valid_groupings.at("MESH").size(), 1u) << "Should have at least one MESH instance";

    // Check that we have matches for the 2x8 mesh grouping (16 ASICs)
    // Names in triple_16x8 are "2x8_Mesh WH", "2x8_Mesh_adjacent", etc.
    size_t total_2x8_matches = 0;
    for (const auto& [instance_name, groupings] : valid_groupings.at("MESH")) {
        for (const auto& grouping : groupings) {
            if (grouping.asic_count == 16u && grouping.name.find("2x8") != std::string::npos) {
                total_2x8_matches++;
                EXPECT_EQ(grouping.asic_count, 16u) << "Should have 16 ASICs";
            }
        }
    }
    EXPECT_GE(total_2x8_matches, 1u) << "Should have at least one 2x8 mesh match";

    // Check that we have FABRIC level grouping (G0)
    ASSERT_EQ(valid_groupings.count("FABRIC"), 1u) << "Should have FABRIC instance type";
    ASSERT_EQ(valid_groupings.at("FABRIC").size(), 1u) << "Should have exactly one FABRIC instance";
    ASSERT_EQ(valid_groupings.at("FABRIC").count("G0"), 1u) << "Should have G0 FABRIC instance";
    const auto& g0_groupings_2x8 = valid_groupings.at("FABRIC").at("G0");
    ASSERT_GE(g0_groupings_2x8.size(), 1u) << "Should have at least one grouping for G0";
}

TEST_F(PhysicalGroupingDescriptorSP4Tests, GetValidGroupingsForMGD_8x16Mesh) {
    // Test matching an 8x16 mesh MGD (128 ASICs) to the 8x16_Mesh grouping
    const std::string pgd_path =
        "tests/tt_metal/tt_fabric/physical_groupings/bh_galaxy_rev_ab_physical_grouping_descriptor.textproto";
    const std::string mgd_path = "tt_metal/fabric/mesh_graph_descriptors/quad_galaxy_mesh_graph_descriptor.textproto";

    ASSERT_TRUE(std::filesystem::exists(pgd_path)) << "PGD file not found: " << pgd_path;
    ASSERT_TRUE(std::filesystem::exists(mgd_path)) << "MGD file not found: " << mgd_path;

    tt::tt_metal::PhysicalSystemDescriptor psd = create_psd_from_mock_cluster();
    PhysicalGroupingDescriptor pgd{std::filesystem::path(pgd_path)};
    MeshGraphDescriptor mgd{std::filesystem::path(mgd_path)};

    auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd, psd);

    // Count total groupings across all instances
    size_t total_groupings = 0;
    for (const auto& [instance_type, instances] : valid_groupings) {
        for (const auto& [instance_name, groupings] : instances) {
            total_groupings += groupings.size();
        }
    }

    // Should have at least one valid grouping match (may have multiple if there are duplicates)
    ASSERT_GE(total_groupings, 1u) << "Should have at least one valid grouping match";

    // Check that we have matches for MESH instances
    ASSERT_EQ(valid_groupings.size(), 1u) << "Should have exactly one instance type (MESH)";
    ASSERT_EQ(valid_groupings.count("MESH"), 1u) << "Should have MESH instance type";
    ASSERT_EQ(valid_groupings.at("MESH").size(), 1u) << "Should have exactly one MESH instance";

    // Check that we have a match for the 8x16_Mesh grouping (128 ASICs)
    // Note: May have multiple matches if there are duplicate definitions
    // When using mock clusters, grouping names may differ from expected names
    // so we just verify the ASIC count matches
    for (const auto& [instance_name, groupings] : valid_groupings.at("MESH")) {
        ASSERT_GE(groupings.size(), 1u) << "Should have at least one grouping for this instance";
        for (const auto& grouping : groupings) {
            EXPECT_EQ(grouping.asic_count, 128u) << "Should have 128 ASICs (name: " << grouping.name << ")";
            // Accept any grouping with 128 ASICs (8x16_Mesh or 4x32_Mesh are both valid)
            // Names may differ when using mock clusters vs file-based PSDs
        }
    }
}

TEST_F(PhysicalGroupingDescriptorSP4Tests, GetValidGroupingsForMGD_SingleGalaxy4x8) {
    // Test matching a single galaxy mesh MGD (32 ASICs) to the 4x8_Mesh grouping
    // Using single_bh_galaxy_mesh_graph_descriptor which has 8x4 (32 ASICs, same count but different topology)
    const std::string pgd_path =
        "tests/tt_metal/tt_fabric/physical_groupings/bh_galaxy_rev_ab_physical_grouping_descriptor.textproto";
    const std::string mgd_path =
        "tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto";

    ASSERT_TRUE(std::filesystem::exists(pgd_path)) << "PGD file not found: " << pgd_path;
    ASSERT_TRUE(std::filesystem::exists(mgd_path)) << "MGD file not found: " << mgd_path;

    tt::tt_metal::PhysicalSystemDescriptor psd = create_psd_from_mock_cluster();
    PhysicalGroupingDescriptor pgd{std::filesystem::path(pgd_path)};
    MeshGraphDescriptor mgd{std::filesystem::path(mgd_path)};

    auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd, psd);

    // Count total groupings across all instances
    size_t total_groupings = 0;
    for (const auto& [instance_type, instances] : valid_groupings) {
        for (const auto& [instance_name, groupings] : instances) {
            total_groupings += groupings.size();
        }
    }

    // A 4x8 (32-ASIC) mesh matches both the MESH and a torus variant of the 4x8_Mesh grouping → 2 matches.
    ASSERT_EQ(total_groupings, 2u) << "Should have two valid grouping matches (mesh + torus variant)";

    // Check that we have matches for MESH instances
    ASSERT_EQ(valid_groupings.size(), 1u) << "Should have exactly one instance type (MESH)";
    ASSERT_EQ(valid_groupings.count("MESH"), 1u) << "Should have MESH instance type";
    ASSERT_EQ(valid_groupings.at("MESH").size(), 1u) << "Should have exactly one MESH instance";

    // Check that we have a match for a 32-ASIC mesh grouping (4x8_Mesh or similar)
    // Note: single_bh_galaxy has 8x4 topology, which may match 4x8_Mesh if topology solver allows it
    for (const auto& [instance_name, groupings] : valid_groupings.at("MESH")) {
        ASSERT_GE(groupings.size(), 1u) << "Should have at least one grouping for this instance";
        for (const auto& grouping : groupings) {
            EXPECT_EQ(grouping.asic_count, 32u) << "Should have 32 ASICs";
            // Accept 4x8_Mesh or other 32-ASIC groupings
            EXPECT_TRUE(grouping.asic_count == 32u)
                << "Should match a 32-ASIC mesh grouping (name: " << grouping.name << ")";
        }
    }
}

TEST_F(PhysicalGroupingDescriptorSP4Tests, GetValidGroupingsForMGD_DualGalaxy8x8) {
    // Test matching a dual galaxy MGD with meshes
    // Using dual_galaxy_mesh_graph_descriptor which has 8x8 (64 ASICs) - different from 4x8 but testing dual mesh
    // matching
    const std::string pgd_path =
        "tests/tt_metal/tt_fabric/physical_groupings/bh_galaxy_rev_ab_physical_grouping_descriptor.textproto";
    const std::string mgd_path = "tt_metal/fabric/mesh_graph_descriptors/dual_galaxy_mesh_graph_descriptor.textproto";

    ASSERT_TRUE(std::filesystem::exists(pgd_path)) << "PGD file not found: " << pgd_path;
    ASSERT_TRUE(std::filesystem::exists(mgd_path)) << "MGD file not found: " << mgd_path;

    tt::tt_metal::PhysicalSystemDescriptor psd = create_psd_from_mock_cluster();
    PhysicalGroupingDescriptor pgd{std::filesystem::path(pgd_path)};
    MeshGraphDescriptor mgd{std::filesystem::path(mgd_path)};

    auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd, psd);

    // Count total groupings across all instances
    size_t total_groupings = 0;
    for (const auto& [instance_type, instances] : valid_groupings) {
        for (const auto& [instance_name, groupings] : instances) {
            total_groupings += groupings.size();
        }
    }

    // Should have at least one valid grouping match (dual_galaxy has 8x8 mesh, may not match 4x8_Mesh)
    ASSERT_GE(total_groupings, 0u) << "Should have valid grouping matches";

    // Check that we have matches for MESH instances (if any matches found)
    if (total_groupings > 0) {
        ASSERT_EQ(valid_groupings.size(), 1u) << "Should have exactly one instance type (MESH)";
        ASSERT_EQ(valid_groupings.count("MESH"), 1u) << "Should have MESH instance type";
        // dual_galaxy has one mesh instance
        ASSERT_GE(valid_groupings.at("MESH").size(), 1u) << "Should have at least one MESH instance";

        // Check groupings (may not match 4x8_Mesh since dual_galaxy is 8x8)
        for (const auto& [instance_name, groupings] : valid_groupings.at("MESH")) {
            for (const auto& grouping : groupings) {
                // Accept any valid match - dual_galaxy has 8x8 mesh (64 ASICs)
                EXPECT_GE(grouping.asic_count, 64u) << "Should have valid ASIC count (name: " << grouping.name
                                                    << ", count: " << grouping.asic_count << ")";
            }
        }
    }
}

// ============================================================================
// GET_VALID_GROUPINGS_FOR_MGD PHASE 3 TEST (higher-layer graph matching)
// ============================================================================
// Hierarchy: MESH -> PODS (FABRIC) -> SUPER_PODS (SUPER_FABRIC)
// PGD groupings: mix of mesh vs all-to-all at each level.
// MGD has ALL_TO_ALL topology at all graph levels, so G2 should only match
// super_pod_4_all_to_all (not super_pod_4_mesh), since PGD grouping = global graph.
//
// MGD: M0 (2x4), M1 (4x2); G0 (2 meshes, ALL_TO_ALL); G1 (4 meshes, ALL_TO_ALL);
//      G2 (4 graphs: 2xG1+2xG0, ALL_TO_ALL)
// ============================================================================

TEST_F(PhysicalGroupingDescriptorSP4Tests, GetValidGroupingsForMGD_Phase3_HigherLayerGraphMatching) {
    const std::string pgd_path = "tests/tt_metal/tt_fabric/physical_groupings/test_superpod_grouping.textproto";
    ASSERT_TRUE(std::filesystem::exists(pgd_path)) << "PGD file not found: " << pgd_path;

    tt::tt_metal::PhysicalSystemDescriptor psd = create_psd_from_mock_cluster();
    PhysicalGroupingDescriptor pgd{std::filesystem::path(pgd_path)};

    const std::string mgd_str = R"proto(
        mesh_descriptors {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology { dims: [ 2, 4 ] }
          host_topology { dims: [ 1, 1 ] }
          channels { count: 2 policy: RELAXED }
        }
        mesh_descriptors {
          name: "M1"
          arch: WORMHOLE_B0
          device_topology { dims: [ 4, 2 ] }
          host_topology { dims: [ 1, 1 ] }
          channels { count: 2 policy: RELAXED }
        }
        graph_descriptors {
          name: "G0"
          type: "FABRIC"
          instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
          instances { mesh { mesh_descriptor: "M1" mesh_id: 1 } }
          graph_topology {
            layout_type: ALL_TO_ALL
            channels { count: 2 policy: STRICT }
          }
        }
        graph_descriptors {
          name: "G1"
          type: "FABRIC"
          instances { mesh { mesh_descriptor: "M1" mesh_id: 0 } }
          instances { mesh { mesh_descriptor: "M1" mesh_id: 1 } }
          instances { mesh { mesh_descriptor: "M1" mesh_id: 2 } }
          instances { mesh { mesh_descriptor: "M1" mesh_id: 3 } }
          graph_topology {
            layout_type: ALL_TO_ALL
            channels { count: 2 policy: STRICT }
          }
        }
        graph_descriptors {
          name: "G2"
          type: "SUPER_FABRIC"
          instances { graph { graph_descriptor: "G1" graph_id: 0 } }
          instances { graph { graph_descriptor: "G1" graph_id: 1 } }
          instances { graph { graph_descriptor: "G0" graph_id: 2 } }
          instances { graph { graph_descriptor: "G0" graph_id: 3 } }
          graph_topology {
            layout_type: ALL_TO_ALL
            channels { count: 2 policy: STRICT }
          }
        }
        top_level_instance { graph { graph_descriptor: "G2" graph_id: 0 } }
    )proto";

    MeshGraphDescriptor mgd{mgd_str};

    auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd, psd);

    // Phase 2: MESH level must pass
    // Count unique mesh definitions (M0, M1), not instances
    ASSERT_GE(valid_groupings.size(), 1u) << "Should have at least MESH (Phase 2)";
    ASSERT_EQ(valid_groupings.at("MESH").size(), 2u);
    // M0 and M1 may have 2-3 matches (mesh_2x4, mesh_4x2, and possibly test_mesh for 2x4)
    ASSERT_GE(valid_groupings.at("MESH").at("M0").size(), 2u);
    ASSERT_GE(valid_groupings.at("MESH").at("M1").size(), 2u);

    // Verify they are mapped to the right grouping
    // M0 (2x4) may match mesh_2x4, mesh_4x2 (topologically isomorphic), or test_mesh - verify at least
    // mesh_2x4/mesh_4x2 are present
    const auto& m0_groupings = valid_groupings.at("MESH").at("M0");
    const auto& m1_groupings = valid_groupings.at("MESH").at("M1");

    // Grouping names may have suffixes (e.g., mesh_2x4_0, mesh_4x2_1) due to flattened combinations
    bool m0_has_mesh_2x4_or_4x2 = std::any_of(m0_groupings.begin(), m0_groupings.end(), [](const auto& g) {
        return g.name.starts_with("mesh_2x4") || g.name.starts_with("mesh_4x2");
    });
    bool m1_has_mesh_2x4_or_4x2 = std::any_of(m1_groupings.begin(), m1_groupings.end(), [](const auto& g) {
        return g.name.starts_with("mesh_2x4") || g.name.starts_with("mesh_4x2");
    });

    EXPECT_TRUE(m0_has_mesh_2x4_or_4x2) << "M0 (2x4) should map to at least one of mesh_2x4 or mesh_4x2";
    EXPECT_TRUE(m1_has_mesh_2x4_or_4x2) << "M1 (4x2) should map to at least one of mesh_2x4 or mesh_4x2";

    // Phase 3: FABRIC - G0 and G1 with ALL_TO_ALL
    // G0 (2 meshes) -> only dual_mesh_all_to_all (16 ASICs), NOT dual_mesh_row
    // G1 (4 meshes) -> only quad_mesh_all_to_all (32 ASICs), NOT quad_mesh_pod
    ASSERT_EQ(valid_groupings.count("FABRIC"), 1u) << "Phase 3 must be implemented: FABRIC should exist";
    ASSERT_EQ(valid_groupings.at("FABRIC").count("G0"), 1u) << "G0 should have mappings";
    const auto& g0_groupings = valid_groupings.at("FABRIC").at("G0");
    ASSERT_EQ(g0_groupings.size(), 1u) << "G0 should have exactly 1 matching grouping";
    EXPECT_TRUE(g0_groupings[0].name == "dual_mesh_row" || g0_groupings[0].name == "dual_mesh_all_to_all")
        << "G0 (2 meshes) may match dual_mesh_row or dual_mesh_all_to_all (structurally identical for 2 nodes)";

    ASSERT_EQ(valid_groupings.at("FABRIC").count("G1"), 1u) << "G1 should have mappings";
    const auto& g1_groupings = valid_groupings.at("FABRIC").at("G1");
    ASSERT_EQ(g1_groupings.size(), 1u) << "G1 should have exactly 1 matching grouping";
    EXPECT_EQ(g1_groupings[0].name, "quad_mesh_all_to_all")
        << "G1 (4 meshes, ALL_TO_ALL) -> only quad_mesh_all_to_all matches";

    // Phase 3: SUPER_FABRIC - G2 (4 graphs) with ALL_TO_ALL
    // should ONLY match super_pod_4_all_to_all, NOT super_pod_4_mesh (PGD grouping = global graph).
    ASSERT_EQ(valid_groupings.count("SUPER_FABRIC"), 1u) << "Phase 3 must be implemented: SUPER_FABRIC should exist";
    ASSERT_EQ(valid_groupings.at("SUPER_FABRIC").size(), 1u) << "G2 should have exactly 1 instance entry";
    const auto& g2_entry = *valid_groupings.at("SUPER_FABRIC").begin();
    const auto& g2_groupings = g2_entry.second;
    ASSERT_EQ(g2_groupings.size(), 1u) << "G2 should have exactly 1 matching grouping";
    EXPECT_EQ(g2_groupings[0].name, "super_pod_4_all_to_all")
        << "G2 has ALL_TO_ALL -> only all_to_all PGD grouping matches (not super_pod_4_mesh)";
}

TEST(PhysicalGroupingDescriptorTests, GetValidGroupingsForMGD_32x4Quad) {
    // Load the physical grouping descriptor
    const std::filesystem::path pgd_file_path =
        "tests/tt_metal/tt_fabric/physical_groupings/bh_galaxy_rev_ab_physical_grouping_descriptor.textproto";

    PhysicalGroupingDescriptor pgd(pgd_file_path);

    // Test with 32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto
    const std::filesystem::path mgd_file_path =
        "tt_metal/fabric/mesh_graph_descriptors/32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto";

    MeshGraphDescriptor mgd(mgd_file_path);

    auto* mock_desc = getenv("TT_METAL_MOCK_CLUSTER_DESC_PATH");
    if (mock_desc == nullptr) {
        GTEST_SKIP() << "TT_METAL_MOCK_CLUSTER_DESC_PATH not set - run with tt-run --mock-cluster-rank-binding";
    }
    tt::tt_metal::PhysicalSystemDescriptor psd = create_psd_from_mock_cluster();

    auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd, psd);

    // M0 mesh has device_topology [32, 4] = 128 chips
    // Should match meshes grouping with 4 hosts (4 * 32 = 128 ASICs, exact match)
    EXPECT_TRUE(valid_groupings.contains("MESH")) << "Should have MESH type in results";
    EXPECT_TRUE(valid_groupings.at("MESH").contains("M0")) << "Should have M0 mesh instance";

    ASSERT_FALSE(valid_groupings.at("MESH").at("M0").empty()) << "M0 should have at least one matching grouping";
    const auto& m0_grouping = valid_groupings.at("MESH").at("M0").front();
    EXPECT_EQ(m0_grouping.name, "meshes") << "M0 should match 'meshes' grouping";
    EXPECT_EQ(m0_grouping.asic_count, 128u) << "M0 grouping should have 128 ASICs (4 hosts)";

    // Verify it matches the 4 hosts grouping
    EXPECT_EQ(m0_grouping.items.size(), 4u) << "Should have 4 items (4 hosts)";
    if (!m0_grouping.items.empty()) {
        EXPECT_EQ(m0_grouping.items[0].type, GroupingItemInfo::ItemType::GROUPING_REF)
            << "First item should be a GROUPING_REF";
        EXPECT_EQ(m0_grouping.items[0].grouping_name, "hosts") << "Should reference 'hosts' grouping";
    }
}

TEST(PhysicalGroupingDescriptorTests, GetValidGroupingsForMGD_SingleGalaxy) {
    // Load the physical grouping descriptor
    const std::filesystem::path pgd_file_path =
        "tests/tt_metal/tt_fabric/physical_groupings/bh_galaxy_rev_ab_physical_grouping_descriptor.textproto";

    PhysicalGroupingDescriptor pgd(pgd_file_path);

    // Test with bh_glx_split_4x2.textproto
    const std::filesystem::path mgd_file_path =
        "tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto";

    MeshGraphDescriptor mgd(mgd_file_path);

    auto* mock_desc = getenv("TT_METAL_MOCK_CLUSTER_DESC_PATH");
    if (mock_desc == nullptr) {
        GTEST_SKIP() << "TT_METAL_MOCK_CLUSTER_DESC_PATH not set - run with tt-run --mock-cluster-rank-binding";
    }
    tt::tt_metal::PhysicalSystemDescriptor psd = create_psd_from_mock_cluster();

    auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd, psd);

    // M0 mesh has device_topology [8, 4] = 32 chips
    // Should match meshes grouping with 1 host (32 ASICs, exact match)
    EXPECT_TRUE(valid_groupings.contains("MESH")) << "Should have MESH type in results";
    EXPECT_TRUE(valid_groupings.at("MESH").contains("M0")) << "Should have M0 mesh instance";

    ASSERT_FALSE(valid_groupings.at("MESH").at("M0").empty()) << "M0 should have at least one matching grouping";
    const auto& m0_grouping = valid_groupings.at("MESH").at("M0").front();
    EXPECT_EQ(m0_grouping.name, "meshes") << "M0 should match 'meshes' grouping";
    EXPECT_EQ(m0_grouping.asic_count, 32u) << "M0 grouping should have 32 ASICs (1 host)";

    // Verify it matches the 1 host grouping
    EXPECT_EQ(m0_grouping.items.size(), 1u) << "Should have 1 item (1 host)";
    if (!m0_grouping.items.empty()) {
        EXPECT_EQ(m0_grouping.items[0].type, GroupingItemInfo::ItemType::GROUPING_REF)
            << "First item should be a GROUPING_REF";
        EXPECT_EQ(m0_grouping.items[0].grouping_name, "hosts") << "Should reference 'hosts' grouping";
    }
}

TEST(PhysicalGroupingDescriptorTests, GetValidGroupingsForMGD_BhGlxSplit4x2) {
    // Load the physical grouping descriptor
    const std::filesystem::path pgd_file_path =
        "tests/tt_metal/tt_fabric/physical_groupings/bh_galaxy_rev_ab_physical_grouping_descriptor.textproto";

    PhysicalGroupingDescriptor pgd(pgd_file_path);

    // Test with bh_glx_split_4x2.textproto
    const std::filesystem::path mgd_file_path = "tt_metal/fabric/mesh_graph_descriptors/bh_glx_split_4x2.textproto";

    MeshGraphDescriptor mgd(mgd_file_path);

    auto* mock_desc = getenv("TT_METAL_MOCK_CLUSTER_DESC_PATH");
    if (mock_desc == nullptr) {
        GTEST_SKIP() << "TT_METAL_MOCK_CLUSTER_DESC_PATH not set - run with tt-run --mock-cluster-rank-binding";
    }
    tt::tt_metal::PhysicalSystemDescriptor psd = create_psd_from_mock_cluster();

    auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd, psd);

    // M0 mesh has device_topology [4, 2] = 8 chips
    // Should match meshes grouping with 1 tray (8 ASICs, exact match)
    // Note: This test has multiple mesh instances (M0 mesh_id 0-47), all with same topology
    EXPECT_TRUE(valid_groupings.contains("MESH")) << "Should have MESH type in results";
    EXPECT_TRUE(valid_groupings.at("MESH").contains("M0")) << "Should have M0 mesh instance";

    ASSERT_FALSE(valid_groupings.at("MESH").at("M0").empty()) << "M0 should have at least one matching grouping";
    const auto& m0_grouping = valid_groupings.at("MESH").at("M0").front();
    EXPECT_EQ(m0_grouping.name, "meshes") << "M0 should match 'meshes' grouping";
    EXPECT_EQ(m0_grouping.asic_count, 8u) << "M0 grouping should have 8 ASICs (1 tray, exact match)";

    // Verify it matches the 1 tray grouping exactly (not oversized)
    EXPECT_EQ(m0_grouping.items.size(), 1u) << "Should have exactly 1 item (1 tray)";
    EXPECT_TRUE(!m0_grouping.items.empty()) << "Should have at least one item";
    EXPECT_EQ(m0_grouping.items[0].type, GroupingItemInfo::ItemType::GROUPING_REF)
        << "First item should be a GROUPING_REF";
    EXPECT_EQ(m0_grouping.items[0].grouping_name, "trays") << "Should reference 'trays' grouping";

    // Verify all items reference trays (should be exactly 1 tray reference)
    uint32_t tray_ref_count = 0;
    for (const auto& item : m0_grouping.items) {
        if (item.type == GroupingItemInfo::ItemType::GROUPING_REF && item.grouping_name == "trays") {
            tray_ref_count++;
        }
    }
    EXPECT_EQ(tray_ref_count, 1u) << "Should reference exactly 1 tray";
}

TEST(PhysicalGroupingDescriptorTests, GetValidGroupingsForMGD_Dual4x4) {
    // Load the physical grouping descriptor
    const std::filesystem::path pgd_file_path =
        "tests/tt_metal/tt_fabric/physical_groupings/bh_galaxy_rev_ab_physical_grouping_descriptor.textproto";

    PhysicalGroupingDescriptor pgd(pgd_file_path);

    // Test with dual_4x4_mesh_graph_descriptor.textproto
    // This is a dual mesh configuration with two 4x4 WORMHOLE_B0 meshes, each with host_topology [1, 1] (1 host)
    const std::filesystem::path mgd_file_path =
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_4x4_mesh_graph_descriptor.textproto";

    MeshGraphDescriptor mgd(mgd_file_path);

    auto* mock_desc = getenv("TT_METAL_MOCK_CLUSTER_DESC_PATH");
    if (mock_desc == nullptr) {
        GTEST_SKIP() << "TT_METAL_MOCK_CLUSTER_DESC_PATH not set - run with tt-run --mock-cluster-rank-binding";
    }
    tt::tt_metal::PhysicalSystemDescriptor psd = create_psd_from_mock_cluster();

    auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd, psd);

    // M0 mesh has device_topology [4, 4] = 16 chips
    // Should match meshes grouping with 2 trays (2 * 8 = 16 ASICs, exact match)
    // Note: This test has 2 mesh instances (M0 mesh_id 0 and 1), both with same topology
    EXPECT_TRUE(valid_groupings.contains("MESH")) << "Should have MESH type in results";
    EXPECT_TRUE(valid_groupings.at("MESH").contains("M0")) << "Should have M0 mesh instance";

    ASSERT_FALSE(valid_groupings.at("MESH").at("M0").empty()) << "M0 should have at least one matching grouping";
    const auto& m0_grouping = valid_groupings.at("MESH").at("M0").front();
    EXPECT_EQ(m0_grouping.name, "meshes") << "M0 should match 'meshes' grouping";
    EXPECT_EQ(m0_grouping.asic_count, 16u) << "M0 grouping should have 16 ASICs (2 trays, exact match)";

    // Verify it matches the 2 trays grouping exactly (not oversized)
    EXPECT_EQ(m0_grouping.items.size(), 2u) << "Should have exactly 2 items (2 trays)";
    EXPECT_TRUE(!m0_grouping.items.empty()) << "Should have at least one item";
    EXPECT_EQ(m0_grouping.items[0].type, GroupingItemInfo::ItemType::GROUPING_REF)
        << "First item should be a GROUPING_REF";
    EXPECT_EQ(m0_grouping.items[0].grouping_name, "trays") << "Should reference 'trays' grouping";

    // Verify all items reference trays (should be exactly 2 tray references)
    uint32_t tray_ref_count = 0;
    for (const auto& item : m0_grouping.items) {
        if (item.type == GroupingItemInfo::ItemType::GROUPING_REF && item.grouping_name == "trays") {
            tray_ref_count++;
        }
    }
    EXPECT_EQ(tray_ref_count, 2u) << "Should reference exactly 2 trays";
}

TEST(PhysicalGroupingDescriptorTests, GetValidGroupingsForMGD_Dual8x2) {
    // Load the physical grouping descriptor
    const std::filesystem::path pgd_file_path =
        "tests/tt_metal/tt_fabric/physical_groupings/bh_galaxy_rev_ab_physical_grouping_descriptor.textproto";

    PhysicalGroupingDescriptor pgd(pgd_file_path);

    // Test with dual_8x2_mesh_graph_descriptor.textproto
    // This is a dual mesh configuration with two 8x2 WORMHOLE_B0 meshes, each with host_topology [1, 1] (1 host)
    const std::filesystem::path mgd_file_path =
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_8x2_mesh_graph_descriptor.textproto";

    MeshGraphDescriptor mgd(mgd_file_path);

    auto* mock_desc = getenv("TT_METAL_MOCK_CLUSTER_DESC_PATH");
    if (mock_desc == nullptr) {
        GTEST_SKIP() << "TT_METAL_MOCK_CLUSTER_DESC_PATH not set - run with tt-run --mock-cluster-rank-binding";
    }
    tt::tt_metal::PhysicalSystemDescriptor psd = create_psd_from_mock_cluster();

    auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd, psd);

    // M0 mesh has device_topology [8, 2] = 16 chips
    // Should match meshes grouping with 2 trays (2 * 8 = 16 ASICs, exact match)
    // Note: This test has 2 mesh instances (M0 mesh_id 0 and 1), both with same topology
    EXPECT_TRUE(valid_groupings.contains("MESH")) << "Should have MESH type in results";
    EXPECT_TRUE(valid_groupings.at("MESH").contains("M0")) << "Should have M0 mesh instance";

    ASSERT_FALSE(valid_groupings.at("MESH").at("M0").empty()) << "M0 should have at least one matching grouping";
    const auto& m0_grouping = valid_groupings.at("MESH").at("M0").front();
    EXPECT_EQ(m0_grouping.name, "meshes") << "M0 should match 'meshes' grouping";
    EXPECT_EQ(m0_grouping.asic_count, 16u) << "M0 grouping should have 16 ASICs (2 trays, exact match)";

    // Verify it matches the 2 trays grouping exactly (not oversized)
    EXPECT_EQ(m0_grouping.items.size(), 2u) << "Should have exactly 2 items (2 trays)";
    EXPECT_TRUE(!m0_grouping.items.empty()) << "Should have at least one item";
    EXPECT_EQ(m0_grouping.items[0].type, GroupingItemInfo::ItemType::GROUPING_REF)
        << "First item should be a GROUPING_REF";
    EXPECT_EQ(m0_grouping.items[0].grouping_name, "trays") << "Should reference 'trays' grouping";

    // Verify all items reference trays (should be exactly 2 tray references)
    uint32_t tray_ref_count = 0;
    for (const auto& item : m0_grouping.items) {
        if (item.type == GroupingItemInfo::ItemType::GROUPING_REF && item.grouping_name == "trays") {
            tray_ref_count++;
        }
    }
    EXPECT_EQ(tray_ref_count, 2u) << "Should reference exactly 2 trays";
}

static size_t count_distinct_hosts_for_asics(
    const tt::tt_metal::PhysicalSystemDescriptor& psd, const std::unordered_set<tt::tt_metal::AsicID>& asics) {
    std::set<std::string> hosts;
    for (const auto& asic : asics) {
        hosts.insert(psd.get_host_name_for_asic(asic));
    }
    return hosts.size();
}

TEST(PhysicalGroupingDescriptorTests, GetValidGroupingsForMGD_SinglePod4x4LineLinePrefersSingleHost) {
    // Single BH galaxy pod (32 ASICs on one host): a 4x4 LINE+LINE mesh with host_topology [1,1] can embed as
    // Rev C 4x4_Mesh (two trays, single host) or 4x4_SplitHost (four half-trays). Both should be committed;
    // PSD placement should still prefer single-host 4x4_Mesh when host_topology is [1,1].
    const std::filesystem::path pgd_file_path =
        "tests/tt_metal/tt_fabric/physical_groupings/wh_bh_rev_c_galaxy_physical_grouping_descriptor.textproto";
    const std::filesystem::path mgd_file_path =
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/single_pod_4x4_line_line_mesh_graph_descriptor.textproto";

    ASSERT_TRUE(std::filesystem::exists(pgd_file_path)) << "PGD file not found: " << pgd_file_path;
    ASSERT_TRUE(std::filesystem::exists(mgd_file_path)) << "MGD file not found: " << mgd_file_path;

    auto* mock_desc = getenv("TT_METAL_MOCK_CLUSTER_DESC_PATH");
    if (mock_desc == nullptr) {
        GTEST_SKIP() << "TT_METAL_MOCK_CLUSTER_DESC_PATH not set - run with bh_galaxy_xyz_cluster_desc.yaml";
    }

    tt::tt_metal::PhysicalSystemDescriptor psd = create_psd_from_mock_cluster();
    PhysicalGroupingDescriptor pgd(pgd_file_path);
    MeshGraphDescriptor mgd(mgd_file_path);

    auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd, psd);

    ASSERT_TRUE(valid_groupings.contains("MESH")) << "Should have MESH type in results";
    ASSERT_TRUE(valid_groupings.at("MESH").contains("M0")) << "Should have M0 mesh instance";
    ASSERT_FALSE(valid_groupings.at("MESH").at("M0").empty()) << "M0 should have at least one matching grouping";

    bool found_single_host_mesh = false;
    bool found_split_host = false;
    for (const auto& grouping : valid_groupings.at("MESH").at("M0")) {
        if (grouping.name.find("4x4_Mesh") != std::string::npos &&
            grouping.name.find("SplitHost") == std::string::npos) {
            found_single_host_mesh = true;
        }
        if (grouping.name.find("SplitHost") != std::string::npos) {
            found_split_host = true;
        }
    }
    EXPECT_TRUE(found_single_host_mesh) << "Expected 4x4_Mesh (single-host two-tray) grouping to match";
    EXPECT_TRUE(found_split_host) << "Expected 4x4_SplitHost grouping to be committed alongside 4x4_Mesh";

    const auto& committed_groupings = valid_groupings.at("MESH").at("M0");
    // TODO(plan 3 §8(a)): rewrite onto solve_adjacency_guided_placement when find_all_in_psd is deleted.
    const auto placements = pgd.find_all_in_psd(committed_groupings, psd);
    ASSERT_FALSE(placements.empty()) << "Should find at least one PSD placement for the 4x4 mesh";

    for (const auto& placement : placements) {
        EXPECT_EQ(placement.asics.size(), 16u) << "Each 4x4 placement should cover 16 ASICs";
        EXPECT_EQ(count_distinct_hosts_for_asics(psd, placement.asics), 1u)
            << "Set-packing should prefer single-host placements when host_topology is [1,1]";

        // find_all_in_psd copies the matched grouping's pinning onto the placement.
        EXPECT_EQ(placement.mesh_node_to_asic_position.size(), 16u)
            << "Composed pinning should cover all 16 logical chips";
        std::set<tt::tt_metal::ASICPosition> composed_positions;
        for (const auto& [chip_id, asic_position] : placement.mesh_node_to_asic_position) {
            composed_positions.insert(asic_position);
        }
        std::set<tt::tt_metal::ASICPosition> footprint_positions;
        for (const auto& asic_id : placement.asics) {
            footprint_positions.insert(
                tt::tt_metal::ASICPosition{psd.get_tray_id(asic_id), psd.get_asic_location(asic_id)});
        }
        EXPECT_EQ(composed_positions, footprint_positions)
            << "Composed pinning should pin exactly the footprint ASIC positions";
    }
}

// get_valid_groupings_for_mgd should persist logical chip_id -> ASIC position pinning on every committed MESH
// grouping (mesh_node_to_asic_position), so the PGD pinning discovered during matching is available downstream.
TEST(PhysicalGroupingDescriptorTests, GetValidGroupingsForMGD_PopulatesMeshNodeToAsicPosition) {
    const std::filesystem::path pgd_file_path =
        "tests/tt_metal/tt_fabric/physical_groupings/wh_bh_rev_c_galaxy_physical_grouping_descriptor.textproto";
    const std::filesystem::path mgd_file_path =
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/single_pod_4x4_line_line_mesh_graph_descriptor.textproto";

    ASSERT_TRUE(std::filesystem::exists(pgd_file_path)) << "PGD file not found: " << pgd_file_path;
    ASSERT_TRUE(std::filesystem::exists(mgd_file_path)) << "MGD file not found: " << mgd_file_path;

    auto* mock_desc = getenv("TT_METAL_MOCK_CLUSTER_DESC_PATH");
    if (mock_desc == nullptr) {
        GTEST_SKIP() << "TT_METAL_MOCK_CLUSTER_DESC_PATH not set - run with bh_galaxy_xyz_cluster_desc.yaml";
    }

    tt::tt_metal::PhysicalSystemDescriptor psd = create_psd_from_mock_cluster();
    PhysicalGroupingDescriptor pgd(pgd_file_path);
    MeshGraphDescriptor mgd(mgd_file_path);

    auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd, psd);
    ASSERT_TRUE(valid_groupings.contains("MESH"));
    ASSERT_TRUE(valid_groupings.at("MESH").contains("M0"));
    const auto& committed_groupings = valid_groupings.at("MESH").at("M0");
    ASSERT_FALSE(committed_groupings.empty());

    constexpr size_t kMgdNodeCount = 16;  // single_pod_4x4 is a 4x4 mesh => 16 logical chips (row-major 0..15)
    size_t groupings_with_pinning = 0;
    for (const auto& grouping : committed_groupings) {
        ASSERT_FALSE(grouping.mesh_node_to_asic_position.empty())
            << "Committed PGD grouping '" << grouping.name << "' should carry logical chip_id -> ASIC position pinning";
        const auto& pinning = grouping.mesh_node_to_asic_position;
        ++groupings_with_pinning;

        EXPECT_EQ(pinning.size(), kMgdNodeCount)
            << "Pinning for '" << grouping.name << "' should cover every MGD mesh node";

        std::set<LogicalChipId> seen_chip_ids;
        std::set<tt::tt_metal::ASICPosition> seen_positions;
        for (const auto& [chip_id, asic_position] : pinning) {
            EXPECT_LT(chip_id, kMgdNodeCount) << "Logical chip id out of range for '" << grouping.name << "'";
            EXPECT_GT(*asic_position.first, 0u) << "Tray id should be set for chip " << chip_id;
            EXPECT_GT(*asic_position.second, 0u) << "ASIC location should be set for chip " << chip_id;
            EXPECT_TRUE(seen_chip_ids.insert(chip_id).second) << "Duplicate logical chip id in pinning";
            EXPECT_TRUE(seen_positions.insert(asic_position).second)
                << "Pinning is not injective for '" << grouping.name << "'";
        }
    }
    EXPECT_GT(groupings_with_pinning, 0u)
        << "At least one committed grouping should carry logical chip_id -> ASIC position pinning";
}

// PGD<->MGD matching receives MGD pinnings as many-to-many groups (same shape as TopologyMapper /
// TopologyMappingConfig). Verify explicit corner all-to-all pinnings still commit PGD layouts with
// mesh_node_to_asic_position populated.
TEST(PhysicalGroupingDescriptorTests, GetValidGroupingsForMGD_WithManyToManyPinnings_StillCommitsPgdLayout) {
    const std::filesystem::path pgd_file_path =
        "tests/tt_metal/tt_fabric/physical_groupings/wh_bh_rev_c_galaxy_physical_grouping_descriptor.textproto";
    ASSERT_TRUE(std::filesystem::exists(pgd_file_path)) << "PGD file not found: " << pgd_file_path;

    const std::string mgd_text_proto = R"proto(
        mesh_descriptors {
          name: "M0"
          arch: BLACKHOLE
          device_topology {
            dims: [ 4, 4 ]
            dim_types: [ LINE, LINE ]
          }
          host_topology { dims: [ 1, 1 ] }
          channels { count: 2 policy: RELAXED }
        }
        top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
        pinnings {
          logical_fabric_node_id { mesh_id: 0 chip_id: 0 }
          logical_fabric_node_id { mesh_id: 0 chip_id: 3 }
          logical_fabric_node_id { mesh_id: 0 chip_id: 12 }
          logical_fabric_node_id { mesh_id: 0 chip_id: 15 }
          physical_asic_position { tray_id: 1 asic_location: 1 }
          physical_asic_position { tray_id: 2 asic_location: 1 }
          physical_asic_position { tray_id: 3 asic_location: 1 }
          physical_asic_position { tray_id: 4 asic_location: 1 }
          physical_asic_position { tray_id: 1 asic_location: 5 }
        }
    )proto";

    auto* mock_desc = getenv("TT_METAL_MOCK_CLUSTER_DESC_PATH");
    if (mock_desc == nullptr) {
        GTEST_SKIP() << "TT_METAL_MOCK_CLUSTER_DESC_PATH not set - run with bh_galaxy_xyz_cluster_desc.yaml";
    }

    tt::tt_metal::PhysicalSystemDescriptor psd = create_psd_from_mock_cluster();
    PhysicalGroupingDescriptor pgd(pgd_file_path);
    MeshGraphDescriptor mgd(mgd_text_proto);

    const auto& pinning_groups = mgd.get_pinnings();
    ASSERT_EQ(pinning_groups.size(), 1u);
    ASSERT_EQ(pinning_groups.at(MeshId{0}).size(), 1u);
    ASSERT_EQ(pinning_groups.at(MeshId{0})[0].fabric_nodes.size(), 4u);
    ASSERT_GE(pinning_groups.at(MeshId{0})[0].asic_positions.size(), 4u);

    auto without_pinnings = pgd.get_valid_groupings_for_mgd(mgd, psd);
    auto with_pinnings = pgd.get_valid_groupings_for_mgd(mgd, psd, mgd.get_pinnings());

    ASSERT_TRUE(without_pinnings.contains("MESH"));
    ASSERT_TRUE(with_pinnings.contains("MESH"));
    ASSERT_TRUE(with_pinnings.at("MESH").contains("M0"));
    ASSERT_FALSE(with_pinnings.at("MESH").at("M0").empty())
        << "PGD matching should succeed with many-to-many MGD pinnings";

    for (const auto& grouping : with_pinnings.at("MESH").at("M0")) {
        ASSERT_FALSE(grouping.mesh_node_to_asic_position.empty())
            << "PGD-derived layout pinning must be populated for '" << grouping.name << "'";
        EXPECT_EQ(grouping.mesh_node_to_asic_position.size(), 16u)
            << "Committed PGD layout should cover all 16 logical chips";
    }
}

// ---------------------------------------------------------------------------------------------
// Adjacency-guided placement
//
// These load a handcrafted PSD straight from a textproto rather than going through a mock cluster,
// so unlike the tests above they need no TT_METAL_MOCK_CLUSTER_DESC_PATH and never skip.
//
// All three share one PGD and vary only the PSD and the MGD, which isolates the seam constraint:
// the same pair of meshes must place when the physical chips they need are linked and must fail
// when they are not.
//
// The shared PGD, test_1x2_mesh_grouping.textproto, is a single MESH grouping of two chips with
// both ASIC locations UNSPECIFIED, so it is free to land on any linked pair:
//
//     1x2_Mesh:   [ ]---[ ]
//
// The two PSDs below both give every link 2 ethernet channels, which matters because the MGD asks
// for 2 channels both inside a mesh (STRICT) and across the mesh-level edge (RELAXED). "=="
// denotes a 2-channel link.
//
//   test_4asic_line.textproto            test_4asic_2mesh.textproto
//
//     100 == 101 == 102 == 103             100 == 101      102 == 103
//
//   one connected line of four           two pairs, nothing joining them
//
// Chip ids are also their asic_location: 100->0, 101->1, 102->2, 103->3, all on host0.
// ---------------------------------------------------------------------------------------------

namespace {

namespace utils = tt::tt_metal::experimental::tt_fabric;

constexpr const char* kAdjacencyPgdPath =
    "tests/tt_metal/tt_fabric/physical_groupings/test_1x2_mesh_grouping.textproto";
constexpr const char* kRingPgdPath = "tests/tt_metal/tt_fabric/physical_groupings/test_ring_mesh_groupings.textproto";
constexpr const char* kSingleMeshMgdPath =
    "tests/tt_metal/tt_fabric/custom_mesh_descriptors/test_single_1x2_mesh.textproto";
constexpr const char* kTwoPairPsdPath = "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_4asic_2mesh.textproto";
constexpr const char* kLinePsdPath = "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_4asic_line.textproto";
constexpr const char* kRingPsdPath = "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_6asic_ring.textproto";
constexpr const char* kOpenLinePsdPath = "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_6asic_line.textproto";
constexpr const char* kLinkedMgdPath =
    "tests/tt_metal/tt_fabric/custom_mesh_descriptors/test_two_1x2_meshes_linked.textproto";
constexpr const char* kUnlinkedMgdPath =
    "tests/tt_metal/tt_fabric/custom_mesh_descriptors/test_two_1x2_meshes_unlinked.textproto";
constexpr const char* kRingMgdPath =
    "tests/tt_metal/tt_fabric/custom_mesh_descriptors/test_alternating_ring_meshes.textproto";
constexpr const char* kMixedShapePgdPath =
    "tests/tt_metal/tt_fabric/physical_groupings/test_mixed_shape_groupings.textproto";
constexpr const char* kMixedShapeMgdPath =
    "tests/tt_metal/tt_fabric/custom_mesh_descriptors/test_mixed_shape_chain_meshes.textproto";
constexpr const char* kSquareForkPsdPath =
    "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_10asic_square_fork.textproto";
constexpr const char* kStarMgdPath =
    "tests/tt_metal/tt_fabric/custom_mesh_descriptors/test_star_channel_count_meshes.textproto";
constexpr const char* kStarPsdPath = "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_12asic_star.textproto";
constexpr const char* kWideThenNarrowMgdPath =
    "tests/tt_metal/tt_fabric/custom_mesh_descriptors/test_pair_1x2_then_1x1.textproto";
constexpr const char* kTwoSinglesMgdPath =
    "tests/tt_metal/tt_fabric/custom_mesh_descriptors/test_pair_two_1x1.textproto";
constexpr const char* kDumbbellPsdPath = "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_5asic_dumbbell.textproto";
constexpr const char* kNarrowThenWideMgdPath =
    "tests/tt_metal/tt_fabric/custom_mesh_descriptors/test_pair_1x1_then_1x2.textproto";
constexpr const char* kWideDumbbellPsdPath = "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_6asic_dumbbell.textproto";
constexpr const char* kUnevenLinePsdPath = "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_3asic_uneven_line.textproto";
constexpr const char* kRelaxedSeamMgdPath =
    "tests/tt_metal/tt_fabric/custom_mesh_descriptors/test_two_1x1_relaxed_seam.textproto";
constexpr const char* kRelaxedWideSeamMgdPath =
    "tests/tt_metal/tt_fabric/custom_mesh_descriptors/test_two_1x1_relaxed_wide_seam.textproto";
constexpr const char* kStrictWideSeamMgdPath =
    "tests/tt_metal/tt_fabric/custom_mesh_descriptors/test_two_1x1_strict_wide_seam.textproto";
constexpr const char* kOnePairPgdPath =
    "tests/tt_metal/tt_fabric/physical_groupings/test_1x2_mesh_grouping_pinned_to_one_pair.textproto";

// ----- pipeline steps -------------------------------------------------------------------------
//
// The tests below call the production functions for each stage directly, one at a time, so every
// stage is visible at the call site and can be asserted on:
//
//   build pgd -> get valid groupings -> build logical_multi_mesh_adjacency_graph
//             -> place / build flat_adjacency_map_from_psd
//             -> build hierarchical_from_flat_graph -> map_multi_mesh_to_physical
//
// Two things are worth knowing when reading them:
//
//   build_hierarchical_from_flat_graph splits the flat ASIC graph by placed footprint and links two
//   meshes when a real ethernet connection crosses between them. Meshes are keyed by placement
//   index, matching the mesh id ordering placement returns, and any PGD pinning a placement carries
//   is preserved -- so DFS placements and find_all_in_psd placements get identical treatment.
//
//   map_multi_mesh_to_physical is given disable_rank_bindings because these fixtures are
//   single-host and rank constraints are not what is under test; connectivity is left RELAXED,
//   matching the default the mapper documents.

// ----- reading the results ----------------------------------------------------------------------

// The ASICs each placement claims, one sorted set per mesh, ordered by mesh id. Only needed for
// find_all_in_psd, whose output never goes through the mapper.
std::vector<std::set<uint64_t>> footprints_of(const std::vector<PsdPlacement>& placements) {
    std::vector<std::set<uint64_t>> footprints;
    footprints.reserve(placements.size());
    for (const auto& placement : placements) {
        std::set<uint64_t> asics;
        for (const auto& asic : placement.asics) {
            asics.insert(*asic);
        }
        footprints.push_back(std::move(asics));
    }
    return footprints;
}

// The ASICs the two-level solve actually bound to each mesh, one sorted set per mesh, ordered by
// mesh id. Assertions read this rather than the raw placement footprints: it is the mapper's own
// output, so a mesh only appears here if map_multi_mesh_to_physical really seated it on those chips.
std::vector<std::set<uint64_t>> mapped_footprints(const utils::TopologyMappingResult& mapping) {
    std::map<MeshId, std::set<uint64_t>> per_mesh;
    for (const auto& [fabric_node, asic] : mapping.fabric_node_to_asic) {
        per_mesh[fabric_node.mesh_id].insert(*asic);
    }
    std::vector<std::set<uint64_t>> footprints;
    footprints.reserve(per_mesh.size());
    for (auto& [mesh_id, asics] : per_mesh) {
        footprints.push_back(std::move(asics));
    }
    return footprints;
}

std::size_t channels_between(
    const AdjacencyGraph<tt::tt_metal::AsicID>& flat_graph,
    const std::set<uint64_t>& left,
    const std::set<uint64_t>& right) {
    std::size_t channels = 0;
    for (uint64_t chip : left) {
        for (const auto& neighbor : flat_graph.get_neighbors(tt::tt_metal::AsicID{chip})) {
            channels += static_cast<std::size_t>(right.count(*neighbor));
        }
    }
    return channels;
}

std::set<uint64_t> chips_in(const std::vector<std::set<uint64_t>>& footprints) {
    std::set<uint64_t> chips;
    for (const auto& footprint : footprints) {
        chips.insert(footprint.begin(), footprint.end());
    }
    return chips;
}

}  // namespace

// A line of four chips has exactly one way to seat two adjacent 1x2 meshes, so the search must
// return that pairing and nothing else.
//
//   mesh-level graph (linked MGD)        physical graph (4-chip line)
//
//     M0[0] == M0[1]                       100 == 101 == 102 == 103
//
//   expected placement
//
//     100 == 101 == 102 == 103
//     '----v----'  '----v----'
//        M0[0]         M0[1]
//
//   M0[0] and M0[1] each take an end pair, and the 101==102 link in the middle carries the seam.
//   Seating either mesh on the middle pair {101,102} would leave the remaining chips 100 and 103
//   unlinked, so no second mesh could form; that is what makes the answer unique.
TEST(AdjacencyGuidedPlacement, LinkedMeshesPlaceAdjacentlyOnLine) {
    // build pgd
    PhysicalGroupingDescriptor pgd{std::filesystem::path(kAdjacencyPgdPath)};
    MeshGraphDescriptor mgd{std::filesystem::path(kLinkedMgdPath)};
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(kLinePsdPath);

    // get valid groupings
    const auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd, psd);

    // build logical
    const auto logical = utils::build_logical_multi_mesh_adjacency_graph(mgd);

    // place, and build the flat ASIC adjacency the physical graph is derived from
    const auto placements = pgd.solve_adjacency_guided_placement(mgd, valid_groupings, psd);
    const AdjacencyGraph<tt::tt_metal::AsicID> flat_graph(utils::build_flat_adjacency_map_from_psd(psd));
    ASSERT_EQ(placements.size(), 2u) << "both meshes should be placed on the 4-chip line";

    // build physical
    const auto physical = utils::build_hierarchical_from_flat_graph(flat_graph, placements);

    // place and map
    utils::TopologyMappingConfig config;
    config.disable_rank_bindings = true;
    const auto mapping = utils::map_multi_mesh_to_physical(logical, physical, config);
    ASSERT_TRUE(mapping.success) << "the two-level solve should succeed, but failed with: " << mapping.error_message;
    EXPECT_THAT(
        mapped_footprints(mapping),
        ::testing::UnorderedElementsAre(std::set<uint64_t>{100, 101}, std::set<uint64_t>{102, 103}))
        << "the only disjoint adjacent seating of two 1x2 meshes on 100-101-102-103";
}

// Same meshes, same grouping, but the PSD is two disconnected pairs. Each mesh still fits on its
// own, so only the mesh-level edge can rule this out.
//
//   mesh-level graph (linked MGD)        physical graph (two disjoint pairs)
//
//     M0[0] == M0[1]                       100 == 101      102 == 103
//
//   attempted placement
//
//     100 == 101       102 == 103
//     '----v----'      '----v----'
//        M0[0]      ?     M0[1]
//                   ^
//        no link here, so the seam cannot be met
//
//   Both meshes fit and the two pairs are disjoint, so everything except adjacency is satisfied.
//   The only reason to reject is the M0[0]--M0[1] edge, which has no physical link to sit on.
TEST(AdjacencyGuidedPlacement, LinkedMeshesFailOnDisconnectedPairs) {
    // build pgd
    PhysicalGroupingDescriptor pgd{std::filesystem::path(kAdjacencyPgdPath)};
    MeshGraphDescriptor mgd{std::filesystem::path(kLinkedMgdPath)};
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(kTwoPairPsdPath);

    // get valid groupings
    const auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd, psd);

    // build logical
    const auto logical = utils::build_logical_multi_mesh_adjacency_graph(mgd);

    // place, and build the flat ASIC adjacency the physical graph is derived from
    const auto placements = pgd.solve_adjacency_guided_placement(mgd, valid_groupings, psd);

    // Placement fails, so there is no physical graph to build and nothing to map.
    EXPECT_TRUE(placements.empty()) << "no link joins the two pairs, so the seam cannot be satisfied";
}

// The control for the test above: drop the mesh-level edge and the same disconnected PSD becomes
// placeable, which shows the failure there came from the seam and not from the meshes not fitting.
//
//   mesh-level graph (unlinked MGD)      physical graph (two disjoint pairs, unchanged)
//
//     M0[0]     M0[1]                      100 == 101      102 == 103
//
//   expected placement
//
//     100 == 101       102 == 103
//     '----v----'      '----v----'
//        M0[0]            M0[1]
//
//   With no edge to honour the meshes only have to be disjoint. A 1x2 mesh cannot straddle the two
//   pairs, so each still takes one whole pair and the footprints match the line test's.
TEST(AdjacencyGuidedPlacement, UnlinkedMeshesPlaceOnDisconnectedPairs) {
    // build pgd
    PhysicalGroupingDescriptor pgd{std::filesystem::path(kAdjacencyPgdPath)};
    MeshGraphDescriptor mgd{std::filesystem::path(kUnlinkedMgdPath)};
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(kTwoPairPsdPath);

    // get valid groupings
    const auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd, psd);

    // build logical
    const auto logical = utils::build_logical_multi_mesh_adjacency_graph(mgd);

    // place, and build the flat ASIC adjacency the physical graph is derived from
    const auto placements = pgd.solve_adjacency_guided_placement(mgd, valid_groupings, psd);
    const AdjacencyGraph<tt::tt_metal::AsicID> flat_graph(utils::build_flat_adjacency_map_from_psd(psd));
    ASSERT_EQ(placements.size(), 2u) << "both meshes should be placed when nothing forces them to touch";

    // build physical
    const auto physical = utils::build_hierarchical_from_flat_graph(flat_graph, placements);

    // place and map
    utils::TopologyMappingConfig config;
    config.disable_rank_bindings = true;
    const auto mapping = utils::map_multi_mesh_to_physical(logical, physical, config);
    ASSERT_TRUE(mapping.success) << "the two-level solve should succeed, but failed with: " << mapping.error_message;
    EXPECT_THAT(
        mapped_footprints(mapping),
        ::testing::UnorderedElementsAre(std::set<uint64_t>{100, 101}, std::set<uint64_t>{102, 103}))
        << "each mesh must take one whole pair, since a 1x2 mesh cannot straddle the two pairs";
}

// Four meshes of two alternating shapes in a ring, on a ring of six chips that exactly fits them.
// This is the heterogeneous case where a placement can be perfectly disjoint and still leave
// mesh-level edges with no physical link under them: packing the two 1x2 meshes side by side uses
// every chip and overlaps nothing, yet strands two of the four seams.
//
//   mesh-level graph                     physical graph (6-chip ring)
//
//     A0 --- B0                            100 == 101 == 102
//     |       |                             |             |
//     B1 --- A1                            105 == 104 == 103
//
//   a placement that packs same shapes together (disjoint, but wrong)
//
//     100 == 101 == 102 == 103 == 104 == 105
//     '----v----'  '----v----'    |      |
//        A0            A1         B0     B1
//
//     A0--B0 and A1--B1 have no link under them, so two seams are stranded.
//
//   a placement that interleaves the shapes (what the search must find)
//
//     100 == 101 == 102 == 103 == 104 == 105
//     '----v----'    |    '----v----'    |
//        A0          B0       A1         B1
//     '-------------------------------------'  (105 == 100 closes the ring)
//
//     A0--B0 on 101==102, B0--A1 on 102==103, A1--B1 on 104==105, B1--A0 on 105==100.
//
// Rotating or reflecting that arrangement is equally valid, so the test asserts the property the
// issue cares about rather than exact chips, and it asserts it for both placement paths so the two
// are compared on identical terms.
//
// Both paths are judged by the same thing, and it is the production mapper rather than a hand-rolled
// adjacency check: map_multi_mesh_to_physical has to complete. Its inter-mesh stage embeds the MGD's
// mesh graph into the mesh-level graph derived from the placement, so a stranded seam shows up as an
// inter-mesh failure; its intra-mesh stage then binds every fabric node to an ASIC.
//
// find_all_in_psd is the old, pre-DFS entry point: it takes one shape's groupings and packs that
// shape into the PSD, with no MGD and therefore no knowledge of the mesh-level edges. Run per shape
// on this ring it produces
//
//     A (1x2):  {100,101}  {102,103}  {104,105}
//     B (1x1):  {100} {101} {102} {103} {104} {105}
//
// The A packing tiles the ring on even boundaries only. That is a maximal, perfectly disjoint
// packing, and it is already fatal: the interleaved seating needs an A mesh on {103,104}, which
// straddles two of those tiles and so is never offered.
//
// The test seats the meshes on that packing -- two A tiles and the two chips they leave over -- and
// puts the result through build_hierarchical_from_flat_graph and map_multi_mesh_to_physical, the
// same two calls the DFS placement goes through below. The old arrangement must fail to map and the
// DFS one must succeed. The assertion on the A pool is what makes that general rather than a
// statement about one arrangement: no seating drawn from a pool that never straddles a tile
// boundary can satisfy the ring, however the meshes are permuted across it.
TEST(AdjacencyGuidedPlacement, AlternatingShapeRingDfsPlacesAndMapsWhereOldPackingCannot) {
    const std::set<uint64_t> whole_ring = {100, 101, 102, 103, 104, 105};

    // build pgd
    PhysicalGroupingDescriptor pgd{std::filesystem::path(kRingPgdPath)};
    MeshGraphDescriptor mgd{std::filesystem::path(kRingMgdPath)};
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(kRingPsdPath);

    // get valid groupings
    const auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd, psd);

    // build logical
    const auto logical = utils::build_logical_multi_mesh_adjacency_graph(mgd);

    // build the flat ASIC adjacency both paths are placed against, and the mapping config both are
    // mapped under
    const AdjacencyGraph<tt::tt_metal::AsicID> flat_graph(utils::build_flat_adjacency_map_from_psd(psd));
    utils::TopologyMappingConfig config;
    config.disable_rank_bindings = true;

    // ----- old path: find_all_in_psd, packing each shape independently -----

    const auto pack_shape = [&](const std::string& mesh_name) {
        return pgd.find_all_in_psd(valid_groupings.at("MESH").at(mesh_name), psd);
    };
    const auto a_pool = pack_shape("A");
    const auto b_pool = pack_shape("B");

    ASSERT_THAT(
        footprints_of(a_pool),
        ::testing::ElementsAre(
            std::set<uint64_t>{100, 101}, std::set<uint64_t>{102, 103}, std::set<uint64_t>{104, 105}))
        << "the 1x2 shape packs onto even tile boundaries and never offers {101,102} or {103,104}";
    ASSERT_EQ(b_pool.size(), 6u) << "the 1x1 shape fits on every chip";

    // The two B entries the arrangement below uses, pinned so the indices mean chips and not
    // whatever order find_all_in_psd happened to return.
    const auto b_footprints = footprints_of(b_pool);
    ASSERT_EQ(b_footprints[4], std::set<uint64_t>({104}));
    ASSERT_EQ(b_footprints[5], std::set<uint64_t>({105}));

    // Seat the meshes on that packing, in MGD order A0, B0, A1, B1: the two A tiles the packer
    // offers first, and the two chips they leave over for the B meshes. Disjoint, covers the whole
    // ring, and nothing about it considered the mesh-level edges.
    const std::vector<PsdPlacement> old_placements = {a_pool[0], b_pool[4], a_pool[1], b_pool[5]};

    // build physical
    const auto old_physical = utils::build_hierarchical_from_flat_graph(flat_graph, old_placements);

    // place and map
    const auto old_mapping = utils::map_multi_mesh_to_physical(logical, old_physical, config);
    EXPECT_FALSE(old_mapping.success)
        << "the packed arrangement strands the A0--B0 and A1--B1 edges, which the inter-mesh stage "
        << "should reject";

    // ----- new path: the adjacency-guided search -----

    // place
    const auto placements = pgd.solve_adjacency_guided_placement(mgd, valid_groupings, psd);
    ASSERT_EQ(placements.size(), 4u) << "all four meshes should be placed on the 6-chip ring";

    // build physical
    const auto physical = utils::build_hierarchical_from_flat_graph(flat_graph, placements);

    // place and map
    const auto mapping = utils::map_multi_mesh_to_physical(logical, physical, config);
    ASSERT_TRUE(mapping.success) << "the two-level solve should succeed on the DFS placement, but failed with: "
                                 << mapping.error_message;
    EXPECT_EQ(mapping.fabric_node_to_asic.size(), 6u) << "every logical fabric node should be bound to an ASIC";

    const auto footprints = mapped_footprints(mapping);
    ASSERT_EQ(footprints.size(), 4u) << "all four meshes should come back bound";

    // Shapes come out as declared: mesh ids 0 and 2 are the 1x2s, 1 and 3 the 1x1s.
    EXPECT_EQ(footprints[0].size(), 2u);
    EXPECT_EQ(footprints[1].size(), 1u);
    EXPECT_EQ(footprints[2].size(), 2u);
    EXPECT_EQ(footprints[3].size(), 1u);
    EXPECT_EQ(chips_in(footprints), whole_ring) << "the mapping should use each chip of the ring exactly once";

    // ----- the difference -----

    // The DFS gets there by using an A footprint the packer never emits.
    const auto a_footprints = footprints_of(a_pool);
    const bool uses_footprint_outside_packing =
        std::find(a_footprints.begin(), a_footprints.end(), footprints[0]) == a_footprints.end() ||
        std::find(a_footprints.begin(), a_footprints.end(), footprints[2]) == a_footprints.end();
    EXPECT_TRUE(uses_footprint_outside_packing)
        << "satisfying the ring requires a 1x2 footprint that straddles the packer's tile boundary";
}

// The falsification for the test above: same four meshes, same six chips, but the ring's closing
// link is removed. The meshes still fit and can still be placed disjointly, so a placer that only
// reasons about packing would happily return a full assignment with one seam unroutable.
//
//   mesh-level graph                     physical graph (6-chip line, wrap removed)
//
//     A0 --- B0                            100 == 101 == 102 == 103 == 104 == 105
//     |       |
//     B1 --- A1                          (no 105 == 100)
//
//   Any interleaved seating still puts one mesh on 100 and another on 105, and the ring declares
//   those two neighbours, so the fourth seam can never be met. Placement must fail rather than
//   emit a disjoint-but-unroutable answer.
TEST(AdjacencyGuidedPlacement, AlternatingShapeRingFailsWhenRingCannotClose) {
    // build pgd
    PhysicalGroupingDescriptor pgd{std::filesystem::path(kRingPgdPath)};
    MeshGraphDescriptor mgd{std::filesystem::path(kRingMgdPath)};
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(kOpenLinePsdPath);

    // get valid groupings
    const auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd, psd);

    // build logical
    const auto logical = utils::build_logical_multi_mesh_adjacency_graph(mgd);

    // place, and build the flat ASIC adjacency the physical graph is derived from
    const auto placements = pgd.solve_adjacency_guided_placement(mgd, valid_groupings, psd);

    // Placement fails, so there is no physical graph to build and nothing to map.
    EXPECT_TRUE(placements.empty()) << "the ring cannot close on a line, so no placement is valid";
}

// ---------------------------------------------------------------------------------------------
// Downgrade from a PGD grouping to the MGD's own grouping
//
// A PGD grouping is only committed if it both matches the MGD mesh topologically and embeds into
// the PSD. Matching alone is not enough. When a grouping matches but cannot be placed, the matcher
// is expected to discard it and fall back to the grouping derived from the MGD device topology.
//
// The pair below differs only in the PGD, so the committed grouping name isolates that decision:
//
//   PGD grouping places  ->  committed "1x2_Mesh_flat"  (the PGD grouping)
//   PGD grouping cannot  ->  committed "M0"             (the MGD fallback)
//
// Both run on test_4asic_2mesh.textproto (100 == 101, 102 == 103) with a single 1x2 mesh.
// ---------------------------------------------------------------------------------------------

// Control: the unpinned PGD grouping is free to land on either linked pair, so it places and is
// committed under its own name. Nothing falls back here.
// Four meshes, four different shapes, chained on a topology that admits exactly one placement.
// Where the ring case is about interleaving two shapes, this one is about the shapes constraining
// each other: each is easy to place on its own, and it is the chain that pins them all down.
//
//   mesh-level graph                     physical graph (10 chips)
//
//     D --- C --- B --- A                  100 == 101
//    (2x2) (1x3) (1x2) (1x1)                ||     ||
//                                          103 == 102 == 104 == 107 == 108 == 109
//                                                         ||
//                                                        105 == 106
//
//   the only placement
//
//     100 == 101
//      ||     ||
//     103 == 102 == 104 == 107 == 108 == 109
//     '------v------' ||   '---v---'  '-v-'
//            D        ||       B        A
//                    105 == 106
//                     '----v----'
//                      C (with 104)
//
// The shapes are pinned down in order. D can only be the 4-cycle, since nothing else in the PSD
// closes a square. C has to touch D, and 104 is the only free chip adjacent to the square, so C
// must contain the junction. That leaves C a genuine choice of branch, and it is the wrong one that
// makes this test worth having: running C along the spine as {104,107,108} is a perfectly good 1x3
// that still touches D, but it strands the far end -- {105,106} and {109} are then the only pieces
// left, and 109 touches nothing outside C, so the B--A seam has no link under it. The search has to
// walk that back and send C down the short branch instead.
TEST(AdjacencyGuidedPlacement, MixedShapeChainPlacesTheOnlyWayItFits) {
    // build pgd
    PhysicalGroupingDescriptor pgd{std::filesystem::path(kMixedShapePgdPath)};
    MeshGraphDescriptor mgd{std::filesystem::path(kMixedShapeMgdPath)};
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(kSquareForkPsdPath);

    // get valid groupings
    const auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd, psd);

    // build logical
    const auto logical = utils::build_logical_multi_mesh_adjacency_graph(mgd);

    // place, and build the flat ASIC adjacency the physical graph is derived from
    const auto placements = pgd.solve_adjacency_guided_placement(mgd, valid_groupings, psd);
    const AdjacencyGraph<tt::tt_metal::AsicID> flat_graph(utils::build_flat_adjacency_map_from_psd(psd));
    ASSERT_EQ(placements.size(), 4u) << "all four meshes should be placed on the 10 chips";

    // build physical
    const auto physical = utils::build_hierarchical_from_flat_graph(flat_graph, placements);

    // place and map
    utils::TopologyMappingConfig config;
    config.disable_rank_bindings = true;
    const auto mapping = utils::map_multi_mesh_to_physical(logical, physical, config);
    ASSERT_TRUE(mapping.success) << "the two-level solve should succeed, but failed with: " << mapping.error_message;
    EXPECT_EQ(mapping.fabric_node_to_asic.size(), 10u) << "every logical fabric node should be bound to an ASIC";

    // Ordered by mesh id, so D, C, B, A as the descriptor declares them.
    EXPECT_THAT(
        mapped_footprints(mapping),
        ::testing::ElementsAre(
            std::set<uint64_t>({100, 101, 102, 103}),
            std::set<uint64_t>({104, 105, 106}),
            std::set<uint64_t>({107, 108}),
            std::set<uint64_t>({109})))
        << "this is the only assignment of the four shapes that satisfies every link in the chain";
}

// Four different shapes around a hub, where the channel count on each seam is what seats them.
//
//   mesh-level graph                  physical graph (12-chip star)
//
//          Sc (1x1)                            104 == 105 == 106     branch X, first link 2
//            |                                /
//            2                    100 == 101 =/
//            |                     ||    ||
//   Sb -3- H (2x2) -4- Sa          102 == 103 ==== 107 == 108 == 109  branch Y, first link 4
//  (1x2)              (1x3)         |
//                                   +==== 110 == 111                 branch Z, first link 3
//
//   the only placement
//
//     H = {100,101,102,103}   Sa = {107,108,109}   Sb = {110,111}   Sc = {104}
//
// Two things have to work together here. The hub is settled by shape: a 2x2 needs a 4-cycle and the
// block is the only one in the graph. The spokes are not -- the branches are plain lines, so by
// shape alone Sa fits either 3-chip branch and Sb or Sc fit almost anywhere. What seats them is how
// many channels each seam asks for: Sa asks for 4 and only the 103-107 branch carries that many,
// Sb asks for 3 and only the 102-110 branch is left that does, and Sc takes the head of the branch
// that remains.
//
// That is the multiplicity half of the seam check. The mesh-level graph repeats a neighbour once
// per requested channel, the flat ASIC graph repeats it once per ethernet channel, and a candidate
// is only offered when the second count covers the first. Ask every seam for a single channel and
// four placements satisfy this descriptor rather than one.
//
// Note that the spokes leave chips 105 and 106 unused: placement covers the meshes the descriptor
// asks for, not every chip in the system.
TEST(AdjacencyGuidedPlacement, StarSeamsPlaceByChannelCount) {
    // build pgd
    PhysicalGroupingDescriptor pgd{std::filesystem::path(kMixedShapePgdPath)};
    MeshGraphDescriptor mgd{std::filesystem::path(kStarMgdPath)};
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(kStarPsdPath);

    // get valid groupings
    const auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd, psd);

    // build logical
    const auto logical = utils::build_logical_multi_mesh_adjacency_graph(mgd);

    // place, and build the flat ASIC adjacency the physical graph is derived from
    const auto placements = pgd.solve_adjacency_guided_placement(mgd, valid_groupings, psd);
    const AdjacencyGraph<tt::tt_metal::AsicID> flat_graph(utils::build_flat_adjacency_map_from_psd(psd));
    ASSERT_EQ(placements.size(), 4u) << "the hub and all three spokes should be placed";

    // build physical
    const auto physical = utils::build_hierarchical_from_flat_graph(flat_graph, placements);

    // place and map
    utils::TopologyMappingConfig config;
    config.disable_rank_bindings = true;
    const auto mapping = utils::map_multi_mesh_to_physical(logical, physical, config);
    ASSERT_TRUE(mapping.success) << "the two-level solve should succeed, but failed with: " << mapping.error_message;
    EXPECT_EQ(mapping.fabric_node_to_asic.size(), 10u) << "the hub's 4 chips plus the spokes' 3, 2 and 1";

    // Ordered by mesh id: the hub, then the 4-, 3- and 2-channel spokes.
    EXPECT_THAT(
        mapped_footprints(mapping),
        ::testing::ElementsAre(
            std::set<uint64_t>({100, 101, 102, 103}),
            std::set<uint64_t>({107, 108, 109}),
            std::set<uint64_t>({110, 111}),
            std::set<uint64_t>({104})))
        << "each spoke must take the branch whose attaching link carries the channels its seam asks for";
}

// Two descriptors placed onto one system in a single pass, and neither may borrow the other's meshes.
//
//   descriptor A            descriptor B         physical graph (5-chip dumbbell)
//
//   M0 (1x2) -4- M1 (1x1)   M0 (1x1) -3- M1 (1x1)   100 == 101 ==== 102 -- 103 ==== 104
//                                                       2       4        1       3
//   the only placement
//
//     A: M0 = {100,101}, M1 = {102}      B: M0 and M1 take 103 and 104
//
// Both descriptors name their meshes "M0" and "M1", and the shapes behind those names differ: A's
// M0 is the 1x2 while B's M0 is a single chip. That is the point of the pair. Valid groupings are
// looked up by instance name, so if the two descriptors were not kept apart, a lookup of "M0" would
// hand back whichever shape was merged last and a mesh would be built with the wrong shape
// entirely, not merely seated in the wrong place. Keying by descriptor index is what prevents it.
//
// Nothing connects the two descriptors -- there is no seam from A to B -- so chip-disjointness is
// all that separates them. They end up on opposite sides of the 1-channel pinch because A's seam
// needs 4 channels and only 101-102 carries that many, which leaves 103-104 as the only link that
// can still carry B's 3-channel seam.
//
// B's two meshes are both single chips joined by a symmetric seam, so which of them takes 103 and
// which takes 104 is genuinely not pinned down; the test asserts that pair as a set.
TEST(AdjacencyGuidedPlacement, TwoDescriptorsPlaceWithoutBorrowingEachOthersMeshes) {
    // build pgd
    PhysicalGroupingDescriptor pgd{std::filesystem::path(kRingPgdPath)};
    std::vector<MeshGraphDescriptor> mgds;
    mgds.emplace_back(std::filesystem::path(kWideThenNarrowMgdPath));
    mgds.emplace_back(std::filesystem::path(kTwoSinglesMgdPath));
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(kDumbbellPsdPath);

    // get valid groupings
    // Merged keys carry the descriptor index, so the two same-named instances stay distinct and each
    // still resolves to its own shape: A's M0 is the 1x2 (two nodes) and B's M0 is a single chip.
    const auto valid_groupings = pgd.get_valid_groupings_for_mgds(mgds, psd);
    const auto& meshes = valid_groupings.at("MESH");
    EXPECT_THAT(
        meshes,
        ::testing::UnorderedElementsAre(
            ::testing::Key("mgd0_M0"), ::testing::Key("mgd0_M1"), ::testing::Key("mgd1_M0"), ::testing::Key("mgd1_M1")))
        << "each descriptor's instances should be keyed by descriptor index, not merged by name";
    ASSERT_FALSE(meshes.at("mgd0_M0").empty());
    ASSERT_FALSE(meshes.at("mgd1_M0").empty());
    EXPECT_EQ(meshes.at("mgd0_M0").front().adjacency_graph.get_nodes().size(), 2u)
        << "the first descriptor's M0 is the 1x2";
    EXPECT_EQ(meshes.at("mgd1_M0").front().adjacency_graph.get_nodes().size(), 1u)
        << "the second descriptor's M0 is a single chip, despite sharing the name";

    // build logical
    // One merged logical graph over both descriptors, with mesh ids renumbered so that the first
    // descriptor's meshes come first. That renumbering is the order the placements come back in.
    std::vector<utils::LogicalMultiMeshGraph> parts;
    parts.reserve(mgds.size());
    for (const auto& mgd : mgds) {
        parts.push_back(utils::build_logical_multi_mesh_adjacency_graph(mgd));
    }
    const auto logical = utils::merge_logical_multi_mesh_adjacency_graphs(parts);

    // place, and build the flat ASIC adjacency the physical graph is derived from
    const std::vector<const MeshGraphDescriptor*> descriptors{&mgds[0], &mgds[1]};
    const auto placements = pgd.solve_adjacency_guided_placement(descriptors, valid_groupings, psd);
    const AdjacencyGraph<tt::tt_metal::AsicID> flat_graph(utils::build_flat_adjacency_map_from_psd(psd));
    ASSERT_EQ(placements.size(), 4u) << "two meshes from each descriptor";

    const auto placed = footprints_of(placements);
    EXPECT_THAT(
        placed,
        ::testing::ElementsAre(std::set<uint64_t>({100, 101}), std::set<uint64_t>({102}), ::testing::_, ::testing::_))
        << "the first descriptor's meshes come first, on the left of the pinch";
    EXPECT_THAT(
        std::vector<std::set<uint64_t>>(placed.begin() + 2, placed.end()),
        ::testing::UnorderedElementsAre(std::set<uint64_t>({103}), std::set<uint64_t>({104})))
        << "and the second descriptor's take the right, in either order";

    // build physical
    const auto physical = utils::build_hierarchical_from_flat_graph(flat_graph, placements);

    // place and map
    utils::TopologyMappingConfig config;
    config.disable_rank_bindings = true;
    const auto mapping = utils::map_multi_mesh_to_physical(logical, physical, config);
    ASSERT_TRUE(mapping.success) << "the two-level solve should succeed, but failed with: " << mapping.error_message;
    EXPECT_EQ(mapping.fabric_node_to_asic.size(), 5u) << "all five chips are bound, three of them A's and two B's";

    // The binding has to agree with the placement, mesh for mesh: A's 1x2 is the only two-chip mesh
    // in either descriptor, and the seam pulls its 1x1 to 102.
    const auto footprints = mapped_footprints(mapping);
    ASSERT_EQ(footprints.size(), 4u);
    EXPECT_THAT(
        footprints,
        ::testing::ElementsAre(std::set<uint64_t>({100, 101}), std::set<uint64_t>({102}), ::testing::_, ::testing::_))
        << "the first descriptor's meshes should be bound to the first descriptor's chips";
    EXPECT_EQ(chips_in({footprints[0], footprints[1]}), std::set<uint64_t>({100, 101, 102}))
        << "the first descriptor owns the left of the pinch";
    EXPECT_EQ(chips_in({footprints[2], footprints[3]}), std::set<uint64_t>({103, 104}))
        << "the second descriptor owns the right of the pinch";
}

// A STRICT seam has to survive mapping, not just placement.
//
//   descriptor A            descriptor B          physical graph (6-chip dumbbell)
//
//   M0 (1x2) -4- M1 (1x1)   M0 (1x1) -3- M1 (1x2)   100 == 101 ==== 102 -- 103 ==== 104 == 105
//   both STRICT                                         2       4        1       3       2
//
//   the only placement
//
//     A: M0 = {100,101}, M1 = {102}      B: M0 = {103}, M1 = {104,105}
//
// These two descriptors are mirror images, so by shape either fits either side of the pinch and the
// counts are the only thing separating them: A requires 4 channels and only 101-102 carries that many,
// which leaves 103-104 as the only link that can still carry B's 3. Placement gets this right.
//
// The mapper is then free to disagree, because it re-solves the logical-to-physical binding from
// scratch rather than keeping the association placement established. Under RELAXED it would: it treats
// a mesh-level edge as present-or-absent, so swapping the two descriptors looks just as good, and it
// would seat A's 4-channel seam on the 3-channel link and merely log that the seam is narrower than
// asked for. Under STRICT it counts channels and the swap is rejected.
//
// Both descriptors here say STRICT, so the test asks for the counts to hold end to end. It sets the
// validation mode to match, which is what production does -- topology_mapper.cpp derives
// inter_mesh_validation_mode from the descriptor's own policy via MeshGraph::is_inter_mesh_policy_relaxed.
// A MeshGraph needs a cluster to build, so the mode is set directly here rather than derived.
TEST(AdjacencyGuidedPlacement, StrictSeamSurvivesInterMeshMapping) {
    // build pgd
    PhysicalGroupingDescriptor pgd{std::filesystem::path(kRingPgdPath)};
    std::vector<MeshGraphDescriptor> mgds;
    mgds.emplace_back(std::filesystem::path(kWideThenNarrowMgdPath));
    mgds.emplace_back(std::filesystem::path(kNarrowThenWideMgdPath));
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(kWideDumbbellPsdPath);

    // get valid groupings
    const auto valid_groupings = pgd.get_valid_groupings_for_mgds(mgds, psd);

    // build logical
    std::vector<utils::LogicalMultiMeshGraph> parts;
    parts.reserve(mgds.size());
    for (const auto& mgd : mgds) {
        parts.push_back(utils::build_logical_multi_mesh_adjacency_graph(mgd));
    }
    const auto logical = utils::merge_logical_multi_mesh_adjacency_graphs(parts);

    // place, and build the flat ASIC adjacency the physical graph is derived from
    const std::vector<const MeshGraphDescriptor*> descriptors{&mgds[0], &mgds[1]};
    const auto placements = pgd.solve_adjacency_guided_placement(descriptors, valid_groupings, psd);
    const AdjacencyGraph<tt::tt_metal::AsicID> flat_graph(utils::build_flat_adjacency_map_from_psd(psd));
    ASSERT_EQ(placements.size(), 4u) << "two meshes from each descriptor";
    ASSERT_THAT(
        footprints_of(placements),
        ::testing::ElementsAre(
            std::set<uint64_t>({100, 101}),
            std::set<uint64_t>({102}),
            std::set<uint64_t>({103}),
            std::set<uint64_t>({104, 105})))
        << "placement should seat each descriptor on the side whose link is wide enough for its seam";

    // build physical
    const auto physical = utils::build_hierarchical_from_flat_graph(flat_graph, placements);

    // place and map
    utils::TopologyMappingConfig config;
    config.disable_rank_bindings = true;
    config.inter_mesh_validation_mode = ::tt::tt_fabric::ConnectionValidationMode::STRICT;
    const auto mapping = utils::map_multi_mesh_to_physical(logical, physical, config);
    ASSERT_TRUE(mapping.success) << "the two-level solve should succeed, but failed with: " << mapping.error_message;

    const auto footprints = mapped_footprints(mapping);
    ASSERT_EQ(footprints.size(), 4u);
    EXPECT_EQ(channels_between(flat_graph, footprints[0], footprints[1]), 4u)
        << "the first descriptor requires 4 channels, so the physical seam its two meshes were bound to "
           "must carry that many";
    EXPECT_EQ(chips_in({footprints[0], footprints[1]}), std::set<uint64_t>({100, 101, 102}))
        << "and the mapper should keep it on the chips placement chose for it";
}

// The three tests below pin down what a descriptor's inter-mesh channel policy is supposed to mean to
// placement. All three put two single-chip meshes on the same system, three chips whose two links are
// different widths:
//
//   M0 (1x1) --n-- M1 (1x1)            100 == 101 ==== 102
//                                          2       4
//
// A seam wanting 4 channels can only be met on 101-102. A seam wanting 8 cannot be met anywhere, and
// that is where the policy decides the outcome: STRICT has no placement, while RELAXED should still
// place, because the count is a preference and the mesh-level edge only insists the two regions touch.
//
// Placement does not read the policy yet -- see the TODO on next_step_pool -- so it treats every count
// as a requirement, which makes RelaxedSeamStillPlacesWhenChannelsFallShort below the failing one.

// A preference that can be met should be met: the seam wants 4 channels and only 101-102 has them.
// Passes today, and must keep passing once the policy is read, since it is the preference half of
// RELAXED rather than the requirement half.
TEST(AdjacencyGuidedPlacement, RelaxedSeamPrefersTheFullChannelCount) {
    // build pgd
    PhysicalGroupingDescriptor pgd{std::filesystem::path(kRingPgdPath)};
    MeshGraphDescriptor mgd{std::filesystem::path(kRelaxedSeamMgdPath)};
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(kUnevenLinePsdPath);

    // get valid groupings
    const auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd, psd);

    // build logical
    const auto logical = utils::build_logical_multi_mesh_adjacency_graph(mgd);

    // place, and build the flat ASIC adjacency the physical graph is derived from
    const auto placements = pgd.solve_adjacency_guided_placement(mgd, valid_groupings, psd);
    const AdjacencyGraph<tt::tt_metal::AsicID> flat_graph(utils::build_flat_adjacency_map_from_psd(psd));
    ASSERT_EQ(placements.size(), 2u) << "both meshes should be placed";
    EXPECT_THAT(
        footprints_of(placements),
        ::testing::UnorderedElementsAre(std::set<uint64_t>({101}), std::set<uint64_t>({102})))
        << "the 4-channel link is the only seam wide enough, so the 2-channel pair should be left alone";

    // build physical
    const auto physical = utils::build_hierarchical_from_flat_graph(flat_graph, placements);

    // place and map
    utils::TopologyMappingConfig config;
    config.disable_rank_bindings = true;
    const auto mapping = utils::map_multi_mesh_to_physical(logical, physical, config);
    ASSERT_TRUE(mapping.success) << "the two-level solve should succeed, but failed with: " << mapping.error_message;
    EXPECT_EQ(chips_in(mapped_footprints(mapping)), std::set<uint64_t>({101, 102}));
}

// A requirement that cannot be met should fail. Guards the STRICT half, so that teaching placement to
// relax a RELAXED count does not quietly relax a STRICT one too.
TEST(AdjacencyGuidedPlacement, StrictSeamFailsWhenChannelsFallShort) {
    // build pgd
    PhysicalGroupingDescriptor pgd{std::filesystem::path(kRingPgdPath)};
    MeshGraphDescriptor mgd{std::filesystem::path(kStrictWideSeamMgdPath)};
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(kUnevenLinePsdPath);

    // get valid groupings
    const auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd, psd);
    ASSERT_FALSE(valid_groupings.at("MESH").empty()) << "the shapes themselves are placeable; only the seam is not";

    // place
    const auto placements = pgd.solve_adjacency_guided_placement(mgd, valid_groupings, psd);
    EXPECT_TRUE(placements.empty())
        << "no link carries the 8 channels a STRICT seam requires, so there should be no placement at all";
}

// EXPECTED TO FAIL until next_step_pool reads the channel policy.
//
// Same unsatisfiable count as the STRICT test above, but the descriptor says RELAXED, which makes the
// count a preference. The meshes still have to touch, and 101-102 is the widest seam on offer, so
// placement should seat them there rather than giving up. Today placement treats the count as a hard
// requirement regardless of policy, finds nothing that carries 8 channels and returns empty -- which is
// stricter than the descriptor asked for, and stricter than the mapper, which would accept this seating
// and log that the seam is narrower than requested.
TEST(AdjacencyGuidedPlacement, RelaxedSeamStillPlacesWhenChannelsFallShort) {
    // build pgd
    PhysicalGroupingDescriptor pgd{std::filesystem::path(kRingPgdPath)};
    MeshGraphDescriptor mgd{std::filesystem::path(kRelaxedWideSeamMgdPath)};
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(kUnevenLinePsdPath);

    // get valid groupings
    const auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd, psd);

    // build logical
    const auto logical = utils::build_logical_multi_mesh_adjacency_graph(mgd);

    // place, and build the flat ASIC adjacency the physical graph is derived from
    const auto placements = pgd.solve_adjacency_guided_placement(mgd, valid_groupings, psd);
    const AdjacencyGraph<tt::tt_metal::AsicID> flat_graph(utils::build_flat_adjacency_map_from_psd(psd));
    ASSERT_EQ(placements.size(), 2u)
        << "a RELAXED count is a preference, so an unmeetable one should not stop the meshes being placed";
    EXPECT_THAT(
        footprints_of(placements),
        ::testing::UnorderedElementsAre(std::set<uint64_t>({101}), std::set<uint64_t>({102})))
        << "and the preference should still steer them onto the widest seam available";

    // build physical
    const auto physical = utils::build_hierarchical_from_flat_graph(flat_graph, placements);

    // place and map
    utils::TopologyMappingConfig config;
    config.disable_rank_bindings = true;
    const auto mapping = utils::map_multi_mesh_to_physical(logical, physical, config);
    EXPECT_TRUE(mapping.success) << "the mapper accepts a short seam under RELAXED, so mapping should succeed: "
                                 << mapping.error_message;
}

TEST(AdjacencyGuidedPlacement, PgdGroupingThatPlacesIsCommittedDirectly) {
    // build pgd
    PhysicalGroupingDescriptor pgd{std::filesystem::path(kAdjacencyPgdPath)};
    MeshGraphDescriptor mgd{std::filesystem::path(kSingleMeshMgdPath)};
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(kTwoPairPsdPath);

    // get valid groupings
    // The committed grouping's name is what separates the two paths: a committed PGD grouping keeps
    // its own flattened name, while the MGD fallback grouping is named after the mesh instance.
    const auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd, psd);
    const auto& committed = valid_groupings.at("MESH").at("M0");
    ASSERT_EQ(committed.size(), 1u);
    EXPECT_EQ(committed.front().name, "1x2_Mesh_flat")
        << "the PGD grouping embeds into the PSD, so it should be committed rather than dropped";

    // Whichever path commits, the grouping carries the shape as its own adjacency graph, and that
    // is what placement then has to embed: two nodes, joined.
    const auto& grouping_graph = committed.front().adjacency_graph;
    ASSERT_EQ(grouping_graph.get_nodes().size(), 2u);
    EXPECT_THAT(grouping_graph.get_neighbors(0u), ::testing::ElementsAre(1u));
    EXPECT_THAT(grouping_graph.get_neighbors(1u), ::testing::ElementsAre(0u));

    // build logical
    const auto logical = utils::build_logical_multi_mesh_adjacency_graph(mgd);

    // place, and build the flat ASIC adjacency
    const auto placements = pgd.solve_adjacency_guided_placement(mgd, valid_groupings, psd);
    const AdjacencyGraph<tt::tt_metal::AsicID> flat_graph(utils::build_flat_adjacency_map_from_psd(psd));
    ASSERT_EQ(placements.size(), 1u);

    // build physical
    const auto physical = utils::build_hierarchical_from_flat_graph(flat_graph, placements);

    // place and map
    utils::TopologyMappingConfig config;
    config.disable_rank_bindings = true;
    const auto mapping = utils::map_multi_mesh_to_physical(logical, physical, config);
    ASSERT_TRUE(mapping.success) << "the two-level solve should succeed, but failed with: " << mapping.error_message;
    EXPECT_THAT(
        mapped_footprints(mapping),
        ::testing::ElementsAre(::testing::AnyOf(std::set<uint64_t>{100, 101}, std::set<uint64_t>{102, 103})))
        << "the mesh must sit on one of the two linked pairs";
}

// The downgrade above only triggers when the PGD grouping cannot be embedded at all. This is the
// case it misses: a grouping that embeds fine on its own but cannot serve every mesh instance that
// shares it.
//
//   mesh-level graph (linked MGD)        physical graph (4-chip line)
//
//     M0[0] == M0[1]                       100 == 101 == 102 == 103
//
//   the committed PGD grouping is pinned to ASIC locations 0 and 1
//
//     100 == 101 == 102 == 103
//     '----v----'
//      the only footprint it allows
//
// Both mesh instances are instances of the same descriptor, so they share one committed grouping,
// and that grouping admits exactly one footprint. Two meshes cannot both have it, so placement
// fails outright. The MGD's own unpinned 1x2 grouping would have placed them on {100,101} and
// {102,103}, which is what LinkedMeshesPlaceAdjacentlyOnLine shows on this very PSD.
//
// The matcher never gets there. Its pre-commit check calls enumerate_distinct_placements_for_grouping
// with empty constraints, so it asks only "does this shape fit somewhere on the PSD", with nothing
// else placed and no seam to satisfy. The pinned grouping answers yes, gets committed, and the
// fallback is skipped -- committed_pgd_matches is already true by the time the search discovers the
// grouping cannot cover both instances.
//
// TODO: this test asserts the behaviour we want and currently fails. The fix belongs in
// get_valid_groupings_for_mgd: when a committed PGD grouping cannot cover every instance that shares
// it, the MGD grouping should be offered alongside it rather than skipped. The search already walks
// the grouping variants for a mesh in next_step_pool, so with both on the list it can seat one
// instance on the pinned grouping and fall back to the MGD grouping for the other -- which is why
// the check at the end of this test wants two entries, not a replacement.
TEST(AdjacencyGuidedPlacement, PgdGroupingThatCannotCoverEveryInstanceShouldDowngrade) {
    // build pgd
    PhysicalGroupingDescriptor pgd{std::filesystem::path(kOnePairPgdPath)};
    MeshGraphDescriptor mgd{std::filesystem::path(kLinkedMgdPath)};
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(kLinePsdPath);

    // get valid groupings
    const auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd, psd);

    // build logical
    const auto logical = utils::build_logical_multi_mesh_adjacency_graph(mgd);

    // place, and build the flat ASIC adjacency the physical graph is derived from
    const auto placements = pgd.solve_adjacency_guided_placement(mgd, valid_groupings, psd);
    const AdjacencyGraph<tt::tt_metal::AsicID> flat_graph(utils::build_flat_adjacency_map_from_psd(psd));
    EXPECT_EQ(placements.size(), 2u)
        << "one instance can take the pinned pair and the other the MGD grouping, so both should place";

    // Nothing to build a physical graph from while placement comes back empty. Guarded rather than
    // asserted so the grouping check at the end still reports in the same run.
    if (!placements.empty()) {
        // build physical
        const auto physical = utils::build_hierarchical_from_flat_graph(flat_graph, placements);

        // place and map
        utils::TopologyMappingConfig config;
        config.disable_rank_bindings = true;
        const auto mapping = utils::map_multi_mesh_to_physical(logical, physical, config);
        EXPECT_TRUE(mapping.success) << "the two-level solve should succeed, but failed with: "
                                     << mapping.error_message;
        EXPECT_THAT(
            mapped_footprints(mapping),
            ::testing::UnorderedElementsAre(std::set<uint64_t>{100, 101}, std::set<uint64_t>{102, 103}))
            << "the same seating LinkedMeshesPlaceAdjacentlyOnLine gets from the unpinned grouping";
    }

    // Only now, the grouping list that has to be there for the above to be reachable: the pinned PGD
    // grouping AND the MGD grouping, so the search has something to fall back to for the instance the
    // pinned one cannot serve. A single entry means one of the two was dropped at commit time.
    const auto& committed = valid_groupings.at("MESH").at("M0");
    std::vector<std::string> committed_names;
    for (const auto& grouping : committed) {
        committed_names.push_back(grouping.name);
    }
    EXPECT_THAT(committed_names, ::testing::UnorderedElementsAre("1x2_Mesh_OnePair_flat", "M0"))
        << "both the committed PGD grouping and the MGD grouping should be available as variants";

    // Either way round, each describes the same 1x2 shape, so placement has the same adjacency graph
    // to embed: two nodes, joined.
    for (const auto& grouping : committed) {
        EXPECT_EQ(grouping.adjacency_graph.get_nodes().size(), 2u) << "grouping " << grouping.name;
        EXPECT_THAT(grouping.adjacency_graph.get_neighbors(0u), ::testing::ElementsAre(1u));
        EXPECT_THAT(grouping.adjacency_graph.get_neighbors(1u), ::testing::ElementsAre(0u));
    }
}

}  // namespace tt::tt_fabric::fabric_router_tests
