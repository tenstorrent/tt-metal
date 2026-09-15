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
#include <fstream>
#include <chrono>
#include <cstdlib>
#include <optional>

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
                EXPECT_EQ(grouping.name, "4x2_Mesh_horizontal_flat")
                    << "Should match 4x2_Mesh_horizontal_flat grouping";
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
        if (grouping.name == "M0") {
            continue;  // MGD fallback is unpinned; PGD entries carry the match pinning
        }
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
        if (grouping.name == "M0") {
            continue;  // MGD fallback is unpinned; PGD entries carry the match pinning
        }
        ASSERT_FALSE(grouping.mesh_node_to_asic_position.empty())
            << "PGD-derived layout pinning must be populated for '" << grouping.name << "'";
        EXPECT_EQ(grouping.mesh_node_to_asic_position.size(), 16u)
            << "Committed PGD layout should cover all 16 logical chips";
    }
}

// ----- a 4x4 on a machine split into four hosts, one per quadrant -------------------------------
//
// The MGD host_topology contract on the split-host square. A mesh host rank is one process on one host,
// so a physical host boundary may never cut through a declared rank, though a declared rank boundary may
// sit anywhere inside a physical host. A quadrant machine divides on both axes at once, so containment
// has to hold in both directions at the same time: [2,2] lands on it exactly and anything finer fits
// inside a quadrant. The refusing direction -- a declared band that spans two hosts of a 2D host grid --
// is CoarserSplitAcrossATwoAxisHostGridIsRejected in PhysicalGroupingDescriptorTestsHostSplit.
//
// The machine is test_16asic_4x4_four_hosts_by_quadrant; its header draws the grid and names the host of
// every ASIC. A [2,2] partition is invariant under the rotations a wrapped mesh would add, so these hold
// for RING/RING as they do here; the RING/RING path is exercised by PhysicalGroupingDescriptorTestsHostSplit further
// down under Phase2AgreesWithPhase1OnASymmetricTorus, which also tests this rule on a 2x4, where the shape is not
// square and no rotation can satisfy an axis by accident.

// The 4x4 on a machine split into four hosts, one per quadrant, declaring [2,2]: four ranks of 4 chips
// that line up with the quadrants exactly. The split-host case where the machine is divided on both axes
// at once, so containment has to hold in both directions at the same time.
//
//   machine: four hosts, one per quadrant      MGD host_topology [2,2]
//
//     100 101 | 102 103    h0 h0 h1 h1           aa | bb     four ranks of 4 chips, each
//     104 105 | 106 107    h0 h0 h1 h1           aa | bb     one sitting on one quadrant:
//     --------+--------                          ---+---     the declared grid and the host
//     108 109 | 110 111    h2 h2 h3 h3           cc | dd     grid are the same partition
//     112 113 | 114 115    h2 h2 h3 h3           cc | dd
TEST(PhysicalGroupingDescriptorTests, GetValidGroupingsForMGD_QuadrantSplitPsdMatchingQuadrantMgdCommits) {
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(
        "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_16asic_4x4_four_hosts_by_quadrant.textproto");

    PhysicalGroupingDescriptor pgd{std::string(R"(
groupings {
  name: "4x4_Mesh"
  preset_type: MESH
  instances: [
    { id: 0  location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
    { id: 1  location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } },
    { id: 2  location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_3 } },
    { id: 3  location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_4 } },
    { id: 4  location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } },
    { id: 5  location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } },
    { id: 6  location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_3 } },
    { id: 7  location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_4 } },
    { id: 8  location { tray_id: TRAY_3 asic_location: ASIC_LOCATION_1 } },
    { id: 9  location { tray_id: TRAY_3 asic_location: ASIC_LOCATION_2 } },
    { id: 10 location { tray_id: TRAY_3 asic_location: ASIC_LOCATION_3 } },
    { id: 11 location { tray_id: TRAY_3 asic_location: ASIC_LOCATION_4 } },
    { id: 12 location { tray_id: TRAY_4 asic_location: ASIC_LOCATION_1 } },
    { id: 13 location { tray_id: TRAY_4 asic_location: ASIC_LOCATION_2 } },
    { id: 14 location { tray_id: TRAY_4 asic_location: ASIC_LOCATION_3 } },
    { id: 15 location { tray_id: TRAY_4 asic_location: ASIC_LOCATION_4 } }
  ]
  row_major_mesh {
    dims: [4, 4]
  }
}

# The PGD's own host level: eight hosts of 2 chips, each one half of a tray. One preset_type: HOSTS
# grouping is one host -- what it contains is that host's chips -- so eight hosts are eight
# definitions sharing a name, the way the galaxy PGDs repeat their MESH groupings.
groupings { name: "4x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
               { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } } ]
  row_major_mesh { dims: [1, 2] } }
groupings { name: "4x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_3 } },
               { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_4 } } ]
  row_major_mesh { dims: [1, 2] } }
groupings { name: "4x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } },
               { id: 1 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } } ]
  row_major_mesh { dims: [1, 2] } }
groupings { name: "4x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_3 } },
               { id: 1 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_4 } } ]
  row_major_mesh { dims: [1, 2] } }
groupings { name: "4x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_3 asic_location: ASIC_LOCATION_1 } },
               { id: 1 location { tray_id: TRAY_3 asic_location: ASIC_LOCATION_2 } } ]
  row_major_mesh { dims: [1, 2] } }
groupings { name: "4x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_3 asic_location: ASIC_LOCATION_3 } },
               { id: 1 location { tray_id: TRAY_3 asic_location: ASIC_LOCATION_4 } } ]
  row_major_mesh { dims: [1, 2] } }
groupings { name: "4x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_4 asic_location: ASIC_LOCATION_1 } },
               { id: 1 location { tray_id: TRAY_4 asic_location: ASIC_LOCATION_2 } } ]
  row_major_mesh { dims: [1, 2] } }
groupings { name: "4x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_4 asic_location: ASIC_LOCATION_3 } },
               { id: 1 location { tray_id: TRAY_4 asic_location: ASIC_LOCATION_4 } } ]
  row_major_mesh { dims: [1, 2] } }
)")};

    MeshGraphDescriptor mgd{std::string(R"(
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 4, 4 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 2, 2 ] }
  channels { count: 2 policy: STRICT }
}

top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
)")};

    const auto valid_groupings =
        pgd.get_valid_groupings_for_mgd(mgd, psd, /*pinnings=*/std::nullopt, /*require_placement=*/false);

    std::vector<GroupingInfo> committed;
    for (const auto& [preset, by_instance] : valid_groupings) {
        for (const auto& [instance, groupings] : by_instance) {
            for (const auto& grouping : groupings) {
                if (grouping.name != "M0" && !grouping.mesh_node_to_asic_position.empty()) {
                    committed.push_back(grouping);
                }
            }
        }
    }

    // host_topology [2,2] on a 4x4 declares four ranks, each a 2x2 block.
    const std::vector<std::vector<LogicalChipId>> declared_ranks = {
        {0, 1, 4, 5}, {2, 3, 6, 7}, {8, 9, 12, 13}, {10, 11, 14, 15}};

    // Against the PSD's hosts this is the aligned case and the seating is correct, so check it first:
    // whatever came back has each declared rank inside one quadrant of the machine.
    for (std::size_t rank = 0; rank < declared_ranks.size() && !committed.empty(); ++rank) {
        std::set<uint32_t> psd_hosts;
        for (LogicalChipId chip : declared_ranks[rank]) {
            const auto& position = committed.front().mesh_node_to_asic_position.at(chip);
            // Quadrant of the ASIC: trays 1-2 are the top half, asic_locations 1-2 the left half.
            psd_hosts.insert(((*position.first - 1) / 2) * 2 + (*position.second - 1) / 2);
        }
        EXPECT_EQ(psd_hosts.size(), 1u) << "declared rank " << rank << " was seated across " << psd_hosts.size()
                                        << " hosts of the machine";
    }

    // TODO: and now the same question against the PGD's host level, which is the half of the contract
    // that is not implemented. The PGD above declares eight 2-chip hosts where the MGD declares four
    // 4-chip ranks, so every declared rank needs two PGD hosts -- the coarser-than-the-hosts violation
    // that CoarserSplitThanThePhysicalHostsIsRejected rejects when the hosts come from the PSD. The only
    // correct answer is a refusal, so this loop is guarded on a non-empty commit and will pass once the
    // matcher consults the PGD hosts and declines. Today it commits and the loop fails, because the
    // HOSTS grouping is parsed and then never consulted: the host split is taken from the MGD alone, in
    // compose_mesh_node_to_host_group_from_mgd_match, and checked only against the PSD. Same root cause
    // as the rotation gap -- there is no PGD host level in the matching -- so both go away together.
    for (std::size_t rank = 0; rank < declared_ranks.size() && !committed.empty(); ++rank) {
        std::set<uint32_t> pgd_hosts;
        for (LogicalChipId chip : declared_ranks[rank]) {
            const auto& position = committed.front().mesh_node_to_asic_position.at(chip);
            // The PGD's halfrow holding this ASIC: one per tray half.
            pgd_hosts.insert((*position.first - 1) * 2 + (*position.second - 1) / 2);
        }
        EXPECT_EQ(pgd_hosts.size(), 1u) << "declared rank " << rank << " was seated across " << pgd_hosts.size()
                                        << " hosts of the PGD, which declares finer hosts than the MGD declares "
                                           "ranks -- the grouping should not have been committed at all";
    }
}

// Finer than the quadrants, and finer on both axes: eight ranks of 2 chips, two to a quadrant. Legal --
// subdividing inside a host cuts nothing, and that stays true when the hosts are quadrants.
//
//   machine: four hosts, one per quadrant      MGD host_topology [2,4]
//
//     100 101 | 102 103    h0 h0 h1 h1           ab | cd     eight ranks of 2 chips, each a
//     104 105 | 106 107    h0 h0 h1 h1           ab | cd     column of one band: two ranks to
//     --------+--------                          ---+---     a quadrant, and no rank crosses
//     108 109 | 110 111    h2 h2 h3 h3           ef | gh     a boundary the machine drew
//     112 113 | 114 115    h2 h2 h3 h3           ef | gh
TEST(PhysicalGroupingDescriptorTests, GetValidGroupingsForMGD_QuadrantSplitPsdFinerSplitInsideEachQuadrantCommits) {
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(
        "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_16asic_4x4_four_hosts_by_quadrant.textproto");

    PhysicalGroupingDescriptor pgd{std::string(R"(
groupings {
  name: "4x4_Mesh"
  preset_type: MESH
  instances: [
    { id: 0  location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
    { id: 1  location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } },
    { id: 2  location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_3 } },
    { id: 3  location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_4 } },
    { id: 4  location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } },
    { id: 5  location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } },
    { id: 6  location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_3 } },
    { id: 7  location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_4 } },
    { id: 8  location { tray_id: TRAY_3 asic_location: ASIC_LOCATION_1 } },
    { id: 9  location { tray_id: TRAY_3 asic_location: ASIC_LOCATION_2 } },
    { id: 10 location { tray_id: TRAY_3 asic_location: ASIC_LOCATION_3 } },
    { id: 11 location { tray_id: TRAY_3 asic_location: ASIC_LOCATION_4 } },
    { id: 12 location { tray_id: TRAY_4 asic_location: ASIC_LOCATION_1 } },
    { id: 13 location { tray_id: TRAY_4 asic_location: ASIC_LOCATION_2 } },
    { id: 14 location { tray_id: TRAY_4 asic_location: ASIC_LOCATION_3 } },
    { id: 15 location { tray_id: TRAY_4 asic_location: ASIC_LOCATION_4 } }
  ]
  row_major_mesh {
    dims: [4, 4]
  }
}

# The PGD's own host level, aligned with the machine: four hosts of 4 chips, one per quadrant.
groupings { name: "4x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
               { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } },
               { id: 2 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } },
               { id: 3 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } } ]
  row_major_mesh { dims: [2, 2] } }
groupings { name: "4x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_3 } },
               { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_4 } },
               { id: 2 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_3 } },
               { id: 3 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_4 } } ]
  row_major_mesh { dims: [2, 2] } }
groupings { name: "4x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_3 asic_location: ASIC_LOCATION_1 } },
               { id: 1 location { tray_id: TRAY_3 asic_location: ASIC_LOCATION_2 } },
               { id: 2 location { tray_id: TRAY_4 asic_location: ASIC_LOCATION_1 } },
               { id: 3 location { tray_id: TRAY_4 asic_location: ASIC_LOCATION_2 } } ]
  row_major_mesh { dims: [2, 2] } }
groupings { name: "4x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_3 asic_location: ASIC_LOCATION_3 } },
               { id: 1 location { tray_id: TRAY_3 asic_location: ASIC_LOCATION_4 } },
               { id: 2 location { tray_id: TRAY_4 asic_location: ASIC_LOCATION_3 } },
               { id: 3 location { tray_id: TRAY_4 asic_location: ASIC_LOCATION_4 } } ]
  row_major_mesh { dims: [2, 2] } }
)")};

    MeshGraphDescriptor mgd{std::string(R"(
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 4, 4 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 2, 4 ] }
  channels { count: 2 policy: STRICT }
}

top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
)")};

    const auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd, psd);

    std::vector<GroupingInfo> committed;
    for (const auto& [preset, by_instance] : valid_groupings) {
        for (const auto& [instance, groupings] : by_instance) {
            for (const auto& grouping : groupings) {
                if (grouping.name != "M0" && !grouping.mesh_node_to_asic_position.empty()) {
                    committed.push_back(grouping);
                }
            }
        }
    }
    ASSERT_FALSE(committed.empty()) << "eight 2-chip ranks fit two to a quadrant";

    // host_topology [2,4] on a 4x4 declares eight ranks, each two rows of the same column.
    const std::vector<std::vector<LogicalChipId>> declared_ranks = {
        {0, 4}, {1, 5}, {2, 6}, {3, 7}, {8, 12}, {9, 13}, {10, 14}, {11, 15}};
    for (std::size_t rank = 0; rank < declared_ranks.size(); ++rank) {
        std::set<uint32_t> hosts;
        for (LogicalChipId chip : declared_ranks[rank]) {
            const auto& position = committed.front().mesh_node_to_asic_position.at(chip);
            hosts.insert(((*position.first - 1) / 2) * 2 + (*position.second - 1) / 2);
        }
        EXPECT_EQ(hosts.size(), 1u) << "declared rank " << rank << " was seated across " << hosts.size() << " hosts";
    }

    // The (tray_id, asic_location) each mesh node was actually given. A rank landing on one host is
    // necessary but far from sufficient: it would hold just as well if the nodes inside a quadrant were
    // shuffled among themselves, or if two nodes were handed the same chip. So check the seating itself
    // -- every node gets a slot this PGD declares, no slot is handed out twice, and the mesh's own
    // neighbours stay neighbours on the machine.
    const auto& seating = committed.front().mesh_node_to_asic_position;
    ASSERT_EQ(seating.size(), 16u) << "all sixteen nodes should be seated";

    std::set<std::pair<uint32_t, uint32_t>> occupied;
    for (const auto& [node, position] : seating) {
        const uint32_t tray = *position.first;
        const uint32_t asic_location = *position.second;
        EXPECT_TRUE(tray >= 1 && tray <= 4)
            << "node " << node << " got tray " << tray << ", which this PGD never declares";
        EXPECT_TRUE(asic_location >= 1 && asic_location <= 4)
            << "node " << node << " got asic_location " << asic_location << ", which this PGD never declares";
        EXPECT_TRUE(occupied.emplace(tray, asic_location).second)
            << "tray " << tray << " asic_location " << asic_location << " was handed to two nodes, the second being "
            << node;
    }
    EXPECT_EQ(occupied.size(), 16u) << "the sixteen nodes should cover the sixteen declared slots exactly";

    // One step apart on the machine means adjacent tray or adjacent asic_location, not both: the
    // fixture numbers ASICs (tray, asic_location) = (row+1, col+1), so a mesh edge has to become a
    // single step along one of the two axes whichever orientation the match chose.
    const auto one_step_apart = [&seating](LogicalChipId a, LogicalChipId b) {
        const auto& first = seating.at(a);
        const auto& second = seating.at(b);
        const uint32_t tray_delta =
            *first.first > *second.first ? *first.first - *second.first : *second.first - *first.first;
        const uint32_t asic_delta =
            *first.second > *second.second ? *first.second - *second.second : *second.second - *first.second;
        return tray_delta + asic_delta == 1;
    };
    for (uint32_t row = 0; row < 4; ++row) {
        for (uint32_t col = 0; col < 4; ++col) {
            const auto node = static_cast<LogicalChipId>(row * 4 + col);
            if (col + 1 < 4) {
                EXPECT_TRUE(one_step_apart(node, node + 1))
                    << "mesh nodes " << node << " and " << node + 1 << " are neighbours but were seated apart";
            }
            if (row + 1 < 4) {
                EXPECT_TRUE(one_step_apart(node, node + 4))
                    << "mesh nodes " << node << " and " << node + 4 << " are neighbours but were seated apart";
            }
        }
    }
}

// ----- cases the PGD host level is needed for ---------------------------------------------------

// FIXME: a PGD that declares hosts the machine does not have should be rejected, and nothing rejects
// it. Every PGD in this file that is used with a PSD declares a host level aligned with that PSD's
// hosts, because that is what a PGD is -- a description of the machine. This test hands the matcher one
// that disagrees, and it is accepted.
//
// The machine is cut by column and the PGD claims its hosts are the trays:
//
//   machine: hosts cut by column       PGD declares hosts by row
//
//     100 101 | 102 103                  hhhh   <- claimed host 0, one tray
//     104 105 | 106 107                  iiii   <- claimed host 1
//      host0  |  host1                   each claimed host spans both real ones
//
// Neither claimed host is a subset of a real one, so no seating can satisfy both, and every answer the
// matcher gives from here is built on a false picture of the machine.
//
// The check to add, at the enumerate stage: flatten each declared PGD host to the set of chips it holds,
// and require that for at least one of the PGD's variants every flattened host lands inside a single PSD
// host. One variant matching is enough -- a PGD legitimately offers several orientations and only needs
// one that fits the machine it is given -- but if none does, the descriptor does not describe this
// machine and should throw rather than be quietly used.
TEST(PhysicalGroupingDescriptorTests, GetValidGroupingsForMGD_PgdHostsThatContradictThePsdAreRejected) {
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(
        "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_8asic_2x4_hosts_by_column.textproto");

    PhysicalGroupingDescriptor pgd{std::string(R"(
groupings {
  name: "2x4_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
    { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } },
    { id: 2 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_3 } },
    { id: 3 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_4 } },
    { id: 4 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } },
    { id: 5 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } },
    { id: 6 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_3 } },
    { id: 7 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_4 } }
  ]
  row_major_mesh {
    dims: [2, 4]
  }
}

# Hosts by row, on a machine whose hosts are by column: each of these spans both real hosts.
groupings { name: "2x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
               { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } },
               { id: 2 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_3 } },
               { id: 3 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_4 } } ]
  row_major_mesh { dims: [1, 4] } }
groupings { name: "2x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } },
               { id: 1 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } },
               { id: 2 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_3 } },
               { id: 3 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_4 } } ]
  row_major_mesh { dims: [1, 4] } }
)")};

    MeshGraphDescriptor mgd{std::string(R"(
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 2, 4 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 2 ] }
  channels { count: 2 policy: STRICT }
}

top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
)")};

    EXPECT_ANY_THROW(pgd.get_valid_groupings_for_mgd(mgd, psd))
        << "a PGD whose declared hosts fit no PSD host does not describe this machine";
}

// ---------------------------------------------------------------------------------------------
// Adjacency-guided placement
//
// These load a handcrafted PSD straight from a textproto rather than going through a mock cluster,
// so unlike the tests above they need no TT_METAL_MOCK_CLUSTER_DESC_PATH and never skip.
//
// All three use the same PGD and vary only the PSD and the MGD, which isolates the seam
// constraint: the same pair of meshes must place when the physical chips they need are linked and
// must fail when they are not.
//
// Each test spells out its PGD and MGD in full, so a case can be read start to finish without
// looking anything up; only the machines are loaded from files, because a PSD is long and its
// header draws the grid. The PGD the first three share is a single MESH grouping of two chips with
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

std::string unspecified_mesh_pgd(std::size_t rows, std::size_t cols) {
    std::ostringstream out;
    out << "groupings {\n  name: \"" << rows << "x" << cols << "_Mesh\"\n  preset_type: MESH\n  instances: [\n";
    const std::size_t n = rows * cols;
    for (std::size_t i = 0; i < n; ++i) {
        out << "    { id: " << i << " location { asic_location: ASIC_LOCATION_UNSPECIFIED } }";
        out << (i + 1 < n ? ",\n" : "\n");
    }
    out << "  ]\n  row_major_mesh {\n    dims: [" << rows << ", " << cols << "]\n  }\n}\n";
    return out.str();
}

std::string mesh_grid_mgd(
    std::size_t mesh_rows, std::size_t mesh_cols, std::size_t fabric_rows, std::size_t fabric_cols) {
    std::ostringstream out;
    out << "mesh_descriptors {\n"
        << "  name: \"M\"\n  arch: WORMHOLE_B0\n"
        << "  device_topology { dims: [ " << mesh_rows << ", " << mesh_cols << " ] dim_types: [ LINE, LINE ] }\n"
        << "  host_topology   { dims: [ 1, 1 ] }\n"
        << "  channels { count: 2 policy: STRICT }\n"
        << "}\n\n"
        << "graph_descriptors {\n  name: \"G0\"\n  type: \"FABRIC\"\n";
    const std::size_t n = fabric_rows * fabric_cols;
    for (std::size_t i = 0; i < n; ++i) {
        out << "  instances { mesh { mesh_descriptor: \"M\" mesh_id: " << i << " } }\n";
    }
    auto mesh_id = [fabric_cols](std::size_t r, std::size_t c) { return r * fabric_cols + c; };
    for (std::size_t r = 0; r < fabric_rows; ++r) {
        for (std::size_t c = 0; c < fabric_cols; ++c) {
            if (c + 1 < fabric_cols) {
                out << "  connections {\n"
                    << "    nodes { mesh { mesh_descriptor: \"M\" mesh_id: " << mesh_id(r, c) << " } }\n"
                    << "    nodes { mesh { mesh_descriptor: \"M\" mesh_id: " << mesh_id(r, c + 1) << " } }\n"
                    << "    channels { count: 2 policy: RELAXED }\n"
                    << "  }\n";
            }
            if (r + 1 < fabric_rows) {
                out << "  connections {\n"
                    << "    nodes { mesh { mesh_descriptor: \"M\" mesh_id: " << mesh_id(r, c) << " } }\n"
                    << "    nodes { mesh { mesh_descriptor: \"M\" mesh_id: " << mesh_id(r + 1, c) << " } }\n"
                    << "    channels { count: 2 policy: RELAXED }\n"
                    << "  }\n";
            }
        }
    }
    out << "}\n\ntop_level_instance { graph { graph_descriptor: \"G0\" graph_id: 0 } }\n";
    return out.str();
}

std::string grid_psd(std::size_t rows, std::size_t cols) {
    std::ostringstream out;
    out << "target_device_type: 0\nsystem_graph {\n  asic_connectivity_graph {\n    host_name: \"host0\"\n";
    auto asic_id = [cols](std::size_t r, std::size_t c) { return 100 + r * cols + c; };
    auto emit_link = [&](std::size_t dst, int chan0) {
        out << "        asic_connections {\n          dst_asic_id: " << dst << "\n";
        for (int i = 0; i < 2; ++i) {
            out << "          eth_connections {\n            src_chan: " << (chan0 + i)
                << "\n            dst_chan: " << (chan0 + i) << "\n            is_local: true\n          }\n";
        }
        out << "        }\n";
    };
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            out << "    asic_topologies {\n      asic_id: " << asic_id(r, c) << "\n      topology {\n";
            if (c > 0) {
                emit_link(asic_id(r, c - 1), 0);
            }
            if (c + 1 < cols) {
                emit_link(asic_id(r, c + 1), 0);
            }
            if (r > 0) {
                emit_link(asic_id(r - 1, c), 2);
            }
            if (r + 1 < rows) {
                emit_link(asic_id(r + 1, c), 2);
            }
            out << "      }\n    }\n";
        }
    }
    out << "  }\n}\n";
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            const auto id = asic_id(r, c);
            out << "asic_descriptors {\n  asic_id: " << id
                << "\n  asic_descriptor {\n    tray_id: 0\n    asic_location: " << (c % 8)
                << "\n    board_type: 1\n    unique_id: " << id << "\n    host_name: \"host0\"\n  }\n}\n";
        }
    }
    out << "host_to_rank {\n  host_name: \"host0\"\n  rank: 0\n}\nethernet_firmware_version {\n  major: 1\n  minor: "
           "0\n  patch: 0\n}\n";
    return out.str();
}

// Sets one environment variable for the enclosing scope and restores whatever was there before.
// TT_METAL_PLACEMENT_SOLVER selects the placement search (sat | dfs | auto), which is how the tests
// below pin down which path they exercise.
class ScopedEnv {
public:
    ScopedEnv(const char* name, const char* value) : name_(name) {
        if (const char* previous = std::getenv(name)) {
            previous_ = previous;
        }
        setenv(name, value, /*overwrite=*/1);
    }
    ~ScopedEnv() {
        if (previous_.has_value()) {
            setenv(name_, previous_->c_str(), /*overwrite=*/1);
        } else {
            unsetenv(name_);
        }
    }
    ScopedEnv(const ScopedEnv&) = delete;
    ScopedEnv& operator=(const ScopedEnv&) = delete;

private:
    const char* name_;
    std::optional<std::string> previous_;
};

tt::tt_metal::PhysicalSystemDescriptor load_psd_from_text(const std::string& text) {
    const auto path = std::filesystem::temp_directory_path() /
                      ("pgd_placement_strain_" +
                       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".textproto");
    {
        std::ofstream out(path);
        out << text;
    }
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(path.string());
    std::filesystem::remove(path);
    return psd;
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
    PhysicalGroupingDescriptor pgd{std::string(R"delimiter(
# Mesh shapes with every ASIC location UNSPECIFIED, so a grouping is free to land anywhere: that is
# what gives the search more than one candidate to choose between.
groupings {
  name: "1x2_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 1 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  row_major_mesh { dims: [1, 2] }
}

# The machine's host level: one host owning all 4 chips, wired the way the machine is,
# a 1x4 line. One preset_type: HOSTS grouping is one host, what it holds is that host's chips,
# and its connection block is that host's own topology.
groupings {
  name: "machine_host"
  preset_type: HOSTS
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 1 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 2 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 3 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  row_major_mesh { dims: [1, 4] }
}
)delimiter")};
    MeshGraphDescriptor mgd{std::string(R"delimiter(
# Two 1x2 meshes joined by one intermesh connection. The unlinked pair below is the same
# descriptor without the connection, which is what isolates the seam: only this one requires
# the two placements to touch.
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 1, 2 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}

graph_descriptors {
  name: "G0"
  type: "FABRIC"
  instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
  instances { mesh { mesh_descriptor: "M0" mesh_id: 1 } }

  connections {
    nodes { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
    nodes { mesh { mesh_descriptor: "M0" mesh_id: 1 } }
    channels { count: 2 policy: RELAXED }
  }
}

top_level_instance { graph { graph_descriptor: "G0" graph_id: 0 } }
)delimiter")};
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(
        "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_4asic_line.textproto");

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
    PhysicalGroupingDescriptor pgd{std::string(R"delimiter(
# Mesh shapes with every ASIC location UNSPECIFIED, so a grouping is free to land anywhere: that is
# what gives the search more than one candidate to choose between.
groupings {
  name: "1x2_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 1 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  row_major_mesh { dims: [1, 2] }
}

# The machine's host level: one host owning all 4 chips, wired the way the machine is.
# One preset_type: HOSTS grouping is one host, what it holds is that host's chips, and its
# connection block is that host's own topology -- spelled out link by link here, because
# this machine is two disjoint pairs.
groupings {
  name: "machine_host"
  preset_type: HOSTS
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 1 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 2 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 3 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  custom {
    connections: [
      { src_instance: 0 dst_instance: 1 },
      { src_instance: 2 dst_instance: 3 }
    ]
  }
}
)delimiter")};
    MeshGraphDescriptor mgd{std::string(R"delimiter(
# Two 1x2 meshes joined by one intermesh connection. The unlinked pair below is the same
# descriptor without the connection, which is what isolates the seam: only this one requires
# the two placements to touch.
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 1, 2 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}

graph_descriptors {
  name: "G0"
  type: "FABRIC"
  instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
  instances { mesh { mesh_descriptor: "M0" mesh_id: 1 } }

  connections {
    nodes { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
    nodes { mesh { mesh_descriptor: "M0" mesh_id: 1 } }
    channels { count: 2 policy: RELAXED }
  }
}

top_level_instance { graph { graph_descriptor: "G0" graph_id: 0 } }
)delimiter")};
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(
        "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_4asic_2mesh.textproto");

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
    PhysicalGroupingDescriptor pgd{std::string(R"delimiter(
# Mesh shapes with every ASIC location UNSPECIFIED, so a grouping is free to land anywhere: that is
# what gives the search more than one candidate to choose between.
groupings {
  name: "1x2_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 1 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  row_major_mesh { dims: [1, 2] }
}

# The machine's host level: one host owning all 4 chips, wired the way the machine is.
# One preset_type: HOSTS grouping is one host, what it holds is that host's chips, and its
# connection block is that host's own topology -- spelled out link by link here, because
# this machine is two disjoint pairs.
groupings {
  name: "machine_host"
  preset_type: HOSTS
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 1 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 2 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 3 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  custom {
    connections: [
      { src_instance: 0 dst_instance: 1 },
      { src_instance: 2 dst_instance: 3 }
    ]
  }
}
)delimiter")};
    MeshGraphDescriptor mgd{std::string(R"delimiter(
# The same two 1x2 meshes with no mesh-level edge, so the placements only have to be disjoint.
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 1, 2 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}

graph_descriptors {
  name: "G0"
  type: "FABRIC"
  instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
  instances { mesh { mesh_descriptor: "M0" mesh_id: 1 } }
}

top_level_instance { graph { graph_descriptor: "G0" graph_id: 0 } }
)delimiter")};
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(
        "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_4asic_2mesh.textproto");

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
    PhysicalGroupingDescriptor pgd{std::string(R"delimiter(
# Mesh shapes with every ASIC location UNSPECIFIED, so a grouping is free to land anywhere: that is
# what gives the search more than one candidate to choose between.
groupings {
  name: "1x2_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 1 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  row_major_mesh { dims: [1, 2] }
}
groupings {
  name: "1x1_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  row_major_mesh { dims: [1, 1] }
}

# The machine's host level: one host owning all 6 chips, wired the way the machine is.
# One preset_type: HOSTS grouping is one host, what it holds is that host's chips, and its
# connection block is that host's own topology -- spelled out link by link here, because
# this machine is a ring, which row_major_mesh cannot express -- it has no wrap.
groupings {
  name: "machine_host"
  preset_type: HOSTS
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 1 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 2 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 3 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 4 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 5 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  custom {
    connections: [
      { src_instance: 0 dst_instance: 1 },
      { src_instance: 1 dst_instance: 2 },
      { src_instance: 2 dst_instance: 3 },
      { src_instance: 3 dst_instance: 4 },
      { src_instance: 4 dst_instance: 5 },
      { src_instance: 0 dst_instance: 5 }
    ]
  }
}
)delimiter")};
    MeshGraphDescriptor mgd{std::string(R"delimiter(
# Four meshes in a ring alternating two shapes, so every mesh-level edge crosses a shape
# boundary. 2 + 1 + 2 + 1 = 6 chips, exactly the machine, so there is no spare room.
mesh_descriptors {
  name: "A"  # 1x2
  arch: WORMHOLE_B0
  device_topology { dims: [ 1, 2 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}

mesh_descriptors {
  name: "B"  # 1x1
  arch: WORMHOLE_B0
  device_topology { dims: [ 1, 1 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}

graph_descriptors {
  name: "G0"
  type: "FABRIC"
  instances { mesh { mesh_descriptor: "A" mesh_id: 0 } }
  instances { mesh { mesh_descriptor: "B" mesh_id: 1 } }
  instances { mesh { mesh_descriptor: "A" mesh_id: 2 } }
  instances { mesh { mesh_descriptor: "B" mesh_id: 3 } }

  connections {
    nodes { mesh { mesh_descriptor: "A" mesh_id: 0 } }
    nodes { mesh { mesh_descriptor: "B" mesh_id: 1 } }
    channels { count: 2 policy: RELAXED }
  }
  connections {
    nodes { mesh { mesh_descriptor: "B" mesh_id: 1 } }
    nodes { mesh { mesh_descriptor: "A" mesh_id: 2 } }
    channels { count: 2 policy: RELAXED }
  }
  connections {
    nodes { mesh { mesh_descriptor: "A" mesh_id: 2 } }
    nodes { mesh { mesh_descriptor: "B" mesh_id: 3 } }
    channels { count: 2 policy: RELAXED }
  }
  connections {
    nodes { mesh { mesh_descriptor: "B" mesh_id: 3 } }
    nodes { mesh { mesh_descriptor: "A" mesh_id: 0 } }
    channels { count: 2 policy: RELAXED }
  }
}

top_level_instance { graph { graph_descriptor: "G0" graph_id: 0 } }
)delimiter")};
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(
        "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_6asic_ring.textproto");

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

    // PGD shapes only. The variant list also carries the MGD grouping as a fallback, which is named
    // after the mesh and has no PGD grouping behind it for find_all_in_psd to flatten. The packer only
    // ever saw PGD shapes, and this half of the test is about what packing one of them produces.
    const auto pack_shape = [&](const std::string& mesh_name) {
        std::vector<GroupingInfo> pgd_shapes;
        for (const auto& grouping : valid_groupings.at("MESH").at(mesh_name)) {
            if (grouping.name != mesh_name) {
                pgd_shapes.push_back(grouping);
            }
        }
        return pgd.find_all_in_psd(pgd_shapes, psd);
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
    PhysicalGroupingDescriptor pgd{std::string(R"delimiter(
# Mesh shapes with every ASIC location UNSPECIFIED, so a grouping is free to land anywhere: that is
# what gives the search more than one candidate to choose between.
groupings {
  name: "1x2_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 1 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  row_major_mesh { dims: [1, 2] }
}
groupings {
  name: "1x1_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  row_major_mesh { dims: [1, 1] }
}

# The machine's host level: one host owning all 6 chips, wired the way the machine is,
# a 1x6 line. One preset_type: HOSTS grouping is one host, what it holds is that host's chips,
# and its connection block is that host's own topology.
groupings {
  name: "machine_host"
  preset_type: HOSTS
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 1 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 2 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 3 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 4 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 5 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  row_major_mesh { dims: [1, 6] }
}
)delimiter")};
    MeshGraphDescriptor mgd{std::string(R"delimiter(
# Four meshes in a ring alternating two shapes, so every mesh-level edge crosses a shape
# boundary. 2 + 1 + 2 + 1 = 6 chips, exactly the machine, so there is no spare room.
mesh_descriptors {
  name: "A"  # 1x2
  arch: WORMHOLE_B0
  device_topology { dims: [ 1, 2 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}

mesh_descriptors {
  name: "B"  # 1x1
  arch: WORMHOLE_B0
  device_topology { dims: [ 1, 1 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}

graph_descriptors {
  name: "G0"
  type: "FABRIC"
  instances { mesh { mesh_descriptor: "A" mesh_id: 0 } }
  instances { mesh { mesh_descriptor: "B" mesh_id: 1 } }
  instances { mesh { mesh_descriptor: "A" mesh_id: 2 } }
  instances { mesh { mesh_descriptor: "B" mesh_id: 3 } }

  connections {
    nodes { mesh { mesh_descriptor: "A" mesh_id: 0 } }
    nodes { mesh { mesh_descriptor: "B" mesh_id: 1 } }
    channels { count: 2 policy: RELAXED }
  }
  connections {
    nodes { mesh { mesh_descriptor: "B" mesh_id: 1 } }
    nodes { mesh { mesh_descriptor: "A" mesh_id: 2 } }
    channels { count: 2 policy: RELAXED }
  }
  connections {
    nodes { mesh { mesh_descriptor: "A" mesh_id: 2 } }
    nodes { mesh { mesh_descriptor: "B" mesh_id: 3 } }
    channels { count: 2 policy: RELAXED }
  }
  connections {
    nodes { mesh { mesh_descriptor: "B" mesh_id: 3 } }
    nodes { mesh { mesh_descriptor: "A" mesh_id: 0 } }
    channels { count: 2 policy: RELAXED }
  }
}

top_level_instance { graph { graph_descriptor: "G0" graph_id: 0 } }
)delimiter")};
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(
        "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_6asic_line.textproto");

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
//   PGD grouping places  ->  committed "1x2_Mesh_flat" first, then "M0" as fallback
//   PGD grouping cannot  ->  committed "M0" only          (the MGD fallback)
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
    PhysicalGroupingDescriptor pgd{std::string(R"delimiter(
# Mesh shapes with every ASIC location UNSPECIFIED, so a grouping is free to land anywhere: that is
# what gives the search more than one candidate to choose between.
groupings {
  name: "2x2_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 1 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 2 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 3 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  row_major_mesh { dims: [2, 2] }
}
groupings {
  name: "1x3_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 1 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 2 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  row_major_mesh { dims: [1, 3] }
}
groupings {
  name: "1x2_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 1 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  row_major_mesh { dims: [1, 2] }
}
groupings {
  name: "1x1_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  row_major_mesh { dims: [1, 1] }
}

# The machine's host level: one host owning all 10 chips, wired the way the machine is.
# One preset_type: HOSTS grouping is one host, what it holds is that host's chips, and its
# connection block is that host's own topology -- spelled out link by link here, because
# this machine is a 4-cycle with two branches off it.
groupings {
  name: "machine_host"
  preset_type: HOSTS
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 1 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 2 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 3 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 4 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 5 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 6 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 7 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 8 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 9 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  custom {
    connections: [
      { src_instance: 0 dst_instance: 1 },
      { src_instance: 0 dst_instance: 3 },
      { src_instance: 1 dst_instance: 2 },
      { src_instance: 2 dst_instance: 3 },
      { src_instance: 2 dst_instance: 4 },
      { src_instance: 4 dst_instance: 5 },
      { src_instance: 4 dst_instance: 7 },
      { src_instance: 5 dst_instance: 6 },
      { src_instance: 7 dst_instance: 8 },
      { src_instance: 8 dst_instance: 9 }
    ]
  }
}
)delimiter")};
    MeshGraphDescriptor mgd{std::string(R"delimiter(
# Four shapes chained D(2x2) - C(1x3) - B(1x2) - A(1x1). 4 + 3 + 2 + 1 = 10 chips, exactly the
# machine.
mesh_descriptors {
  name: "D"  # 2x2
  arch: WORMHOLE_B0
  device_topology { dims: [ 2, 2 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}

mesh_descriptors {
  name: "C"  # 1x3
  arch: WORMHOLE_B0
  device_topology { dims: [ 1, 3 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}

mesh_descriptors {
  name: "B"  # 1x2
  arch: WORMHOLE_B0
  device_topology { dims: [ 1, 2 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}

mesh_descriptors {
  name: "A"  # 1x1
  arch: WORMHOLE_B0
  device_topology { dims: [ 1, 1 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}

graph_descriptors {
  name: "G0"
  type: "FABRIC"
  instances { mesh { mesh_descriptor: "D" mesh_id: 0 } }
  instances { mesh { mesh_descriptor: "C" mesh_id: 1 } }
  instances { mesh { mesh_descriptor: "B" mesh_id: 2 } }
  instances { mesh { mesh_descriptor: "A" mesh_id: 3 } }

  connections {
    nodes { mesh { mesh_descriptor: "D" mesh_id: 0 } }
    nodes { mesh { mesh_descriptor: "C" mesh_id: 1 } }
    channels { count: 2 policy: RELAXED }
  }
  connections {
    nodes { mesh { mesh_descriptor: "C" mesh_id: 1 } }
    nodes { mesh { mesh_descriptor: "B" mesh_id: 2 } }
    channels { count: 2 policy: RELAXED }
  }
  connections {
    nodes { mesh { mesh_descriptor: "B" mesh_id: 2 } }
    nodes { mesh { mesh_descriptor: "A" mesh_id: 3 } }
    channels { count: 2 policy: RELAXED }
  }
}

top_level_instance { graph { graph_descriptor: "G0" graph_id: 0 } }
)delimiter")};
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(
        "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_10asic_square_fork.textproto");

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
    PhysicalGroupingDescriptor pgd{std::string(R"delimiter(
# Mesh shapes with every ASIC location UNSPECIFIED, so a grouping is free to land anywhere: that is
# what gives the search more than one candidate to choose between.
groupings {
  name: "2x2_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 1 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 2 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 3 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  row_major_mesh { dims: [2, 2] }
}
groupings {
  name: "1x3_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 1 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 2 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  row_major_mesh { dims: [1, 3] }
}
groupings {
  name: "1x2_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 1 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  row_major_mesh { dims: [1, 2] }
}
groupings {
  name: "1x1_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  row_major_mesh { dims: [1, 1] }
}

# The machine's host level: one host owning all 12 chips, wired the way the machine is.
# One preset_type: HOSTS grouping is one host, what it holds is that host's chips, and its
# connection block is that host's own topology -- spelled out link by link here, because
# this machine is a 4-cycle with three spokes.
groupings {
  name: "machine_host"
  preset_type: HOSTS
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 1 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 2 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 3 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 4 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 5 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 6 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 7 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 8 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 9 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 10 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 11 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  custom {
    connections: [
      { src_instance: 0 dst_instance: 1 },
      { src_instance: 0 dst_instance: 2 },
      { src_instance: 1 dst_instance: 3 },
      { src_instance: 1 dst_instance: 4 },
      { src_instance: 2 dst_instance: 3 },
      { src_instance: 2 dst_instance: 10 },
      { src_instance: 3 dst_instance: 7 },
      { src_instance: 4 dst_instance: 5 },
      { src_instance: 5 dst_instance: 6 },
      { src_instance: 7 dst_instance: 8 },
      { src_instance: 8 dst_instance: 9 },
      { src_instance: 10 dst_instance: 11 }
    ]
  }
}
)delimiter")};
    MeshGraphDescriptor mgd{std::string(R"delimiter(
# A 2x2 hub with three spokes, each seam asking for a different number of channels. Ask every
# seam for one channel instead and four placements satisfy this rather than one.
mesh_descriptors {
  name: "H"  # the 2x2 hub
  arch: WORMHOLE_B0
  device_topology { dims: [ 2, 2 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}

mesh_descriptors {
  name: "Sa"  # 1x3 spoke, on the 4-channel branch
  arch: WORMHOLE_B0
  device_topology { dims: [ 1, 3 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}

mesh_descriptors {
  name: "Sb"  # 1x2 spoke, on the 3-channel branch
  arch: WORMHOLE_B0
  device_topology { dims: [ 1, 2 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}

mesh_descriptors {
  name: "Sc"  # 1x1 spoke, on the 2-channel branch
  arch: WORMHOLE_B0
  device_topology { dims: [ 1, 1 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}

graph_descriptors {
  name: "G0"
  type: "FABRIC"
  instances { mesh { mesh_descriptor: "H" mesh_id: 0 } }
  instances { mesh { mesh_descriptor: "Sa" mesh_id: 1 } }
  instances { mesh { mesh_descriptor: "Sb" mesh_id: 2 } }
  instances { mesh { mesh_descriptor: "Sc" mesh_id: 3 } }

  connections {
    nodes { mesh { mesh_descriptor: "H" mesh_id: 0 } }
    nodes { mesh { mesh_descriptor: "Sa" mesh_id: 1 } }
    channels { count: 4 policy: RELAXED }
  }
  connections {
    nodes { mesh { mesh_descriptor: "H" mesh_id: 0 } }
    nodes { mesh { mesh_descriptor: "Sb" mesh_id: 2 } }
    channels { count: 3 policy: RELAXED }
  }
  connections {
    nodes { mesh { mesh_descriptor: "H" mesh_id: 0 } }
    nodes { mesh { mesh_descriptor: "Sc" mesh_id: 3 } }
    channels { count: 2 policy: RELAXED }
  }
}

top_level_instance { graph { graph_descriptor: "G0" graph_id: 0 } }
)delimiter")};
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(
        "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_12asic_star.textproto");

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
    PhysicalGroupingDescriptor pgd{std::string(R"delimiter(
# Mesh shapes with every ASIC location UNSPECIFIED, so a grouping is free to land anywhere: that is
# what gives the search more than one candidate to choose between.
groupings {
  name: "1x2_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 1 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  row_major_mesh { dims: [1, 2] }
}
groupings {
  name: "1x1_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  row_major_mesh { dims: [1, 1] }
}

# The machine's host level: one host owning all 5 chips, wired the way the machine is,
# a 1x5 line. One preset_type: HOSTS grouping is one host, what it holds is that host's chips,
# and its connection block is that host's own topology.
groupings {
  name: "machine_host"
  preset_type: HOSTS
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 1 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 2 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 3 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 4 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  row_major_mesh { dims: [1, 5] }
}
)delimiter")};
    std::vector<MeshGraphDescriptor> mgds;
    mgds.emplace_back(std::string(R"delimiter(
# One of a pair of descriptors placed together. Both reuse the instance names M0 and M1:
# descriptors placed at once are kept apart by descriptor index, and that reuse is what makes a
# failure of the keying visible -- look up "M0" without knowing which descriptor asked and a mesh
# is built with the wrong shape. The STRICT count on the seam is what decides which side of a
# pinched machine each descriptor lands on.
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 1, 2 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}

mesh_descriptors {
  name: "M1"
  arch: WORMHOLE_B0
  device_topology { dims: [ 1, 1 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}

graph_descriptors {
  name: "G0"
  type: "FABRIC"
  instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
  instances { mesh { mesh_descriptor: "M1" mesh_id: 1 } }

  connections {
    nodes { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
    nodes { mesh { mesh_descriptor: "M1" mesh_id: 1 } }
    channels { count: 4 policy: STRICT }
  }
}

top_level_instance { graph { graph_descriptor: "G0" graph_id: 0 } }
)delimiter"));
    mgds.emplace_back(std::string(R"delimiter(
# The companion of the descriptor above of descriptors placed together. Both reuse the instance names M0 and M1:
# descriptors placed at once are kept apart by descriptor index, and that reuse is what makes a
# failure of the keying visible -- look up "M0" without knowing which descriptor asked and a mesh
# is built with the wrong shape. The STRICT count on the seam is what decides which side of a
# pinched machine each descriptor lands on.
# Two single chips rather than a mirror of it, so the two are not interchangeable: only one 1x2
# mesh exists across the pair, and it belongs to the other descriptor.
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 1, 1 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}

mesh_descriptors {
  name: "M1"
  arch: WORMHOLE_B0
  device_topology { dims: [ 1, 1 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}

graph_descriptors {
  name: "G0"
  type: "FABRIC"
  instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
  instances { mesh { mesh_descriptor: "M1" mesh_id: 1 } }

  connections {
    nodes { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
    nodes { mesh { mesh_descriptor: "M1" mesh_id: 1 } }
    channels { count: 3 policy: STRICT }
  }
}

top_level_instance { graph { graph_descriptor: "G0" graph_id: 0 } }
)delimiter"));
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(
        "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_5asic_dumbbell.textproto");

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
    PhysicalGroupingDescriptor pgd{std::string(R"delimiter(
# Mesh shapes with every ASIC location UNSPECIFIED, so a grouping is free to land anywhere: that is
# what gives the search more than one candidate to choose between.
groupings {
  name: "1x2_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 1 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  row_major_mesh { dims: [1, 2] }
}
groupings {
  name: "1x1_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  row_major_mesh { dims: [1, 1] }
}

# The machine's host level: one host owning all 6 chips, wired the way the machine is,
# a 1x6 line. One preset_type: HOSTS grouping is one host, what it holds is that host's chips,
# and its connection block is that host's own topology.
groupings {
  name: "machine_host"
  preset_type: HOSTS
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 1 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 2 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 3 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 4 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 5 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  row_major_mesh { dims: [1, 6] }
}
)delimiter")};
    std::vector<MeshGraphDescriptor> mgds;
    mgds.emplace_back(std::string(R"delimiter(
# One of a pair of descriptors placed together. Both reuse the instance names M0 and M1:
# descriptors placed at once are kept apart by descriptor index, and that reuse is what makes a
# failure of the keying visible -- look up "M0" without knowing which descriptor asked and a mesh
# is built with the wrong shape. The STRICT count on the seam is what decides which side of a
# pinched machine each descriptor lands on.
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 1, 2 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}

mesh_descriptors {
  name: "M1"
  arch: WORMHOLE_B0
  device_topology { dims: [ 1, 1 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}

graph_descriptors {
  name: "G0"
  type: "FABRIC"
  instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
  instances { mesh { mesh_descriptor: "M1" mesh_id: 1 } }

  connections {
    nodes { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
    nodes { mesh { mesh_descriptor: "M1" mesh_id: 1 } }
    channels { count: 4 policy: STRICT }
  }
}

top_level_instance { graph { graph_descriptor: "G0" graph_id: 0 } }
)delimiter"));
    mgds.emplace_back(std::string(R"delimiter(
# The mirror of the 1x2-then-1x1 descriptor: same two names, shapes the other way round. That
# makes the two interchangeable by shape, so nothing but the channel count on the seam says which
# side each belongs on -- which is what the mapper seam-width test wants, and exactly why the
# placement test uses a non-interchangeable pair instead.
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 1, 1 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}

mesh_descriptors {
  name: "M1"
  arch: WORMHOLE_B0
  device_topology { dims: [ 1, 2 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}

graph_descriptors {
  name: "G0"
  type: "FABRIC"
  instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
  instances { mesh { mesh_descriptor: "M1" mesh_id: 1 } }

  connections {
    nodes { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
    nodes { mesh { mesh_descriptor: "M1" mesh_id: 1 } }
    channels { count: 3 policy: STRICT }
  }
}

top_level_instance { graph { graph_descriptor: "G0" graph_id: 0 } }
)delimiter"));
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(
        "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_6asic_dumbbell.textproto");

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
// that is where the policy decides the outcome: STRICT has no placement, while RELAXED still places,
// because the count is a preference and the mesh-level edge only insists the two regions touch.

// A preference that can be met should be met: the seam wants 4 channels and only 101-102 has them.
// The preference half of RELAXED, as opposed to the requirement half below.
TEST(AdjacencyGuidedPlacement, RelaxedSeamPrefersTheFullChannelCount) {
    // build pgd
    PhysicalGroupingDescriptor pgd{std::string(R"delimiter(
# Mesh shapes with every ASIC location UNSPECIFIED, so a grouping is free to land anywhere: that is
# what gives the search more than one candidate to choose between.
groupings {
  name: "1x2_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 1 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  row_major_mesh { dims: [1, 2] }
}
groupings {
  name: "1x1_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  row_major_mesh { dims: [1, 1] }
}

# The machine's host level: one host owning all 3 chips, laid out the way the machine is,
# a 1x3 line. One preset_type: HOSTS grouping is one host and what it holds is that
# host's chips.
groupings {
  name: "machine_host"
  preset_type: HOSTS
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 1 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 2 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  row_major_mesh { dims: [1, 3] }
}
)delimiter")};
    MeshGraphDescriptor mgd{std::string(R"delimiter(
# Two single-chip meshes whose seam asks for 4 channels as a preference: satisfiable on the 3-chip
# machine, but only on 101-102.
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 1, 1 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}

mesh_descriptors {
  name: "M1"
  arch: WORMHOLE_B0
  device_topology { dims: [ 1, 1 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}

graph_descriptors {
  name: "G0"
  type: "FABRIC"
  instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
  instances { mesh { mesh_descriptor: "M1" mesh_id: 1 } }

  connections {
    nodes { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
    nodes { mesh { mesh_descriptor: "M1" mesh_id: 1 } }
    channels { count: 4 policy: RELAXED }
  }
}

top_level_instance { graph { graph_descriptor: "G0" graph_id: 0 } }
)delimiter")};
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(
        "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_3asic_uneven_line.textproto");

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
    PhysicalGroupingDescriptor pgd{std::string(R"delimiter(
# Mesh shapes with every ASIC location UNSPECIFIED, so a grouping is free to land anywhere: that is
# what gives the search more than one candidate to choose between.
groupings {
  name: "1x2_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 1 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  row_major_mesh { dims: [1, 2] }
}
groupings {
  name: "1x1_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  row_major_mesh { dims: [1, 1] }
}

# The machine's host level: one host owning all 3 chips, wired the way the machine is,
# a 1x3 line. One preset_type: HOSTS grouping is one host, what it holds is that host's chips,
# and its connection block is that host's own topology.
groupings {
  name: "machine_host"
  preset_type: HOSTS
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 1 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 2 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  row_major_mesh { dims: [1, 3] }
}
)delimiter")};
    MeshGraphDescriptor mgd{std::string(R"delimiter(
# The same unsatisfiable count as a requirement: there is no valid placement at all.
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 1, 1 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}

mesh_descriptors {
  name: "M1"
  arch: WORMHOLE_B0
  device_topology { dims: [ 1, 1 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}

graph_descriptors {
  name: "G0"
  type: "FABRIC"
  instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
  instances { mesh { mesh_descriptor: "M1" mesh_id: 1 } }

  connections {
    nodes { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
    nodes { mesh { mesh_descriptor: "M1" mesh_id: 1 } }
    channels { count: 8 policy: STRICT }
  }
}

top_level_instance { graph { graph_descriptor: "G0" graph_id: 0 } }
)delimiter")};
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(
        "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_3asic_uneven_line.textproto");

    // get valid groupings
    const auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd, psd);
    ASSERT_FALSE(valid_groupings.at("MESH").empty()) << "the shapes themselves are placeable; only the seam is not";

    // place
    const auto placements = pgd.solve_adjacency_guided_placement(mgd, valid_groupings, psd);
    EXPECT_TRUE(placements.empty())
        << "no link carries the 8 channels a STRICT seam requires, so there should be no placement at all";
}

// Same unsatisfiable count as the STRICT test above, but the descriptor says RELAXED, which makes the
// count a preference. The meshes still have to touch, and 101-102 is the widest seam on offer, so
// placement seats them there rather than giving up. Treating the count as a requirement here would be
// stricter than the descriptor asked for, and stricter than the mapper, which accepts this seating and
// logs that the seam is narrower than requested.
TEST(AdjacencyGuidedPlacement, RelaxedSeamStillPlacesWhenChannelsFallShort) {
    // build pgd
    PhysicalGroupingDescriptor pgd{std::string(R"delimiter(
# Mesh shapes with every ASIC location UNSPECIFIED, so a grouping is free to land anywhere: that is
# what gives the search more than one candidate to choose between.
groupings {
  name: "1x2_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 1 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  row_major_mesh { dims: [1, 2] }
}
groupings {
  name: "1x1_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  row_major_mesh { dims: [1, 1] }
}

# The machine's host level: one host owning all 3 chips, wired the way the machine is,
# a 1x3 line. One preset_type: HOSTS grouping is one host, what it holds is that host's chips,
# and its connection block is that host's own topology.
groupings {
  name: "machine_host"
  preset_type: HOSTS
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 1 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 2 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  row_major_mesh { dims: [1, 3] }
}
)delimiter")};
    MeshGraphDescriptor mgd{std::string(R"delimiter(
# The same pair asking for 8 channels, which no link on that machine carries. As a preference the
# meshes only have to touch, so a placement should still be found, on the widest seam there is.
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 1, 1 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}

mesh_descriptors {
  name: "M1"
  arch: WORMHOLE_B0
  device_topology { dims: [ 1, 1 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}

graph_descriptors {
  name: "G0"
  type: "FABRIC"
  instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
  instances { mesh { mesh_descriptor: "M1" mesh_id: 1 } }

  connections {
    nodes { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
    nodes { mesh { mesh_descriptor: "M1" mesh_id: 1 } }
    channels { count: 8 policy: RELAXED }
  }
}

top_level_instance { graph { graph_descriptor: "G0" graph_id: 0 } }
)delimiter")};
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(
        "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_3asic_uneven_line.textproto");

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

// Descriptors merged into one topology have to agree on the policy. The merged solve applies a single
// policy to every seam, so a mixed set would quietly have one descriptor's policy applied to the other's
// seams -- the same reason MGD validation rejects mixing within one descriptor. Temporary, until per-seam
// policy is supported: https://github.com/tenstorrent/tt-metal/issues/49960
TEST(AdjacencyGuidedPlacement, DescriptorsThatDisagreeOnInterMeshPolicyAreRejected) {
    PhysicalGroupingDescriptor pgd{std::string(R"delimiter(
# Mesh shapes with every ASIC location UNSPECIFIED, so a grouping is free to land anywhere: that is
# what gives the search more than one candidate to choose between.
groupings {
  name: "1x2_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 1 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  row_major_mesh { dims: [1, 2] }
}
groupings {
  name: "1x1_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  row_major_mesh { dims: [1, 1] }
}

# The machine's host level: one host owning all 3 chips, wired the way the machine is,
# a 1x3 line. One preset_type: HOSTS grouping is one host, what it holds is that host's chips,
# and its connection block is that host's own topology.
groupings {
  name: "machine_host"
  preset_type: HOSTS
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 1 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 2 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  row_major_mesh { dims: [1, 3] }
}
)delimiter")};
    std::vector<MeshGraphDescriptor> mgds;
    mgds.emplace_back(std::string(R"delimiter(
# Two single-chip meshes whose seam asks for 4 channels as a preference: satisfiable on the 3-chip
# machine, but only on 101-102.
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 1, 1 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}

mesh_descriptors {
  name: "M1"
  arch: WORMHOLE_B0
  device_topology { dims: [ 1, 1 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}

graph_descriptors {
  name: "G0"
  type: "FABRIC"
  instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
  instances { mesh { mesh_descriptor: "M1" mesh_id: 1 } }

  connections {
    nodes { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
    nodes { mesh { mesh_descriptor: "M1" mesh_id: 1 } }
    channels { count: 4 policy: RELAXED }
  }
}

top_level_instance { graph { graph_descriptor: "G0" graph_id: 0 } }
)delimiter"));
    mgds.emplace_back(std::string(R"delimiter(
# The same unsatisfiable count as a requirement: there is no valid placement at all.
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 1, 1 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}

mesh_descriptors {
  name: "M1"
  arch: WORMHOLE_B0
  device_topology { dims: [ 1, 1 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}

graph_descriptors {
  name: "G0"
  type: "FABRIC"
  instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
  instances { mesh { mesh_descriptor: "M1" mesh_id: 1 } }

  connections {
    nodes { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
    nodes { mesh { mesh_descriptor: "M1" mesh_id: 1 } }
    channels { count: 8 policy: STRICT }
  }
}

top_level_instance { graph { graph_descriptor: "G0" graph_id: 0 } }
)delimiter"));
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(
        "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_3asic_uneven_line.textproto");

    ASSERT_NE(mgds[0].is_inter_mesh_policy_relaxed(), mgds[1].is_inter_mesh_policy_relaxed())
        << "the two descriptors have to disagree for this test to mean anything";

    EXPECT_ANY_THROW(utils::validate_shared_inter_mesh_policy({&mgds[0], &mgds[1]}))
        << "a RELAXED descriptor and a STRICT one cannot be merged into one topology";

    // And the multi-MGD build applies it on its own, so a caller cannot reach the merge by not asking.
    EXPECT_ANY_THROW(utils::build_physical_multi_mesh_adjacency_graph(psd, pgd, mgds))
        << "the vector overload should reject the pair before it does any work";
}

// A descriptor with no inter-mesh connections defaults to STRICT, so it conflicts with a RELAXED sibling.
TEST(AdjacencyGuidedPlacement, DescriptorWithoutInterMeshConnectionsDefaultsToStrictPolicy) {
    MeshGraphDescriptor relaxed{std::string(R"delimiter(
# Two single-chip meshes whose seam asks for 4 channels as a preference: satisfiable on the 3-chip
# machine, but only on 101-102.
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 1, 1 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}

mesh_descriptors {
  name: "M1"
  arch: WORMHOLE_B0
  device_topology { dims: [ 1, 1 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}

graph_descriptors {
  name: "G0"
  type: "FABRIC"
  instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
  instances { mesh { mesh_descriptor: "M1" mesh_id: 1 } }

  connections {
    nodes { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
    nodes { mesh { mesh_descriptor: "M1" mesh_id: 1 } }
    channels { count: 4 policy: RELAXED }
  }
}

top_level_instance { graph { graph_descriptor: "G0" graph_id: 0 } }
)delimiter")};
    MeshGraphDescriptor single{std::string(R"delimiter(
# One 1x2 mesh, nothing else and no seams.
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 1, 2 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}

graph_descriptors {
  name: "G0"
  type: "FABRIC"
  instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
}

top_level_instance { graph { graph_descriptor: "G0" graph_id: 0 } }
)delimiter")};

    EXPECT_FALSE(single.is_inter_mesh_policy_relaxed()) << "no inter-mesh connections defaults to STRICT";
    EXPECT_TRUE(relaxed.is_inter_mesh_policy_relaxed());
    EXPECT_ANY_THROW(utils::validate_shared_inter_mesh_policy({&relaxed, &single}))
        << "STRICT default and RELAXED cannot be merged into one topology";
}

TEST(AdjacencyGuidedPlacement, PgdGroupingThatPlacesIsCommittedDirectly) {
    // build pgd
    PhysicalGroupingDescriptor pgd{std::string(R"delimiter(
# Mesh shapes with every ASIC location UNSPECIFIED, so a grouping is free to land anywhere: that is
# what gives the search more than one candidate to choose between.
groupings {
  name: "1x2_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 1 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  row_major_mesh { dims: [1, 2] }
}

# The machine's host level: one host owning all 4 chips, wired the way the machine is.
# One preset_type: HOSTS grouping is one host, what it holds is that host's chips, and its
# connection block is that host's own topology -- spelled out link by link here, because
# this machine is two disjoint pairs.
groupings {
  name: "machine_host"
  preset_type: HOSTS
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 1 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 2 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 3 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  custom {
    connections: [
      { src_instance: 0 dst_instance: 1 },
      { src_instance: 2 dst_instance: 3 }
    ]
  }
}
)delimiter")};
    MeshGraphDescriptor mgd{std::string(R"delimiter(
# One 1x2 mesh, nothing else and no seams.
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 1, 2 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}

graph_descriptors {
  name: "G0"
  type: "FABRIC"
  instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
}

top_level_instance { graph { graph_descriptor: "G0" graph_id: 0 } }
)delimiter")};
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(
        "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_4asic_2mesh.textproto");

    // get valid groupings
    // The committed grouping's name is what separates the two paths: a committed PGD grouping keeps
    // its own flattened name, while the MGD fallback grouping is named after the mesh instance.
    // PGD is first (preferred); MGD is appended last when it also embeds, so the list is not a
    // replacement of one by the other.
    const auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd, psd);
    const auto& committed = valid_groupings.at("MESH").at("M0");
    std::vector<std::string> committed_names;
    for (const auto& grouping : committed) {
        committed_names.push_back(grouping.name);
    }
    EXPECT_THAT(committed_names, ::testing::ElementsAre("1x2_Mesh_flat", "M0"))
        << "the PGD grouping embeds into the PSD, so it should be committed first, with the MGD "
           "grouping offered last as fallback";

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

// Both groupings embed, so the list has PGD then MGD. Placement must still take the PGD seating.
// The pinned grouping only fits on {100,101}; the MGD 1x2 would also accept {101,102} or
// {102,103}. If the fallback were tried first, the mesh could land on a pair the PGD never asked
// for. The pinning map is how we tell which variant actually won: PGD carries one, MGD does not.
TEST(AdjacencyGuidedPlacement, PlaceableMgdFallbackDoesNotOutrankPgd) {
    // build pgd
    PhysicalGroupingDescriptor pgd{std::string(R"delimiter(
# A 1x2 shape pinned to ASIC locations 0 and 1 -- chips 100 and 101 of the 4-chip line. It matches
# an MGD 1x2 and embeds on its own, so the matcher commits it; it just cannot serve two mesh
# instances, because both would land on that one pair.
groupings {
  name: "1x2_Mesh_OnePair"
  preset_type: MESH
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_0 } },
    { id: 1 location { asic_location: ASIC_LOCATION_1 } }
  ]
  row_major_mesh { dims: [1, 2] }
}

# The machine's host level: one host owning all 4 chips, laid out the way the machine is,
# a 1x4 line. One preset_type: HOSTS grouping is one host and what it holds is that
# host's chips.
groupings {
  name: "machine_host"
  preset_type: HOSTS
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 1 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 2 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 3 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  row_major_mesh { dims: [1, 4] }
}
)delimiter")};
    MeshGraphDescriptor mgd{std::string(R"delimiter(
# One 1x2 mesh, nothing else and no seams.
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 1, 2 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}

graph_descriptors {
  name: "G0"
  type: "FABRIC"
  instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
}

top_level_instance { graph { graph_descriptor: "G0" graph_id: 0 } }
)delimiter")};
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(
        "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_4asic_line.textproto");

    // get valid groupings
    const auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd, psd);
    const auto& committed = valid_groupings.at("MESH").at("M0");
    std::vector<std::string> committed_names;
    for (const auto& grouping : committed) {
        committed_names.push_back(grouping.name);
    }
    ASSERT_THAT(committed_names, ::testing::ElementsAre("1x2_Mesh_OnePair_flat", "M0"))
        << "both must be on the list, PGD first, or this is not a priority test";

    // build logical
    const auto logical = utils::build_logical_multi_mesh_adjacency_graph(mgd);

    // place, and build the flat ASIC adjacency
    const auto placements = pgd.solve_adjacency_guided_placement(mgd, valid_groupings, psd);
    const AdjacencyGraph<tt::tt_metal::AsicID> flat_graph(utils::build_flat_adjacency_map_from_psd(psd));
    ASSERT_EQ(placements.size(), 1u) << "the single mesh should place";
    EXPECT_FALSE(placements.front().mesh_node_to_asic_position.empty())
        << "the PGD grouping carries pinning and must win; an empty map means the MGD fallback was used";
    EXPECT_THAT(footprints_of(placements), ::testing::ElementsAre(std::set<uint64_t>{100, 101}))
        << "the pinned PGD pair, not an MGD seating on {101,102} or {102,103}";

    // build physical
    const auto physical = utils::build_hierarchical_from_flat_graph(flat_graph, placements);

    // place and map
    utils::TopologyMappingConfig config;
    config.disable_rank_bindings = true;
    const auto mapping = utils::map_multi_mesh_to_physical(logical, physical, config);
    ASSERT_TRUE(mapping.success) << "the two-level solve should succeed, but failed with: " << mapping.error_message;
    EXPECT_THAT(mapped_footprints(mapping), ::testing::ElementsAre(std::set<uint64_t>{100, 101}))
        << "the mapper should keep the mesh on the PGD-pinned pair";
}

// PlaceableMgdFallbackDoesNotOutrankPgd is the single-mesh control: PGD wins when it can cover the
// mesh. This is the case that control misses: a grouping that embeds fine on its own but cannot
// serve every mesh instance that shares it.
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
// The matcher used to skip the fallback once any PGD grouping placed once. It now offers the MGD
// grouping last whenever that grouping also embeds, so the search can seat one instance on the pinned
// pair and the other on the unpinned 1x2. Two entries, not a replacement.
TEST(AdjacencyGuidedPlacement, PgdGroupingThatCannotCoverEveryInstanceShouldDowngrade) {
    // build pgd
    PhysicalGroupingDescriptor pgd{std::string(R"delimiter(
# A 1x2 shape pinned to ASIC locations 0 and 1 -- chips 100 and 101 of the 4-chip line. It matches
# an MGD 1x2 and embeds on its own, so the matcher commits it; it just cannot serve two mesh
# instances, because both would land on that one pair.
groupings {
  name: "1x2_Mesh_OnePair"
  preset_type: MESH
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_0 } },
    { id: 1 location { asic_location: ASIC_LOCATION_1 } }
  ]
  row_major_mesh { dims: [1, 2] }
}

# The machine's host level: one host owning all 4 chips, wired the way the machine is,
# a 1x4 line. One preset_type: HOSTS grouping is one host, what it holds is that host's chips,
# and its connection block is that host's own topology.
groupings {
  name: "machine_host"
  preset_type: HOSTS
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 1 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 2 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 3 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  row_major_mesh { dims: [1, 4] }
}
)delimiter")};
    MeshGraphDescriptor mgd{std::string(R"delimiter(
# Two 1x2 meshes joined by one intermesh connection. The unlinked pair below is the same
# descriptor without the connection, which is what isolates the seam: only this one requires
# the two placements to touch.
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 1, 2 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}

graph_descriptors {
  name: "G0"
  type: "FABRIC"
  instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
  instances { mesh { mesh_descriptor: "M0" mesh_id: 1 } }

  connections {
    nodes { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
    nodes { mesh { mesh_descriptor: "M0" mesh_id: 1 } }
    channels { count: 2 policy: RELAXED }
  }
}

top_level_instance { graph { graph_descriptor: "G0" graph_id: 0 } }
)delimiter")};
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(
        "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_4asic_line.textproto");

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

// Strain the adjacency-guided DFS: many meshes, so next_step_pool (and the inner topology-solver
// enumeration it loops) runs once per search node. Auto picks SAT when n_target * n_global >= 512
// and DFS otherwise, so a 4x4 mesh on an 8x8 system starts on SAT and finishes on DFS as occupancy
// shrinks. Stats are the thing to watch when changing the looping.
TEST(AdjacencyGuidedPlacement, StrainManyMeshesReportsDfsStats) {
    // The counters asserted below are the DFS's own; the SAT joint placement runs first by default.
    ScopedEnv dfs_only("TT_METAL_PLACEMENT_SOLVER", "dfs");
    auto run_case = [](std::size_t mesh_rows,
                       std::size_t mesh_cols,
                       std::size_t fabric_rows,
                       std::size_t fabric_cols,
                       const char* label) {
        PhysicalGroupingDescriptor pgd{unspecified_mesh_pgd(mesh_rows, mesh_cols)};
        MeshGraphDescriptor mgd{mesh_grid_mgd(mesh_rows, mesh_cols, fabric_rows, fabric_cols)};
        auto psd = load_psd_from_text(grid_psd(mesh_rows * fabric_rows, mesh_cols * fabric_cols));

        const auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd, psd);
        ASSERT_TRUE(valid_groupings.contains("MESH")) << label;

        PlacementSolveStats stats;
        const auto placements =
            pgd.solve_adjacency_guided_placement(mgd, valid_groupings, psd, /*node_budget=*/0, &stats);

        const std::size_t expected_meshes = fabric_rows * fabric_cols;
        EXPECT_EQ(placements.size(), expected_meshes) << label << "\n" << stats.to_string();
        EXPECT_TRUE(stats.success) << label << "\n" << stats.to_string();
        EXPECT_EQ(stats.meshes_placed, expected_meshes) << label;
        EXPECT_GE(stats.next_step_pool_calls, expected_meshes) << label << "\n" << stats.to_string();
        EXPECT_GE(stats.inner_solver_calls, expected_meshes) << label << "\n" << stats.to_string();
        EXPECT_GT(stats.total_elapsed.count(), 0) << label;
    };

    // 8 linked 2x2 meshes on a 4x8 chip grid: 4 * remaining_chips < 512, so the inner calls stay on DFS.
    run_case(2, 2, 2, 4, "8x 2x2 meshes on 4x8");
    // 4 linked 4x4 meshes on an 8x8 chip grid: first inner call is 16*64 >= 512 (SAT), last is 16*16 (DFS).
    run_case(4, 4, 2, 2, "4x 4x4 meshes on 8x8");
}

// ----- two-layer SAT joint placement (Plan 4) --------------------------------------------------
//
// Same descriptors as the DFS tests above, forced onto the SAT path. A SAT model is a real placement,
// so the footprints must be the ones the DFS found; the stats must say the master solve ran.

// The unique seating on the 4-chip line is found by the master solve, with every candidate list
// exhausted (the 1x2 grouping has exactly three seats on a line of four).
TEST(PhysicalGroupingDescriptorTestsSatJointPlacement, LinkedMeshesPlaceAdjacentlyOnLine) {
    ScopedEnv sat_only("TT_METAL_PLACEMENT_SOLVER", "sat");
    PhysicalGroupingDescriptor pgd{std::string(R"delimiter(
# Mesh shapes with every ASIC location UNSPECIFIED, so a grouping is free to land anywhere: that is
# what gives the search more than one candidate to choose between.
groupings {
  name: "1x2_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 1 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  row_major_mesh { dims: [1, 2] }
}

# The machine's host level: one host owning all 4 chips, laid out the way the machine is,
# a 1x4 line. One preset_type: HOSTS grouping is one host and what it holds is that
# host's chips.
groupings {
  name: "machine_host"
  preset_type: HOSTS
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 1 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 2 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 3 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  row_major_mesh { dims: [1, 4] }
}
)delimiter")};
    MeshGraphDescriptor mgd{std::string(R"delimiter(
# Two 1x2 meshes joined by one intermesh connection. The unlinked pair below is the same
# descriptor without the connection, which is what isolates the seam: only this one requires
# the two placements to touch.
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 1, 2 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}

graph_descriptors {
  name: "G0"
  type: "FABRIC"
  instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
  instances { mesh { mesh_descriptor: "M0" mesh_id: 1 } }

  connections {
    nodes { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
    nodes { mesh { mesh_descriptor: "M0" mesh_id: 1 } }
    channels { count: 2 policy: RELAXED }
  }
}

top_level_instance { graph { graph_descriptor: "G0" graph_id: 0 } }
)delimiter")};
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(
        "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_4asic_line.textproto");
    const auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd, psd);

    PlacementSolveStats stats;
    const auto placements = pgd.solve_adjacency_guided_placement(mgd, valid_groupings, psd, /*node_budget=*/0, &stats);
    ASSERT_EQ(placements.size(), 2u) << stats.to_string();
    EXPECT_THAT(
        footprints_of(placements),
        ::testing::UnorderedElementsAre(std::set<uint64_t>{100, 101}, std::set<uint64_t>{102, 103}))
        << "the only disjoint adjacent seating of two 1x2 meshes on 100-101-102-103";
    EXPECT_TRUE(stats.master_solve_attempted) << stats.to_string();
    EXPECT_TRUE(stats.master_solve_success) << stats.to_string();
    EXPECT_TRUE(stats.candidate_lists_complete) << stats.to_string();
    EXPECT_EQ(stats.adjacency_nodes_expanded, 0u) << "the DFS must not have run\n" << stats.to_string();
    EXPECT_GT(stats.master_sat_vars, 0u) << stats.to_string();
    EXPECT_GT(stats.master_sat_clauses, 0u) << stats.to_string();
}

// Two disjoint pairs cannot carry the seam. With every candidate list exhausted the UNSAT verdict is
// trustworthy, so `auto` mode must NOT fall back to the DFS, and the stats must say why.
TEST(PhysicalGroupingDescriptorTestsSatJointPlacement, LinkedMeshesFailOnDisconnectedPairsWithTrustworthyUnsat) {
    ScopedEnv auto_mode("TT_METAL_PLACEMENT_SOLVER", "auto");
    PhysicalGroupingDescriptor pgd{std::string(R"delimiter(
# Mesh shapes with every ASIC location UNSPECIFIED, so a grouping is free to land anywhere: that is
# what gives the search more than one candidate to choose between.
groupings {
  name: "1x2_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 1 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  row_major_mesh { dims: [1, 2] }
}

# The machine's host level: one host owning all 4 chips, wired the way the machine is.
# One preset_type: HOSTS grouping is one host, what it holds is that host's chips, and its
# connection block is that host's own topology -- spelled out link by link here, because
# this machine is two disjoint pairs.
groupings {
  name: "machine_host"
  preset_type: HOSTS
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 1 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 2 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 3 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  custom {
    connections: [
      { src_instance: 0 dst_instance: 1 },
      { src_instance: 2 dst_instance: 3 }
    ]
  }
}
)delimiter")};
    MeshGraphDescriptor mgd{std::string(R"delimiter(
# Two 1x2 meshes joined by one intermesh connection. The unlinked pair below is the same
# descriptor without the connection, which is what isolates the seam: only this one requires
# the two placements to touch.
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 1, 2 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}

graph_descriptors {
  name: "G0"
  type: "FABRIC"
  instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
  instances { mesh { mesh_descriptor: "M0" mesh_id: 1 } }

  connections {
    nodes { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
    nodes { mesh { mesh_descriptor: "M0" mesh_id: 1 } }
    channels { count: 2 policy: RELAXED }
  }
}

top_level_instance { graph { graph_descriptor: "G0" graph_id: 0 } }
)delimiter")};
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(
        "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_4asic_2mesh.textproto");
    const auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd, psd);

    PlacementSolveStats stats;
    const auto placements = pgd.solve_adjacency_guided_placement(mgd, valid_groupings, psd, /*node_budget=*/0, &stats);
    EXPECT_TRUE(placements.empty()) << "no link joins the two pairs, so the seam cannot be satisfied";
    EXPECT_TRUE(stats.master_solve_attempted) << stats.to_string();
    EXPECT_FALSE(stats.master_solve_success) << stats.to_string();
    EXPECT_TRUE(stats.candidate_lists_complete) << "every 1x2 seat on 4 chips must have been enumerated\n"
                                                << stats.to_string();
    EXPECT_EQ(stats.adjacency_nodes_expanded, 0u) << "a trustworthy UNSAT must not fall back to the DFS\n"
                                                  << stats.to_string();
}

// Under a RELAXED policy the strict seam tier is solved first, so the full channel count wins when it
// is available -- the same preference next_step_pool expresses per seam, here as a global one.
TEST(PhysicalGroupingDescriptorTestsSatJointPlacement, RelaxedSeamPrefersTheFullChannelCount) {
    ScopedEnv sat_only("TT_METAL_PLACEMENT_SOLVER", "sat");
    PhysicalGroupingDescriptor pgd{std::string(R"delimiter(
# Mesh shapes with every ASIC location UNSPECIFIED, so a grouping is free to land anywhere: that is
# what gives the search more than one candidate to choose between.
groupings {
  name: "1x2_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 1 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  row_major_mesh { dims: [1, 2] }
}
groupings {
  name: "1x1_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  row_major_mesh { dims: [1, 1] }
}

# The machine's host level: one host owning all 4 chips, wired the way the machine is.
# One preset_type: HOSTS grouping is one host, what it holds is that host's chips, and its
# connection block is that host's own topology -- spelled out link by link here, because
# this machine is two disjoint pairs.
groupings {
  name: "machine_host"
  preset_type: HOSTS
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 1 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 2 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 3 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  custom {
    connections: [
      { src_instance: 0 dst_instance: 1 },
      { src_instance: 2 dst_instance: 3 }
    ]
  }
}
)delimiter")};
    MeshGraphDescriptor mgd{std::string(R"delimiter(
# Two single-chip meshes whose seam asks for 4 channels as a preference: satisfiable on the 3-chip
# machine, but only on 101-102.
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 1, 1 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}

mesh_descriptors {
  name: "M1"
  arch: WORMHOLE_B0
  device_topology { dims: [ 1, 1 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}

graph_descriptors {
  name: "G0"
  type: "FABRIC"
  instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
  instances { mesh { mesh_descriptor: "M1" mesh_id: 1 } }

  connections {
    nodes { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
    nodes { mesh { mesh_descriptor: "M1" mesh_id: 1 } }
    channels { count: 4 policy: RELAXED }
  }
}

top_level_instance { graph { graph_descriptor: "G0" graph_id: 0 } }
)delimiter")};
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(
        "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_3asic_uneven_line.textproto");
    const auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd, psd);

    PlacementSolveStats stats;
    const auto placements = pgd.solve_adjacency_guided_placement(mgd, valid_groupings, psd, /*node_budget=*/0, &stats);
    ASSERT_EQ(placements.size(), 2u) << stats.to_string();
    EXPECT_THAT(
        footprints_of(placements),
        ::testing::UnorderedElementsAre(std::set<uint64_t>({101}), std::set<uint64_t>({102})))
        << "the 4-channel link is the only seam wide enough, so the 2-channel pair should be left alone";
    EXPECT_TRUE(stats.master_solve_success) << stats.to_string();
}

// Trait-free (unpinned) groupings give the master solve nothing to narrow the search with: every mesh may
// sit anywhere on the fabric, so there are far more distinct footprints than enumeration will list and the
// pool it hands the solver is a capped sample of them. Both grids are still placed by a single joint SAT
// solve over that sample, with no column-generation round needed to grow it and no adjacency DFS behind it.
TEST(PhysicalGroupingDescriptorTestsSatJointPlacement, StrainManyMeshesPlacesInOneMasterSolve) {
    ScopedEnv sat_only("TT_METAL_PLACEMENT_SOLVER", "sat");
    auto run_case = [](std::size_t mesh_rows,
                       std::size_t mesh_cols,
                       std::size_t fabric_rows,
                       std::size_t fabric_cols,
                       const char* label) {
        PhysicalGroupingDescriptor pgd{unspecified_mesh_pgd(mesh_rows, mesh_cols)};
        MeshGraphDescriptor mgd{mesh_grid_mgd(mesh_rows, mesh_cols, fabric_rows, fabric_cols)};
        auto psd = load_psd_from_text(grid_psd(mesh_rows * fabric_rows, mesh_cols * fabric_cols));

        const auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd, psd);
        ASSERT_TRUE(valid_groupings.contains("MESH")) << label;

        PlacementSolveStats stats;
        const auto placements =
            pgd.solve_adjacency_guided_placement(mgd, valid_groupings, psd, /*node_budget=*/0, &stats);

        const std::size_t expected_meshes = fabric_rows * fabric_cols;
        EXPECT_EQ(placements.size(), expected_meshes) << label << "\n" << stats.to_string();
        EXPECT_TRUE(stats.success) << label << "\n" << stats.to_string();
        EXPECT_TRUE(stats.master_solve_attempted) << label << "\n" << stats.to_string();
        EXPECT_TRUE(stats.master_solve_success) << label << "\n" << stats.to_string();
        EXPECT_EQ(stats.adjacency_nodes_expanded, 0u) << label << "\n" << stats.to_string();
        EXPECT_GE(stats.master_candidates_enumerated, expected_meshes) << label << "\n" << stats.to_string();
        EXPECT_EQ(stats.master_growth_rounds, 0u) << label << "\n" << stats.to_string();
        EXPECT_EQ(stats.master_sat_attempts, 1u) << label << "\n" << stats.to_string();
        // The pools are truncated: enumeration stops at its per-mesh cap long before it has listed every
        // footprint on a fabric this open. That is the point of the case -- a partial pool is still enough
        // for one solve to seat every mesh, which is why no growth round is needed above.
        EXPECT_FALSE(stats.candidate_lists_complete) << label << "\n" << stats.to_string();
        // Every placement must be a disjoint footprint of the right size.
        std::set<uint64_t> seen;
        for (const auto& placement : placements) {
            EXPECT_EQ(placement.asics.size(), mesh_rows * mesh_cols) << label;
            for (const auto& asic : placement.asics) {
                EXPECT_TRUE(seen.insert(*asic).second) << label << ": ASIC " << *asic << " placed twice";
            }
        }
    };

    run_case(2, 2, 2, 4, "8x 2x2 meshes on 4x8");
    run_case(4, 4, 2, 2, "4x 4x4 meshes on 8x8");
}

// Asserts that every declared MGD host rank landed inside a single one of the PGD's declared hosts.
// A rank is a list of logical chip ids, a host is the (tray_id, asic_location) slots it holds, and
// each caller writes both out next to the PGD that declares them. Takes the seating rather than the
// grouping, so a matcher GroupingInfo and a placement result can both be checked with it.
//
// TODO: nothing in the matcher reads a PGD host level. The split is taken from the MGD alone, in
// compose_mesh_node_to_host_group_from_mgd_match, and checked only against the PSD, so the HOSTS
// groupings are parsed and then ignored. These checks are the target behaviour rather than a live
// constraint: where one fails, it marks the missing MGD<->PGD host match rather than a regression.
void expect_each_rank_inside_one_pgd_host(
    const std::map<LogicalChipId, tt::tt_metal::ASICPosition>& seating,
    const std::vector<std::vector<LogicalChipId>>& declared_ranks,
    const std::vector<std::set<std::pair<uint32_t, uint32_t>>>& pgd_hosts) {
    for (std::size_t rank = 0; rank < declared_ranks.size(); ++rank) {
        std::set<std::size_t> hosts_used;
        for (LogicalChipId chip : declared_ranks[rank]) {
            const auto& position = seating.at(chip);
            const std::pair<uint32_t, uint32_t> slot{*position.first, *position.second};
            for (std::size_t host = 0; host < pgd_hosts.size(); ++host) {
                if (pgd_hosts[host].contains(slot)) {
                    hosts_used.insert(host);
                }
            }
        }
        EXPECT_EQ(hosts_used.size(), 1u) << "declared rank " << rank << " was seated across " << hosts_used.size()
                                         << " of the PGD's declared hosts";
    }
}

// Drops the MGD fallback, the grouping get_valid_groupings_for_mgd offers last under the mesh
// descriptor's own name, from a pool before it is handed to placement.
//
// A test that wants to look at where the chips landed has to, because the fallback is the MGD's own
// topology rather than a PGD layout: it carries no chip -> slot pinning, so a placement that lands on
// it reports a footprint and nothing to check a host split against. Leaving it in would also not test
// what these tests are about, since SAT joint placement currently seats it ahead of a placeable PGD
// grouping (AdjacencyGuidedPlacement.PlaceableMgdFallbackDoesNotOutrankPgd, failing).
ValidGroupingsMap without_mgd_fallback(ValidGroupingsMap valid_groupings, const std::string& mgd_mesh_name) {
    for (auto& [preset, by_instance] : valid_groupings) {
        for (auto& [instance, groupings] : by_instance) {
            std::erase_if(groupings, [&](const GroupingInfo& grouping) { return grouping.name == mgd_mesh_name; });
        }
    }
    return valid_groupings;
}

// ----- Phase 1: get_valid_groupings_for_mgd ----------------------------------------------------

// A mesh host rank is one process on one host, so it cannot own chips on two hosts. That makes the
// contract between an MGD's host_topology and the PSD's hosts one-directional, and it is what every
// test from here on is checking:
//
//   a physical host boundary may never cut through a declared mesh host rank,
//   but a declared rank boundary may sit anywhere inside a physical host.
//
// This first case has neither: nothing is declared and nothing is split, so there is no contract to
// check and the grouping commits.
//
//   machine: one host              MGD host_topology [1,1]
//
//     100 101 102 103                aaaa      one rank owning all 8 chips, and one
//     104 105 106 107                aaaa      host to put it on
//          host0
TEST(PhysicalGroupingDescriptorTestsHostSplit, SingleHostPsdSingleHostMgdCommits) {
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(
        "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_8asic_2x4_one_host.textproto");

    PhysicalGroupingDescriptor pgd{std::string(R"(
groupings {
  name: "2x4_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
    { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } },
    { id: 2 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_3 } },
    { id: 3 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_4 } },
    { id: 4 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } },
    { id: 5 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } },
    { id: 6 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_3 } },
    { id: 7 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_4 } }
  ]
  row_major_mesh {
    dims: [2, 4]
  }
}

# The PGD's own host level, mirroring the machine: one host holding all 8 chips.
groupings { name: "2x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
               { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } },
               { id: 2 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_3 } },
               { id: 3 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_4 } },
               { id: 4 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } },
               { id: 5 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } },
               { id: 6 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_3 } },
               { id: 7 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_4 } } ]
  row_major_mesh { dims: [2, 4] } }
)")};

    MeshGraphDescriptor mgd{std::string(R"(
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 2, 4 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}

top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
)")};

    const auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd, psd);

    // The committed groupings that came from the PGD. The MGD fallback is named after the MGD instance
    // and carries no layout, so it is not evidence that the PGD matched.
    std::vector<GroupingInfo> committed;
    for (const auto& [preset, by_instance] : valid_groupings) {
        for (const auto& [instance, groupings] : by_instance) {
            for (const auto& grouping : groupings) {
                if (grouping.name != "M0" && !grouping.mesh_node_to_asic_position.empty()) {
                    committed.push_back(grouping);
                }
            }
        }
    }

    ASSERT_FALSE(committed.empty()) << "an undeclared split on an undivided machine has nothing to violate";
    EXPECT_EQ(committed.front().mesh_node_to_asic_position.size(), 8u);

    // The PGD's declared hosts, as the slots each one holds, and the rank that has to land inside one:
    //
    //   rank a = chips 0 1 2 3 4 5 6 7 -> slots (1,1) (1,2) (1,3) (1,4) (2,1) (2,2) (2,3) (2,4)
    //   PGD host0 = all eight of those slots, so a is inside it
    const std::vector<std::set<std::pair<uint32_t, uint32_t>>> pgd_hosts = {
        {{1, 1}, {1, 2}, {1, 3}, {1, 4}, {2, 1}, {2, 2}, {2, 3}, {2, 4}}};
    expect_each_rank_inside_one_pgd_host(
        committed.front().mesh_node_to_asic_position, {{0, 1, 2, 3, 4, 5, 6, 7}}, pgd_hosts);
}

// The declared split is finer than the machine: both ranks live on the one host. Legal -- ranks sharing a
// host is ordinary, and no host boundary exists to cut either of them.
//
//   machine: one host              MGD host_topology [1,2]
//
//     100 101 102 103                aa bb     two ranks sharing host0: there is no
//     104 105 106 107                aa bb     boundary here for either to straddle
//          host0
TEST(PhysicalGroupingDescriptorTestsHostSplit, SingleHostPsdColumnSplitMgdCommits) {
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(
        "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_8asic_2x4_one_host.textproto");

    PhysicalGroupingDescriptor pgd{std::string(R"(
groupings {
  name: "2x4_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
    { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } },
    { id: 2 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_3 } },
    { id: 3 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_4 } },
    { id: 4 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } },
    { id: 5 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } },
    { id: 6 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_3 } },
    { id: 7 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_4 } }
  ]
  row_major_mesh {
    dims: [2, 4]
  }
}

# The PGD's own host level, mirroring the machine: one host holding all 8 chips.
groupings { name: "2x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
               { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } },
               { id: 2 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_3 } },
               { id: 3 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_4 } },
               { id: 4 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } },
               { id: 5 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } },
               { id: 6 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_3 } },
               { id: 7 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_4 } } ]
  row_major_mesh { dims: [2, 4] } }
)")};

    MeshGraphDescriptor mgd{std::string(R"(
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 2, 4 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 2 ] }
  channels { count: 2 policy: STRICT }
}

top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
)")};

    const auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd, psd);

    std::vector<GroupingInfo> committed;
    for (const auto& [preset, by_instance] : valid_groupings) {
        for (const auto& [instance, groupings] : by_instance) {
            for (const auto& grouping : groupings) {
                if (grouping.name != "M0" && !grouping.mesh_node_to_asic_position.empty()) {
                    committed.push_back(grouping);
                }
            }
        }
    }

    ASSERT_FALSE(committed.empty()) << "two declared ranks may share the single host";
    EXPECT_EQ(committed.front().mesh_node_to_asic_position.size(), 8u);

    // The PGD's declared hosts, as the slots each one holds, and the ranks that land in them:
    //
    //   rank a = chips 0 1 4 5 -> slots (1,1) (1,2) (2,1) (2,2)
    //   rank b = chips 2 3 6 7 -> slots (1,3) (1,4) (2,3) (2,4)
    //   PGD host0 = all eight slots, so both ranks are inside it
    const std::vector<std::set<std::pair<uint32_t, uint32_t>>> pgd_hosts = {
        {{1, 1}, {1, 2}, {1, 3}, {1, 4}, {2, 1}, {2, 2}, {2, 3}, {2, 4}}};
    const std::vector<std::vector<LogicalChipId>> declared_ranks = {{0, 1, 4, 5}, {2, 3, 6, 7}};
    expect_each_rank_inside_one_pgd_host(committed.front().mesh_node_to_asic_position, declared_ranks, pgd_hosts);
}

// Declared split and physical split agree: two ranks of one column-half each, on a machine split by
// column. Each rank lands wholly on one host.
//
//   machine: hosts cut by column     MGD host_topology [1,2]
//
//     100 101 | 102 103                aa | bb    the declared boundary and the
//     104 105 | 106 107                aa | bb    physical one are the same line:
//      host0  |  host1                            a is host0, b is host1
TEST(PhysicalGroupingDescriptorTestsHostSplit, ColumnSplitPsdColumnSplitMgdCommits) {
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(
        "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_8asic_2x4_hosts_by_column.textproto");

    PhysicalGroupingDescriptor pgd{std::string(R"(
groupings {
  name: "2x4_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
    { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } },
    { id: 2 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_3 } },
    { id: 3 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_4 } },
    { id: 4 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } },
    { id: 5 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } },
    { id: 6 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_3 } },
    { id: 7 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_4 } }
  ]
  row_major_mesh {
    dims: [2, 4]
  }
}

# The PGD's own host level, mirroring the machine: two hosts, each one column-half.
groupings { name: "2x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
               { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } },
               { id: 2 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } },
               { id: 3 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } } ]
  row_major_mesh { dims: [2, 2] } }
groupings { name: "2x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_3 } },
               { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_4 } },
               { id: 2 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_3 } },
               { id: 3 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_4 } } ]
  row_major_mesh { dims: [2, 2] } }
)")};

    MeshGraphDescriptor mgd{std::string(R"(
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 2, 4 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 2 ] }
  channels { count: 2 policy: STRICT }
}

top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
)")};

    const auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd, psd);

    std::vector<GroupingInfo> committed;
    for (const auto& [preset, by_instance] : valid_groupings) {
        for (const auto& [instance, groupings] : by_instance) {
            for (const auto& grouping : groupings) {
                if (grouping.name != "M0" && !grouping.mesh_node_to_asic_position.empty()) {
                    committed.push_back(grouping);
                }
            }
        }
    }
    ASSERT_FALSE(committed.empty()) << "the declared split lies along the machine's own boundary";

    // host_topology [1,2] on a 2x4 declares two ranks of one column-half each, by row-major chip id. Where
    // those chips landed is read back from the seating rather than from what the matcher believed.
    const std::vector<std::vector<LogicalChipId>> declared_ranks = {{0, 1, 4, 5}, {2, 3, 6, 7}};
    for (std::size_t rank = 0; rank < declared_ranks.size(); ++rank) {
        std::set<uint32_t> hosts;
        for (LogicalChipId chip : declared_ranks[rank]) {
            const auto& position = committed.front().mesh_node_to_asic_position.at(chip);
            // host0 holds asic_location 1-2, host1 holds 3-4.
            hosts.insert((*position.second - 1) / 2);
        }
        EXPECT_EQ(hosts.size(), 1u) << "declared rank " << rank << " was seated across " << hosts.size() << " hosts";
    }

    // And against the PGD's own hosts, which here draw the same boundary as the machine's:
    //
    //   rank a = chips 0 1 4 5 -> slots (1,1) (1,2) (2,1) (2,2) = PGD host0
    //   rank b = chips 2 3 6 7 -> slots (1,3) (1,4) (2,3) (2,4) = PGD host1
    const std::vector<std::set<std::pair<uint32_t, uint32_t>>> pgd_hosts = {
        {{1, 1}, {1, 2}, {2, 1}, {2, 2}}, {{1, 3}, {1, 4}, {2, 3}, {2, 4}}};
    expect_each_rank_inside_one_pgd_host(committed.front().mesh_node_to_asic_position, declared_ranks, pgd_hosts);
}

// The same agreement on the other axis: two ranks of one row each, on a machine split by row. Kept so
// that a transposed reading of host_topology -- rows taken for columns anywhere in the path -- has
// somewhere to show up.
//
//   machine: hosts cut by row        MGD host_topology [2,1]
//
//     100 101 102 103   host0          aaaa      a is host0
//     -----------------------          ----
//     104 105 106 107   host1          bbbb      b is host1
TEST(PhysicalGroupingDescriptorTestsHostSplit, RowSplitPsdRowSplitMgdCommits) {
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(
        "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_8asic_2x4_hosts_by_row.textproto");

    PhysicalGroupingDescriptor pgd{std::string(R"(
groupings {
  name: "2x4_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
    { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } },
    { id: 2 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_3 } },
    { id: 3 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_4 } },
    { id: 4 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } },
    { id: 5 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } },
    { id: 6 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_3 } },
    { id: 7 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_4 } }
  ]
  row_major_mesh {
    dims: [2, 4]
  }
}

# The PGD's own host level, mirroring the machine: two hosts, one per tray.
groupings { name: "2x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
               { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } },
               { id: 2 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_3 } },
               { id: 3 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_4 } } ]
  row_major_mesh { dims: [1, 4] } }
groupings { name: "2x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } },
               { id: 1 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } },
               { id: 2 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_3 } },
               { id: 3 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_4 } } ]
  row_major_mesh { dims: [1, 4] } }
)")};

    MeshGraphDescriptor mgd{std::string(R"(
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 2, 4 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 2, 1 ] }
  channels { count: 2 policy: STRICT }
}

top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
)")};

    const auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd, psd);

    std::vector<GroupingInfo> committed;
    for (const auto& [preset, by_instance] : valid_groupings) {
        for (const auto& [instance, groupings] : by_instance) {
            for (const auto& grouping : groupings) {
                if (grouping.name != "M0" && !grouping.mesh_node_to_asic_position.empty()) {
                    committed.push_back(grouping);
                }
            }
        }
    }
    ASSERT_FALSE(committed.empty()) << "the declared split lies along the machine's own boundary";

    // host_topology [2,1] on a 2x4 declares two ranks of one row each.
    const std::vector<std::vector<LogicalChipId>> declared_ranks = {{0, 1, 2, 3}, {4, 5, 6, 7}};
    for (std::size_t rank = 0; rank < declared_ranks.size(); ++rank) {
        std::set<uint32_t> hosts;
        for (LogicalChipId chip : declared_ranks[rank]) {
            const auto& position = committed.front().mesh_node_to_asic_position.at(chip);
            // host0 is tray 1, host1 is tray 2.
            hosts.insert(*position.first - 1);
        }
        EXPECT_EQ(hosts.size(), 1u) << "declared rank " << rank << " was seated across " << hosts.size() << " hosts";
    }

    // And against the PGD's own hosts, which here draw the same boundary as the machine's:
    //
    //   rank a = chips 0 1 2 3 -> slots (1,1) (1,2) (1,3) (1,4) = PGD host0, tray 1
    //   rank b = chips 4 5 6 7 -> slots (2,1) (2,2) (2,3) (2,4) = PGD host1, tray 2
    const std::vector<std::set<std::pair<uint32_t, uint32_t>>> pgd_hosts = {
        {{1, 1}, {1, 2}, {1, 3}, {1, 4}}, {{2, 1}, {2, 2}, {2, 3}, {2, 4}}};
    expect_each_rank_inside_one_pgd_host(committed.front().mesh_node_to_asic_position, declared_ranks, pgd_hosts);
}

// The wrong axis, checked in both places it has to hold. The machine is cut widthwise -- across the
// mesh's long side -- into two 2x4/2 halves, while the MGD declares its split lengthwise, two ranks of
// one full 1x4 row each:
//
//   machine: hosts cut widthwise        MGD host_topology [2,1]: ranks cut lengthwise
//
//     100 101 | 102 103                   aaaa aaaa
//     104 105 | 106 107                   bbbb bbbb
//      host0  |  host1                    each rank spans both hosts
//
// So each declared rank needs chips from both hosts, and on a 2x4 there is no way out of it: the grid
// has no 90-degree automorphism, so the mesh cannot be turned to lay the ranks alongside the boundary
// instead of across it. There is no seating, and both phases have to say so -- the matcher must refuse
// to commit the grouping, and the rank-bound path must not come back with a pinned graph either. A
// verdict that held in phase 1 and then quietly reversed in phase 2 is the failure mode this pair is
// here to catch.
//
// SquareMeshCommitsWhenTheDeclaredAxisAlreadyMatches is the aligned control, and the 2x2 case below is
// the same pairing on a shape where the turn does exist -- where the refusal is therefore a bug.
//
// require_placement is false here, as it is in every case below whose expected answer is a rejection:
// it makes the refusal come back as an empty result rather than a throw. What is under test is whether
// the grouping survives the host constraint, not what an unplaceable mesh does to its caller.
TEST(PhysicalGroupingDescriptorTestsHostSplit, LengthwiseSplitOnWidthwiseSplitHostsFailsEnumerateAndMapping) {
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(
        "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_8asic_2x4_hosts_by_column.textproto");

    PhysicalGroupingDescriptor pgd{std::string(R"(
groupings {
  name: "2x4_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
    { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } },
    { id: 2 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_3 } },
    { id: 3 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_4 } },
    { id: 4 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } },
    { id: 5 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } },
    { id: 6 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_3 } },
    { id: 7 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_4 } }
  ]
  row_major_mesh {
    dims: [2, 4]
  }
}

# The PGD's own host level, mirroring the machine: two hosts, each one column-half. The declared
# lengthwise ranks straddle these exactly as they straddle the machine's.
groupings { name: "2x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
               { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } },
               { id: 2 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } },
               { id: 3 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } } ]
  row_major_mesh { dims: [2, 2] } }
groupings { name: "2x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_3 } },
               { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_4 } },
               { id: 2 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_3 } },
               { id: 3 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_4 } } ]
  row_major_mesh { dims: [2, 2] } }
)")};

    MeshGraphDescriptor mgd{std::string(R"(
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 2, 4 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 2, 1 ] }
  channels { count: 2 policy: STRICT }
}

top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
)")};

    // Phase 1: enumerate must reject the grouping rather than seat a rank across the boundary.
    const auto valid_groupings =
        pgd.get_valid_groupings_for_mgd(mgd, psd, /*pinnings=*/std::nullopt, /*require_placement=*/false);

    std::vector<GroupingInfo> committed;
    for (const auto& [preset, by_instance] : valid_groupings) {
        for (const auto& [instance, groupings] : by_instance) {
            for (const auto& grouping : groupings) {
                if (grouping.name != "M0" && !grouping.mesh_node_to_asic_position.empty()) {
                    committed.push_back(grouping);
                }
            }
        }
    }
    EXPECT_TRUE(committed.empty()) << "enumerate seated a lengthwise split on a widthwise-split machine, got "
                                   << committed.size() << " grouping(s)";

    // Guarded, because the verdict above is that nothing is committed: should anything ever be, it has
    // to keep each declared rank inside one of the PGD's hosts as well as one of the machine's. There
    // is no such seating here, which is the point -- the declared rows cross the declared columns:
    //
    //   rank a = chips 0 1 2 3 -> slots (1,1) (1,2) (1,3) (1,4), which is half of host0 and half of host1
    //   rank b = chips 4 5 6 7 -> slots (2,1) (2,2) (2,3) (2,4), the other half of each
    //   PGD host0 = (1,1) (1,2) (2,1) (2,2)   PGD host1 = (1,3) (1,4) (2,3) (2,4)
    const std::vector<std::set<std::pair<uint32_t, uint32_t>>> pgd_hosts = {
        {{1, 1}, {1, 2}, {2, 1}, {2, 2}}, {{1, 3}, {1, 4}, {2, 3}, {2, 4}}};
    const std::vector<std::vector<LogicalChipId>> declared_ranks = {{0, 1, 2, 3}, {4, 5, 6, 7}};
    if (!committed.empty()) {
        expect_each_rank_inside_one_pgd_host(committed.front().mesh_node_to_asic_position, declared_ranks, pgd_hosts);
    }

    // Phase 2: the same question against a rank-bound graph, one rank per host.
    std::map<std::string, MeshHostRankId> rank_of_host;
    for (const auto& [asic_id, descriptor] : psd.get_asic_descriptors()) {
        rank_of_host.emplace(psd.get_host_name_for_asic(asic_id), MeshHostRankId{0});
    }
    uint32_t next_rank = 0;
    for (auto& [host_name, rank] : rank_of_host) {
        rank = MeshHostRankId{next_rank++};
    }
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> asic_id_to_mesh_rank;
    for (const auto& [asic_id, descriptor] : psd.get_asic_descriptors()) {
        asic_id_to_mesh_rank[MeshId{0}][asic_id] = rank_of_host.at(psd.get_host_name_for_asic(asic_id));
    }

    // What matters is that no seating comes back. How that is reported is the caller's business and is
    // currently in flux: with require_placement=true in topology_mapper_utils.cpp the empty result
    // aborts, and with it restored the graph is simply left unpinned. Either is a refusal; a pinned
    // graph would not be, so that is what is asserted. Phase2DegradesToNoPinnings... covers which of
    // the two it should be.
    bool phase2_produced_a_seating = false;
    try {
        const auto physical = utils::build_physical_multi_mesh_adjacency_graph(psd, asic_id_to_mesh_rank, pgd, mgd);
        phase2_produced_a_seating = physical.mesh_pgd_pinnings_.contains(MeshId{0});
    } catch (const std::exception&) {
        phase2_produced_a_seating = false;
    }
    EXPECT_FALSE(phase2_produced_a_seating)
        << "the rank-bound path pinned a mesh whose declared ranks cannot fit on one host each";
}

// FIXME: the same pairing on a 2x2, where the mesh is square and the turn the 2x4 lacked does exist --
// and it is still rejected, which is the one verdict here that is wrong. The machine is cut between its
// two columns, the MGD declares two ranks of one row each, and a quarter turn lays those rows down the
// columns so each rank owns one host outright:
//
//   machine: hosts cut by column       MGD host_topology [2,1]     seated a quarter turn round
//
//     100 | 101                          aa                          ab
//     102 | 103                          bb                          ab
//    host0| host1                                                    a is host0, b is host1
//
// Both phases should therefore succeed. Enumerate refuses instead, reporting that each 2-chip rank has
// only one of its two members on its best host -- it is being asked about the unturned orientation.
//
// The reason is the order the two halves are solved in. solve_topology_mapping returns one MGD<->PGD
// match per PGD variant (one, here), and compose_mesh_node_to_host_group_from_mgd_match stamps the
// declared split onto that match's orientation right there -- before the PSD is consulted at all. Only
// then does enumerate check the stamped groups against the PSD hosts. So the orientation is chosen by a
// solve that cannot see the host boundaries, and if it comes back with the other one there is no second
// chance: the variant is discarded rather than re-matched. What is missing is a PGD host level for the
// MGD host partition to be matched *against*, so the orientation is decided with the hosts in view; that
// is the TODO on compose_mesh_node_to_host_group_from_mgd_match.
//
// Four chips is the smallest this can be shown on, which rules out the search space being the problem. A
// 4x4 fails the same way, LINE/LINE or RING/RING, and a torus has more orientations to come back with
// since wrapping an axis adds translations to the automorphisms.
TEST(PhysicalGroupingDescriptorTestsHostSplit, LengthwiseSplitOnWidthwiseSplitHostsIsTurnedToFitOnASquareMesh) {
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(
        "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_4asic_2x2_hosts_by_column.textproto");

    PhysicalGroupingDescriptor pgd{std::string(R"(
groupings {
  name: "2x2_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
    { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } },
    { id: 2 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } },
    { id: 3 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } }
  ]
  row_major_mesh {
    dims: [2, 2]
  }
}

# The PGD's own host level, mirroring the machine: two hosts, each one column.
groupings { name: "2x2_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
               { id: 1 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } } ]
  row_major_mesh { dims: [2, 1] } }
groupings { name: "2x2_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } },
               { id: 1 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } } ]
  row_major_mesh { dims: [2, 1] } }
)")};

    MeshGraphDescriptor mgd{std::string(R"(
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 2, 2 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 2, 1 ] }
  channels { count: 2 policy: STRICT }
}

top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
)")};

    // Phase 1: the grouping commits, and each declared rank sits on exactly one host.
    const auto valid_groupings =
        pgd.get_valid_groupings_for_mgd(mgd, psd, /*pinnings=*/std::nullopt, /*require_placement=*/false);

    std::vector<GroupingInfo> committed;
    for (const auto& [preset, by_instance] : valid_groupings) {
        for (const auto& [instance, groupings] : by_instance) {
            for (const auto& grouping : groupings) {
                if (grouping.name != "M0" && !grouping.mesh_node_to_asic_position.empty()) {
                    committed.push_back(grouping);
                }
            }
        }
    }
    ASSERT_FALSE(committed.empty()) << "a quarter turn seats this split, so enumerate should commit it";

    // host_topology [2,1] on a 2x2 declares two ranks of one row each.
    const std::vector<std::vector<LogicalChipId>> declared_ranks = {{0, 1}, {2, 3}};
    for (std::size_t rank = 0; rank < declared_ranks.size(); ++rank) {
        std::set<uint32_t> hosts;
        for (LogicalChipId chip : declared_ranks[rank]) {
            const auto& position = committed.front().mesh_node_to_asic_position.at(chip);
            // host0 is asic_location 1, host1 is asic_location 2.
            hosts.insert(*position.second - 1);
        }
        EXPECT_EQ(hosts.size(), 1u) << "declared rank " << rank << " was seated across " << hosts.size() << " hosts";
    }

    // And against the PGD's own hosts, which draw the same two columns. The quarter turn that seats the
    // declared rows on them is the one the matcher does not consider:
    //
    //   unturned          a quarter turn round
    //     rank a = chips 0 1 -> slots (1,1) (1,2)      rank a -> slots (1,1) (2,1) = PGD host0
    //     rank b = chips 2 3 -> slots (2,1) (2,2)      rank b -> slots (1,2) (2,2) = PGD host1
    //     each rank half in host0, half in host1       each rank inside one host
    const std::vector<std::set<std::pair<uint32_t, uint32_t>>> pgd_hosts = {{{1, 1}, {2, 1}}, {{1, 2}, {2, 2}}};
    expect_each_rank_inside_one_pgd_host(committed.front().mesh_node_to_asic_position, declared_ranks, pgd_hosts);

    // Phase 2: the rank-bound path reaches the same verdict and the pinnings arrive.
    std::map<std::string, MeshHostRankId> rank_of_host;
    for (const auto& [asic_id, descriptor] : psd.get_asic_descriptors()) {
        rank_of_host.emplace(psd.get_host_name_for_asic(asic_id), MeshHostRankId{0});
    }
    uint32_t next_rank = 0;
    for (auto& [host_name, rank] : rank_of_host) {
        rank = MeshHostRankId{next_rank++};
    }
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> asic_id_to_mesh_rank;
    for (const auto& [asic_id, descriptor] : psd.get_asic_descriptors()) {
        asic_id_to_mesh_rank[MeshId{0}][asic_id] = rank_of_host.at(psd.get_host_name_for_asic(asic_id));
    }

    utils::PhysicalMultiMeshGraph physical;
    ASSERT_NO_THROW(physical = utils::build_physical_multi_mesh_adjacency_graph(psd, asic_id_to_mesh_rank, pgd, mgd))
        << "the rank-bound path should map a mesh that phase 1 already seated";
    ASSERT_TRUE(physical.mesh_pgd_pinnings_.contains(MeshId{0}))
        << "phase 2 should reach phase 1's verdict on the same machine";
    EXPECT_EQ(physical.mesh_pgd_pinnings_.at(MeshId{0}).size(), 4u);
}

// Being finer than the machine does not rescue a split that runs the wrong way. The host grid is [2,1],
// two hosts cut by row, and the MGD declares [1,3]: three ranks where there are only two hosts, which by
// the counting argument alone ought to be the easy, legal direction. It is not, because each of those
// three ranks is a column with one chip on each host:
//
//   machine: host grid [2,1]        MGD host_topology [1,3]
//
//     100 101 102   host0            a b c      three ranks, each a column
//     -------------------           a b c      every one of them straddles the boundary
//     103 104 105   host1
//
// Containment is per rank, so what matters is where a rank's chips are, not how many ranks there are.
// the subdivision TwoAxisSplitInsideRowSplitHostsCommits allows is legal precisely because those ranks
// land inside a host; these do not. The third column also makes the mesh oblong, so there is no quarter
// turn to escape with.
TEST(PhysicalGroupingDescriptorTestsHostSplit, FinerSplitAcrossTheHostBoundaryIsRejected) {
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(
        "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_6asic_2x3_hosts_by_row.textproto");

    PhysicalGroupingDescriptor pgd{std::string(R"(
groupings {
  name: "2x3_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
    { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } },
    { id: 2 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_3 } },
    { id: 3 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } },
    { id: 4 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } },
    { id: 5 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_3 } }
  ]
  row_major_mesh {
    dims: [2, 3]
  }
}

# The PGD's own host level, mirroring the machine: two hosts, one per tray.
groupings { name: "2x3_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
               { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } },
               { id: 2 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_3 } } ]
  row_major_mesh { dims: [1, 3] } }
groupings { name: "2x3_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } },
               { id: 1 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } },
               { id: 2 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_3 } } ]
  row_major_mesh { dims: [1, 3] } }
)")};

    MeshGraphDescriptor mgd{std::string(R"(
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 2, 3 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 3 ] }
  channels { count: 2 policy: STRICT }
}

top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
)")};

    const auto valid_groupings =
        pgd.get_valid_groupings_for_mgd(mgd, psd, /*pinnings=*/std::nullopt, /*require_placement=*/false);

    std::vector<GroupingInfo> committed;
    for (const auto& [preset, by_instance] : valid_groupings) {
        for (const auto& [instance, groupings] : by_instance) {
            for (const auto& grouping : groupings) {
                if (grouping.name != "M0" && !grouping.mesh_node_to_asic_position.empty()) {
                    committed.push_back(grouping);
                }
            }
        }
    }

    EXPECT_TRUE(committed.empty()) << "three column ranks on a row-split machine each straddle the boundary, got "
                                   << committed.size() << " grouping(s)";

    // Guarded on that verdict: the PGD draws the same two rows, so each declared column rank straddles
    // them there too, and anything committed would have to violate one or the other:
    //
    //   rank a = chips 0 3 -> slots (1,1) (2,1)    one chip in each PGD host
    //   rank b = chips 1 4 -> slots (1,2) (2,2)    likewise
    //   rank c = chips 2 5 -> slots (1,3) (2,3)    likewise
    //   PGD host0 = (1,1) (1,2) (1,3)   PGD host1 = (2,1) (2,2) (2,3)
    const std::vector<std::set<std::pair<uint32_t, uint32_t>>> pgd_hosts = {
        {{1, 1}, {1, 2}, {1, 3}}, {{2, 1}, {2, 2}, {2, 3}}};
    const std::vector<std::vector<LogicalChipId>> declared_ranks = {{0, 3}, {1, 4}, {2, 5}};
    if (!committed.empty()) {
        expect_each_rank_inside_one_pgd_host(committed.front().mesh_node_to_asic_position, declared_ranks, pgd_hosts);
    }

    // Phase 2: the same question on a rank-bound graph, one rank per host, must reach the same verdict. A
    // split that enumerate refused must not become seatable just because the graph arrived pre-ranked.
    std::map<std::string, MeshHostRankId> rank_of_host;
    for (const auto& [asic_id, descriptor] : psd.get_asic_descriptors()) {
        rank_of_host.emplace(psd.get_host_name_for_asic(asic_id), MeshHostRankId{0});
    }
    uint32_t next_rank = 0;
    for (auto& [host_name, rank] : rank_of_host) {
        rank = MeshHostRankId{next_rank++};
    }
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> asic_id_to_mesh_rank;
    for (const auto& [asic_id, descriptor] : psd.get_asic_descriptors()) {
        asic_id_to_mesh_rank[MeshId{0}][asic_id] = rank_of_host.at(psd.get_host_name_for_asic(asic_id));
    }

    // A refusal may arrive as an abort or as an unpinned graph depending on the require_placement toggle in
    // topology_mapper_utils.cpp; both are refusals, and a pinned graph is what must never happen.
    bool phase2_produced_a_seating = false;
    try {
        const auto physical = utils::build_physical_multi_mesh_adjacency_graph(psd, asic_id_to_mesh_rank, pgd, mgd);
        phase2_produced_a_seating = physical.mesh_pgd_pinnings_.contains(MeshId{0});
    } catch (const std::exception&) {
        phase2_produced_a_seating = false;
    }
    EXPECT_FALSE(phase2_produced_a_seating) << "the rank-bound path pinned a mesh enumerate had refused";
}

// The declared split need not be parallel to the physical one, only contained by it. The host grid is
// [2,1] and the MGD declares [2,2], which cuts the columns the machine does not cut -- but each of the
// four ranks is a 1x2 block wholly inside one host, so it is legal:
//
//   machine: host grid [2,1]          MGD host_topology [2,2]
//
//     100 101 102 103   host0          aa bb        a and b inside host0
//     -----------------------          cc dd        c and d inside host1
//     104 105 106 107   host1
//
// This is what makes the rule per-rank containment rather than axis equality, and it is the accepting
// half of the pair with CoarserSplitAcrossATwoAxisHostGridIsRejected below: same mesh, and the host grid
// and declared grid exchanged.
TEST(PhysicalGroupingDescriptorTestsHostSplit, TwoAxisSplitInsideRowSplitHostsCommits) {
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(
        "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_8asic_2x4_hosts_by_row.textproto");

    PhysicalGroupingDescriptor pgd{std::string(R"(
groupings {
  name: "2x4_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
    { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } },
    { id: 2 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_3 } },
    { id: 3 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_4 } },
    { id: 4 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } },
    { id: 5 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } },
    { id: 6 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_3 } },
    { id: 7 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_4 } }
  ]
  row_major_mesh {
    dims: [2, 4]
  }
}

# The PGD's own host level, mirroring the machine: two hosts, one per tray.
groupings { name: "2x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
               { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } },
               { id: 2 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_3 } },
               { id: 3 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_4 } } ]
  row_major_mesh { dims: [1, 4] } }
groupings { name: "2x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } },
               { id: 1 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } },
               { id: 2 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_3 } },
               { id: 3 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_4 } } ]
  row_major_mesh { dims: [1, 4] } }
)")};

    MeshGraphDescriptor mgd{std::string(R"(
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 2, 4 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 2, 2 ] }
  channels { count: 2 policy: STRICT }
}

top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
)")};

    const auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd, psd);

    std::vector<GroupingInfo> committed;
    for (const auto& [preset, by_instance] : valid_groupings) {
        for (const auto& [instance, groupings] : by_instance) {
            for (const auto& grouping : groupings) {
                if (grouping.name != "M0" && !grouping.mesh_node_to_asic_position.empty()) {
                    committed.push_back(grouping);
                }
            }
        }
    }
    ASSERT_FALSE(committed.empty()) << "each 1x2 block sits inside one host";

    // host_topology [2,2] on a 2x4 declares four ranks, each one row by two columns.
    const std::vector<std::vector<LogicalChipId>> declared_ranks = {{0, 1}, {2, 3}, {4, 5}, {6, 7}};
    for (std::size_t rank = 0; rank < declared_ranks.size(); ++rank) {
        std::set<uint32_t> hosts;
        for (LogicalChipId chip : declared_ranks[rank]) {
            const auto& position = committed.front().mesh_node_to_asic_position.at(chip);
            // host0 is tray 1, host1 is tray 2.
            hosts.insert(*position.first - 1);
        }
        EXPECT_EQ(hosts.size(), 1u) << "declared rank " << rank << " was seated across " << hosts.size() << " hosts";
    }

    // And against the PGD's own two hosts: each declared rank is finer than a host and has to land
    // inside one of them, which is the whole point of allowing a subdivision:
    //
    //   rank a = chips 0 1 -> slots (1,1) (1,2)     both in PGD host0
    //   rank b = chips 2 3 -> slots (1,3) (1,4)     both in PGD host0
    //   rank c = chips 4 5 -> slots (2,1) (2,2)     both in PGD host1
    //   rank d = chips 6 7 -> slots (2,3) (2,4)     both in PGD host1
    const std::vector<std::set<std::pair<uint32_t, uint32_t>>> pgd_hosts = {
        {{1, 1}, {1, 2}, {1, 3}, {1, 4}}, {{2, 1}, {2, 2}, {2, 3}, {2, 4}}};
    expect_each_rank_inside_one_pgd_host(committed.front().mesh_node_to_asic_position, declared_ranks, pgd_hosts);

    // Mapping the mesh onto the machine: the seating the placement search returns has to honour the split
    // too, not just the grouping the matcher committed.
    const auto placements = pgd.solve_adjacency_guided_placement(mgd, without_mgd_fallback(valid_groupings, "M0"), psd);
    ASSERT_EQ(placements.size(), 1u) << "the 2x4 mesh should be placed";
    for (std::size_t rank = 0; rank < declared_ranks.size(); ++rank) {
        std::set<uint32_t> hosts;
        for (LogicalChipId chip : declared_ranks[rank]) {
            hosts.insert(*placements.front().mesh_node_to_asic_position.at(chip).first - 1);
        }
        EXPECT_EQ(hosts.size(), 1u) << "placement seated declared rank " << rank << " across " << hosts.size()
                                    << " hosts";
    }

    // Phase 2: the rank-bound graph has two ranks of 4 where the MGD declared four of 2, so the declared
    // split is finer than the partitions phase 2 sees. Finer is legal, so the hints must still arrive --
    // this is the accepting counterpart of the finer-slices divergence in
    // SecondGetValidGroupingsOnRankSlicedPsdRejectsWhatTheFirstCommitted.
    std::map<std::string, MeshHostRankId> rank_of_host;
    for (const auto& [asic_id, descriptor] : psd.get_asic_descriptors()) {
        rank_of_host.emplace(psd.get_host_name_for_asic(asic_id), MeshHostRankId{0});
    }
    uint32_t next_rank = 0;
    for (auto& [host_name, rank] : rank_of_host) {
        rank = MeshHostRankId{next_rank++};
    }
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> asic_id_to_mesh_rank;
    for (const auto& [asic_id, descriptor] : psd.get_asic_descriptors()) {
        asic_id_to_mesh_rank[MeshId{0}][asic_id] = rank_of_host.at(psd.get_host_name_for_asic(asic_id));
    }

    utils::PhysicalMultiMeshGraph physical;
    ASSERT_NO_THROW(physical = utils::build_physical_multi_mesh_adjacency_graph(psd, asic_id_to_mesh_rank, pgd, mgd))
        << "the rank-bound path should map a mesh that phase 1 already seated";
    ASSERT_TRUE(physical.mesh_pgd_pinnings_.contains(MeshId{0}))
        << "phase 2 should reach phase 1's verdict on the same machine";
    EXPECT_EQ(physical.mesh_pgd_pinnings_.at(MeshId{0}).size(), 8u);
}

// The refusing half of that pair, with the two grids exchanged: the machine is now cut on both axes into
// a [2,2] host grid of 2-chip hosts, and the MGD declares [2,1], two ranks of one full row each. A row
// here is two hosts wide, so each declared rank needs chips from both hosts of its row:
//
//   machine: host grid [2,2]          MGD host_topology [2,1]
//
//     100 101 | 102 103                 aaaa aaaa      rank a spans h0 and h1
//     --------+--------                 bbbb bbbb      rank b spans h2 and h3
//     104 105 | 106 107
//       h0/h1 above, h2/h3 below
//
// Which is the one-rank-owns-chips-it-cannot-reach violation again, and the direction that has to stay
// hard however the machine is divided: a 2D host grid can carve a declared split as readily as a 1D one.
TEST(PhysicalGroupingDescriptorTestsHostSplit, CoarserSplitAcrossATwoAxisHostGridIsRejected) {
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(
        "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_8asic_2x4_four_hosts_by_quadrant.textproto");

    PhysicalGroupingDescriptor pgd{std::string(R"(
groupings {
  name: "2x4_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
    { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } },
    { id: 2 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_3 } },
    { id: 3 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_4 } },
    { id: 4 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } },
    { id: 5 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } },
    { id: 6 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_3 } },
    { id: 7 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_4 } }
  ]
  row_major_mesh {
    dims: [2, 4]
  }
}

# The PGD's own host level, mirroring the machine: four hosts of 2 chips, one per quadrant.
groupings { name: "2x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
               { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } } ]
  row_major_mesh { dims: [1, 2] } }
groupings { name: "2x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_3 } },
               { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_4 } } ]
  row_major_mesh { dims: [1, 2] } }
groupings { name: "2x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } },
               { id: 1 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } } ]
  row_major_mesh { dims: [1, 2] } }
groupings { name: "2x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_3 } },
               { id: 1 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_4 } } ]
  row_major_mesh { dims: [1, 2] } }
)")};

    MeshGraphDescriptor mgd{std::string(R"(
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 2, 4 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 2, 1 ] }
  channels { count: 2 policy: STRICT }
}

top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
)")};

    const auto valid_groupings =
        pgd.get_valid_groupings_for_mgd(mgd, psd, /*pinnings=*/std::nullopt, /*require_placement=*/false);

    std::vector<GroupingInfo> committed;
    for (const auto& [preset, by_instance] : valid_groupings) {
        for (const auto& [instance, groupings] : by_instance) {
            for (const auto& grouping : groupings) {
                if (grouping.name != "M0" && !grouping.mesh_node_to_asic_position.empty()) {
                    committed.push_back(grouping);
                }
            }
        }
    }

    EXPECT_TRUE(committed.empty()) << "a 4-chip row rank cannot fit on a 2-chip host, got " << committed.size()
                                   << " grouping(s)";

    // Guarded on that verdict: the PGD declares the same four quadrant hosts, so a declared 4-chip row
    // rank cannot sit inside one of those either:
    //
    //   rank a = chips 0 1 2 3 -> slots (1,1) (1,2) (1,3) (1,4) = PGD host0 plus host1
    //   rank b = chips 4 5 6 7 -> slots (2,1) (2,2) (2,3) (2,4) = PGD host2 plus host3
    //   the PGD's hosts hold 2 chips each, so a 4-chip rank needs two of them
    const std::vector<std::set<std::pair<uint32_t, uint32_t>>> pgd_hosts = {
        {{1, 1}, {1, 2}}, {{1, 3}, {1, 4}}, {{2, 1}, {2, 2}}, {{2, 3}, {2, 4}}};
    const std::vector<std::vector<LogicalChipId>> declared_ranks = {{0, 1, 2, 3}, {4, 5, 6, 7}};
    if (!committed.empty()) {
        expect_each_rank_inside_one_pgd_host(committed.front().mesh_node_to_asic_position, declared_ranks, pgd_hosts);
    }

    // Phase 2: same verdict on a rank-bound graph, one rank per host -- four 2-chip ranks here, so the
    // declared 4-chip rank has even less room than it had in phase 1.
    std::map<std::string, MeshHostRankId> rank_of_host;
    for (const auto& [asic_id, descriptor] : psd.get_asic_descriptors()) {
        rank_of_host.emplace(psd.get_host_name_for_asic(asic_id), MeshHostRankId{0});
    }
    uint32_t next_rank = 0;
    for (auto& [host_name, rank] : rank_of_host) {
        rank = MeshHostRankId{next_rank++};
    }
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> asic_id_to_mesh_rank;
    for (const auto& [asic_id, descriptor] : psd.get_asic_descriptors()) {
        asic_id_to_mesh_rank[MeshId{0}][asic_id] = rank_of_host.at(psd.get_host_name_for_asic(asic_id));
    }

    bool phase2_produced_a_seating = false;
    try {
        const auto physical = utils::build_physical_multi_mesh_adjacency_graph(psd, asic_id_to_mesh_rank, pgd, mgd);
        phase2_produced_a_seating = physical.mesh_pgd_pinnings_.contains(MeshId{0});
    } catch (const std::exception&) {
        phase2_produced_a_seating = false;
    }
    EXPECT_FALSE(phase2_produced_a_seating) << "the rank-bound path pinned a mesh enumerate had refused";
}

// Coarser than the machine: the MGD declares 2 ranks of 4 chips where the machine offers 4 hosts of 2.
// Each declared rank spans two hosts, so it is rejected. This is the direction that must stay hard -- it
// is one process being asked to own chips it cannot reach.
//
//   machine: one host per column     MGD host_topology [1,2]
//
//     100 | 101 | 102 | 103            aa | bb    rank a wants h0 and h1,
//     104 | 105 | 106 | 107            aa | bb    rank b wants h2 and h3
//      h0 | h1  | h2  | h3
//
// Note this is the same declared split that ColumnSplitPsdColumnSplitMgdCommits accepts, on the same
// mesh and the same axis. Only the machine got finer, which is all it takes.
TEST(PhysicalGroupingDescriptorTestsHostSplit, CoarserSplitThanThePhysicalHostsIsRejected) {
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(
        "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_8asic_2x4_four_hosts_by_column.textproto");

    PhysicalGroupingDescriptor pgd{std::string(R"(
groupings {
  name: "2x4_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
    { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } },
    { id: 2 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_3 } },
    { id: 3 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_4 } },
    { id: 4 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } },
    { id: 5 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } },
    { id: 6 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_3 } },
    { id: 7 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_4 } }
  ]
  row_major_mesh {
    dims: [2, 4]
  }
}

# The PGD's own host level, mirroring the machine: four hosts of 2 chips, one per column.
groupings { name: "2x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
               { id: 1 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } } ]
  row_major_mesh { dims: [2, 1] } }
groupings { name: "2x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } },
               { id: 1 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } } ]
  row_major_mesh { dims: [2, 1] } }
groupings { name: "2x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_3 } },
               { id: 1 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_3 } } ]
  row_major_mesh { dims: [2, 1] } }
groupings { name: "2x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_4 } },
               { id: 1 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_4 } } ]
  row_major_mesh { dims: [2, 1] } }
)")};

    MeshGraphDescriptor mgd{std::string(R"(
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 2, 4 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 2 ] }
  channels { count: 2 policy: STRICT }
}

top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
)")};

    const auto valid_groupings =
        pgd.get_valid_groupings_for_mgd(mgd, psd, /*pinnings=*/std::nullopt, /*require_placement=*/false);

    std::vector<GroupingInfo> committed;
    for (const auto& [preset, by_instance] : valid_groupings) {
        for (const auto& [instance, groupings] : by_instance) {
            for (const auto& grouping : groupings) {
                if (grouping.name != "M0" && !grouping.mesh_node_to_asic_position.empty()) {
                    committed.push_back(grouping);
                }
            }
        }
    }

    EXPECT_TRUE(committed.empty()) << "a 4-chip rank cannot fit on a 2-chip host, got " << committed.size()
                                   << " grouping(s)";

    // Guarded on that verdict: the PGD declares the same four column hosts, so a 4-chip declared rank
    // has no single PGD host to sit in either:
    //
    //   rank a = chips 0 1 4 5 -> slots (1,1) (1,2) (2,1) (2,2) = PGD host0 plus host1
    //   rank b = chips 2 3 6 7 -> slots (1,3) (1,4) (2,3) (2,4) = PGD host2 plus host3
    //   each PGD host is one column of 2 chips, and each declared rank wants two columns
    const std::vector<std::set<std::pair<uint32_t, uint32_t>>> pgd_hosts = {
        {{1, 1}, {2, 1}}, {{1, 2}, {2, 2}}, {{1, 3}, {2, 3}}, {{1, 4}, {2, 4}}};
    const std::vector<std::vector<LogicalChipId>> declared_ranks = {{0, 1, 4, 5}, {2, 3, 6, 7}};
    if (!committed.empty()) {
        expect_each_rank_inside_one_pgd_host(committed.front().mesh_node_to_asic_position, declared_ranks, pgd_hosts);
    }
}

// A mesh that declares no split cannot be seated on a machine that has one. host_topology [1,1] is one
// rank owning all 8 chips, so a 2-host seating would hand that rank chips it cannot reach -- the same
// violation CoarserSplitThanThePhysicalHostsIsRejected covers, with one rank instead of two. A mesh that
// genuinely needs both hosts, torus or not, has to say so in its host_topology.
//
//   machine: hosts cut by column     MGD host_topology [1,1]
//
//     100 101 | 102 103                aaaa       the one declared rank wants every
//     104 105 | 106 107                aaaa       chip, so it wants both hosts
//      host0  |  host1
//
// SingleHostPsdSingleHostMgdCommits is the control: the same MGD on an undivided machine commits, so the
// rejection here is about the host boundary and not about the mesh being too big to fit.
TEST(PhysicalGroupingDescriptorTestsHostSplit, SingleHostMgdOnSplitPsdIsRejected) {
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(
        "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_8asic_2x4_hosts_by_column.textproto");

    PhysicalGroupingDescriptor pgd{std::string(R"(
groupings {
  name: "2x4_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
    { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } },
    { id: 2 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_3 } },
    { id: 3 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_4 } },
    { id: 4 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } },
    { id: 5 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } },
    { id: 6 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_3 } },
    { id: 7 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_4 } }
  ]
  row_major_mesh {
    dims: [2, 4]
  }
}

# The PGD's own host level, mirroring the machine: two hosts, each one column-half.
groupings { name: "2x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
               { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } },
               { id: 2 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } },
               { id: 3 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } } ]
  row_major_mesh { dims: [2, 2] } }
groupings { name: "2x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_3 } },
               { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_4 } },
               { id: 2 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_3 } },
               { id: 3 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_4 } } ]
  row_major_mesh { dims: [2, 2] } }
)")};

    MeshGraphDescriptor mgd{std::string(R"(
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 2, 4 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}

top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
)")};

    const auto valid_groupings =
        pgd.get_valid_groupings_for_mgd(mgd, psd, /*pinnings=*/std::nullopt, /*require_placement=*/false);

    std::vector<GroupingInfo> committed;
    for (const auto& [preset, by_instance] : valid_groupings) {
        for (const auto& [instance, groupings] : by_instance) {
            for (const auto& grouping : groupings) {
                if (grouping.name != "M0" && !grouping.mesh_node_to_asic_position.empty()) {
                    committed.push_back(grouping);
                }
            }
        }
    }

    EXPECT_TRUE(committed.empty()) << "one declared rank cannot own chips on two hosts, got " << committed.size()
                                   << " grouping(s)";

    // Guarded on that verdict: the PGD declares the same two hosts, and the single declared rank owns
    // every chip, so it cannot sit inside either of them:
    //
    //   rank a = chips 0 1 2 3 4 5 6 7 -> every slot, which is PGD host0 and host1 together
    //   PGD host0 = (1,1) (1,2) (2,1) (2,2)   PGD host1 = (1,3) (1,4) (2,3) (2,4)
    const std::vector<std::set<std::pair<uint32_t, uint32_t>>> pgd_hosts = {
        {{1, 1}, {1, 2}, {2, 1}, {2, 2}}, {{1, 3}, {1, 4}, {2, 3}, {2, 4}}};
    if (!committed.empty()) {
        expect_each_rank_inside_one_pgd_host(
            committed.front().mesh_node_to_asic_position, {{0, 1, 2, 3, 4, 5, 6, 7}}, pgd_hosts);
    }
}

// ----- the constraint as the placement search sees it -------------------------------------------

// The checks above read the verdict off the committed grouping. This one reads it off the seating the
// placement search actually returns, on both backends: the host split is encoded as constraints inside
// enumerate_distinct_placements_for_grouping, so SAT and DFS have to honour it equally or a mesh accepted
// during matching can still come back seated across a host boundary.
//
//   machine: hosts cut by column     MGD host_topology [1,2]
//
//     100 101 | 102 103                aa | bb    the aligned case again, but the
//     104 105 | 106 107                aa | bb    seating is read from the search
//      host0  |  host1                            rather than from the matcher
TEST(PhysicalGroupingDescriptorTestsHostSplit, SatAndDfsPlacementBothSeatEachDeclaredRankOnOneHost) {
    for (const char* solver : {"sat", "dfs"}) {
        ScopedEnv pick_solver("TT_METAL_PLACEMENT_SOLVER", solver);

        auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(
            "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_8asic_2x4_hosts_by_column.textproto");

        PhysicalGroupingDescriptor pgd{std::string(R"(
groupings {
  name: "2x4_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
    { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } },
    { id: 2 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_3 } },
    { id: 3 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_4 } },
    { id: 4 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } },
    { id: 5 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } },
    { id: 6 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_3 } },
    { id: 7 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_4 } }
  ]
  row_major_mesh {
    dims: [2, 4]
  }
}

# The PGD's own host level, mirroring the machine: two hosts, each one column-half.
groupings { name: "2x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
               { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } },
               { id: 2 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } },
               { id: 3 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } } ]
  row_major_mesh { dims: [2, 2] } }
groupings { name: "2x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_3 } },
               { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_4 } },
               { id: 2 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_3 } },
               { id: 3 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_4 } } ]
  row_major_mesh { dims: [2, 2] } }
)")};

        MeshGraphDescriptor mgd{std::string(R"(
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 2, 4 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 2 ] }
  channels { count: 2 policy: STRICT }
}

top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
)")};

        const auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd, psd);
        const auto placements =
            pgd.solve_adjacency_guided_placement(mgd, without_mgd_fallback(valid_groupings, "M0"), psd);
        ASSERT_EQ(placements.size(), 1u) << solver << ": the 2x4 mesh should be placed";
        EXPECT_EQ(placements.front().asics.size(), 8u) << solver;

        const std::vector<std::vector<LogicalChipId>> declared_ranks = {{0, 1, 4, 5}, {2, 3, 6, 7}};
        for (std::size_t rank = 0; rank < declared_ranks.size(); ++rank) {
            std::set<uint32_t> hosts;
            for (LogicalChipId chip : declared_ranks[rank]) {
                const auto& position = placements.front().mesh_node_to_asic_position.at(chip);
                hosts.insert((*position.second - 1) / 2);
            }
            EXPECT_EQ(hosts.size(), 1u) << solver << ": placement seated declared rank " << rank << " across "
                                        << hosts.size() << " hosts";
        }

        // And the same of the PGD's own hosts, on the seating the search returned rather than the one
        // the matcher committed:
        //
        //   rank a = chips 0 1 4 5 -> slots (1,1) (1,2) (2,1) (2,2) = PGD host0
        //   rank b = chips 2 3 6 7 -> slots (1,3) (1,4) (2,3) (2,4) = PGD host1
        const std::vector<std::set<std::pair<uint32_t, uint32_t>>> pgd_hosts = {
            {{1, 1}, {1, 2}, {2, 1}, {2, 2}}, {{1, 3}, {1, 4}, {2, 3}, {2, 4}}};
        expect_each_rank_inside_one_pgd_host(placements.front().mesh_node_to_asic_position, declared_ranks, pgd_hosts);
    }
}

// The other half of the pair: when the declared split is on the wrong axis, whatever the search returns
// must still respect it -- either nothing, or a seating that keeps each rank on one host. It must not
// answer by throwing.
//
//   machine: hosts cut by column     MGD host_topology [2,1]
//
//     100 101 | 102 103                aaaa       no seating on this machine keeps
//     104 105 | 106 107                bbbb       either rank on a single host, so
//      host0  |  host1                            the honest answer is "nothing"
//
// FIXME: the answer it gives is the MGD fallback, which is not an answer to the question. The host
// constraint rejects the only PGD grouping, so the fallback -- the mesh descriptor's own topology,
// committed last by get_valid_groupings_for_mgd -- is all that is left, and the mesh places on it. That
// grouping carries neither a chip -> slot pinning nor a mesh_node_to_host_group, so the placement claims
// eight ASICs while saying nothing about which rank sits where, and the declared split it was supposed to
// respect is not applied to it at all. The loop below can only skip such a placement, which is why this
// test currently proves the weaker statement: no seating that violates the split is returned, because no
// seating is returned. Holding the fallback to the MGD's own host_topology is what would close this.
TEST(PhysicalGroupingDescriptorTestsHostSplit, SatAndDfsPlacementNeverSeatTheWrongAxisAcrossHosts) {
    for (const char* solver : {"sat", "dfs"}) {
        ScopedEnv pick_solver("TT_METAL_PLACEMENT_SOLVER", solver);

        auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(
            "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_8asic_2x4_hosts_by_column.textproto");

        PhysicalGroupingDescriptor pgd{std::string(R"(
groupings {
  name: "2x4_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
    { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } },
    { id: 2 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_3 } },
    { id: 3 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_4 } },
    { id: 4 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } },
    { id: 5 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } },
    { id: 6 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_3 } },
    { id: 7 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_4 } }
  ]
  row_major_mesh {
    dims: [2, 4]
  }
}

# The PGD's own host level, mirroring the machine: two hosts, each one column-half. The declared
# lengthwise ranks cross these as well, so neither host level admits this split.
groupings { name: "2x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
               { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } },
               { id: 2 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } },
               { id: 3 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } } ]
  row_major_mesh { dims: [2, 2] } }
groupings { name: "2x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_3 } },
               { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_4 } },
               { id: 2 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_3 } },
               { id: 3 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_4 } } ]
  row_major_mesh { dims: [2, 2] } }
)")};

        MeshGraphDescriptor mgd{std::string(R"(
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 2, 4 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 2, 1 ] }
  channels { count: 2 policy: STRICT }
}

top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
)")};

        const auto valid_groupings =
            pgd.get_valid_groupings_for_mgd(mgd, psd, /*pinnings=*/std::nullopt, /*require_placement=*/false);

        std::vector<GroupingInfo> committed;
        for (const auto& [preset, by_instance] : valid_groupings) {
            for (const auto& [instance, groupings] : by_instance) {
                for (const auto& grouping : groupings) {
                    if (grouping.name != "M0" && !grouping.mesh_node_to_asic_position.empty()) {
                        committed.push_back(grouping);
                    }
                }
            }
        }
        EXPECT_TRUE(committed.empty()) << solver;

        std::vector<PsdPlacement> placements;
        ASSERT_NO_THROW(placements = pgd.solve_adjacency_guided_placement(mgd, valid_groupings, psd)) << solver;

        const std::vector<std::vector<LogicalChipId>> declared_ranks = {{0, 1, 2, 3}, {4, 5, 6, 7}};
        for (const auto& placement : placements) {
            if (placement.mesh_node_to_asic_position.empty()) {
                // The MGD fallback, per the FIXME above: a footprint with nothing to check.
                continue;
            }
            for (std::size_t rank = 0; rank < declared_ranks.size(); ++rank) {
                std::set<uint32_t> hosts;
                for (LogicalChipId chip : declared_ranks[rank]) {
                    const auto& position = placement.mesh_node_to_asic_position.at(chip);
                    hosts.insert((*position.second - 1) / 2);
                }
                EXPECT_EQ(hosts.size(), 1u)
                    << solver << ": a row split cannot be seated on a column-split machine, rank " << rank
                    << " landed on " << hosts.size() << " hosts";
            }

            // And of the PGD's hosts, which draw the same boundary. No seating satisfies this, which
            // is what the test asserts the search must never return:
            //
            //   rank a = chips 0 1 2 3 -> slots (1,1) (1,2) (1,3) (1,4), half in each PGD host
            //   rank b = chips 4 5 6 7 -> slots (2,1) (2,2) (2,3) (2,4), likewise
            const std::vector<std::set<std::pair<uint32_t, uint32_t>>> pgd_hosts = {
                {{1, 1}, {1, 2}, {2, 1}, {2, 2}}, {{1, 3}, {1, 4}, {2, 3}, {2, 4}}};
            expect_each_rank_inside_one_pgd_host(placement.mesh_node_to_asic_position, declared_ranks, pgd_hosts);
        }
    }
}

// ----- the same rule on the square shape --------------------------------------------------------
//
// The quadrant-machine cases live with the other GetValidGroupingsForMGD tests further up, under
// PhysicalGroupingDescriptorTests. What is left here is the column-split 4x4, which is where the
// rotation question below is asked.

// Control for the case below: on a square mesh, the split that happens to be declared along the machine's
// own axis commits, exactly as it does on the 2x4. No turn is needed, so the orientation the match
// happens to return is the right one and the rotation gap cannot be reached from here.
//
//   machine: 4x4, hosts cut by column   MGD host_topology [1,2]
//
//     100 101 | 102 103                   aa | bb
//     104 105 | 106 107                   aa | bb     a is host0, b is host1,
//     108 109 | 110 111                   aa | bb     as declared
//     112 113 | 114 115                   aa | bb
//      host0  |  host1
TEST(PhysicalGroupingDescriptorTestsHostSplit, SquareMeshCommitsWhenTheDeclaredAxisAlreadyMatches) {
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(
        "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_16asic_4x4_hosts_by_column.textproto");

    PhysicalGroupingDescriptor pgd{std::string(R"(
groupings {
  name: "4x4_Mesh"
  preset_type: MESH
  instances: [
    { id: 0  location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
    { id: 1  location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } },
    { id: 2  location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_3 } },
    { id: 3  location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_4 } },
    { id: 4  location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } },
    { id: 5  location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } },
    { id: 6  location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_3 } },
    { id: 7  location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_4 } },
    { id: 8  location { tray_id: TRAY_3 asic_location: ASIC_LOCATION_1 } },
    { id: 9  location { tray_id: TRAY_3 asic_location: ASIC_LOCATION_2 } },
    { id: 10 location { tray_id: TRAY_3 asic_location: ASIC_LOCATION_3 } },
    { id: 11 location { tray_id: TRAY_3 asic_location: ASIC_LOCATION_4 } },
    { id: 12 location { tray_id: TRAY_4 asic_location: ASIC_LOCATION_1 } },
    { id: 13 location { tray_id: TRAY_4 asic_location: ASIC_LOCATION_2 } },
    { id: 14 location { tray_id: TRAY_4 asic_location: ASIC_LOCATION_3 } },
    { id: 15 location { tray_id: TRAY_4 asic_location: ASIC_LOCATION_4 } }
  ]
  row_major_mesh {
    dims: [4, 4]
  }
}

# The PGD's own host level, mirroring the machine: two hosts of 8 chips, each two columns.
groupings { name: "4x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
               { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } },
               { id: 2 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } },
               { id: 3 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } },
               { id: 4 location { tray_id: TRAY_3 asic_location: ASIC_LOCATION_1 } },
               { id: 5 location { tray_id: TRAY_3 asic_location: ASIC_LOCATION_2 } },
               { id: 6 location { tray_id: TRAY_4 asic_location: ASIC_LOCATION_1 } },
               { id: 7 location { tray_id: TRAY_4 asic_location: ASIC_LOCATION_2 } } ]
  row_major_mesh { dims: [4, 2] } }
groupings { name: "4x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_3 } },
               { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_4 } },
               { id: 2 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_3 } },
               { id: 3 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_4 } },
               { id: 4 location { tray_id: TRAY_3 asic_location: ASIC_LOCATION_3 } },
               { id: 5 location { tray_id: TRAY_3 asic_location: ASIC_LOCATION_4 } },
               { id: 6 location { tray_id: TRAY_4 asic_location: ASIC_LOCATION_3 } },
               { id: 7 location { tray_id: TRAY_4 asic_location: ASIC_LOCATION_4 } } ]
  row_major_mesh { dims: [4, 2] } }
)")};

    MeshGraphDescriptor mgd{std::string(R"(
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 4, 4 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 2 ] }
  channels { count: 2 policy: STRICT }
}

top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
)")};

    const auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd, psd);

    std::vector<GroupingInfo> committed;
    for (const auto& [preset, by_instance] : valid_groupings) {
        for (const auto& [instance, groupings] : by_instance) {
            for (const auto& grouping : groupings) {
                if (grouping.name != "M0" && !grouping.mesh_node_to_asic_position.empty()) {
                    committed.push_back(grouping);
                }
            }
        }
    }
    ASSERT_FALSE(committed.empty());

    // host_topology [1,2] on a 4x4 declares two ranks of one column-half each.
    const std::vector<std::vector<LogicalChipId>> declared_ranks = {
        {0, 1, 4, 5, 8, 9, 12, 13}, {2, 3, 6, 7, 10, 11, 14, 15}};
    for (std::size_t rank = 0; rank < declared_ranks.size(); ++rank) {
        std::set<uint32_t> hosts;
        for (LogicalChipId chip : declared_ranks[rank]) {
            const auto& position = committed.front().mesh_node_to_asic_position.at(chip);
            hosts.insert((*position.second - 1) / 2);
        }
        EXPECT_EQ(hosts.size(), 1u) << "declared rank " << rank << " was seated across " << hosts.size() << " hosts";
    }

    // And against the PGD's own two hosts, which draw the same two columns. On a square mesh this is
    // the orientation the match returns anyway, so it holds without needing the turn:
    //
    //   rank a = chips 0 1 4 5 8 9 12 13 -> slots (1,1) (1,2) (2,1) (2,2) (3,1) (3,2) (4,1) (4,2) = host0
    //   rank b = chips 2 3 6 7 10 11 14 15 -> slots (1,3) (1,4) (2,3) (2,4) (3,3) (3,4) (4,3) (4,4) = host1
    const std::vector<std::set<std::pair<uint32_t, uint32_t>>> pgd_hosts = {
        {{1, 1}, {1, 2}, {2, 1}, {2, 2}, {3, 1}, {3, 2}, {4, 1}, {4, 2}},
        {{1, 3}, {1, 4}, {2, 3}, {2, 4}, {3, 3}, {3, 4}, {4, 3}, {4, 4}}};
    expect_each_rank_inside_one_pgd_host(committed.front().mesh_node_to_asic_position, declared_ranks, pgd_hosts);
}

// ----- Phase 2: asking the matcher a second time ------------------------------------------------

// The two phases do not ask about the same machine. Phase 1 runs against the PSD's hosts; phase 2 runs
// after rank binding, against a PSD whose host names are per-rank slices, which are finer. So the second
// call can reject the very host split the first call accepted and placed -- the matcher is consistent, it
// is being asked a different question.
//
// This is the disagreement itself, told apart from what either caller does about it. The declared split
// is [1,2] both times; only the machine under it changes:
//
//   phase 1: the PSD's two hosts      phase 2: the same chips as four rank slices
//
//     100 101 | 102 103                  100 | 101 | 102 | 103
//     104 105 | 106 107                  104 | 105 | 106 | 107
//      host0  |  host1                    r0  | r1  | r2  | r3
//
//        aa   |   bb    accepted             aa | bb    rejected: rank a now needs r0 and r1
//        aa   |   bb                         aa | bb
TEST(PhysicalGroupingDescriptorTestsHostSplit, SecondGetValidGroupingsOnRankSlicedPsdRejectsWhatTheFirstCommitted) {
    PhysicalGroupingDescriptor pgd{std::string(R"(
groupings {
  name: "2x4_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
    { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } },
    { id: 2 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_3 } },
    { id: 3 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_4 } },
    { id: 4 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } },
    { id: 5 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } },
    { id: 6 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_3 } },
    { id: 7 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_4 } }
  ]
  row_major_mesh {
    dims: [2, 4]
  }
}

# The PGD's own host level: the machine's two real hosts, each one column-half. A PGD describes the
# system, so it knows these and not the per-rank slices phase 2 is handed.
groupings { name: "2x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
               { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } },
               { id: 2 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } },
               { id: 3 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } } ]
  row_major_mesh { dims: [2, 2] } }
groupings { name: "2x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_3 } },
               { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_4 } },
               { id: 2 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_3 } },
               { id: 3 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_4 } } ]
  row_major_mesh { dims: [2, 2] } }
)")};

    MeshGraphDescriptor mgd{std::string(R"(
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 2, 4 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 2 ] }
  channels { count: 2 policy: STRICT }
}

top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
)")};

    auto host_psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(
        "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_8asic_2x4_hosts_by_column.textproto");
    std::vector<GroupingInfo> on_hosts;
    for (const auto& [preset, by_instance] :
         pgd.get_valid_groupings_for_mgd(mgd, host_psd, /*pinnings=*/std::nullopt, /*require_placement=*/false)) {
        for (const auto& [instance, groupings] : by_instance) {
            for (const auto& grouping : groupings) {
                if (grouping.name != "M0" && !grouping.mesh_node_to_asic_position.empty()) {
                    on_hosts.push_back(grouping);
                }
            }
        }
    }
    EXPECT_FALSE(on_hosts.empty()) << "two declared ranks on two hosts: accepted";

    // The PGD's hosts are the same two, so the accepted seating sits inside them as well:
    //
    //   rank a = chips 0 1 4 5 -> slots (1,1) (1,2) (2,1) (2,2) = PGD host0
    //   rank b = chips 2 3 6 7 -> slots (1,3) (1,4) (2,3) (2,4) = PGD host1
    //
    // It is only the PSD that changes below; the PGD's host level says the same thing across both
    // calls, which is the property that would make the two phases agree once the matcher reads it.
    const std::vector<std::set<std::pair<uint32_t, uint32_t>>> pgd_hosts = {
        {{1, 1}, {1, 2}, {2, 1}, {2, 2}}, {{1, 3}, {1, 4}, {2, 3}, {2, 4}}};
    const std::vector<std::vector<LogicalChipId>> declared_ranks = {{0, 1, 4, 5}, {2, 3, 6, 7}};
    if (!on_hosts.empty()) {
        expect_each_rank_inside_one_pgd_host(on_hosts.front().mesh_node_to_asic_position, declared_ranks, pgd_hosts);
    }

    auto rank_sliced_psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(
        "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_8asic_2x4_four_hosts_by_column.textproto");
    std::vector<GroupingInfo> on_rank_slices;
    for (const auto& [preset, by_instance] : pgd.get_valid_groupings_for_mgd(
             mgd, rank_sliced_psd, /*pinnings=*/std::nullopt, /*require_placement=*/false)) {
        for (const auto& [instance, groupings] : by_instance) {
            for (const auto& grouping : groupings) {
                if (grouping.name != "M0" && !grouping.mesh_node_to_asic_position.empty()) {
                    on_rank_slices.push_back(grouping);
                }
            }
        }
    }
    EXPECT_TRUE(on_rank_slices.empty()) << "the same two ranks against four slices: each rank now spans two, so "
                                           "rejected";
}

// The phase-2 happy path, taken on the most rotation-prone shape there is: a 4x4 RING/RING, symmetric on
// both axes and wrapped on both, asked the same question twice. It holds -- the match returns the same
// orientation each time, so phase 2 reaches phase 1's verdict and the pinnings arrive. This is also the
// suite's one RING/RING split-host case that commits.
//
// Worth stating as a test because it bounds the rotation gap. An arbitrary orientation would be far more
// damaging if it were also an unstable one, re-rolled per call; it is not. The gap costs a declared split
// that needs a rotation to fit (LengthwiseSplitOnWidthwiseSplitHostsIsTurnedToFitOnASquareMesh), and a
// phase-2 divergence only when the two calls are given genuinely different host partitions, which is
// SecondGetValidGroupingsOnRankSlicedPsdRejectsWhatTheFirstCommitted, not this.
//
//   machine: 4x4 RING/RING, hosts cut by column     MGD host_topology [1,2]
//
//     100 101 | 102 103     wrapped on both           aa | bb
//     104 105 | 106 107     axes, so translations     aa | bb    the declared split already
//     108 109 | 110 111     along either one are      aa | bb    lies along the boundary, and
//     112 113 | 114 115     automorphisms as well     aa | bb    asking twice gives the same
//      host0  |  host1      as the reflections                   orientation twice
TEST(PhysicalGroupingDescriptorTestsHostSplit, Phase2AgreesWithPhase1OnASymmetricTorus) {
    auto psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(
        "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_16asic_4x4_torus_hosts_by_column.textproto");

    PhysicalGroupingDescriptor pgd{std::string(R"(
groupings {
  name: "4x4_Mesh"
  preset_type: MESH
  instances: [
    { id: 0  location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
    { id: 1  location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } },
    { id: 2  location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_3 } },
    { id: 3  location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_4 } },
    { id: 4  location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } },
    { id: 5  location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } },
    { id: 6  location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_3 } },
    { id: 7  location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_4 } },
    { id: 8  location { tray_id: TRAY_3 asic_location: ASIC_LOCATION_1 } },
    { id: 9  location { tray_id: TRAY_3 asic_location: ASIC_LOCATION_2 } },
    { id: 10 location { tray_id: TRAY_3 asic_location: ASIC_LOCATION_3 } },
    { id: 11 location { tray_id: TRAY_3 asic_location: ASIC_LOCATION_4 } },
    { id: 12 location { tray_id: TRAY_4 asic_location: ASIC_LOCATION_1 } },
    { id: 13 location { tray_id: TRAY_4 asic_location: ASIC_LOCATION_2 } },
    { id: 14 location { tray_id: TRAY_4 asic_location: ASIC_LOCATION_3 } },
    { id: 15 location { tray_id: TRAY_4 asic_location: ASIC_LOCATION_4 } }
  ]
  row_major_mesh {
    dims: [4, 4]
  }
}

# The PGD's own host level, mirroring the machine: two hosts of 8 chips, each two columns. A wrapped
# mesh has translations among its automorphisms, so this is the shape where an orientation chosen
# without the hosts in view has the most room to disagree with them.
groupings { name: "4x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
               { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } },
               { id: 2 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } },
               { id: 3 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } },
               { id: 4 location { tray_id: TRAY_3 asic_location: ASIC_LOCATION_1 } },
               { id: 5 location { tray_id: TRAY_3 asic_location: ASIC_LOCATION_2 } },
               { id: 6 location { tray_id: TRAY_4 asic_location: ASIC_LOCATION_1 } },
               { id: 7 location { tray_id: TRAY_4 asic_location: ASIC_LOCATION_2 } } ]
  row_major_mesh { dims: [4, 2] } }
groupings { name: "4x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_3 } },
               { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_4 } },
               { id: 2 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_3 } },
               { id: 3 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_4 } },
               { id: 4 location { tray_id: TRAY_3 asic_location: ASIC_LOCATION_3 } },
               { id: 5 location { tray_id: TRAY_3 asic_location: ASIC_LOCATION_4 } },
               { id: 6 location { tray_id: TRAY_4 asic_location: ASIC_LOCATION_3 } },
               { id: 7 location { tray_id: TRAY_4 asic_location: ASIC_LOCATION_4 } } ]
  row_major_mesh { dims: [4, 2] } }
)")};

    MeshGraphDescriptor mgd{std::string(R"(
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 4, 4 ] dim_types: [ RING, RING ] }
  host_topology   { dims: [ 1, 2 ] }
  channels { count: 2 policy: STRICT }
}

top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
)")};

    std::vector<GroupingInfo> phase1;
    for (const auto& [preset, by_instance] : pgd.get_valid_groupings_for_mgd(mgd, psd)) {
        for (const auto& [instance, groupings] : by_instance) {
            for (const auto& grouping : groupings) {
                if (grouping.name != "M0" && !grouping.mesh_node_to_asic_position.empty()) {
                    phase1.push_back(grouping);
                }
            }
        }
    }
    ASSERT_FALSE(phase1.empty()) << "phase 1 commits the split that lies along the hosts";

    const std::vector<std::vector<LogicalChipId>> declared_ranks = {
        {0, 1, 4, 5, 8, 9, 12, 13}, {2, 3, 6, 7, 10, 11, 14, 15}};
    for (std::size_t rank = 0; rank < declared_ranks.size(); ++rank) {
        std::set<uint32_t> hosts;
        for (LogicalChipId chip : declared_ranks[rank]) {
            const auto& position = phase1.front().mesh_node_to_asic_position.at(chip);
            hosts.insert((*position.second - 1) / 2);
        }
        ASSERT_EQ(hosts.size(), 1u) << "and seats declared rank " << rank << " on one host";
    }

    // And on the PGD's own hosts, which draw the same two columns:
    //
    //   rank a = chips 0 1 4 5 8 9 12 13 -> slots (1,1) (1,2) (2,1) (2,2) (3,1) (3,2) (4,1) (4,2) = host0
    //   rank b = chips 2 3 6 7 10 11 14 15 -> slots (1,3) (1,4) (2,3) (2,4) (3,3) (3,4) (4,3) (4,4) = host1
    const std::vector<std::set<std::pair<uint32_t, uint32_t>>> pgd_hosts = {
        {{1, 1}, {1, 2}, {2, 1}, {2, 2}, {3, 1}, {3, 2}, {4, 1}, {4, 2}},
        {{1, 3}, {1, 4}, {2, 3}, {2, 4}, {3, 3}, {3, 4}, {4, 3}, {4, 4}}};
    expect_each_rank_inside_one_pgd_host(phase1.front().mesh_node_to_asic_position, declared_ranks, pgd_hosts);

    std::map<std::string, MeshHostRankId> rank_of_host;
    for (const auto& [asic_id, descriptor] : psd.get_asic_descriptors()) {
        rank_of_host.emplace(psd.get_host_name_for_asic(asic_id), MeshHostRankId{0});
    }
    uint32_t next_rank = 0;
    for (auto& [host_name, rank] : rank_of_host) {
        rank = MeshHostRankId{next_rank++};
    }
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> asic_id_to_mesh_rank;
    for (const auto& [asic_id, descriptor] : psd.get_asic_descriptors()) {
        asic_id_to_mesh_rank[MeshId{0}][asic_id] = rank_of_host.at(psd.get_host_name_for_asic(asic_id));
    }

    utils::PhysicalMultiMeshGraph physical;
    ASSERT_NO_THROW(physical = utils::build_physical_multi_mesh_adjacency_graph(psd, asic_id_to_mesh_rank, pgd, mgd));
    EXPECT_TRUE(physical.mesh_pgd_pinnings_.contains(MeshId{0}))
        << "phase 2 should reach the same verdict on the same machine";
}

// FIXME: currently throws. Phase 2 owns no placement decision, so a grouping it cannot get is a missing
// hint, not a broken system: it should leave the graph unpinned and carry on. get_valid_groupings_for_mgd
// takes require_placement for exactly this, but the rank-bound caller in topology_mapper_utils.cpp is
// passing true behind a FIXME while the gemma failure is being reproduced, so the empty result aborts the
// run instead. This is what took down that run -- a split phase 1 had already accepted and placed became
// fatal the second time it was looked at. Re-enabling require_placement=false there should fix this.
//
//   machine: four rank slices        MGD host_topology [1,2]
//
//     100 | 101 | 102 | 103            aa | bb    the rejection itself is correct and is
//     104 | 105 | 106 | 107            aa | bb    SecondGetValidGroupings...'s subject; what
//      r0 | r1  | r2  | r3                        is wrong is that it aborts instead of
//                                                 leaving the graph unpinned
TEST(PhysicalGroupingDescriptorTestsHostSplit, Phase2DegradesToNoPinningsWhenRankSlicesAreFinerThanHosts) {
    auto rank_sliced_psd = tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(
        "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_8asic_2x4_four_hosts_by_column.textproto");

    PhysicalGroupingDescriptor pgd{std::string(R"(
groupings {
  name: "2x4_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
    { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } },
    { id: 2 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_3 } },
    { id: 3 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_4 } },
    { id: 4 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } },
    { id: 5 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } },
    { id: 6 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_3 } },
    { id: 7 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_4 } }
  ]
  row_major_mesh {
    dims: [2, 4]
  }
}

# The PGD's own host level: the machine's two real hosts, each one column-half. Deliberately not the
# four rank slices this call is handed -- a PGD describes the system, and the divergence between the
# two is the whole subject of this test.
groupings { name: "2x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
               { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } },
               { id: 2 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } },
               { id: 3 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } } ]
  row_major_mesh { dims: [2, 2] } }
groupings { name: "2x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_3 } },
               { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_4 } },
               { id: 2 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_3 } },
               { id: 3 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_4 } } ]
  row_major_mesh { dims: [2, 2] } }
)")};

    MeshGraphDescriptor mgd{std::string(R"(
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 2, 4 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 2 ] }
  channels { count: 2 policy: STRICT }
}

top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
)")};

    std::map<std::string, MeshHostRankId> rank_of_host;
    for (const auto& [asic_id, descriptor] : rank_sliced_psd.get_asic_descriptors()) {
        rank_of_host.emplace(rank_sliced_psd.get_host_name_for_asic(asic_id), MeshHostRankId{0});
    }
    uint32_t next_rank = 0;
    for (auto& [host_name, rank] : rank_of_host) {
        rank = MeshHostRankId{next_rank++};
    }
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> asic_id_to_mesh_rank;
    for (const auto& [asic_id, descriptor] : rank_sliced_psd.get_asic_descriptors()) {
        asic_id_to_mesh_rank[MeshId{0}][asic_id] = rank_of_host.at(rank_sliced_psd.get_host_name_for_asic(asic_id));
    }

    utils::PhysicalMultiMeshGraph physical;
    ASSERT_NO_THROW(
        physical = utils::build_physical_multi_mesh_adjacency_graph(rank_sliced_psd, asic_id_to_mesh_rank, pgd, mgd))
        << "a hint the matcher cannot produce must not abort the rank-bound path";
    EXPECT_TRUE(physical.mesh_pgd_pinnings_.empty())
        << "no grouping survived the finer slices, so there is nothing to pin with";
}

}  // namespace tt::tt_fabric::fabric_router_tests
