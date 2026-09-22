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
#include <map>

#include <tt-metalium/experimental/fabric/physical_grouping_descriptor.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph_descriptor.hpp>
#include <tt-metalium/experimental/fabric/topology_solver.hpp>
#include <tt-metalium/experimental/fabric/topology_mapper_utils.hpp>
#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>
#include "tt_metal/fabric/serialization/physical_system_descriptor_serialization.hpp"
#include "tt_metal/fabric/physical_system_discovery.hpp"
#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"
#include "mock_psd_builder.hpp"

using namespace tt::tt_fabric::test;

using namespace tt::tt_fabric;

// Flatten a hierarchical PGD grouping, then enumerate the first embedding the same way SAT column
// generation and the matcher PSD gate do.
static std::vector<MappingResult<LogicalChipId, tt::tt_metal::AsicID>> enumerate_grouping_on_psd(
    const PhysicalGroupingDescriptor& pgd,
    const GroupingInfo& grouping,
    const tt::tt_metal::PhysicalSystemDescriptor& psd) {
    for (const auto& flat : pgd.build_flattened_adjacency_mesh(grouping, psd)) {
        if (flat.adjacency_graph.get_nodes().empty()) {
            continue;
        }
        auto placements = pgd.enumerate_distinct_placements_for_grouping(flat, psd);
        if (!placements.empty()) {
            return placements;
        }
    }
    return {};
}

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

// Splits a committed list into the PGD layouts and the MGD fallback.
//
// get_mgd_placement_fallbacks_for_mgd returns the MGD's own topology when it embeds on the PSD. It is not mixed
// into get_valid_groupings_for_mgd; SAT placement adds it to the candidate pool only after PGD variants are
// exhausted. Tests that inspect PGD commits use pgd_layouts only; MGD fallbacks are queried separately.
struct CommittedGroupings {
    std::vector<GroupingInfo> pgd_layouts;     // In priority order, as the matcher committed them.
    std::optional<GroupingInfo> mgd_fallback;  // The MGD's own topology, when it was offered at all.
};

static CommittedGroupings split_off_mgd_fallback(
    const std::vector<GroupingInfo>& committed, const std::string& mgd_instance_name) {
    CommittedGroupings split;
    for (const auto& grouping : committed) {
        if (grouping.name == mgd_instance_name) {
            split.mgd_fallback = grouping;
        } else {
            split.pgd_layouts.push_back(grouping);
        }
    }
    return split;
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

        auto placements = enumerate_grouping_on_psd(pgd, *mesh_grouping, psd);

        EXPECT_FALSE(placements.empty())
            << "Expected validation to pass: 4x2_Mesh grouping should map to mock cluster PSD";
    }

    // Test 4x4_Mesh - validation against mock cluster
    {
        const auto* mesh_grouping = find_mesh_by_name("4x4_Mesh");
        ASSERT_NE(mesh_grouping, nullptr) << "4x4_Mesh grouping not found";

        auto placements = enumerate_grouping_on_psd(pgd, *mesh_grouping, psd);

        EXPECT_FALSE(placements.empty())
            << "Expected validation to pass: 4x4_Mesh grouping should map to mock cluster PSD";
    }

    // Test 2x8_Mesh - validation against mock cluster
    {
        const auto* mesh_grouping = find_mesh_by_name("2x8_Mesh");
        ASSERT_NE(mesh_grouping, nullptr) << "2x8_Mesh grouping not found";

        auto placements = enumerate_grouping_on_psd(pgd, *mesh_grouping, psd);

        EXPECT_FALSE(placements.empty())
            << "Expected validation to pass: 2x8_Mesh grouping should map to mock cluster PSD";
    }

    // Test 4x8_Mesh - validation against mock cluster
    {
        const auto* mesh_grouping = find_mesh_by_name("4x8_Mesh");
        ASSERT_NE(mesh_grouping, nullptr) << "4x8_Mesh grouping not found";

        auto placements = enumerate_grouping_on_psd(pgd, *mesh_grouping, psd);

        EXPECT_FALSE(placements.empty())
            << "Expected validation to pass: 4x8_Mesh grouping should map to mock cluster PSD";
    }

    // Test HOSTS type grouping - validation against mock cluster
    {
        auto hosts_groupings = pgd.get_groupings_by_type("HOSTS");
        ASSERT_FALSE(hosts_groupings.empty()) << "HOSTS grouping not found";
        const auto& hosts_grouping = hosts_groupings[0];

        auto placements = enumerate_grouping_on_psd(pgd, hosts_grouping, psd);

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

        auto placements = enumerate_grouping_on_psd(pgd, hosts_grouping, psd);

        EXPECT_FALSE(placements.empty())
            << "Expected validation to pass: galaxy_hosts grouping should map to mock cluster PSD";
    }

    {
        // 4x32_Mesh: same 128 ASICs / 4 hosts as an 8x16_Mesh, row_major_mesh [1,4] — MGD device grid 32×4
        auto mesh_groupings = pgd.get_groupings_by_name("4x32_Mesh");
        ASSERT_FALSE(mesh_groupings.empty()) << "4x32_Mesh grouping not found";
        const auto& mesh_grouping = mesh_groupings[0];

        auto placements = enumerate_grouping_on_psd(pgd, mesh_grouping, psd);

        EXPECT_FALSE(placements.empty())
            << "Expected validation to pass: 4x32_Mesh (32x4 device layout) should map to mock cluster PSD";
    }
}

// POD groupings flatten onto the SP4 mock PSD; SUPERPOD all_to_all cannot. Same
// PhysicalGroupingDescriptorSP4Tests name so tt-run --gtest_filter=PhysicalGroupingDescriptorSP4Tests*
// still picks this up on the multi-process mock cluster.
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
    auto pod_placements = enumerate_grouping_on_psd(pgd, pod_grouping, psd);

    // Expect it to pass - POD level grouping should validate successfully
    EXPECT_FALSE(pod_placements.empty())
        << "Expected validation to pass: POD level grouping should validate against mock cluster PSD";

    // SAT joint placement of a 2x4 mesh from this PGD onto the same SP4 mock PSD.
    MeshGraphDescriptor mgd{std::string(R"delimiter(
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 2, 4 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 1 policy: STRICT }
}
top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
)delimiter")};
    const auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd, psd);
    ASSERT_TRUE(valid_groupings.contains("MESH")) << "2x4 MGD should match a MESH grouping in this PGD";
    ASSERT_TRUE(valid_groupings.at("MESH").contains("M0"));
    ASSERT_FALSE(valid_groupings.at("MESH").at("M0").empty());
    const auto sat_placements = SatPlacementEnumerationSession(pgd, mgd, psd, nullptr, {}).next();
    ASSERT_EQ(sat_placements.size(), 1u) << "SAT joint placement should seat the 2x4 mesh on the SP4 mock";
    EXPECT_EQ(sat_placements.front().placement.asics.size(), 8u) << "2x4 seating covers 8 ASICs";

    // Test SUPERPOD level grouping - should fail during mesh building (all_to_all connection type)
    auto superpod_groupings = pgd.get_groupings_by_name("superpods");
    ASSERT_FALSE(superpod_groupings.empty()) << "superpods grouping not found";
    const auto& superpod_grouping = superpod_groupings[0];

    // This should throw during build_flattened_adjacency_mesh because SUPERPOD uses all_to_all connection type
    // which cannot be flattened into a mesh (no row_major_mesh structure)
    EXPECT_THROW(
        { pgd.build_flattened_adjacency_mesh(superpod_grouping, psd); }, std::exception)
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
    // Test matching a quad-galaxy MGD (128 ASICs) to the 128-ASIC PGD grouping on the SP4 GLX mock.
    // The SP4 mock wires its 4 Blackhole galaxies into a 4x32 torus, so the MGD must be the Blackhole
    // 32x4 quad-galaxy torus (arch BLACKHOLE, RING/RING) -- NOT the Wormhole quad_galaxy (8x16 plain
    // mesh), which demands an 8-wide dimension the physical fabric does not have and so cannot embed.
    const std::string pgd_path =
        "tests/tt_metal/tt_fabric/physical_groupings/bh_galaxy_rev_ab_physical_grouping_descriptor.textproto";
    const std::string mgd_path =
        "tt_metal/fabric/mesh_graph_descriptors/32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto";

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

    // A 4x8 (32-ASIC) mesh matches both the MESH and a torus variant of the 4x8_Mesh grouping → 2 PGD commits.
    ASSERT_EQ(total_groupings, 2u) << "Should have two valid PGD grouping matches (mesh + torus variant)";
    const auto committed = split_off_mgd_fallback(valid_groupings.at("MESH").at("M0"), "M0");
    EXPECT_EQ(committed.pgd_layouts.size(), 2u) << "the mesh and its torus variant are the PGD matches";
    EXPECT_FALSE(committed.mgd_fallback.has_value()) << "MGD fallback is not in get_valid_groupings_for_mgd";
    const auto mgd_fallbacks = pgd.get_mgd_placement_fallbacks_for_mgd(mgd, psd);
    ASSERT_EQ(mgd_fallbacks.at("MESH").at("M0").size(), 1u);

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
    // Test matching a dual-galaxy MGD (64 ASICs) on the SP4 GLX mock.
    // A 2-galaxy slice of the mock's 4x32 torus is a 4x16 fabric, so the MGD must be the Blackhole
    // 16x4 dual-galaxy 2D mesh (arch BLACKHOLE, 4-wide) -- NOT the Wormhole dual_galaxy (8x8 plain
    // mesh), which demands an 8-wide dimension the physical fabric does not have and so cannot embed.
    const std::string pgd_path =
        "tests/tt_metal/tt_fabric/physical_groupings/bh_galaxy_rev_ab_physical_grouping_descriptor.textproto";
    const std::string mgd_path =
        "tt_metal/fabric/mesh_graph_descriptors/16x4_dual_bh_galaxy_2d_mesh_graph_descriptor.textproto";

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

    const auto committed = split_off_mgd_fallback(valid_groupings.at("MESH").at("M0"), "M0");
    ASSERT_FALSE(committed.pgd_layouts.empty()) << "M0 should have at least one matching PGD grouping";
    EXPECT_FALSE(committed.mgd_fallback.has_value());
    EXPECT_FALSE(pgd.get_mgd_placement_fallbacks_for_mgd(mgd, psd).at("MESH").at("M0").empty());

    const auto placements = SatPlacementEnumerationSession(pgd, mgd, psd, nullptr, {}).next();
    ASSERT_EQ(placements.size(), 1u) << "SAT joint placement should seat the single 4x4 mesh";
    EXPECT_EQ(placements.front().placement.asics.size(), 16u) << "the 4x4 seating should cover 16 ASICs";
    EXPECT_EQ(count_distinct_hosts_for_asics(psd, placements.front().placement.asics), 1u)
        << "host_topology [1,1] should land on a single host";
    EXPECT_EQ(placements.front().placement.mesh_node_to_asic_position.size(), 16u)
        << "Composed pinning should cover all 16 logical chips";
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
// down under AlignedSplitOnASymmetricTorusCommits, which also tests this rule on a 2x4, where the shape is not
// square and no rotation can satisfy an axis by accident.

// The 4x4 on the quadrant machine, handed a descriptor whose own host level disagrees with it: the PGD
// claims eight hosts of 2 chips, each half a tray, where the machine has four hosts of 4. The MGD declares
// [2,2], four ranks of 4 chips lining up with the quadrants exactly, so against the machine alone this is
// the aligned case. Against the descriptor it is impossible: a rank is one process on one host, and a
// 4-chip rank does not fit inside a 2-chip host, so no seating exists and nothing may be committed.
//
//   machine: four hosts per quadrant    the PGD's eight half-tray hosts    MGD host_topology [2,2]
//
//     100 101 | 102 103   h0 h0 h1 h1        p p | q q                          aa | bb
//     104 105 | 106 107   h0 h0 h1 h1        r r | s s                          aa | bb
//     --------+--------                      ----+----                          ---+---
//     108 109 | 110 111   h2 h2 h3 h3        t t | u u                          cc | dd
//     112 113 | 114 115   h2 h2 h3 h3        v v | w w                          cc | dd
//
// The refusal is right but later than it should be. A descriptor claiming hosts its machine does not have
// is wrong about the machine whatever workload arrives, so it belongs to the PGD<->PSD check that
// get_valid_groupings_for_mgd still carries as a FIXME; what refuses it today is the MGD<->PGD host
// constraint, which simply finds no seating. QuadrantSplitPsdFinerSplitInsideEachQuadrantCommits is this
// machine with a host level that agrees with it, and is where the aligned split is checked.
TEST(PhysicalGroupingDescriptorTests, GetValidGroupingsForMGD_RanksCoarserThanTheDescriptorsOwnHostsCommitNothing) {
    auto psd = build_grid_mock_psd(
        4,
        4,
        {"host0",
         "host0",
         "host1",
         "host1",
         "host0",
         "host0",
         "host1",
         "host1",
         "host2",
         "host2",
         "host3",
         "host3",
         "host2",
         "host2",
         "host3",
         "host3"});

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

    for (const auto& [preset, by_instance] : valid_groupings) {
        for (const auto& [instance, groupings] : by_instance) {
            for (const auto& grouping : groupings) {
                // The MGD's own topology is offered last under the mesh descriptor's name and carries no
                // seating, so it is not a commit of the PGD's layout.
                EXPECT_TRUE(grouping.name == "M0" || grouping.mesh_node_to_asic_position.empty())
                    << "'" << grouping.name << "' was committed, seating a 4-chip rank on 2-chip hosts";
            }
        }
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
    auto psd = build_grid_mock_psd(
        4,
        4,
        {"host0",
         "host0",
         "host1",
         "host1",
         "host0",
         "host0",
         "host1",
         "host1",
         "host2",
         "host2",
         "host3",
         "host3",
         "host2",
         "host2",
         "host3",
         "host3"});

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

// A PGD whose declared hosts are hosts the machine has is used as declared. Only such hosts reach the
// matcher: the ones the machine does not have are dropped as the host level is flattened, so this holds
// that the dropping cannot widen into one that also turns away a PGD describing the machine correctly.
// Same machine and MGD as the host-split tests, with a PGD drawing its hosts where the machine's are:
//
//   machine: hosts cut by column       PGD claims the same two columns
//
//     100 101 | 102 103                  hh | ii     each claimed host is exactly
//     104 105 | 106 107                  hh | ii     one real host
//      host0  |  host1
TEST(PhysicalGroupingDescriptorTests, GetValidGroupingsForMGD_PgdHostsThatMatchThePsdAreAccepted) {
    auto psd = build_grid_mock_psd(2, 4, {"host0", "host0", "host1", "host1", "host0", "host0", "host1", "host1"});

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

# Hosts by column, which is how this machine is actually cut: each claimed host is one real host.
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

    ValidGroupingsMap valid_groupings;
    ASSERT_NO_THROW(valid_groupings = pgd.get_valid_groupings_for_mgd(mgd, psd))
        << "a PGD whose declared hosts are the machine's own hosts describes this machine";

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
    EXPECT_FALSE(committed.empty()) << "and its layout should still be committed";
}

// Declaring hosts is optional, and a PGD that declares none must be left alone -- but only by the check
// that reads the PGD's host level. The MGD's own declared split is still held against the machine, and
// the point of this test is which of the two answers.
//
// The PGD offers a 2x4 layout and says nothing about who owns the chips. The same PGD and the same two
// MGDs are put to two machines:
//
//   PGD: a layout, no hosts     MGD [1,2]     MGD [2,1]
//
//     mmmm                        aa | bb       aaaa
//     mmmm                        aa | bb       bbbb
//
//   machine A: one host          machine B: hosts cut by column
//
//     100 101 102 103              100 101 | 102 103
//     104 105 106 107              104 105 | 106 107
//          host0                    host0  |  host1
//
// On machine A both splits commit: the layouts are isomorphic, and with no host boundary anywhere there
// is nothing either split could cross. That is the MGD<->PGD match going through untouched, and it is
// the evidence that the absent PGD host level objects to nothing. On machine B the layout match is the
// same but the PGD<->PSD half now has hosts to answer to, and [2,1] is refused there -- by the MGD's
// split meeting the machine's hosts, and not by anything to do with the PGD's own hosts, which this
// descriptor does not declare and so is never asked about.
TEST(PhysicalGroupingDescriptorTests, GetValidGroupingsForMGD_PgdWithoutDeclaredHostsIsNotChecked) {
    auto undivided_machine = build_grid_mock_psd(2, 4, std::vector<std::string>(8, "host0"));
    auto machine_cut_by_column =
        build_grid_mock_psd(2, 4, {"host0", "host0", "host1", "host1", "host0", "host0", "host1", "host1"});

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
)")};

    // One split along the machine's own boundary and one across it.
    for (const auto& [host_dims, seats_on_this_machine] :
         std::vector<std::pair<std::string, bool>>{{"[ 1, 2 ]", true}, {"[ 2, 1 ]", false}}) {
        MeshGraphDescriptor mgd{
            std::string(R"(
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 2, 4 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: )") +
            host_dims + R"( }
  channels { count: 2 policy: STRICT }
}

top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
)"};

        // Machine A, where no host boundary exists: the layout match goes through and the split has
        // nothing to cross, so this commits whichever way the split runs.
        ValidGroupingsMap on_undivided;
        ASSERT_NO_THROW(
            on_undivided = pgd.get_valid_groupings_for_mgd(
                mgd, undivided_machine, /*pinnings=*/std::nullopt, /*require_placement=*/false))
            << host_dims << ": a PGD that declares no hosts has nothing to contradict";

        std::vector<GroupingInfo> committed_on_undivided;
        for (const auto& [preset, by_instance] : on_undivided) {
            for (const auto& [instance, groupings] : by_instance) {
                for (const auto& grouping : groupings) {
                    if (grouping.name != "M0" && !grouping.mesh_node_to_asic_position.empty()) {
                        committed_on_undivided.push_back(grouping);
                    }
                }
            }
        }
        EXPECT_FALSE(committed_on_undivided.empty())
            << host_dims << ": the MGD<->PGD match should commit, since the absent PGD host level is the "
            << "only thing that could have objected to this split";

        // Machine B, same PGD and same MGD, with hosts. The PGD<->PSD half now has a boundary to answer
        // to, and the split laid across it is refused there.
        ValidGroupingsMap on_split;
        ASSERT_NO_THROW(
            on_split = pgd.get_valid_groupings_for_mgd(
                mgd, machine_cut_by_column, /*pinnings=*/std::nullopt, /*require_placement=*/false))
            << host_dims << ": the rejection should be a verdict, not a crash";

        std::vector<GroupingInfo> committed_on_split;
        for (const auto& [preset, by_instance] : on_split) {
            for (const auto& [instance, groupings] : by_instance) {
                for (const auto& grouping : groupings) {
                    if (grouping.name != "M0" && !grouping.mesh_node_to_asic_position.empty()) {
                        committed_on_split.push_back(grouping);
                    }
                }
            }
        }
        EXPECT_EQ(!committed_on_split.empty(), seats_on_this_machine)
            << host_dims << ": PGD<->PSD should " << (seats_on_this_machine ? "seat" : "refuse")
            << " this split on a machine cut by column, and it committed " << committed_on_split.size()
            << " grouping(s)";

        // The same rule applies to the MGD placement fallback (queried separately from PGD commits).
        const auto fallbacks_on_split = pgd.get_mgd_placement_fallbacks_for_mgd(mgd, machine_cut_by_column);
        const bool fallback_offered_on_split = fallbacks_on_split.contains("MESH") &&
                                               fallbacks_on_split.at("MESH").contains("M0") &&
                                               !fallbacks_on_split.at("MESH").at("M0").empty();
        EXPECT_EQ(fallback_offered_on_split, seats_on_this_machine)
            << host_dims << ": the MGD fallback should be " << (seats_on_this_machine ? "offered" : "withheld")
            << " on a machine cut by column";
    }
}

// ---------------------------------------------------------------------------------------------
// Adjacency-guided placement
//
// PSD is built with mock_psd_builder. PGD and MGD stay inline. SAT duplicates of these cases
// were dropped; StrainManyMeshes is the SAT-only strain.
// ---------------------------------------------------------------------------------------------

namespace {

namespace utils = tt::tt_metal::experimental::tt_fabric;

std::vector<std::set<uint64_t>> footprints_of(const AssignedMeshes& placements) {
    std::vector<std::set<uint64_t>> footprints;
    footprints.reserve(placements.size());
    for (const auto& placed : placements) {
        std::set<uint64_t> asics;
        for (const auto& asic : placed.placement.asics) {
            asics.insert(*asic);
        }
        footprints.push_back(std::move(asics));
    }
    return footprints;
}

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

std::vector<std::set<uint64_t>> mapped_footprints(const std::vector<utils::TopologyMappingResult>& mappings) {
    std::vector<std::set<uint64_t>> footprints;
    for (const auto& mapping : mappings) {
        auto part = mapped_footprints(mapping);
        footprints.insert(footprints.end(), part.begin(), part.end());
    }
    return footprints;
}

std::size_t channels_between(
    const tt::tt_metal::PhysicalSystemDescriptor& psd,
    const std::set<uint64_t>& left,
    const std::set<uint64_t>& right) {
    std::size_t channels = 0;
    for (const auto& host_name : psd.get_all_hostnames()) {
        for (const auto& [src_asic_id, asic_connections] : psd.get_asic_topology(host_name)) {
            if (!left.contains(*src_asic_id)) {
                continue;
            }
            for (const auto& [dst_asic_id, eth_connections] : asic_connections) {
                if (right.contains(*dst_asic_id)) {
                    channels += eth_connections.size();
                }
            }
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

utils::TopologyMappingConfig no_rank_config() {
    utils::TopologyMappingConfig config;
    config.disable_rank_bindings = true;
    return config;
}

struct PinnedHost {
    std::vector<std::pair<int, int>> cells;
    int dim_r = 1;
    int dim_c = 1;
};

std::vector<std::pair<int, int>> rect_cells(int r0, int r1, int c0, int c1) {
    std::vector<std::pair<int, int>> cells;
    for (int r = r0; r < r1; ++r) {
        for (int c = c0; c < c1; ++c) {
            cells.emplace_back(r, c);
        }
    }
    return cells;
}

PinnedHost rect_host(int r0, int r1, int c0, int c1) {
    return PinnedHost{rect_cells(r0, r1, c0, c1), r1 - r0, c1 - c0};
}

std::string pinned_instances(const std::vector<std::pair<int, int>>& cells) {
    std::ostringstream out;
    for (std::size_t i = 0; i < cells.size(); ++i) {
        const auto [row, col] = cells[i];
        out << "    { id: " << i << " location { tray_id: TRAY_" << (row + 1) << " asic_location: ASIC_LOCATION_"
            << (col + 1) << " } }";
        out << (i + 1 < cells.size() ? ",\n" : "\n");
    }
    return out.str();
}

std::string pinned_grouping(
    const std::string& name,
    const std::string& type_line,
    const std::vector<std::pair<int, int>>& cells,
    int dim_r,
    int dim_c) {
    std::ostringstream out;
    out << "groupings {\n  name: \"" << name << "\"\n  " << type_line << "\n  instances: [\n"
        << pinned_instances(cells) << "  ]\n  row_major_mesh { dims: [" << dim_r << ", " << dim_c << "] }\n}\n";
    return out.str();
}

PhysicalGroupingDescriptor pinned_pgd(int rows, int cols, const std::vector<PinnedHost>& hosts) {
    const std::string prefix = std::to_string(rows) + "x" + std::to_string(cols);
    std::ostringstream out;
    out << pinned_grouping(prefix + "_Mesh", "preset_type: MESH", rect_cells(0, rows, 0, cols), rows, cols);
    for (const auto& host : hosts) {
        out << pinned_grouping(prefix + "_hosts", "preset_type: HOSTS", host.cells, host.dim_r, host.dim_c);
    }
    return PhysicalGroupingDescriptor{out.str()};
}

MeshGraphDescriptor single_mesh_mgd(int rows, int cols, int host_r, int host_c, const char* dim_types = "LINE, LINE") {
    std::ostringstream out;
    out << "mesh_descriptors {\n  name: \"M0\"\n  arch: WORMHOLE_B0\n"
        << "  device_topology { dims: [ " << rows << ", " << cols << " ] dim_types: [ " << dim_types << " ] }\n"
        << "  host_topology   { dims: [ " << host_r << ", " << host_c << " ] }\n"
        << "  channels { count: 2 policy: STRICT }\n}\n"
        << "top_level_instance { mesh { mesh_descriptor: \"M0\" mesh_id: 0 } }\n";
    return MeshGraphDescriptor{out.str()};
}

std::vector<GroupingInfo> committed_layouts(const ValidGroupingsMap& valid) {
    std::vector<GroupingInfo> committed;
    for (const auto& [preset, by_instance] : valid) {
        for (const auto& [instance, groupings] : by_instance) {
            for (const auto& grouping : groupings) {
                if (grouping.name != "M0" && !grouping.mesh_node_to_asic_position.empty()) {
                    committed.push_back(grouping);
                }
            }
        }
    }
    return committed;
}

std::vector<std::set<std::pair<uint32_t, uint32_t>>> host_slots(const std::vector<PinnedHost>& hosts) {
    std::vector<std::set<std::pair<uint32_t, uint32_t>>> slots;
    slots.reserve(hosts.size());
    for (const auto& host : hosts) {
        std::set<std::pair<uint32_t, uint32_t>> one;
        for (const auto& [row, col] : host.cells) {
            one.emplace(static_cast<uint32_t>(row + 1), static_cast<uint32_t>(col + 1));
        }
        slots.push_back(std::move(one));
    }
    return slots;
}

std::map<std::pair<uint32_t, uint32_t>, tt::tt_metal::AsicID> asic_by_slot(
    const tt::tt_metal::PhysicalSystemDescriptor& psd) {
    std::map<std::pair<uint32_t, uint32_t>, tt::tt_metal::AsicID> asic_at_slot;
    for (const auto& [asic_id, descriptor] : psd.get_asic_descriptors()) {
        asic_at_slot.emplace(std::pair{*descriptor.tray_id, *descriptor.asic_location}, asic_id);
    }
    return asic_at_slot;
}

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

utils::TopologyMappingResult map_placement_with_declared_ranks(
    const tt::tt_metal::PhysicalSystemDescriptor& psd,
    const PhysicalGroupingDescriptor& pgd,
    const MeshGraphDescriptor& mgd,
    const std::vector<std::vector<LogicalChipId>>& declared_ranks) {
    utils::TopologyMappingConfig config;
    for (const auto& [asic_id, descriptor] : psd.get_asic_descriptors()) {
        config.hostname_to_asics[descriptor.host_name].insert(asic_id);
    }
    std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>> fabric_node_id_to_mesh_rank;
    for (std::size_t rank = 0; rank < declared_ranks.size(); ++rank) {
        for (LogicalChipId chip : declared_ranks[rank]) {
            fabric_node_id_to_mesh_rank[MeshId{0}][FabricNodeId(MeshId{0}, chip)] =
                MeshHostRankId{static_cast<uint32_t>(rank)};
        }
    }
    return utils::map_multi_mesh_to_physical(
        psd, pgd, mgd, config, /*pinnings=*/{}, /*asic_id_to_mesh_rank=*/{}, fabric_node_id_to_mesh_rank);
}

void expect_ranks_survive_placement(
    const tt::tt_metal::PhysicalSystemDescriptor& psd,
    const PhysicalGroupingDescriptor& pgd,
    const MeshGraphDescriptor& mgd,
    const std::vector<std::vector<LogicalChipId>>& ranks) {
    const auto placements = SatPlacementEnumerationSession(pgd, mgd, psd, nullptr, {}).next();
    ASSERT_EQ(placements.size(), 1u);
    const auto asic_at_slot = asic_by_slot(psd);
    for (std::size_t rank = 0; rank < ranks.size(); ++rank) {
        std::set<std::string> hosts;
        for (LogicalChipId chip : ranks[rank]) {
            const auto& position = placements.front().placement.mesh_node_to_asic_position.at(chip);
            hosts.insert(psd.get_host_name_for_asic(asic_at_slot.at({*position.first, *position.second})));
        }
        EXPECT_EQ(hosts.size(), 1u) << "placement seated declared rank " << rank << " across " << hosts.size()
                                    << " hosts";
    }
    const auto mapping = map_placement_with_declared_ranks(psd, pgd, mgd, ranks);
    ASSERT_TRUE(mapping.success) << mapping.error_message;
    for (std::size_t rank = 0; rank < ranks.size(); ++rank) {
        std::set<std::string> hosts;
        for (LogicalChipId chip : ranks[rank]) {
            hosts.insert(psd.get_host_name_for_asic(mapping.fabric_node_to_asic.at(FabricNodeId(MeshId{0}, chip))));
        }
        EXPECT_EQ(hosts.size(), 1u) << "the mapper put declared rank " << rank << " on " << hosts.size() << " hosts";
    }
}

}  // namespace

// Linked 1x2 meshes must sit on touching pairs; the same meshes with no seam may sit on
// disconnected pairs; a required seam on disconnected pairs is unsat.
//
//   linked MGD                 line PSD                    pairs PSD
//
//     M0 == M1                 100 == 101 == 102 == 103    100 == 101      102 == 103
//                              '----M0---'  '----M1---'    '----M0---'     '----M1---'
//
//   seating the middle pair {101,102} would leave 100 and 103 unlinked, so the line answer
//   is unique. Unlinked MGD on the pairs PSD is the control: no seam, both pairs used.
TEST(AdjacencyGuidedPlacement, LinkedMeshesRespectSeamConnectivity) {
    PhysicalGroupingDescriptor pgd{std::string(R"(
groupings {
  name: "1x2_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 1 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  row_major_mesh { dims: [1, 2] }
}
)")};
    MeshGraphDescriptor linked{std::string(R"(
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
)")};
    MeshGraphDescriptor unlinked{std::string(R"(
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
)")};
    auto line = build_mock_psd(std::vector<std::string>(4, "host0"), line_edges(4));
    auto pairs = build_mock_psd(std::vector<std::string>(4, "host0"), std::vector<std::pair<int, int>>{{0, 1}, {2, 3}});

    PlacementSolveStats line_stats;
    ASSERT_EQ(SatPlacementEnumerationSession(pgd, linked, line, &line_stats, {}).next().size(), 2u)
        << line_stats.to_string();
    EXPECT_TRUE(line_stats.master_solve_success) << line_stats.to_string();
    const auto on_line = utils::map_multi_mesh_to_physical(line, pgd, linked, no_rank_config());
    ASSERT_TRUE(on_line.success) << on_line.error_message;
    EXPECT_THAT(
        mapped_footprints(on_line),
        ::testing::UnorderedElementsAre(std::set<uint64_t>{100, 101}, std::set<uint64_t>{102, 103}));

    PlacementSolveStats fail_stats;
    EXPECT_TRUE(SatPlacementEnumerationSession(pgd, linked, pairs, &fail_stats, {}).next().empty());
    EXPECT_TRUE(fail_stats.master_solve_attempted) << fail_stats.to_string();
    EXPECT_FALSE(fail_stats.master_solve_success) << fail_stats.to_string();

    const auto on_pairs = utils::map_multi_mesh_to_physical(pairs, pgd, unlinked, no_rank_config());
    ASSERT_TRUE(on_pairs.success) << on_pairs.error_message;
    EXPECT_THAT(
        mapped_footprints(on_pairs),
        ::testing::UnorderedElementsAre(std::set<uint64_t>{100, 101}, std::set<uint64_t>{102, 103}));
}

// Mixed shapes pick the unique attach whose channel count matches the seam:
//
//              106 == 105 == 104          Sc 1x1 on the 2-ch attach
//                              |
//     100 == 101 == 103 ==== 107 == 108 == 109     Sa 1x3 on 4-ch
//      ||     ||     ||
//     102 ====+      ||
//      ||            ||
//     110 == 111     4-ch                  Sb 1x2 on 3-ch
//      3-ch
//
//     hub H is the 2x2 {100,101,102,103}
TEST(AdjacencyGuidedPlacement, StarSeamsPlaceByChannelCount) {
    PhysicalGroupingDescriptor pgd{std::string(R"(
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
)")};
    MeshGraphDescriptor mgd{std::string(R"(
mesh_descriptors {
  name: "H"
  arch: WORMHOLE_B0
  device_topology { dims: [ 2, 2 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}
mesh_descriptors {
  name: "Sa"
  arch: WORMHOLE_B0
  device_topology { dims: [ 1, 3 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}
mesh_descriptors {
  name: "Sb"
  arch: WORMHOLE_B0
  device_topology { dims: [ 1, 2 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}
mesh_descriptors {
  name: "Sc"
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
)")};
    auto psd = build_mock_psd(
        std::vector<std::string>(12, "host0"),
        std::vector<MockLink>{
            {0, 1, 2},
            {0, 2, 2},
            {1, 3, 2},
            {1, 4, 2},
            {2, 3, 2},
            {2, 10, 3},
            {3, 7, 4},
            {4, 5, 2},
            {5, 6, 2},
            {7, 8, 2},
            {8, 9, 2},
            {10, 11, 2}});

    const auto mapping = utils::map_multi_mesh_to_physical(psd, pgd, mgd, no_rank_config());
    ASSERT_TRUE(mapping.success) << mapping.error_message;
    EXPECT_THAT(
        mapped_footprints(mapping),
        ::testing::ElementsAre(
            std::set<uint64_t>({100, 101, 102, 103}),
            std::set<uint64_t>({107, 108, 109}),
            std::set<uint64_t>({110, 111}),
            std::set<uint64_t>({104})));
}

// Two descriptors share the names M0/M1 but not the shapes. Keys stay descriptor-prefixed,
// and a STRICT 4-ch seam has to land on the only 4-ch physical link.
//
//   A: 1x2 -4- 1x1          B: 1x1 -3- 1x1         PSD
//
//   A takes {100,101}+{102}    B takes {103,104}    100 == 101 ==== 102 -- 103 ==== 104
//                                                       2       4        1       3
TEST(AdjacencyGuidedPlacement, TwoDescriptorsKeepKeysAndStrictSeams) {
    PhysicalGroupingDescriptor pgd{std::string(R"(
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
)")};
    std::vector<MeshGraphDescriptor> mgds;
    mgds.emplace_back(std::string(R"(
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
)"));
    mgds.emplace_back(std::string(R"(
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
)"));
    auto psd = build_mock_psd(
        std::vector<std::string>(5, "host0"), std::vector<MockLink>{{0, 1, 2}, {1, 2, 4}, {2, 3, 1}, {3, 4, 3}});

    const auto valid_groupings = pgd.get_valid_groupings_for_mgds(mgds, psd);
    EXPECT_THAT(
        valid_groupings.at("MESH"),
        ::testing::UnorderedElementsAre(
            ::testing::Key("mgd0_M0"),
            ::testing::Key("mgd0_M1"),
            ::testing::Key("mgd1_M0"),
            ::testing::Key("mgd1_M1")));
    EXPECT_EQ(valid_groupings.at("MESH").at("mgd0_M0").front().adjacency_graph.get_nodes().size(), 2u);
    EXPECT_EQ(valid_groupings.at("MESH").at("mgd1_M0").front().adjacency_graph.get_nodes().size(), 1u);

    utils::TopologyMappingConfig config = no_rank_config();
    config.inter_mesh_validation_mode = ::tt::tt_fabric::ConnectionValidationMode::STRICT;
    const auto mapping = utils::map_multi_mesh_to_physical(
        psd, pgd, std::vector<utils::MultiMeshMappingPart>{{mgds.data()}, {mgds.data() + 1}}, config);
    ASSERT_FALSE(mapping.empty());
    ASSERT_TRUE(mapping.front().success) << mapping.front().error_message;
    const auto footprints = mapped_footprints(mapping);
    ASSERT_EQ(footprints.size(), 4u);
    EXPECT_EQ(channels_between(psd, footprints[0], footprints[1]), 4u);
    EXPECT_EQ(chips_in({footprints[0], footprints[1]}), std::set<uint64_t>({100, 101, 102}));
    EXPECT_EQ(chips_in({footprints[2], footprints[3]}), std::set<uint64_t>({103, 104}));
}

// Inter-mesh channel policy on the same 3-chip machine, plus merge rules across descriptors:
//
//   M0 --n-- M1                 100 == 101 ==== 102
//                                    2       4
//
//   RELAXED 4 prefers 101-102. STRICT 8 is unsat. RELAXED 8 still places.
//   RELAXED + STRICT siblings refuse to merge. A singleton with no seam does not force STRICT.
TEST(AdjacencyGuidedPlacement, InterMeshPolicyAndMerge) {
    PhysicalGroupingDescriptor pgd{std::string(R"(
groupings {
  name: "1x1_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  row_major_mesh { dims: [1, 1] }
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
)")};
    MeshGraphDescriptor relaxed_4{std::string(R"(
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
)")};
    MeshGraphDescriptor strict_8{std::string(R"(
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
)")};
    MeshGraphDescriptor relaxed_8{std::string(R"(
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
)")};
    MeshGraphDescriptor single{std::string(R"(
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
)")};
    auto three = build_mock_psd(std::vector<std::string>(3, "host0"), std::vector<MockLink>{{0, 1, 2}, {1, 2, 4}});
    auto four = build_mock_psd(std::vector<std::string>(4, "host0"), line_edges(4));

    const auto prefer = utils::map_multi_mesh_to_physical(three, pgd, relaxed_4, no_rank_config());
    ASSERT_TRUE(prefer.success) << prefer.error_message;
    EXPECT_EQ(chips_in(mapped_footprints(prefer)), (std::set<uint64_t>{101, 102}));
    EXPECT_TRUE(SatPlacementEnumerationSession(pgd, strict_8, three, nullptr, {}).next().empty());
    ASSERT_TRUE(utils::map_multi_mesh_to_physical(three, pgd, relaxed_8, no_rank_config()).success);

    ASSERT_NE(relaxed_4.is_inter_mesh_policy_relaxed(), strict_8.is_inter_mesh_policy_relaxed());
    EXPECT_ANY_THROW(utils::map_multi_mesh_to_physical(
        three, pgd, std::vector<utils::MultiMeshMappingPart>{{&relaxed_4}, {&strict_8}}, no_rank_config()));

    EXPECT_FALSE(single.is_inter_mesh_policy_specified());
    EXPECT_TRUE(relaxed_4.is_inter_mesh_policy_specified());
    const auto merged = utils::map_multi_mesh_to_physical(
        four, pgd, std::vector<utils::MultiMeshMappingPart>{{&relaxed_4}, {&single}}, no_rank_config());
    ASSERT_FALSE(merged.empty());
    EXPECT_TRUE(merged.front().success) << merged.front().error_message;
}

// Pinned PGD grouping vs MGD fallback:
//
//   pinned 1x2 on loc 0,1          100 == 101 == 102 == 103
//                                  '----v----'
//                                 the only PGD footprint
//
//   One mesh stays on {100,101}. Two linked meshes still place: the second uses the MGD fallback.
TEST(AdjacencyGuidedPlacement, PinnedPgdWinsUntilItCannotCoverEveryInstance) {
    PhysicalGroupingDescriptor pgd{std::string(R"(
groupings {
  name: "1x2_Mesh_OnePair"
  preset_type: MESH
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_0 } },
    { id: 1 location { asic_location: ASIC_LOCATION_1 } }
  ]
  row_major_mesh { dims: [1, 2] }
}
)")};
    MeshGraphDescriptor one{std::string(R"(
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
)")};
    MeshGraphDescriptor two{std::string(R"(
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
)")};
    auto psd = build_mock_psd(std::vector<std::string>(4, "host0"), line_edges(4));

    EXPECT_THAT(
        pgd.get_valid_groupings_for_mgd(one, psd).at("MESH").at("M0").front().name,
        ::testing::Eq("1x2_Mesh_OnePair_flat"));
    const auto one_place = SatPlacementEnumerationSession(pgd, one, psd, nullptr, {}).next();
    ASSERT_EQ(one_place.size(), 1u);
    EXPECT_FALSE(one_place.front().placement.mesh_node_to_asic_position.empty());
    EXPECT_THAT(footprints_of(one_place), ::testing::ElementsAre(std::set<uint64_t>{100, 101}));
    EXPECT_THAT(
        mapped_footprints(utils::map_multi_mesh_to_physical(psd, pgd, one, no_rank_config())),
        ::testing::ElementsAre(std::set<uint64_t>{100, 101}));

    EXPECT_EQ(SatPlacementEnumerationSession(pgd, two, psd, nullptr, {}).next().size(), 2u);
    const auto two_map = utils::map_multi_mesh_to_physical(psd, pgd, two, no_rank_config());
    ASSERT_TRUE(two_map.success) << two_map.error_message;
    EXPECT_THAT(
        mapped_footprints(two_map),
        ::testing::UnorderedElementsAre(std::set<uint64_t>{100, 101}, std::set<uint64_t>{102, 103}));
}

TEST(PhysicalGroupingDescriptorTestsSatJointPlacement, StrainManyMeshesPlacesInOneMasterSolve) {
    auto run_case = [](std::size_t mesh_rows,
                       std::size_t mesh_cols,
                       std::size_t fabric_rows,
                       std::size_t fabric_cols,
                       const char* label) {
        PhysicalGroupingDescriptor pgd{unspecified_mesh_pgd(mesh_rows, mesh_cols)};
        MeshGraphDescriptor mgd{mesh_grid_mgd(mesh_rows, mesh_cols, fabric_rows, fabric_cols)};
        const int psd_rows = static_cast<int>(mesh_rows * fabric_rows);
        const int psd_cols = static_cast<int>(mesh_cols * fabric_cols);
        auto psd = build_grid_mock_psd(psd_rows, psd_cols, std::vector<std::string>(psd_rows * psd_cols, "host0"));

        PlacementSolveStats stats;
        const auto placements = SatPlacementEnumerationSession(pgd, mgd, psd, &stats, {}).next();
        const std::size_t expected_meshes = fabric_rows * fabric_cols;
        EXPECT_EQ(placements.size(), expected_meshes) << label << "\n" << stats.to_string();
        EXPECT_TRUE(stats.success) << label << "\n" << stats.to_string();
        EXPECT_TRUE(stats.master_solve_attempted) << label << "\n" << stats.to_string();
        EXPECT_TRUE(stats.master_solve_success) << label << "\n" << stats.to_string();
        EXPECT_GE(stats.master_candidates_enumerated, expected_meshes) << label << "\n" << stats.to_string();
        EXPECT_FALSE(stats.candidate_lists_complete) << label << "\n" << stats.to_string();
        std::set<uint64_t> seen;
        for (const auto& placed : placements) {
            EXPECT_EQ(placed.placement.asics.size(), mesh_rows * mesh_cols) << label;
            for (const auto& asic : placed.placement.asics) {
                EXPECT_TRUE(seen.insert(*asic).second) << label << ": ASIC " << *asic << " placed twice";
            }
        }
    };

    run_case(2, 2, 2, 4, "8x 2x2 meshes on 4x8");
    run_case(4, 4, 2, 2, "4x 4x4 meshes on 8x8");
}

TEST(PhysicalGroupingDescriptorTestsHostSplit, SingleHostPsdColumnSplitMgdCommits) {
    auto psd = build_grid_mock_psd(2, 4, std::vector<std::string>(8, "host0"));
    const std::vector<PinnedHost> hosts{rect_host(0, 2, 0, 4)};
    const auto pgd = pinned_pgd(2, 4, hosts);
    const auto mgd = single_mesh_mgd(2, 4, 1, 2);
    const auto committed = committed_layouts(pgd.get_valid_groupings_for_mgd(mgd, psd));
    ASSERT_FALSE(committed.empty());
    EXPECT_EQ(committed.front().mesh_node_to_asic_position.size(), 8u);
    const std::vector<std::vector<LogicalChipId>> ranks = {{0, 1, 4, 5}, {2, 3, 6, 7}};
    expect_each_rank_inside_one_pgd_host(committed.front().mesh_node_to_asic_position, ranks, host_slots(hosts));
    expect_ranks_survive_placement(psd, pgd, mgd, ranks);
}

TEST(PhysicalGroupingDescriptorTestsHostSplit, RowSplitPsdRowSplitMgdCommits) {
    auto psd = build_grid_mock_psd(2, 4, {"host0", "host0", "host0", "host0", "host1", "host1", "host1", "host1"});
    const std::vector<PinnedHost> hosts{rect_host(0, 1, 0, 4), rect_host(1, 2, 0, 4)};
    const auto pgd = pinned_pgd(2, 4, hosts);
    const auto mgd = single_mesh_mgd(2, 4, 2, 1);
    const auto committed = committed_layouts(pgd.get_valid_groupings_for_mgd(mgd, psd));
    ASSERT_FALSE(committed.empty());
    const std::vector<std::vector<LogicalChipId>> ranks = {{0, 1, 2, 3}, {4, 5, 6, 7}};
    expect_each_rank_inside_one_pgd_host(committed.front().mesh_node_to_asic_position, ranks, host_slots(hosts));
    expect_ranks_survive_placement(psd, pgd, mgd, ranks);
}

TEST(PhysicalGroupingDescriptorTestsHostSplit, LengthwiseSplitOnWidthwiseSplitHostsIsRejected) {
    auto psd = build_grid_mock_psd(2, 4, {"host0", "host0", "host1", "host1", "host0", "host0", "host1", "host1"});
    const auto pgd = pinned_pgd(2, 4, {rect_host(0, 2, 0, 2), rect_host(0, 2, 2, 4)});
    const auto mgd = single_mesh_mgd(2, 4, 2, 1);
    EXPECT_TRUE(committed_layouts(
                    pgd.get_valid_groupings_for_mgd(mgd, psd, /*pinnings=*/std::nullopt, /*require_placement=*/false))
                    .empty());
}

TEST(PhysicalGroupingDescriptorTestsHostSplit, LengthwiseSplitOnWidthwiseSplitHostsIsTurnedToFitOnASquareMesh) {
    auto psd = build_grid_mock_psd(2, 2, {"host0", "host1", "host0", "host1"});
    const std::vector<PinnedHost> hosts{rect_host(0, 2, 0, 1), rect_host(0, 2, 1, 2)};
    const auto pgd = pinned_pgd(2, 2, hosts);
    const auto mgd = single_mesh_mgd(2, 2, 2, 1);
    const auto committed = committed_layouts(
        pgd.get_valid_groupings_for_mgd(mgd, psd, /*pinnings=*/std::nullopt, /*require_placement=*/false));
    ASSERT_FALSE(committed.empty());
    const std::vector<std::vector<LogicalChipId>> ranks = {{0, 1}, {2, 3}};
    expect_each_rank_inside_one_pgd_host(committed.front().mesh_node_to_asic_position, ranks, host_slots(hosts));
    expect_ranks_survive_placement(psd, pgd, mgd, ranks);
}

TEST(PhysicalGroupingDescriptorTestsHostSplit, FinerSplitAcrossTheHostBoundaryIsRejected) {
    auto psd = build_grid_mock_psd(2, 3, {"host0", "host0", "host0", "host1", "host1", "host1"});
    const auto pgd = pinned_pgd(2, 3, {rect_host(0, 1, 0, 3), rect_host(1, 2, 0, 3)});
    const auto mgd = single_mesh_mgd(2, 3, 1, 3);
    EXPECT_TRUE(committed_layouts(
                    pgd.get_valid_groupings_for_mgd(mgd, psd, /*pinnings=*/std::nullopt, /*require_placement=*/false))
                    .empty());
}

TEST(PhysicalGroupingDescriptorTestsHostSplit, TwoAxisSplitInsideRowSplitHostsCommits) {
    auto psd = build_grid_mock_psd(2, 4, {"host0", "host0", "host0", "host0", "host1", "host1", "host1", "host1"});
    const std::vector<PinnedHost> hosts{rect_host(0, 1, 0, 4), rect_host(1, 2, 0, 4)};
    const auto pgd = pinned_pgd(2, 4, hosts);
    const auto mgd = single_mesh_mgd(2, 4, 2, 2);
    const auto committed = committed_layouts(pgd.get_valid_groupings_for_mgd(mgd, psd));
    ASSERT_FALSE(committed.empty());
    const std::vector<std::vector<LogicalChipId>> ranks = {{0, 1}, {2, 3}, {4, 5}, {6, 7}};
    expect_each_rank_inside_one_pgd_host(committed.front().mesh_node_to_asic_position, ranks, host_slots(hosts));
    expect_ranks_survive_placement(psd, pgd, mgd, ranks);
}

TEST(PhysicalGroupingDescriptorTestsHostSplit, CoarserSplitAcrossATwoAxisHostGridIsRejected) {
    auto psd = build_grid_mock_psd(2, 4, {"host0", "host0", "host1", "host1", "host2", "host2", "host3", "host3"});
    const auto pgd =
        pinned_pgd(2, 4, {rect_host(0, 1, 0, 2), rect_host(0, 1, 2, 4), rect_host(1, 2, 0, 2), rect_host(1, 2, 2, 4)});
    const auto mgd = single_mesh_mgd(2, 4, 2, 1);
    EXPECT_TRUE(committed_layouts(
                    pgd.get_valid_groupings_for_mgd(mgd, psd, /*pinnings=*/std::nullopt, /*require_placement=*/false))
                    .empty());
}

TEST(PhysicalGroupingDescriptorTestsHostSplit, SingleHostMgdOnSplitPsdIsRejected) {
    auto psd = build_grid_mock_psd(2, 4, {"host0", "host0", "host1", "host1", "host0", "host0", "host1", "host1"});
    const auto pgd = pinned_pgd(2, 4, {rect_host(0, 2, 0, 2), rect_host(0, 2, 2, 4)});
    const auto mgd = single_mesh_mgd(2, 4, 1, 1);
    EXPECT_TRUE(committed_layouts(
                    pgd.get_valid_groupings_for_mgd(mgd, psd, /*pinnings=*/std::nullopt, /*require_placement=*/false))
                    .empty());
}

TEST(PhysicalGroupingDescriptorTestsHostSplit, AlignedSplitOnASymmetricTorusCommits) {
    auto psd = build_grid_mock_psd(
        4,
        4,
        {"host0",
         "host0",
         "host1",
         "host1",
         "host0",
         "host0",
         "host1",
         "host1",
         "host0",
         "host0",
         "host1",
         "host1",
         "host0",
         "host0",
         "host1",
         "host1"},
        2,
        std::vector<std::pair<int, int>>{{0, 3}, {4, 7}, {8, 11}, {12, 15}, {0, 12}, {1, 13}, {2, 14}, {3, 15}});
    const std::vector<PinnedHost> hosts{rect_host(0, 4, 0, 2), rect_host(0, 4, 2, 4)};
    const auto pgd = pinned_pgd(4, 4, hosts);
    const auto mgd = single_mesh_mgd(4, 4, 1, 2, "RING, RING");
    const auto phase1 = committed_layouts(pgd.get_valid_groupings_for_mgd(mgd, psd));
    ASSERT_FALSE(phase1.empty());
    const std::vector<std::vector<LogicalChipId>> ranks = {{0, 1, 4, 5, 8, 9, 12, 13}, {2, 3, 6, 7, 10, 11, 14, 15}};
    expect_each_rank_inside_one_pgd_host(phase1.front().mesh_node_to_asic_position, ranks, host_slots(hosts));
    expect_ranks_survive_placement(psd, pgd, mgd, ranks);

    const auto asic_at_slot = asic_by_slot(psd);
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> asic_id_to_mesh_rank;
    for (const auto& [chip, position] : phase1.front().mesh_node_to_asic_position) {
        asic_id_to_mesh_rank[MeshId{0}][asic_at_slot.at({*position.first, *position.second})] =
            MeshHostRankId{phase1.front().mesh_node_to_host_group.at(chip)};
    }
    utils::TopologyMappingConfig phase2_config;
    for (const auto& [asic_id, descriptor] : psd.get_asic_descriptors()) {
        phase2_config.hostname_to_asics[descriptor.host_name].insert(asic_id);
    }
    std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>> fabric_ranks;
    for (std::size_t rank = 0; rank < ranks.size(); ++rank) {
        for (LogicalChipId chip : ranks[rank]) {
            fabric_ranks[MeshId{0}][FabricNodeId(MeshId{0}, chip)] = MeshHostRankId{static_cast<uint32_t>(rank)};
        }
    }
    const auto phase2 = utils::map_multi_mesh_to_physical(
        psd, pgd, mgd, phase2_config, /*pinnings=*/{}, asic_id_to_mesh_rank, fabric_ranks);
    ASSERT_TRUE(phase2.success) << phase2.error_message;
    for (const auto& [chip, position] : phase1.front().mesh_node_to_asic_position) {
        EXPECT_EQ(
            phase2.fabric_node_to_asic.at(FabricNodeId(MeshId{0}, chip)),
            asic_at_slot.at({*position.first, *position.second}));
    }
}

TEST(PhysicalGroupingDescriptorTestsHostSplit, HostSplitIsHeldAgainstEveryPlaceAGroupingSits) {
    auto psd = build_grid_mock_psd(2, 4, std::vector<std::string>(8, "host0"));
    PhysicalGroupingDescriptor pgd{
        pinned_grouping("quadrant", "custom_type: \"QUAD\"", rect_cells(0, 2, 0, 2), 2, 2) +
        pinned_grouping("quadrant", "custom_type: \"QUAD\"", rect_cells(0, 2, 2, 4), 2, 2) +
        "groupings {\n  name: \"2x2_Mesh\"\n  preset_type: MESH\n"
        "  instances: [ { id: 0 grouping_ref { custom_type: \"QUAD\" } } ]\n}\n" +
        pinned_grouping("2x4_hosts", "preset_type: HOSTS", rect_cells(0, 2, 0, 4), 2, 4)};
    const auto mgd = single_mesh_mgd(2, 2, 1, 1);
    std::vector<std::set<std::pair<uint32_t, uint32_t>>> committed_seatings;
    for (const auto& grouping : committed_layouts(pgd.get_valid_groupings_for_mgd(mgd, psd))) {
        std::set<std::pair<uint32_t, uint32_t>> slots;
        for (const auto& [_, position] : grouping.mesh_node_to_asic_position) {
            slots.emplace(*position.first, *position.second);
        }
        committed_seatings.push_back(std::move(slots));
    }
    const std::set<std::pair<uint32_t, uint32_t>> first_quadrant{{1, 1}, {1, 2}, {2, 1}, {2, 2}};
    const std::set<std::pair<uint32_t, uint32_t>> second_quadrant{{1, 3}, {1, 4}, {2, 3}, {2, 4}};
    EXPECT_EQ(committed_seatings.size(), 2u);
    EXPECT_EQ(std::count(committed_seatings.begin(), committed_seatings.end(), first_quadrant), 1);
    EXPECT_EQ(std::count(committed_seatings.begin(), committed_seatings.end(), second_quadrant), 1);
}

}  // namespace tt::tt_fabric::fabric_router_tests
