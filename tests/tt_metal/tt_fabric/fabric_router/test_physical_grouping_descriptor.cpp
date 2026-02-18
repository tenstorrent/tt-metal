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
#include <iostream>

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

// Helper to get required groupings (TRAY_1-4, hosts, meshes) - can be prepended to any test proto
static std::string get_required_groupings() {
    return R"proto(
        groupings {
          name: "tray_1"
          preset_type: TRAY_1
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "tray_2"
          preset_type: TRAY_2
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "tray_3"
          preset_type: TRAY_3
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "tray_4"
          preset_type: TRAY_4
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "hosts_required"
          custom_type: "hosts"
          instances:
          [ {
            id: 0
            grouping_ref { preset_type: TRAY_1 }
          }]
        }
    )proto";
}

// Helper to wrap a test proto with required groupings (adds meshes if not present)
static std::string wrap_with_required_groupings(const std::string& test_proto) {
    bool has_meshes = test_proto.find("custom_type: \"meshes\"") != std::string::npos ||
                      test_proto.find("preset_type: MESH") != std::string::npos;

    if (!has_meshes) {
        return get_required_groupings() + R"proto(
                   groupings {
                     name: "meshes_required"
                     custom_type: "meshes"
                     instances:
                     [ { id: 0 asic_location: ASIC_LOCATION_1 }]
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
          all_to_all {}
        }
    )proto");

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
    const std::string text_proto = wrap_with_required_groupings(R"proto(
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
          row_major_mesh { dims: [ 2, 2 ] }
        }
    )proto");

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

TEST(PhysicalGroupingDescriptorTests, AdjacencyGraph_CustomConnections) {
    const std::string text_proto = wrap_with_required_groupings(R"proto(
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
            [ { src_instance: 0 dst_instance: 1 }
              , { src_instance: 0 dst_instance: 2 }
              , { src_instance: 1 dst_instance: 2 }]
          }
        }
    )proto");

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

// ============================================================================
// VALID CONFIGURATION TESTS
// ============================================================================

TEST(PhysicalGroupingDescriptorTests, ParsesValidBasicConfiguration) {
    const std::string text_proto = wrap_with_required_groupings(R"proto(
        groupings {
          name: "trays_1"
          custom_type: "trays"
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }
            , { id: 1 asic_location: ASIC_LOCATION_2 }]
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
        "tests/tt_metal/tt_fabric/physical_groupings/triple_16x8_quad_bh_galaxy_physical_groupings.textproto";
    EXPECT_NO_THROW({ PhysicalGroupingDescriptor desc(text_proto_file_path); });
}

// ============================================================================
// VALIDATION TESTS
// ============================================================================

TEST(PhysicalGroupingDescriptorTests, ValidationSucceedsWithAllRequiredGroupings) {
    // Test that validation passes when all required groupings are present:
    // - Exactly one of each TRAY_1, TRAY_2, TRAY_3, TRAY_4
    // - Exactly one custom_type "hosts"
    // - At least one "meshes" grouping
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

TEST(PhysicalGroupingDescriptorTests, ValidationFailsWhenTRAY1Missing) {
    const std::string text_proto = R"proto(
        groupings {
          name: "tray_2"
          preset_type: TRAY_2
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "tray_3"
          preset_type: TRAY_3
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "tray_4"
          preset_type: TRAY_4
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "hosts_1"
          custom_type: "hosts"
          instances:
          [ {
            id: 0
            grouping_ref { preset_type: TRAY_1 }
          }]
        }
        groupings {
          name: "meshes_1"
          custom_type: "meshes"
          instances:
          [ {
            id: 0
            grouping_ref { custom_type: "hosts" }
          }]
        }
    )proto";

    EXPECT_THAT(
        ([&]() { PhysicalGroupingDescriptor desc(text_proto); }),
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Exactly one grouping with preset_type 'TRAY_1' is required but none are defined")));
}

TEST(PhysicalGroupingDescriptorTests, ValidationFailsWhenHostsMissing) {
    const std::string text_proto = R"proto(groupings {
                                             name: "tray_1"
                                             preset_type: TRAY_1
                                             instances:
                                             [ { id: 0 asic_location: ASIC_LOCATION_1 }]
                                           }
                                           groupings {
                                             name: "tray_2"
                                             preset_type: TRAY_2
                                             instances:
                                             [ { id: 0 asic_location: ASIC_LOCATION_1 }]
                                           }
                                           groupings {
                                             name: "tray_3"
                                             preset_type: TRAY_3
                                             instances:
                                             [ { id: 0 asic_location: ASIC_LOCATION_1 }]
                                           }
                                           groupings {
                                             name: "tray_4"
                                             preset_type: TRAY_4
                                             instances:
                                             [ { id: 0 asic_location: ASIC_LOCATION_1 }]
                                           }
                                           groupings {
                                             name: "meshes_1"
                                             custom_type: "meshes"
                                             instances:
                                             [ {
                                               id: 0
                                               grouping_ref { preset_type: TRAY_1 }
                                             }]
                                           }
    )proto";

    EXPECT_THAT(
        ([&]() { PhysicalGroupingDescriptor desc(text_proto); }),
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Exactly one grouping with custom_type 'hosts' is required but none are defined")));
}

TEST(PhysicalGroupingDescriptorTests, ValidationFailsWhenDuplicateTRAY1) {
    const std::string text_proto = R"proto(
        groupings {
          name: "tray_1_a"
          preset_type: TRAY_1
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "tray_1_b"
          preset_type: TRAY_1
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "tray_2"
          preset_type: TRAY_2
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "tray_3"
          preset_type: TRAY_3
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "tray_4"
          preset_type: TRAY_4
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "hosts_1"
          custom_type: "hosts"
          instances:
          [ {
            id: 0
            grouping_ref { preset_type: TRAY_1 }
          }]
        }
        groupings {
          name: "meshes_1"
          custom_type: "meshes"
          instances:
          [ {
            id: 0
            grouping_ref { custom_type: "hosts" }
          }]
        }
    )proto";

    EXPECT_THAT(
        ([&]() { PhysicalGroupingDescriptor desc(text_proto); }),
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Exactly one grouping with preset_type 'TRAY_1' is required but 2 are defined")));
}

TEST(PhysicalGroupingDescriptorTests, ValidationFailsWhenDuplicateHosts) {
    const std::string text_proto = R"proto(
        groupings {
          name: "tray_1"
          preset_type: TRAY_1
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "tray_2"
          preset_type: TRAY_2
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "tray_3"
          preset_type: TRAY_3
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "tray_4"
          preset_type: TRAY_4
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "hosts_1"
          custom_type: "hosts"
          instances:
          [ {
            id: 0
            grouping_ref { preset_type: TRAY_1 }
          }]
        }
        groupings {
          name: "hosts_2"
          custom_type: "hosts"
          instances:
          [ {
            id: 0
            grouping_ref { preset_type: TRAY_1 }
          }]
        }
        groupings {
          name: "meshes_1"
          custom_type: "meshes"
          instances:
          [ {
            id: 0
            grouping_ref { custom_type: "hosts" }
          }]
        }
    )proto";

    EXPECT_THAT(
        ([&]() { PhysicalGroupingDescriptor desc(text_proto); }),
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Exactly one grouping with custom_type 'hosts' is required but 2 are defined")));
}

TEST(PhysicalGroupingDescriptorTests, ValidationFailsWhenNonLeafGroupingUsesASICLocations) {
    // Test that a non-leaf grouping (one with grouping references) cannot also use ASIC locations
    const std::string text_proto = get_required_groupings() + R"proto(
        groupings {
          name: "meshes_required"
          custom_type: "meshes"
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "pods_bad"
          custom_type: "pods"
          instances:
          [ {
            id: 0
            grouping_ref { custom_type: "meshes" }
          }
            , { id: 1 asic_location: ASIC_LOCATION_2 }]
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
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
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

TEST(PhysicalGroupingDescriptorTests, DuplicateNamesAreUniquified) {
    const std::string text_proto = wrap_with_required_groupings(R"proto(
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
          all_to_all {}
        }
    )proto");
    ;

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
    const std::string text_proto = wrap_with_required_groupings(R"proto(
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
          all_to_all {}
        }
    )proto");
    ;

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
          [ { id: 0 asic_location: ASIC_LOCATION_1 }
            , { id: 1 asic_location: ASIC_LOCATION_2 }]
          row_major_mesh { dims: [ 1, 2 ] }
        }
        groupings {
          name: "halftray_4"
          custom_type: "halftray"
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_3 }
            , { id: 1 asic_location: ASIC_LOCATION_4 }]
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
    ;

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
    const std::string text_proto = wrap_with_required_groupings(R"proto(
        groupings {
          name: "trays_2"
          custom_type: "trays"
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }
            , { id: 1 asic_location: ASIC_LOCATION_2 }]
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
    // Count includes required groupings: TRAY_1-4 (4), hosts (1), plus trays (1), meshes (1), pods (1) = 8 total
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
          [ { id: 0 asic_location: ASIC_LOCATION_1 }
            , { id: 1 asic_location: ASIC_LOCATION_2 }
            , { id: 2 asic_location: ASIC_LOCATION_3 }
            , { id: 3 asic_location: ASIC_LOCATION_4 }]
          row_major_mesh { dims: [ 2, 2 ] }
        }
    )proto");
    ;

    PhysicalGroupingDescriptor desc(text_proto);
    auto meshes = desc.get_groupings_by_name("meshes");
    ASSERT_EQ(meshes.size(), 1);
    EXPECT_EQ(meshes[0].asic_count, 4u);
}

TEST(PhysicalGroupingDescriptorTests, AsicCountCalculation_NestedGroupings) {
    const std::string text_proto = wrap_with_required_groupings(R"proto(
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
    auto trays = desc.get_groupings_by_name("trays");
    auto pods = desc.get_groupings_by_name("pods");
    auto meshes = desc.get_groupings_by_name("meshes");
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
          preset_type: TRAY_1
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "tray_2"
          preset_type: TRAY_2
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "tray_3"
          preset_type: TRAY_3
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "tray_4"
          preset_type: TRAY_4
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "hosts_1"
          custom_type: "hosts"
          instances:
          [ {
            id: 0
            grouping_ref { preset_type: TRAY_1 }
          }]
        }
        groupings {
          name: "meshes_1"
          custom_type: "meshes"
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "tray_2x4"
          custom_type: "tray_2x4"
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }
            , { id: 1 asic_location: ASIC_LOCATION_2 }
            , { id: 2 asic_location: ASIC_LOCATION_3 }
            , { id: 3 asic_location: ASIC_LOCATION_4 }
            , { id: 4 asic_location: ASIC_LOCATION_5 }
            , { id: 5 asic_location: ASIC_LOCATION_6 }
            , { id: 6 asic_location: ASIC_LOCATION_7 }
            , { id: 7 asic_location: ASIC_LOCATION_8 }]
          row_major_mesh { dims: [ 2, 4 ] }
        }
    )proto";

    PhysicalGroupingDescriptor desc_2x4(text_proto_2x4);
    auto trays_2x4 = desc_2x4.get_groupings_by_name("tray_2x4");
    ASSERT_EQ(trays_2x4.size(), 1u) << "Should have one TRAY_1 grouping";
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
          preset_type: TRAY_1
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "tray_2"
          preset_type: TRAY_2
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "tray_3"
          preset_type: TRAY_3
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "tray_4"
          preset_type: TRAY_4
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "hosts_1"
          custom_type: "hosts"
          instances:
          [ {
            id: 0
            grouping_ref { preset_type: TRAY_1 }
          }]
        }
        groupings {
          name: "meshes_1"
          custom_type: "meshes"
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "mesh_1x4"
          custom_type: "mesh"
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }
            , { id: 1 asic_location: ASIC_LOCATION_2 }
            , { id: 2 asic_location: ASIC_LOCATION_3 }
            , { id: 3 asic_location: ASIC_LOCATION_4 }]
          row_major_mesh { dims: [ 1, 4 ] }
        }
    )proto";

    PhysicalGroupingDescriptor desc_1x4(text_proto_1x4);
    auto meshes_1x4 = desc_1x4.get_groupings_by_name("mesh");
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
          preset_type: TRAY_1
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "tray_2"
          preset_type: TRAY_2
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "tray_3"
          preset_type: TRAY_3
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "tray_4"
          preset_type: TRAY_4
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "hosts_1"
          custom_type: "hosts"
          instances:
          [ {
            id: 0
            grouping_ref { preset_type: TRAY_1 }
          }]
        }
        groupings {
          name: "meshes_1"
          custom_type: "meshes"
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "mesh_4x1"
          custom_type: "mesh"
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }
            , { id: 1 asic_location: ASIC_LOCATION_2 }
            , { id: 2 asic_location: ASIC_LOCATION_3 }
            , { id: 3 asic_location: ASIC_LOCATION_4 }]
          row_major_mesh { dims: [ 4, 1 ] }
        }
    )proto";

    PhysicalGroupingDescriptor desc_4x1(text_proto_4x1);
    auto meshes_4x1 = desc_4x1.get_groupings_by_name("mesh");
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
          preset_type: TRAY_1
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "tray_2"
          preset_type: TRAY_2
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "tray_3"
          preset_type: TRAY_3
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "tray_4"
          preset_type: TRAY_4
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "hosts_1"
          custom_type: "hosts"
          instances:
          [ {
            id: 0
            grouping_ref { preset_type: TRAY_1 }
          }]
        }
        groupings {
          name: "mesh_1x1"
          preset_type: MESH
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
          row_major_mesh { dims: [ 1, 1 ] }
        }
    )proto";

    PhysicalGroupingDescriptor desc_1x1(text_proto_1x1);
    auto meshes_1x1 = desc_1x1.get_groupings_by_name("MESH");
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
        "tests/tt_metal/tt_fabric/physical_groupings/triple_16x8_quad_bh_galaxy_physical_groupings.textproto";
    PhysicalGroupingDescriptor desc(text_proto_file_path);

    // Get one of the MESH grouping infos - "8x16_Mesh" which has 4 hosts in a 2x2 grid
    auto mesh_groupings = desc.get_groupings_by_name("MESH");
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

    // Build the flattened adjacency mesh
    // Note: This function is currently not implemented, so the test will fail until it's implemented
    AdjacencyGraph<uint32_t> flattened_mesh = desc.build_flattened_adjacency_mesh(mesh_8x16);

    // Verify the result is a valid adjacency graph
    // The flattened mesh should have 128 nodes (one per ASIC)
    auto nodes = flattened_mesh.get_nodes();
    EXPECT_EQ(nodes.size(), 128u) << "Flattened mesh should have 128 nodes (one per ASIC)";

    // Verify that nodes are connected (each node should have neighbors in a 2D mesh)
    for (uint32_t node : nodes) {
        const auto& neighbors = flattened_mesh.get_neighbors(node);
        EXPECT_GE(neighbors.size(), 2u) << "Node " << node << " should have at least 2 neighbors";
        EXPECT_LE(neighbors.size(), 4u) << "Node " << node << " should have at most 4 neighbors";
    }
}

TEST(PhysicalGroupingDescriptorTests, BuildFlattenedAdjacencyMesh_4x4Mesh) {
    const std::filesystem::path text_proto_file_path =
        "tests/tt_metal/tt_fabric/physical_groupings/triple_16x8_quad_bh_galaxy_physical_groupings.textproto";
    PhysicalGroupingDescriptor desc(text_proto_file_path);

    GroupingInfo mesh_4x4;
    bool found = false;
    for (const auto& mesh : desc.get_groupings_by_name("MESH")) {
        if (mesh.name == "4x4_Mesh") {
            mesh_4x4 = mesh;
            found = true;
            break;
        }
    }
    ASSERT_TRUE(found) << "Expected to find '4x4_Mesh' grouping";

    EXPECT_EQ(mesh_4x4.asic_count, 16u) << "4x4_Mesh should have 16 ASICs (2 trays * 8 ASICs each)";
    EXPECT_EQ(mesh_4x4.items.size(), 2u) << "4x4_Mesh should have 2 instances (trays)";

    AdjacencyGraph<uint32_t> flattened_mesh = desc.build_flattened_adjacency_mesh(mesh_4x4);

    auto nodes = flattened_mesh.get_nodes();
    EXPECT_EQ(nodes.size(), 16u) << "Flattened mesh should have 16 nodes";

    for (uint32_t node : nodes) {
        const auto& neighbors = flattened_mesh.get_neighbors(node);
        EXPECT_GE(neighbors.size(), 2u) << "Node " << node << " should have at least 2 neighbors";
        EXPECT_LE(neighbors.size(), 4u) << "Node " << node << " should have at most 4 neighbors";
    }
}

TEST(PhysicalGroupingDescriptorTests, BuildFlattenedAdjacencyMesh_2x8Mesh) {
    const std::filesystem::path text_proto_file_path =
        "tests/tt_metal/tt_fabric/physical_groupings/triple_16x8_quad_bh_galaxy_physical_groupings.textproto";
    PhysicalGroupingDescriptor desc(text_proto_file_path);

    GroupingInfo mesh_2x8;
    bool found = false;
    for (const auto& mesh : desc.get_groupings_by_name("MESH")) {
        if (mesh.name == "2x8_Mesh") {
            mesh_2x8 = mesh;
            found = true;
            break;
        }
    }
    ASSERT_TRUE(found) << "Expected to find '2x8_Mesh' grouping";

    EXPECT_EQ(mesh_2x8.asic_count, 16u) << "2x8_Mesh should have 16 ASICs (2 trays * 8 ASICs each)";
    EXPECT_EQ(mesh_2x8.items.size(), 2u) << "2x8_Mesh should have 2 instances (trays)";

    AdjacencyGraph<uint32_t> flattened_mesh = desc.build_flattened_adjacency_mesh(mesh_2x8);

    auto nodes = flattened_mesh.get_nodes();
    EXPECT_EQ(nodes.size(), 16u) << "Flattened mesh should have 16 nodes";

    for (uint32_t node : nodes) {
        const auto& neighbors = flattened_mesh.get_neighbors(node);
        EXPECT_GE(neighbors.size(), 2u) << "Node " << node << " should have at least 2 neighbors";
        EXPECT_LE(neighbors.size(), 3u) << "Node " << node << " should have at most 4 neighbors";
    }
}

TEST(PhysicalGroupingDescriptorTests, BuildFlattenedAdjacencyMesh_2x2Halftray) {
    const std::filesystem::path text_proto_file_path =
        "tests/tt_metal/tt_fabric/physical_groupings/triple_16x8_quad_bh_galaxy_physical_groupings.textproto";
    PhysicalGroupingDescriptor desc(text_proto_file_path);

    GroupingInfo mesh_halftray;
    bool found = false;
    for (const auto& mesh : desc.get_groupings_by_name("MESH")) {
        if (mesh.name == "2x2_Mesh_Halftray") {
            mesh_halftray = mesh;
            found = true;
            break;
        }
    }
    ASSERT_TRUE(found) << "Expected to find '2x2_Mesh_Halftray' grouping";

    EXPECT_EQ(mesh_halftray.asic_count, 4u) << "2x2_Mesh_Halftray should have 4 ASICs (1 halftray)";
    EXPECT_EQ(mesh_halftray.items.size(), 1u) << "2x2_Mesh_Halftray should have 1 instance (halftray)";

    AdjacencyGraph<uint32_t> flattened_mesh = desc.build_flattened_adjacency_mesh(mesh_halftray);

    auto nodes = flattened_mesh.get_nodes();
    EXPECT_EQ(nodes.size(), 4u) << "Flattened mesh should have 4 nodes";

    for (uint32_t node : nodes) {
        const auto& neighbors = flattened_mesh.get_neighbors(node);
        EXPECT_GE(neighbors.size(), 2u) << "Node " << node << " should have at least 2 neighbors";
        EXPECT_LE(neighbors.size(), 4u) << "Node " << node << " should have at most 4 neighbors (2x2 mesh)";
    }
}

TEST(PhysicalGroupingDescriptorTests, BuildFlattenedAdjacencyMesh_4x32Mesh) {
    const std::filesystem::path text_proto_file_path =
        "tests/tt_metal/tt_fabric/physical_groupings/triple_16x8_quad_bh_galaxy_physical_groupings.textproto";
    PhysicalGroupingDescriptor desc(text_proto_file_path);

    GroupingInfo mesh_4x32;
    bool found = false;
    for (const auto& mesh : desc.get_groupings_by_name("MESH")) {
        if (mesh.name == "4x32_Mesh") {
            mesh_4x32 = mesh;
            found = true;
            break;
        }
    }
    ASSERT_TRUE(found) << "Expected to find '4x32_Mesh' grouping";

    EXPECT_EQ(mesh_4x32.asic_count, 128u) << "4x32_Mesh should have 128 ASICs (4 hosts * 32 ASICs each)";
    EXPECT_EQ(mesh_4x32.items.size(), 4u) << "4x32_Mesh should have 4 instances (hosts)";

    AdjacencyGraph<uint32_t> flattened_mesh = desc.build_flattened_adjacency_mesh(mesh_4x32);

    auto nodes = flattened_mesh.get_nodes();
    EXPECT_EQ(nodes.size(), 128u) << "Flattened mesh should have 128 nodes";

    for (uint32_t node : nodes) {
        const auto& neighbors = flattened_mesh.get_neighbors(node);
        EXPECT_GE(neighbors.size(), 2u) << "Node " << node << " should have at least 2 neighbors";
        EXPECT_LE(neighbors.size(), 4u) << "Node " << node << " should have at most 4 neighbors";
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
            grouping_ref { preset_type: TRAY_1 }
          }]
        }
        groupings {
          name: "mesh_1x4"
          preset_type: MESH
          instances:
          [ {
            id: 0
            grouping_ref { preset_type: TRAY_1 }
          }
            , {
              id: 1
              grouping_ref { preset_type: TRAY_1 }
            }
            , {
              id: 2
              grouping_ref { preset_type: TRAY_1 }
            }
            , {
              id: 3
              grouping_ref { preset_type: TRAY_1 }
            }]
          row_major_mesh { dims: [ 1, 4 ] }
        }
    )proto");

    PhysicalGroupingDescriptor desc(text_proto);
    auto meshes = desc.get_groupings_by_name("MESH");
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

    auto flat_1x1 = desc.build_flattened_adjacency_mesh(mesh_1x1);
    EXPECT_EQ(flat_1x1.get_nodes().size(), 1u);  // 1 tray with 1 ASIC (from required groupings)
    expect_neighbors(flat_1x1, 0, {});           // Single node has no neighbors

    auto flat_1x4 = desc.build_flattened_adjacency_mesh(mesh_1x4);
    EXPECT_EQ(flat_1x4.get_nodes().size(), 4u);  // 4 trays x 1 ASIC each
    // 1x4 chain: endpoints have 1 neighbor, interior nodes have 2 (row-major IDs 0..3)
    expect_neighbors(flat_1x4, 0, {1});
    expect_neighbors(flat_1x4, 1, {0, 2});
    expect_neighbors(flat_1x4, 2, {1, 3});
    expect_neighbors(flat_1x4, 3, {2});
}
}  // namespace tt::tt_fabric::fabric_router_tests
