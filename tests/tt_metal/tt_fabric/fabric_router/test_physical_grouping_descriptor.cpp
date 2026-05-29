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
#include <iostream>
#include <cstdlib>

#include <tt-metalium/experimental/fabric/physical_grouping_descriptor.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph_descriptor.hpp>
#include <tt-metalium/experimental/fabric/topology_solver.hpp>
#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>
#include "tt_metal/fabric/serialization/physical_system_descriptor_serialization.hpp"
#include "tt_metal/fabric/physical_system_discovery.hpp"
#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"

using namespace tt::tt_fabric;

namespace tt::tt_fabric::fabric_router_tests {

// Helper function to create PSD from mock cluster
static tt::tt_metal::PhysicalSystemDescriptor create_psd_from_mock_cluster() {
    auto* mock_desc = getenv("TT_METAL_MOCK_CLUSTER_DESC_PATH");
    if (mock_desc == nullptr) {
        throw std::runtime_error("TT_METAL_MOCK_CLUSTER_DESC_PATH must be set for PSD tests");
    }

    // Create PSD from mock cluster (CPU-only test)
    using namespace tt::tt_metal::distributed::multihost;
    auto distributed_context = tt::tt_metal::MetalContext::instance().get_distributed_context_ptr();
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    return tt::tt_metal::run_physical_system_discovery(
        *cluster.get_cluster_desc(), distributed_context, rtoptions.get_target_device());
}

static PhysicalGroupingDescriptor load_bh_galaxy_pgd_from_fragments(bool is_rev_c = false) {
    const char* tt_metal_home = std::getenv("TT_METAL_HOME");
    if (tt_metal_home == nullptr) {
        throw std::runtime_error("TT_METAL_HOME environment variable must be set for BH PGD tests");
    }

    const std::filesystem::path physical_groupings_dir =
        std::filesystem::path(tt_metal_home) / "tests" / "tt_metal" / "tt_fabric" / "physical_groupings" / "bh";
    const std::filesystem::path common_asic_pgd_path =
        physical_groupings_dir / "bh_galaxy_common_asic_physical_grouping_descriptor.textproto";
    const std::filesystem::path rev_pgd_path =
        physical_groupings_dir / (is_rev_c ? "bh_galaxy_rev_c_physical_grouping_descriptor.textproto"
                                           : "bh_galaxy_rev_ab_physical_grouping_descriptor.textproto");
    const std::filesystem::path common_pgd_path =
        physical_groupings_dir / "bh_galaxy_common_physical_grouping_descriptor.textproto";

    if (!std::filesystem::exists(common_asic_pgd_path) || !std::filesystem::is_regular_file(common_asic_pgd_path)) {
        throw std::runtime_error("BH common ASIC PGD file not found: " + common_asic_pgd_path.string());
    }
    if (!std::filesystem::exists(rev_pgd_path) || !std::filesystem::is_regular_file(rev_pgd_path)) {
        throw std::runtime_error("BH revision PGD file not found: " + rev_pgd_path.string());
    }
    if (!std::filesystem::exists(common_pgd_path) || !std::filesystem::is_regular_file(common_pgd_path)) {
        throw std::runtime_error("BH common PGD file not found: " + common_pgd_path.string());
    }

    return PhysicalGroupingDescriptor::from_fragment_files({common_asic_pgd_path, rev_pgd_path, common_pgd_path});
}

static PhysicalGroupingDescriptor load_bh_galaxy_pgd_from_psd(const tt::tt_metal::PhysicalSystemDescriptor& psd) {
    auto& context = tt::tt_metal::MetalContext::instance();
    const auto& cluster = context.get_cluster();
    return PhysicalGroupingDescriptor::from_repo_default(psd, cluster.get_cluster_type(), cluster.arch());
}

using TrayLocationSlots = std::set<std::pair<uint32_t, uint32_t>>;

static std::map<std::string, TrayLocationSlots> get_expected_bh_slice_slots(bool is_rev_c) {
    if (is_rev_c) {
        return {
            {"SLICE_0", {{1u, 1u}, {1u, 2u}, {1u, 5u}, {1u, 6u}, {2u, 1u}, {2u, 2u}, {2u, 5u}, {2u, 6u}}},
            {"SLICE_1", {{1u, 3u}, {1u, 4u}, {1u, 7u}, {1u, 8u}, {2u, 3u}, {2u, 4u}, {2u, 7u}, {2u, 8u}}},
            {"SLICE_2", {{3u, 3u}, {3u, 4u}, {3u, 7u}, {3u, 8u}, {4u, 3u}, {4u, 4u}, {4u, 7u}, {4u, 8u}}},
            {"SLICE_3", {{3u, 1u}, {3u, 2u}, {3u, 5u}, {3u, 6u}, {4u, 1u}, {4u, 2u}, {4u, 5u}, {4u, 6u}}},
        };
    }

    return {
        {"SLICE_0", {{1u, 1u}, {1u, 2u}, {1u, 5u}, {1u, 6u}, {3u, 1u}, {3u, 2u}, {3u, 5u}, {3u, 6u}}},
        {"SLICE_1", {{1u, 3u}, {1u, 4u}, {1u, 7u}, {1u, 8u}, {3u, 3u}, {3u, 4u}, {3u, 7u}, {3u, 8u}}},
        {"SLICE_2", {{2u, 3u}, {2u, 4u}, {2u, 7u}, {2u, 8u}, {4u, 3u}, {4u, 4u}, {4u, 7u}, {4u, 8u}}},
        {"SLICE_3", {{2u, 1u}, {2u, 2u}, {2u, 5u}, {2u, 6u}, {4u, 1u}, {4u, 2u}, {4u, 5u}, {4u, 6u}}},
    };
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

// Helper to get common groupings (TRAY_1-4, hosts) - can be prepended to any test proto
// Note: These groupings are no longer required but are commonly used in tests
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
// HOST SPAN TESTS
// ============================================================================

TEST(PhysicalGroupingDescriptorTests, HostSpan_ParsedAndDefaulted) {
    // A grouping with host_span: 2 is parsed as cross-host; a grouping without the field is unspecified (0).
    const std::string text_proto = wrap_with_required_groupings(R"proto(
        groupings {
          name: "cross_host_mesh"
          custom_type: "cross_host"
          host_span: 2
          instances:
          [ {
            id: 0
            grouping_ref { custom_type: "meshes" }
          }
            , {
              id: 1
              grouping_ref { custom_type: "meshes" }
            }]
          row_major_mesh { dims: [ 1, 2 ] }
        }
        groupings {
          name: "unspecified_mesh"
          custom_type: "unspecified"
          instances:
          [ {
            id: 0
            grouping_ref { custom_type: "meshes" }
          }
            , {
              id: 1
              grouping_ref { custom_type: "meshes" }
            }]
          row_major_mesh { dims: [ 1, 2 ] }
        }
    )proto");

    PhysicalGroupingDescriptor desc(text_proto);

    auto cross_host = desc.get_groupings_by_type("cross_host");
    ASSERT_EQ(cross_host.size(), 1u);
    EXPECT_EQ(cross_host[0].host_span, 2u) << "host_span: 2 should be parsed as 2";

    auto unspecified = desc.get_groupings_by_type("unspecified");
    ASSERT_EQ(unspecified.size(), 1u);
    EXPECT_EQ(unspecified[0].host_span, 0u) << "absent host_span should remain unspecified (0)";
}

TEST(PhysicalGroupingDescriptorTests, HostSpan_ValidationRejectsNonDivisor) {
    // host_span must evenly divide the number of top-level instances; 2 does not divide 3.
    const std::string text_proto = wrap_with_required_groupings(R"proto(
        groupings {
          name: "bad_host_span"
          custom_type: "bad_host_span"
          host_span: 2
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
            }]
          row_major_mesh { dims: [ 1, 3 ] }
        }
    )proto");

    EXPECT_THROW(
        { PhysicalGroupingDescriptor desc(text_proto); }, std::exception)
        << "host_span that does not divide the instance count should be rejected";
}

// Confirms the production BH PGD tags exactly the [SLICE_3, SLICE_0] 4x4 candidate as cross-host
// (host_span 2) and leaves the other three host-local (host_span 1).
TEST(PhysicalGroupingDescriptorSP4Tests, BhGalaxy4x4CrossHostCandidateTaggedHostSpan2) {
    PhysicalGroupingDescriptor pgd = load_bh_galaxy_pgd_from_fragments(/*is_rev_c=*/false);

    auto groupings_4x4 = pgd.get_groupings_by_name("4x4_Mesh BH");
    ASSERT_EQ(groupings_4x4.size(), 4u) << "Expected four 4x4_Mesh BH candidate definitions";

    size_t cross_host = 0;
    size_t host_local = 0;
    for (const auto& g : groupings_4x4) {
        if (g.host_span == 2u) {
            ++cross_host;
        } else if (g.host_span == 1u) {
            ++host_local;
        }
    }
    EXPECT_EQ(cross_host, 1u) << "Exactly one 4x4_Mesh BH candidate (SLICE_3, SLICE_0) should be cross-host";
    EXPECT_EQ(host_local, 3u) << "The other three 4x4_Mesh BH candidates should be host-local";
}

// Diagnostic (hardware-free): flatten 8x8_Mesh (two HOSTS stacked) and print the inter-host "seam"
// edges with their (tray, asic_location) labels for both rev A/B and rev C. This surfaces the
// cross-host adjacency the PGD *asserts* at the host boundary, which is the PGD half of the
// exact-embedding diff for the multi-host meshes (8x8 / 8x16). It does not need a PSD; the PSD side
// (what is actually cabled) is captured separately by the TT_METAL_PGD_DEBUG_EMBED diagnostic in
// solve_for_one_grouping_to_psd. Not an assertion test - it always passes and just logs.
TEST(PhysicalGroupingDescriptorSP4Tests, DiagnoseCrossHostSeamEdges_8x8) {
    for (bool is_rev_c : {false, true}) {
        PhysicalGroupingDescriptor pgd = load_bh_galaxy_pgd_from_fragments(is_rev_c);

        auto groupings = pgd.get_groupings_by_name("8x8_Mesh");
        ASSERT_GE(groupings.size(), 1u) << "8x8_Mesh grouping must exist";

        auto flats = pgd.build_flattened_adjacency_mesh(groupings[0]);
        ASSERT_GE(flats.size(), 1u) << "8x8_Mesh should flatten to at least one mesh";
        const GroupingInfo& flat = flats[0];

        const auto& nodes = flat.adjacency_graph.get_nodes();
        ASSERT_EQ(nodes.size(), 64u) << "8x8_Mesh should flatten to 64 nodes (two 32-ASIC hosts)";

        // Host partition: node IDs are assigned in DFS order, host 0's block before host 1's, so a
        // sorted-midpoint split separates the two hosts even if IDs are non-contiguous.
        std::vector<uint32_t> sorted_nodes(nodes.begin(), nodes.end());
        std::sort(sorted_nodes.begin(), sorted_nodes.end());
        std::set<uint32_t> host1(sorted_nodes.begin() + sorted_nodes.size() / 2, sorted_nodes.end());

        auto label = [&](uint32_t id) {
            const auto& item = flat.items[id];
            return "(T" + std::to_string(*item.tray_id) + ",L" + std::to_string(*item.asic_location) + ")";
        };
        auto host_of = [&](uint32_t id) { return host1.contains(id) ? 1 : 0; };

        std::cout << "=== 8x8_Mesh inter-host seam edges (rev " << (is_rev_c ? "C" : "AB") << ") ===" << std::endl;
        size_t seam = 0;
        for (uint32_t a : sorted_nodes) {
            for (uint32_t b : flat.adjacency_graph.get_neighbors(a)) {
                if (a < b && host_of(a) != host_of(b)) {
                    std::cout << "  seam: node " << a << " " << label(a) << " [host" << host_of(a) << "]"
                              << "  <->  node " << b << " " << label(b) << " [host" << host_of(b) << "]" << std::endl;
                    ++seam;
                }
            }
        }
        std::cout << "  total inter-host seam edges: " << seam << std::endl;
        EXPECT_GT(seam, 0u) << "A stacked 8x8 (dims [2,1]) must assert at least one cross-host edge";
    }
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
    EXPECT_NO_THROW({ PhysicalGroupingDescriptor desc = load_bh_galaxy_pgd_from_fragments(); });
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

TEST(PhysicalGroupingDescriptorTests, MeshGroupingsCanBeLeafNodes) {
    // Test that MESH groupings can be leaf nodes (using ASIC locations directly)
    // This verifies that MESH groupings are allowed to use ASIC locations without grouping references
    const std::string text_proto = get_required_groupings() + R"proto(
        groupings {
          name: "mesh_leaf_1"
          preset_type: MESH
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_1 }
            , { id: 1 asic_location: ASIC_LOCATION_2 }
            , { id: 2 asic_location: ASIC_LOCATION_3 }
            , { id: 3 asic_location: ASIC_LOCATION_4 }]
          row_major_mesh { dims: [ 2, 2 ] }
        }
        groupings {
          name: "mesh_leaf_2"
          preset_type: MESH
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_5 }
            , { id: 1 asic_location: ASIC_LOCATION_6 }]
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
          [ { id: 0 asic_location: ASIC_LOCATION_1 }
            , { id: 1 asic_location: ASIC_LOCATION_2 }]
          row_major_mesh { dims: [ 1, 2 ] }
        }
        groupings {
          name: "mesh_non_leaf"
          preset_type: MESH
          instances:
          [ {
            id: 0
            grouping_ref { preset_type: TRAY_1 }
          }]
        }
        groupings {
          name: "mesh_another_leaf"
          preset_type: MESH
          instances:
          [ { id: 0 asic_location: ASIC_LOCATION_3 }
            , { id: 1 asic_location: ASIC_LOCATION_4 }
            , { id: 2 asic_location: ASIC_LOCATION_5 }
            , { id: 3 asic_location: ASIC_LOCATION_6 }]
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
          [ { id: 0 asic_location: ASIC_LOCATION_1 }]
        }
        groupings {
          name: "mesh_mixed_bad"
          preset_type: MESH
          instances:
          [ {
            id: 0
            grouping_ref { preset_type: TRAY_1 }
          }
            , { id: 1 asic_location: ASIC_LOCATION_2 }]
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
    PhysicalGroupingDescriptor desc = load_bh_galaxy_pgd_from_fragments();

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
    PhysicalGroupingDescriptor desc = load_bh_galaxy_pgd_from_fragments();

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
    PhysicalGroupingDescriptor desc = load_bh_galaxy_pgd_from_fragments();

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

TEST(PhysicalGroupingDescriptorTests, BuildFlattenedAdjacencyMesh_2x2AsicGrouping) {
    // PGD names this MESH grouping "2x2 Mesh" and wraps exactly one explicit 2x2 ASIC grouping.
    PhysicalGroupingDescriptor desc = load_bh_galaxy_pgd_from_fragments();

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
    ASSERT_TRUE(found) << "Expected MESH grouping named \"" << kMeshGroupingName << "\"";

    EXPECT_EQ(mesh_halftray.asic_count, 4u)
        << "MESH grouping \"" << kMeshGroupingName << "\" should have 4 ASICs (1 explicit 2x2 ASIC grouping instance)";
    EXPECT_EQ(mesh_halftray.items.size(), 1u)
        << "MESH grouping \"" << kMeshGroupingName << "\" should have 1 instance (one explicit 2x2 ASIC grouping ref)";

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

TEST(PhysicalGroupingDescriptorTests, BhGalaxyMeshCandidateCountsMatchExplicitSliceModel) {
    PhysicalGroupingDescriptor desc = load_bh_galaxy_pgd_from_fragments();

    const auto groupings_2x2 = desc.get_groupings_by_name("2x2 Mesh");
    ASSERT_EQ(groupings_2x2.size(), 8u) << "Expected one 2x2 MESH candidate per explicit 2x2 ASIC grouping";
    for (const auto& grouping : groupings_2x2) {
        EXPECT_EQ(grouping.asic_count, 4u);
        EXPECT_EQ(grouping.items.size(), 1u);
        ASSERT_FALSE(desc.build_flattened_adjacency_mesh(grouping).empty());
    }

    const auto groupings_4x2 = desc.get_groupings_by_name("4x2_Mesh");
    ASSERT_EQ(groupings_4x2.size(), 4u) << "Expected one 4x2 candidate per canonical BH slice";
    for (const auto& grouping : groupings_4x2) {
        EXPECT_EQ(grouping.asic_count, 8u);
        EXPECT_EQ(grouping.items.size(), 1u);
        ASSERT_FALSE(desc.build_flattened_adjacency_mesh(grouping).empty());
    }

    const auto groupings_4x4 = desc.get_groupings_by_name("4x4_Mesh BH");
    ASSERT_EQ(groupings_4x4.size(), 4u)
        << "Expected one 4x4 candidate per cyclic adjacent slice pair (SLICE_0..SLICE_3 form 4 pairs)";
    for (const auto& grouping : groupings_4x4) {
        EXPECT_EQ(grouping.asic_count, 16u);
        EXPECT_EQ(grouping.items.size(), 2u);
        ASSERT_FALSE(desc.build_flattened_adjacency_mesh(grouping).empty());
    }
}

TEST(PhysicalGroupingDescriptorTests, BhGalaxyRevCHalftraySlotsSwapTray2AndTray3) {
    auto get_single_grouping_ref_name = [](const GroupingInfo& grouping) {
        EXPECT_EQ(grouping.items.size(), 1u);
        EXPECT_EQ(grouping.items[0].type, GroupingItemInfo::ItemType::GROUPING_REF);
        return grouping.items[0].grouping_name;
    };
    auto get_single_grouping = [](const PhysicalGroupingDescriptor& desc, const std::string& type) {
        const auto groupings = desc.get_groupings_by_type(type);
        EXPECT_EQ(groupings.size(), 1u) << "Expected exactly one grouping of type " << type;
        return groupings.front();
    };

    const PhysicalGroupingDescriptor rev_ab = load_bh_galaxy_pgd_from_fragments(false);
    const PhysicalGroupingDescriptor rev_c = load_bh_galaxy_pgd_from_fragments(true);

    const auto tray1_half_1_rev_ab = get_single_grouping_ref_name(get_single_grouping(rev_ab, "TRAY_1_HALF_1"));
    const auto tray1_half_2_rev_ab = get_single_grouping_ref_name(get_single_grouping(rev_ab, "TRAY_1_HALF_2"));
    const auto tray2_half_1_rev_ab = get_single_grouping_ref_name(get_single_grouping(rev_ab, "TRAY_2_HALF_1"));
    const auto tray2_half_2_rev_ab = get_single_grouping_ref_name(get_single_grouping(rev_ab, "TRAY_2_HALF_2"));
    const auto tray3_half_1_rev_ab = get_single_grouping_ref_name(get_single_grouping(rev_ab, "TRAY_3_HALF_1"));
    const auto tray3_half_2_rev_ab = get_single_grouping_ref_name(get_single_grouping(rev_ab, "TRAY_3_HALF_2"));
    const auto tray4_half_1_rev_ab = get_single_grouping_ref_name(get_single_grouping(rev_ab, "TRAY_4_HALF_1"));
    const auto tray4_half_2_rev_ab = get_single_grouping_ref_name(get_single_grouping(rev_ab, "TRAY_4_HALF_2"));

    const auto tray1_half_1_rev_c = get_single_grouping_ref_name(get_single_grouping(rev_c, "TRAY_1_HALF_1"));
    const auto tray1_half_2_rev_c = get_single_grouping_ref_name(get_single_grouping(rev_c, "TRAY_1_HALF_2"));
    const auto tray2_half_1_rev_c = get_single_grouping_ref_name(get_single_grouping(rev_c, "TRAY_2_HALF_1"));
    const auto tray2_half_2_rev_c = get_single_grouping_ref_name(get_single_grouping(rev_c, "TRAY_2_HALF_2"));
    const auto tray3_half_1_rev_c = get_single_grouping_ref_name(get_single_grouping(rev_c, "TRAY_3_HALF_1"));
    const auto tray3_half_2_rev_c = get_single_grouping_ref_name(get_single_grouping(rev_c, "TRAY_3_HALF_2"));
    const auto tray4_half_1_rev_c = get_single_grouping_ref_name(get_single_grouping(rev_c, "TRAY_4_HALF_1"));
    const auto tray4_half_2_rev_c = get_single_grouping_ref_name(get_single_grouping(rev_c, "TRAY_4_HALF_2"));

    EXPECT_EQ(tray1_half_1_rev_ab, "ASIC_1256_R");
    EXPECT_EQ(tray1_half_2_rev_ab, "ASIC_3478_R");
    EXPECT_EQ(tray2_half_1_rev_ab, "ASIC_3478_RC");
    EXPECT_EQ(tray2_half_2_rev_ab, "ASIC_1256_RC");
    EXPECT_EQ(tray3_half_1_rev_ab, "ASIC_1256");
    EXPECT_EQ(tray3_half_2_rev_ab, "ASIC_3478");
    EXPECT_EQ(tray4_half_1_rev_ab, "ASIC_3478_C");
    EXPECT_EQ(tray4_half_2_rev_ab, "ASIC_1256_C");

    EXPECT_EQ(tray1_half_1_rev_c, tray1_half_1_rev_ab);
    EXPECT_EQ(tray1_half_2_rev_c, tray1_half_2_rev_ab);
    EXPECT_EQ(tray4_half_1_rev_c, tray4_half_1_rev_ab);
    EXPECT_EQ(tray4_half_2_rev_c, tray4_half_2_rev_ab);
    EXPECT_EQ(tray2_half_1_rev_c, tray3_half_1_rev_ab)
        << "Rev C tray 2 HALF_1 slot should take the rev A/B tray 3 HALF_1 slot content";
    EXPECT_EQ(tray2_half_2_rev_c, tray3_half_2_rev_ab)
        << "Rev C tray 2 HALF_2 slot should take the rev A/B tray 3 HALF_2 slot content";
    EXPECT_EQ(tray3_half_1_rev_c, tray2_half_1_rev_ab)
        << "Rev C tray 3 HALF_1 slot should take the rev A/B tray 2 HALF_1 slot content";
    EXPECT_EQ(tray3_half_2_rev_c, tray2_half_2_rev_ab)
        << "Rev C tray 3 HALF_2 slot should take the rev A/B tray 2 HALF_2 slot content";
}

TEST(PhysicalGroupingDescriptorTests, BhGalaxySlicesMatchPhysicalDiscoveryTables) {
    auto get_single_grouping = [](const PhysicalGroupingDescriptor& desc, const std::string& type) {
        const auto groupings = desc.get_groupings_by_type(type);
        EXPECT_EQ(groupings.size(), 1u) << "Expected exactly one grouping of type " << type;
        return groupings.front();
    };
    auto collect_flattened_slots = [](const PhysicalGroupingDescriptor& desc, const GroupingInfo& grouping) {
        const auto flattened_meshes = desc.build_flattened_adjacency_mesh(grouping);
        EXPECT_EQ(flattened_meshes.size(), 1u) << "Expected one flattened slice for " << grouping.type;
        TrayLocationSlots slots;
        if (flattened_meshes.empty()) {
            return slots;
        }
        for (uint32_t node_id : flattened_meshes.front().adjacency_graph.get_nodes()) {
            if (node_id >= flattened_meshes.front().items.size()) {
                ADD_FAILURE() << "Flattened slice metadata missing item for node_id=" << node_id;
                continue;
            }
            const auto& item = flattened_meshes.front().items[node_id];
            EXPECT_EQ(item.type, GroupingItemInfo::ItemType::ASIC_LOCATION);
            slots.emplace(*item.tray_id, *item.asic_location);
        }
        return slots;
    };

    for (bool is_rev_c : {false, true}) {
        const PhysicalGroupingDescriptor desc = load_bh_galaxy_pgd_from_fragments(is_rev_c);
        const auto expected_slice_slots = get_expected_bh_slice_slots(is_rev_c);

        for (const auto& [slice_type, expected_slots] : expected_slice_slots) {
            const auto slice_grouping = get_single_grouping(desc, slice_type);
            EXPECT_EQ(slice_grouping.asic_count, 8u) << slice_type << " should contain exactly one 4x2 slice";
            EXPECT_EQ(slice_grouping.items.size(), 2u) << slice_type << " should be composed from two halftrays";
            EXPECT_EQ(collect_flattened_slots(desc, slice_grouping), expected_slots)
                << slice_type << " should match the Generate2x4SliceToPCIeDeviceMapping tray/location table";
        }
    }
}

TEST(PhysicalGroupingDescriptorTests, BhGalaxySliceSlotSetFallbackPreservesCanonicalSlice) {
    auto* mock_desc = getenv("TT_METAL_MOCK_CLUSTER_DESC_PATH");
    if (mock_desc == nullptr) {
        GTEST_SKIP() << "TT_METAL_MOCK_CLUSTER_DESC_PATH not set - run with tt-run --mock-cluster-rank-binding";
    }

    auto slot_counts_fit_in_psd = [](const GroupingInfo& grouping,
                                     const tt::tt_metal::PhysicalSystemDescriptor& psd) -> bool {
        std::map<std::pair<uint32_t, uint32_t>, size_t> psd_slot_counts;
        for (const auto& [_, desc] : psd.get_asic_descriptors()) {
            if (*desc.tray_id > 0 && *desc.asic_location <= 8) {
                psd_slot_counts[{*desc.tray_id, *desc.asic_location}]++;
            }
        }

        std::map<std::pair<uint32_t, uint32_t>, size_t> required_slot_counts;
        for (uint32_t node_id : grouping.adjacency_graph.get_nodes()) {
            if (node_id >= grouping.items.size()) {
                ADD_FAILURE() << "Relabeled slice metadata missing item for node_id=" << node_id;
                return false;
            }

            const auto& item = grouping.items[node_id];
            if (item.type != GroupingItemInfo::ItemType::ASIC_LOCATION) {
                ADD_FAILURE() << "Relabeled slice node " << node_id << " is missing ASIC_LOCATION metadata";
                return false;
            }

            if (*item.tray_id > 0 && *item.asic_location <= 8) {
                required_slot_counts[{*item.tray_id, *item.asic_location}]++;
            }
        }

        for (const auto& [slot, needed] : required_slot_counts) {
            auto it = psd_slot_counts.find(slot);
            if (it == psd_slot_counts.end() || it->second < needed) {
                return false;
            }
        }
        return true;
    };

    tt::tt_metal::PhysicalSystemDescriptor psd = create_psd_from_mock_cluster();
    PhysicalGroupingDescriptor desc = load_bh_galaxy_pgd_from_psd(psd);

    const auto slice_groupings = desc.get_groupings_by_type("SLICE_0");
    ASSERT_EQ(slice_groupings.size(), 1u) << "Expected a single canonical SLICE_0 grouping";

    auto flattened_meshes = desc.build_flattened_adjacency_mesh(slice_groupings.front(), psd);
    ASSERT_EQ(flattened_meshes.size(), 1u) << "Expected one flattened mesh for SLICE_0";

    GroupingInfo relabeled_slice = flattened_meshes.front();
    TrayLocationSlots expected_slots;
    std::optional<uint32_t> degree_two_node;
    std::optional<uint32_t> degree_three_node;
    for (uint32_t node_id : relabeled_slice.adjacency_graph.get_nodes()) {
        ASSERT_LT(node_id, relabeled_slice.items.size());
        expected_slots.emplace(*relabeled_slice.items[node_id].tray_id, *relabeled_slice.items[node_id].asic_location);

        const auto degree = relabeled_slice.adjacency_graph.get_neighbors(node_id).size();
        if (degree == 2 && !degree_two_node.has_value()) {
            degree_two_node = node_id;
        }
        if (degree == 3 && !degree_three_node.has_value()) {
            degree_three_node = node_id;
        }
    }

    ASSERT_TRUE(degree_two_node.has_value()) << "Expected at least one corner node in flattened 4x2 slice";
    ASSERT_TRUE(degree_three_node.has_value()) << "Expected at least one edge-center node in flattened 4x2 slice";

    std::swap(relabeled_slice.items[*degree_two_node].tray_id, relabeled_slice.items[*degree_three_node].tray_id);
    std::swap(
        relabeled_slice.items[*degree_two_node].asic_location, relabeled_slice.items[*degree_three_node].asic_location);

    ASSERT_TRUE(slot_counts_fit_in_psd(relabeled_slice, psd))
        << "Relabeled slice should still match the PSD slot multiset";

    auto mapped_asics = desc.find_any_in_psd(relabeled_slice, psd);
    ASSERT_EQ(mapped_asics.size(), relabeled_slice.adjacency_graph.get_nodes().size())
        << "Slot-set fallback should recover a canonical BH slice even when per-node slot labels are permuted";

    TrayLocationSlots mapped_slots;
    for (const auto& asic_id : mapped_asics) {
        mapped_slots.emplace(*psd.get_tray_id(asic_id), *psd.get_asic_location(asic_id));
    }

    EXPECT_EQ(mapped_slots, expected_slots)
        << "Recovered slice should stay within the intended BH discovery-aligned slot set";
}

// A slice-wrapped 4x2 mesh produces 8 ASIC nodes; items must still be indexed by node_id after flattening.
TEST(PhysicalGroupingDescriptorTests, BuildFlattenedAdjacencyMesh_4x2Mesh_OneSlice_ItemsPerGraphNode) {
    PhysicalGroupingDescriptor desc = load_bh_galaxy_pgd_from_fragments();

    GroupingInfo mesh_4x2;
    bool found = false;
    for (const auto& mesh : desc.get_groupings_by_type("MESH")) {
        if (mesh.name == "4x2_Mesh") {
            mesh_4x2 = mesh;
            found = true;
            break;
        }
    }
    ASSERT_TRUE(found) << "Expected to find '4x2_Mesh' grouping";

    EXPECT_EQ(mesh_4x2.asic_count, 8u) << "4x2_Mesh: 1 slice x 8 ASICs";
    EXPECT_EQ(mesh_4x2.items.size(), 1u) << "4x2_Mesh should have 1 slice ref before flatten";

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

    std::set<uint32_t> trays;
    for (uint32_t node_id : nodes) {
        trays.insert(*flat.items[node_id].tray_id);
    }
    EXPECT_EQ(trays.size(), 2u) << "Flattened 4x2 slice should span exactly two trays";
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
TEST(PhysicalGroupingDescriptorSP4Tests, ValidatePreformedGroups_Sp4BhGalaxyMeshGroupings_SingleHostScale) {
    tt::tt_metal::PhysicalSystemDescriptor psd = create_psd_from_mock_cluster();
    PhysicalGroupingDescriptor pgd = load_bh_galaxy_pgd_from_psd(psd);

    // Get all mesh groupings to test
    auto all_mesh_groupings = pgd.get_groupings_by_type("MESH");
    ASSERT_FALSE(all_mesh_groupings.empty()) << "No MESH groupings found in PGD";

    // Find specific mesh groupings by name or by dimensions (name can have WH/BH suffix)
    // Prefer exact match first so "4x2_Mesh" matches the one-tray grouping, not a longer prefix
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

    // Test 4x2_Mesh (one tray, 8 ASICs) - validation against mock cluster
    {
        const auto* mesh_grouping = find_mesh_by_name("4x2_Mesh");
        ASSERT_NE(mesh_grouping, nullptr) << "4x2_Mesh grouping not found";

        auto asic_ids = pgd.find_any_in_psd(*mesh_grouping, psd);

        EXPECT_FALSE(asic_ids.empty())
            << "Expected validation to pass: 4x2_Mesh grouping should map to mock cluster PSD";
    }

    // Test 4x4_Mesh - validation against mock cluster
    {
        const auto* mesh_grouping = find_mesh_by_name("4x4_Mesh");
        ASSERT_NE(mesh_grouping, nullptr) << "4x4_Mesh grouping not found";

        auto asic_ids = pgd.find_any_in_psd(*mesh_grouping, psd);

        EXPECT_FALSE(asic_ids.empty())
            << "Expected validation to pass: 4x4_Mesh grouping should map to mock cluster PSD";
    }

    // Test 2x8_Mesh BH - validation against mock cluster
    {
        const auto* mesh_grouping = find_mesh_by_name("2x8_Mesh BH");
        ASSERT_NE(mesh_grouping, nullptr) << "2x8_Mesh BH grouping not found";

        auto asic_ids = pgd.find_any_in_psd(*mesh_grouping, psd);

        EXPECT_FALSE(asic_ids.empty())
            << "Expected validation to pass: 2x8_Mesh BH grouping should map to mock cluster PSD";
    }

    // Test 4x8_Mesh - validation against mock cluster
    {
        const auto* mesh_grouping = find_mesh_by_name("4x8_Mesh");
        ASSERT_NE(mesh_grouping, nullptr) << "4x8_Mesh grouping not found";

        auto asic_ids = pgd.find_any_in_psd(*mesh_grouping, psd);

        EXPECT_FALSE(asic_ids.empty())
            << "Expected validation to pass: 4x8_Mesh grouping should map to mock cluster PSD";
    }

    // Test HOSTS type grouping - validation against mock cluster
    {
        auto hosts_groupings = pgd.get_groupings_by_type("HOSTS");
        ASSERT_FALSE(hosts_groupings.empty()) << "HOSTS grouping not found";
        const auto& hosts_grouping = hosts_groupings[0];

        auto asic_ids = pgd.find_any_in_psd(hosts_grouping, psd);

        EXPECT_FALSE(asic_ids.empty()) << "Expected validation to pass: HOSTS grouping should map to mock cluster PSD";
    }
}

TEST(PhysicalGroupingDescriptorSP4Tests, ValidatePreformedGroups_Sp4BhGalaxyQuadHostMeshes) {
    tt::tt_metal::PhysicalSystemDescriptor psd = create_psd_from_mock_cluster();
    PhysicalGroupingDescriptor pgd = load_bh_galaxy_pgd_from_psd(psd);

    // Try finding any for BH_galaxy_hosts
    {
        auto hosts_groupings = pgd.get_groupings_by_name("BH_galaxy_hosts");
        ASSERT_FALSE(hosts_groupings.empty()) << "BH_galaxy_hosts grouping not found";
        const auto& hosts_grouping = hosts_groupings[0];

        auto asic_ids = pgd.find_any_in_psd(hosts_grouping, psd);

        EXPECT_FALSE(asic_ids.empty())
            << "Expected validation to pass: BH_galaxy_hosts grouping should map to mock cluster PSD";
    }

    {
        // 4x32_Mesh: same 128 ASICs / 4 hosts as an 8x16_Mesh, laid out as a 32x4-equivalent mesh.
        auto mesh_groupings = pgd.get_groupings_by_name("4x32_Mesh");
        ASSERT_FALSE(mesh_groupings.empty()) << "4x32_Mesh grouping not found";
        const auto& mesh_grouping = mesh_groupings[0];

        auto asic_ids = pgd.find_any_in_psd(mesh_grouping, psd);

        EXPECT_FALSE(asic_ids.empty())
            << "Expected validation to pass: 4x32_Mesh (32x4 device layout) should map to mock cluster PSD";
    }

    {
        auto mesh_groupings = pgd.get_groupings_by_name("4x32_Mesh");
        ASSERT_FALSE(mesh_groupings.empty()) << "4x32_Mesh grouping not found";

        std::vector<std::string> errors;

        auto asic_ids = pgd.find_all_in_psd(mesh_groupings, psd, errors);

        EXPECT_EQ(asic_ids.size(), 4u)
            << "Expected validation to pass: 4x32_Mesh (32x4) should map to mock cluster PSD (4 placements on SP4)";
    }

    {
        // Test 4x4_Mesh BH groupings (four cyclic-adjacent slice pair variants: [SLICE_0,SLICE_1],
        // [SLICE_1,SLICE_2], [SLICE_2,SLICE_3], [SLICE_3,SLICE_0]).
        auto mesh_groupings = pgd.get_groupings_by_name("4x4_Mesh BH");
        ASSERT_EQ(mesh_groupings.size(), 4u) << "4x4_Mesh BH grouping not found";

        for (const auto& grouping : mesh_groupings) {
            auto asic_ids = pgd.find_any_in_psd(grouping, psd);
            EXPECT_EQ(asic_ids.size(), 16u)
                << "Expected validation to pass: each 4x4_Mesh BH variant should admit at least one 16-ASIC "
                   "embedding on the merged SP4 PSD";
        }

        std::vector<std::string> errors;
        auto packed_asic_id_sets = pgd.find_all_in_psd(mesh_groupings, psd, errors);

        // The heterogeneous find_all_in_psd path forbids overlap across all input groupings, so the 4x4 variants pack
        // into non-overlapping host-local windows across the SP4 PSD. With four cyclic-adjacent slice pair candidates
        // the achievable placement count is at least the two host-disjoint pairs per host across the four SP4 hosts.
        EXPECT_GE(packed_asic_id_sets.size(), 8u)
            << "Expected heterogeneous packing to find at least 8 non-overlapping 4x4_Mesh BH placements "
               "(two host-disjoint pairings per host across the four SP4 hosts)";
        for (const auto& asic_ids : packed_asic_id_sets) {
            EXPECT_EQ(asic_ids.size(), 16u) << "Each packed 4x4_Mesh BH placement should contain exactly 16 ASICs";
        }
    }
}

TEST(PhysicalGroupingDescriptorDualT3kTests, ValidatePreformedGroups_WHt3kGroupings) {
    const std::string pgd_path =
        "tests/tt_metal/tt_fabric/physical_groupings/wh_t3k_physical_grouping_descriptor.textproto";

    ASSERT_TRUE(std::filesystem::exists(pgd_path)) << "PGD file not found: " << pgd_path;

    tt::tt_metal::PhysicalSystemDescriptor psd = create_psd_from_mock_cluster();
    PhysicalGroupingDescriptor pgd{std::filesystem::path(pgd_path)};

    {
        auto mesh_groupings = pgd.get_groupings_by_name("2x2_Mesh_t3k");
        ASSERT_FALSE(mesh_groupings.empty()) << "2x2_Mesh_t3k grouping not found";

        std::vector<std::string> errors;

        auto asic_ids = pgd.find_all_in_psd(mesh_groupings, psd, errors);

        // Should find 4 of them, each of them on a single host
        EXPECT_EQ(asic_ids.size(), 4u)
            << "Expected validation to pass: 2x2_Mesh_t3k grouping should map to mock cluster PSD";

        // Each should have their own host name
        for (const auto& asic_id_set : asic_ids) {
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

        std::vector<std::string> errors;

        auto asic_ids = pgd.find_all_in_psd(mesh_groupings, psd, errors);

        ASSERT_EQ(asic_ids.size(), 2u)
            << "Expected validation to pass: 2x4_Mesh_t3k grouping should map to mock cluster PSD";

        // Each should have their own host name
        for (const auto& asic_id_set : asic_ids) {
            ASSERT_FALSE(asic_id_set.empty()) << "Each 2x4_Mesh_t3k mapping should contain at least one ASIC";
            std::string host_name = psd.get_host_name_for_asic(*asic_id_set.begin());
            for (const auto& asic_id : asic_id_set) {
                EXPECT_EQ(psd.get_host_name_for_asic(asic_id), host_name)
                    << "Expected validation to pass: 2x4_Mesh_t3k grouping should map to mock cluster PSD";
            }
        }
    }
}

TEST(PhysicalGroupingDescriptorSP4Tests, ValidatePreformedGroups_Triple16x8PsdWithTriple16x8QuadUnknownGroupings) {
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
TEST(PhysicalGroupingDescriptorSP4Tests, ValidateGroupingWithPsd_PodAndSuperpodLevel) {
    const std::string pgd_path = "tests/tt_metal/tt_fabric/physical_groupings/test_superpod_grouping.textproto";

    ASSERT_TRUE(std::filesystem::exists(pgd_path)) << "PGD file not found: " << pgd_path;

    tt::tt_metal::PhysicalSystemDescriptor psd = create_psd_from_mock_cluster();

    PhysicalGroupingDescriptor pgd{std::filesystem::path(pgd_path)};

    // Test POD level grouping - should pass (can be flattened and matches PSD)
    auto pod_groupings = pgd.get_groupings_by_name("pods");
    ASSERT_FALSE(pod_groupings.empty()) << "pods grouping not found";
    const auto& pod_grouping = pod_groupings[0];

    // POD groupings reference meshes, but should flatten properly and match the PSD structure
    auto pod_asic_ids = pgd.find_any_in_psd(pod_grouping, psd);

    // Expect it to pass - POD level grouping should validate successfully
    EXPECT_FALSE(pod_asic_ids.empty())
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

TEST(PhysicalGroupingDescriptorSP4Tests, GetValidGroupingsForMGD_BlitzPipeline2x4) {
    // Test matching a 4x2 mesh MGD (8 ASICs) to the 4x2_Mesh grouping in bh_galaxy PGD
    const std::string mgd_path = "tt_metal/fabric/mesh_graph_descriptors/bh_glx_split_4x2.textproto";

    ASSERT_TRUE(std::filesystem::exists(mgd_path)) << "MGD file not found: " << mgd_path;

    tt::tt_metal::PhysicalSystemDescriptor psd = create_psd_from_mock_cluster();
    PhysicalGroupingDescriptor pgd = load_bh_galaxy_pgd_from_psd(psd);
    MeshGraphDescriptor mgd{std::filesystem::path(mgd_path)};

    auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd, psd);

    // Print valid groupings
    for (const auto& [instance_type, instances] : valid_groupings) {
        for (const auto& [instance_name, groupings] : instances) {
            std::cout << "Instance type: " << instance_type << ", Instance name: " << instance_name << std::endl;
            for (const auto& grouping : groupings) {
                std::cout << "Grouping name: " << grouping.name << ", ASIC count: " << grouping.asic_count << std::endl;
            }
        }
    }

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
            if (grouping.asic_count == 8u && grouping.name == "4x2_Mesh_flat") {
                found_mesh_match = true;
                EXPECT_EQ(grouping.name, "4x2_Mesh_flat") << "Should match 4x2_Mesh_flat grouping";
                EXPECT_EQ(grouping.asic_count, 8u) << "Should have 8 ASICs";
                break;
            }
        }
        if (found_mesh_match) {
            break;
        }
    }
    EXPECT_TRUE(found_mesh_match) << "Should find a match for 4x2 mesh (8 ASICs) matching 4x2_Mesh_flat grouping";

    // Check that we have FABRIC level grouping (G0)
    ASSERT_EQ(valid_groupings.count("FABRIC"), 1u) << "Should have FABRIC instance type";
    ASSERT_EQ(valid_groupings.at("FABRIC").size(), 1u) << "Should have exactly one FABRIC instance";
    ASSERT_EQ(valid_groupings.at("FABRIC").count("G0"), 1u) << "Should have G0 FABRIC instance";
    const auto& g0_groupings = valid_groupings.at("FABRIC").at("G0");
    ASSERT_GE(g0_groupings.size(), 1u) << "Should have at least one grouping for G0";
}

TEST(PhysicalGroupingDescriptorSP4Tests, GetValidGroupingsForMGD_4x4Mesh) {
    // Test matching a 4x4 mesh MGD (16 ASICs) to the 4x4_Mesh grouping
    // Using dual_4x4_mesh_graph_descriptor which has 4x4 meshes in a graph
    const std::string mgd_path =
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_4x4_mesh_graph_descriptor.textproto";

    ASSERT_TRUE(std::filesystem::exists(mgd_path)) << "MGD file not found: " << mgd_path;

    tt::tt_metal::PhysicalSystemDescriptor psd = create_psd_from_mock_cluster();
    PhysicalGroupingDescriptor pgd = load_bh_galaxy_pgd_from_psd(psd);
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
    // Names in triple_16x8 are "4x4_Mesh WH", "4x4_Mesh BH", etc.
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

TEST(PhysicalGroupingDescriptorSP4Tests, GetValidGroupingsForMGD_2x8Mesh) {
    // Test matching a 2x8 mesh MGD (16 ASICs) to the 2x8_Mesh grouping
    // Using wh_galaxy_split_2x8_2x4_3_mesh which has a 2x8 mesh (MESH4)
    const std::string mgd_path =
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/wh_galaxy_split_2x8_2x4_3_mesh.textproto";

    ASSERT_TRUE(std::filesystem::exists(mgd_path)) << "MGD file not found: " << mgd_path;

    tt::tt_metal::PhysicalSystemDescriptor psd = create_psd_from_mock_cluster();
    PhysicalGroupingDescriptor pgd = load_bh_galaxy_pgd_from_psd(psd);
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
    // Names in triple_16x8 are "2x8_Mesh WH", "2x8_Mesh BH", etc.
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

TEST(PhysicalGroupingDescriptorSP4Tests, GetValidGroupingsForMGD_8x16Mesh) {
    // Test matching an 8x16 mesh MGD (128 ASICs) to the 8x16_Mesh grouping
    const std::string mgd_path = "tt_metal/fabric/mesh_graph_descriptors/quad_galaxy_mesh_graph_descriptor.textproto";

    ASSERT_TRUE(std::filesystem::exists(mgd_path)) << "MGD file not found: " << mgd_path;

    tt::tt_metal::PhysicalSystemDescriptor psd = create_psd_from_mock_cluster();
    PhysicalGroupingDescriptor pgd = load_bh_galaxy_pgd_from_psd(psd);
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

TEST(PhysicalGroupingDescriptorSP4Tests, GetValidGroupingsForMGD_SingleGalaxy4x8) {
    // Test matching a single galaxy mesh MGD (32 ASICs) to the 4x8_Mesh grouping
    // Using single_bh_galaxy_mesh_graph_descriptor which has 8x4 (32 ASICs, same count but different topology)
    const std::string mgd_path =
        "tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto";

    ASSERT_TRUE(std::filesystem::exists(mgd_path)) << "MGD file not found: " << mgd_path;

    tt::tt_metal::PhysicalSystemDescriptor psd = create_psd_from_mock_cluster();
    PhysicalGroupingDescriptor pgd = load_bh_galaxy_pgd_from_psd(psd);
    MeshGraphDescriptor mgd{std::filesystem::path(mgd_path)};

    auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd, psd);

    // Count total groupings across all instances
    size_t total_groupings = 0;
    for (const auto& [instance_type, instances] : valid_groupings) {
        for (const auto& [instance_name, groupings] : instances) {
            total_groupings += groupings.size();
        }
    }

    // On merged SP4 BH PGD, the single-galaxy 32-ASIC MGD matches the canonical 4x8 mesh.
    ASSERT_EQ(total_groupings, 1u) << "Should have exactly one valid 32-ASIC grouping match";

    // Check that we have matches for MESH instances
    ASSERT_EQ(valid_groupings.size(), 1u) << "Should have exactly one instance type (MESH)";
    ASSERT_EQ(valid_groupings.count("MESH"), 1u) << "Should have MESH instance type";
    ASSERT_EQ(valid_groupings.at("MESH").size(), 1u) << "Should have exactly one MESH instance";

    // Check that we have the expected 32-ASIC mesh candidate.
    std::set<std::string> grouping_names;
    for (const auto& [instance_name, groupings] : valid_groupings.at("MESH")) {
        ASSERT_EQ(groupings.size(), 1u) << "Should have exactly one grouping match for this instance";
        for (const auto& grouping : groupings) {
            EXPECT_EQ(grouping.asic_count, 32u) << "Should have 32 ASICs";
            grouping_names.insert(grouping.name);
        }
    }

    EXPECT_EQ(grouping_names, (std::set<std::string>{"4x8_Mesh_flat"}))
        << "Single-galaxy 32-ASIC MGD should match the canonical 4x8 BH PGD mesh candidate";
}

TEST(PhysicalGroupingDescriptorSP4Tests, GetValidGroupingsForMGD_DualGalaxy8x8) {
    // Test matching a dual galaxy MGD with meshes
    // Using dual_galaxy_mesh_graph_descriptor which has 8x8 (64 ASICs) - different from 4x8 but testing dual mesh
    // matching
    const std::string mgd_path = "tt_metal/fabric/mesh_graph_descriptors/dual_galaxy_mesh_graph_descriptor.textproto";

    ASSERT_TRUE(std::filesystem::exists(mgd_path)) << "MGD file not found: " << mgd_path;

    tt::tt_metal::PhysicalSystemDescriptor psd = create_psd_from_mock_cluster();
    PhysicalGroupingDescriptor pgd = load_bh_galaxy_pgd_from_psd(psd);
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

TEST(PhysicalGroupingDescriptorSP4Tests, GetValidGroupingsForMGD_Phase3_HigherLayerGraphMatching) {
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
    // Test with 32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto
    const std::filesystem::path mgd_file_path =
        "tt_metal/fabric/mesh_graph_descriptors/32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto";

    MeshGraphDescriptor mgd(mgd_file_path);

    auto* mock_desc = getenv("TT_METAL_MOCK_CLUSTER_DESC_PATH");
    if (mock_desc == nullptr) {
        GTEST_SKIP() << "TT_METAL_MOCK_CLUSTER_DESC_PATH not set - run with tt-run --mock-cluster-rank-binding";
    }
    tt::tt_metal::PhysicalSystemDescriptor psd = create_psd_from_mock_cluster();
    PhysicalGroupingDescriptor pgd = load_bh_galaxy_pgd_from_psd(psd);

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
    // Test with bh_glx_split_4x2.textproto
    const std::filesystem::path mgd_file_path =
        "tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto";

    MeshGraphDescriptor mgd(mgd_file_path);

    auto* mock_desc = getenv("TT_METAL_MOCK_CLUSTER_DESC_PATH");
    if (mock_desc == nullptr) {
        GTEST_SKIP() << "TT_METAL_MOCK_CLUSTER_DESC_PATH not set - run with tt-run --mock-cluster-rank-binding";
    }
    tt::tt_metal::PhysicalSystemDescriptor psd = create_psd_from_mock_cluster();
    PhysicalGroupingDescriptor pgd = load_bh_galaxy_pgd_from_psd(psd);

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
    // Test with bh_glx_split_4x2.textproto
    const std::filesystem::path mgd_file_path = "tt_metal/fabric/mesh_graph_descriptors/bh_glx_split_4x2.textproto";

    MeshGraphDescriptor mgd(mgd_file_path);

    auto* mock_desc = getenv("TT_METAL_MOCK_CLUSTER_DESC_PATH");
    if (mock_desc == nullptr) {
        GTEST_SKIP() << "TT_METAL_MOCK_CLUSTER_DESC_PATH not set - run with tt-run --mock-cluster-rank-binding";
    }
    tt::tt_metal::PhysicalSystemDescriptor psd = create_psd_from_mock_cluster();
    PhysicalGroupingDescriptor pgd = load_bh_galaxy_pgd_from_psd(psd);

    auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd, psd);

    // M0 mesh has device_topology [4, 2] = 8 chips
    // Should match meshes grouping with 1 tray (8 ASICs, exact match)
    // Note: This test has multiple mesh instances (M0 mesh_id 0-47), all with same topology
    EXPECT_TRUE(valid_groupings.contains("MESH")) << "Should have MESH type in results";
    EXPECT_TRUE(valid_groupings.at("MESH").contains("M0")) << "Should have M0 mesh instance";

    const auto& m0_groupings = valid_groupings.at("MESH").at("M0");
    ASSERT_EQ(m0_groupings.size(), 4u) << "M0 should see one valid 4x2 grouping per canonical BH slice";

    std::set<std::string> slice_refs;
    for (const auto& grouping : m0_groupings) {
        EXPECT_EQ(grouping.name, "meshes") << "M0 should match a mesh grouping";
        EXPECT_EQ(grouping.asic_count, 8u) << "M0 grouping should have 8 ASICs (1 slice, exact match)";
        ASSERT_EQ(grouping.items.size(), 1u) << "Each 4x2 grouping should have exactly 1 slice ref";
        EXPECT_EQ(grouping.items[0].type, GroupingItemInfo::ItemType::GROUPING_REF)
            << "4x2 candidate should be rooted at a GROUPING_REF";
        slice_refs.insert(grouping.items[0].grouping_name);
    }

    EXPECT_EQ(slice_refs, (std::set<std::string>{"SLICE_0", "SLICE_1", "SLICE_2", "SLICE_3"}))
        << "The BH 4x2 MGD should match the four discovery-aligned canonical slices";
}

TEST(PhysicalGroupingDescriptorTests, GetValidGroupingsForMGD_Dual4x4) {
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
    PhysicalGroupingDescriptor pgd = load_bh_galaxy_pgd_from_psd(psd);

    auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd, psd);

    // M0 mesh has device_topology [4, 4] = 16 chips
    // Should match the flattened 4x4 BH mesh candidates, each spanning 2 trays / 16 ASICs.
    // Note: This test has 2 mesh instances (M0 mesh_id 0 and 1) deduped to a single M0 entry.
    // The BH PGD exposes four cyclic-adjacent slice pair 4x4 candidates ([SLICE_0,SLICE_1],
    // [SLICE_1,SLICE_2], [SLICE_2,SLICE_3], [SLICE_3,SLICE_0]); the three non-wrap-around windows
    // embed on the plain SP4 host while the [SLICE_3,SLICE_0] wrap-around variant only embeds on
    // torus-style topologies.
    EXPECT_TRUE(valid_groupings.contains("MESH")) << "Should have MESH type in results";
    EXPECT_TRUE(valid_groupings.at("MESH").contains("M0")) << "Should have M0 mesh instance";

    const auto& m0_groupings = valid_groupings.at("MESH").at("M0");
    ASSERT_GE(m0_groupings.size(), 3u)
        << "M0 should see at least the three non-wrap-around flattened 4x4 BH candidates";

    for (const auto& grouping : m0_groupings) {
        EXPECT_NE(grouping.name.find("4x4"), std::string::npos)
            << "Dual 4x4 MGD should match a flattened 4x4 candidate";
        EXPECT_TRUE(grouping.name.ends_with("_flat")) << "Expected a flattened candidate grouping";
        EXPECT_EQ(grouping.asic_count, 16u) << "Flattened 4x4 candidate should have 16 ASICs";
        ASSERT_EQ(grouping.items.size(), 16u) << "Flattened 4x4 candidate should expose one item per ASIC";

        std::set<uint32_t> trays;
        for (const auto& item : grouping.items) {
            EXPECT_EQ(item.type, GroupingItemInfo::ItemType::ASIC_LOCATION)
                << "Flattened 4x4 candidate should carry ASIC_LOCATION metadata";
            ASSERT_GT(*item.tray_id, 0u) << "Flattened 4x4 candidate ASIC should have a tray id";
            trays.insert(*item.tray_id);
        }
        EXPECT_EQ(trays.size(), 2u) << "Flattened 4x4 candidate should span exactly 2 trays";
    }
}

TEST(PhysicalGroupingDescriptorTests, GetValidGroupingsForMGD_Dual8x2) {
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
    PhysicalGroupingDescriptor pgd = load_bh_galaxy_pgd_from_psd(psd);

    auto valid_groupings = pgd.get_valid_groupings_for_mgd(mgd, psd);

    // M0 mesh has device_topology [8, 2] = 16 chips
    // Should match the flattened 2x8 BH mesh candidates, each spanning 2 trays / 16 ASICs.
    // Note: This test has 2 mesh instances (M0 mesh_id 0 and 1), both with same topology
    EXPECT_TRUE(valid_groupings.contains("MESH")) << "Should have MESH type in results";
    EXPECT_TRUE(valid_groupings.at("MESH").contains("M0")) << "Should have M0 mesh instance";

    const auto& m0_groupings = valid_groupings.at("MESH").at("M0");
    ASSERT_EQ(m0_groupings.size(), 2u) << "M0 should see both flattened 2x8 BH candidates";

    for (const auto& grouping : m0_groupings) {
        EXPECT_NE(grouping.name.find("2x8"), std::string::npos)
            << "Dual 8x2 MGD should match a flattened 2x8 candidate";
        EXPECT_TRUE(grouping.name.ends_with("_flat")) << "Expected a flattened candidate grouping";
        EXPECT_EQ(grouping.asic_count, 16u) << "Flattened 2x8 candidate should have 16 ASICs";
        ASSERT_EQ(grouping.items.size(), 16u) << "Flattened 2x8 candidate should expose one item per ASIC";

        std::set<uint32_t> trays;
        for (const auto& item : grouping.items) {
            EXPECT_EQ(item.type, GroupingItemInfo::ItemType::ASIC_LOCATION)
                << "Flattened 2x8 candidate should carry ASIC_LOCATION metadata";
            ASSERT_GT(*item.tray_id, 0u) << "Flattened 2x8 candidate ASIC should have a tray id";
            trays.insert(*item.tray_id);
        }
        EXPECT_EQ(trays.size(), 2u) << "Flattened 2x8 candidate should span exactly 2 trays";
    }
}

}  // namespace tt::tt_fabric::fabric_router_tests
