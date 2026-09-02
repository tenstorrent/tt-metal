// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Unit tests for the adjacency-guided placement search, driven through the pool entry point declared
// in physical_grouping_descriptor.hpp. That is the pure-graph core, over the same structures
// find_all_in_psd uses (GroupingInfo in, MappingResult candidates, PsdPlacement out), so these run
// offline: no PSD, no mock cluster, no descriptor files. The search internals have internal linkage
// and are deliberately not reachable from here.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/fabric/physical_grouping_descriptor.hpp>
#include <tt-metalium/experimental/fabric/topology_solver.hpp>

namespace tt::tt_fabric::fabric_router_tests {
namespace {

using tt::tt_metal::AsicID;

AsicID chip(std::uint64_t id) { return AsicID{id}; }

// Physical chips in a line: 0 - 1 - 2 - ... - (count-1).
AdjacencyGraph<AsicID> make_chip_line(std::uint64_t count) {
    AdjacencyGraph<AsicID>::AdjacencyMap adjacency;
    for (std::uint64_t i = 0; i < count; ++i) {
        std::vector<AsicID> neighbors;
        if (i > 0) {
            neighbors.push_back(chip(i - 1));
        }
        if (i + 1 < count) {
            neighbors.push_back(chip(i + 1));
        }
        adjacency[chip(i)] = std::move(neighbors);
    }
    return AdjacencyGraph<AsicID>(adjacency);
}

AdjacencyGraph<AsicID> make_chip_graph(
    std::uint64_t count, const std::vector<std::pair<std::uint64_t, std::uint64_t>>& links) {
    AdjacencyGraph<AsicID>::AdjacencyMap adjacency;
    for (std::uint64_t i = 0; i < count; ++i) {
        adjacency[chip(i)];
    }
    for (const auto& [a, b] : links) {
        adjacency[chip(a)].push_back(chip(b));
        adjacency[chip(b)].push_back(chip(a));
    }
    return AdjacencyGraph<AsicID>(adjacency);
}

AdjacencyGraph<LogicalChipId> make_mesh_graph(
    std::uint32_t mesh_count, const std::vector<std::pair<std::uint32_t, std::uint32_t>>& edges) {
    AdjacencyGraph<LogicalChipId>::AdjacencyMap adjacency;
    for (std::uint32_t i = 0; i < mesh_count; ++i) {
        adjacency[i];
    }
    for (const auto& [a, b] : edges) {
        adjacency[a].push_back(b);
        adjacency[b].push_back(a);
    }
    return AdjacencyGraph<LogicalChipId>(adjacency);
}

// A flattened grouping shaped like a 1xN line, which is what find_all_in_psd hands to the packer.
GroupingInfo make_line_grouping(const std::string& name, std::uint32_t width) {
    GroupingInfo grouping;
    grouping.name = name;
    grouping.type = "MESH";
    grouping.asic_count = width;
    AdjacencyGraph<LogicalChipId>::AdjacencyMap adjacency;
    for (std::uint32_t i = 0; i < width; ++i) {
        std::vector<std::uint32_t> neighbors;
        if (i > 0) {
            neighbors.push_back(i - 1);
        }
        if (i + 1 < width) {
            neighbors.push_back(i + 1);
        }
        adjacency[i] = std::move(neighbors);
    }
    grouping.adjacency_graph = AdjacencyGraph<LogicalChipId>(adjacency);
    return grouping;
}

// The solver output for placing a width-N line grouping on chips [start, start + width).
PlacementCandidate make_candidate(std::size_t grouping_index, std::uint64_t start, std::uint32_t width) {
    PlacementCandidate candidate;
    candidate.grouping_index = grouping_index;
    candidate.result.success = true;
    for (std::uint32_t node = 0; node < width; ++node) {
        candidate.result.target_to_global[node] = chip(start + node);
        candidate.result.global_to_target[chip(start + node)] = node;
    }
    return candidate;
}

// Every contiguous window of `width` chips on a line. Stands in for what phase A enumerates for one
// grouping via solve_topology_mapping_n.
void append_line_windows(
    std::vector<PlacementCandidate>& pool, std::size_t grouping_index, std::uint32_t width, std::uint64_t chip_count) {
    for (std::uint64_t start = 0; start + width <= chip_count; ++start) {
        pool.push_back(make_candidate(grouping_index, start, width));
    }
}

std::vector<std::uint64_t> chips_of(const PsdPlacement& placement) {
    std::vector<std::uint64_t> ids;
    for (const auto& asic : placement.asics) {
        ids.push_back(*asic);
    }
    std::sort(ids.begin(), ids.end());
    return ids;
}

// Fails the calling test if any two placed regions share a chip.
void expect_disjoint(const std::vector<PsdPlacement>& placements) {
    std::vector<std::uint64_t> seen;
    for (const auto& placement : placements) {
        for (const auto& asic : placement.asics) {
            EXPECT_EQ(std::count(seen.begin(), seen.end(), *asic), 0) << "chip " << *asic << " placed twice";
            seen.push_back(*asic);
        }
    }
}

// The counterexample from the design plan. Eight chips in a line, a 4-wide mesh flanked by two
// 2-wide meshes. Maximum-coverage tiling picks the two 4-wide windows [0..3] and [4..7] first
// because they cover the most chips, which leaves nowhere for the 2-wide meshes; the only tiling
// that satisfies the logical chain is the one that gives up coverage in the middle.
TEST(AdjacencyGuidedPlacement, ChainOfTwoShapesFindsTheOnlyValidTiling) {
    constexpr std::uint64_t kChipCount = 8;
    const std::vector<GroupingInfo> groupings = {make_line_grouping("S2", 2), make_line_grouping("S4", 4)};

    std::vector<PlacementCandidate> pool;
    append_line_windows(pool, /*grouping_index=*/0, /*width=*/2, kChipCount);
    append_line_windows(pool, /*grouping_index=*/1, /*width=*/4, kChipCount);

    const std::vector<std::vector<std::size_t>> options = {{0}, {1}, {0}};
    const auto mesh_graph = make_mesh_graph(3, {{0, 1}, {1, 2}});

    const auto result =
        solve_adjacency_guided_placement(groupings, pool, options, mesh_graph, make_chip_line(kChipCount));

    ASSERT_TRUE(result.success) << result.error_message;
    ASSERT_EQ(result.placements.size(), 3u);
    expect_disjoint(result.placements);

    // The middle mesh has to sit in the interior, and then the flanks are forced.
    EXPECT_EQ(chips_of(result.placements[1]), (std::vector<std::uint64_t>{2, 3, 4, 5}));
    std::vector<std::vector<std::uint64_t>> flanks = {chips_of(result.placements[0]), chips_of(result.placements[2])};
    std::sort(flanks.begin(), flanks.end());
    EXPECT_EQ(flanks, (std::vector<std::vector<std::uint64_t>>{{0, 1}, {6, 7}}));

    // Loose regression bound. The exact count will move as pruning improves; what matters is that
    // this does not degenerate into walking the product of the two candidate pools.
    EXPECT_LT(result.nodes_expanded, 100u) << "search is not pruning";
}

TEST(AdjacencyGuidedPlacement, PlacementsAreChipDisjoint) {
    constexpr std::uint64_t kChipCount = 4;
    const std::vector<GroupingInfo> groupings = {make_line_grouping("S2", 2)};
    std::vector<PlacementCandidate> pool;
    append_line_windows(pool, 0, 2, kChipCount);

    const auto result = solve_adjacency_guided_placement(
        groupings, pool, {{0}, {0}}, make_mesh_graph(2, {{0, 1}}), make_chip_line(kChipCount));

    ASSERT_TRUE(result.success) << result.error_message;
    expect_disjoint(result.placements);

    // [1,2] overlaps both alternatives, so the only disjoint adjacent pair is the two ends.
    std::vector<std::vector<std::uint64_t>> placed = {chips_of(result.placements[0]), chips_of(result.placements[1])};
    std::sort(placed.begin(), placed.end());
    EXPECT_EQ(placed, (std::vector<std::vector<std::uint64_t>>{{0, 1}, {2, 3}}));
}

// Two regions exist and are individually placeable, but no ethernet link crosses between them. A
// packer that only maximises coverage accepts this; the adjacency constraint rejects it.
TEST(AdjacencyGuidedPlacement, RejectsWhenNoLinkCrossesBetweenRegions) {
    const std::vector<GroupingInfo> groupings = {make_line_grouping("S2", 2)};
    const std::vector<PlacementCandidate> pool = {make_candidate(0, 0, 2), make_candidate(0, 2, 2)};

    const auto result = solve_adjacency_guided_placement(
        groupings, pool, {{0}, {0}}, make_mesh_graph(2, {{0, 1}}), make_chip_graph(4, {{0, 1}, {2, 3}}));

    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.budget_exhausted);
    EXPECT_NE(result.error_message.find("chip-disjoint"), std::string::npos) << result.error_message;
}

// Same pool and same disconnected hardware, but the logical meshes do not talk to each other, so
// there is nothing to satisfy and both placements stand.
TEST(AdjacencyGuidedPlacement, IndependentMeshesNeedNoLinkBetweenThem) {
    const std::vector<GroupingInfo> groupings = {make_line_grouping("S2", 2)};
    const std::vector<PlacementCandidate> pool = {make_candidate(0, 0, 2), make_candidate(0, 2, 2)};

    const auto result = solve_adjacency_guided_placement(
        groupings, pool, {{0}, {0}}, make_mesh_graph(2, {}), make_chip_graph(4, {{0, 1}, {2, 3}}));

    ASSERT_TRUE(result.success) << result.error_message;
    expect_disjoint(result.placements);
}

// A mesh restricted to one grouping cannot be satisfied by another grouping's placements, even when
// those cover the same chips.
TEST(AdjacencyGuidedPlacement, MeshOnlyTakesAGroupingItAccepts) {
    const std::vector<GroupingInfo> groupings = {make_line_grouping("Alpha", 1), make_line_grouping("Beta", 1)};
    const std::vector<PlacementCandidate> pool = {
        make_candidate(0, 0, 1),  // Alpha on chip 0
        make_candidate(0, 1, 1),  // Alpha on chip 1
        make_candidate(1, 2, 1),  // Beta on chip 2
        make_candidate(1, 3, 1),  // Beta on chip 3
    };

    const auto result =
        solve_adjacency_guided_placement(groupings, pool, {{0}, {1}}, make_mesh_graph(2, {{0, 1}}), make_chip_line(4));

    ASSERT_TRUE(result.success) << result.error_message;
    EXPECT_EQ(pool[result.assignment[0]].grouping_index, 0u);
    EXPECT_EQ(pool[result.assignment[1]].grouping_index, 1u);
    // Only the Alpha/Beta pair straddling the middle link satisfies adjacency.
    EXPECT_EQ(chips_of(result.placements[0]), (std::vector<std::uint64_t>{1}));
    EXPECT_EQ(chips_of(result.placements[1]), (std::vector<std::uint64_t>{2}));
}

// The variant case: one MGD mesh accepts several PGD groupings, and which one is right depends on
// what is left after its neighbour is placed. This is the mechanism behind the 4x4_Mesh versus
// 4x4_SplitHost problem, where committing to a single variant up front loses the only solution.
TEST(AdjacencyGuidedPlacement, MeshWithSeveralVariantsFallsBackToTheOneThatFits) {
    const std::vector<GroupingInfo> groupings = {make_line_grouping("Wide", 3), make_line_grouping("Narrow", 1)};
    std::vector<PlacementCandidate> pool;
    append_line_windows(pool, /*grouping_index=*/0, /*width=*/3, /*chip_count=*/4);
    append_line_windows(pool, /*grouping_index=*/1, /*width=*/1, /*chip_count=*/4);

    // Mesh 0 would happily take Wide; mesh 1 can only be Wide. Four chips cannot hold two Wides, so
    // mesh 0 has to give up its preferred variant.
    const std::vector<std::vector<std::size_t>> options = {{0, 1}, {0}};

    const auto result =
        solve_adjacency_guided_placement(groupings, pool, options, make_mesh_graph(2, {{0, 1}}), make_chip_line(4));

    ASSERT_TRUE(result.success) << result.error_message;
    expect_disjoint(result.placements);
    EXPECT_EQ(pool[result.assignment[0]].grouping_index, 1u) << "mesh 0 should have fallen back to Narrow";
    EXPECT_EQ(pool[result.assignment[1]].grouping_index, 0u);
    EXPECT_EQ(chips_of(result.placements[0]), (std::vector<std::uint64_t>{3}));
    EXPECT_EQ(chips_of(result.placements[1]), (std::vector<std::uint64_t>{0, 1, 2}));
}

// find_all_in_psd copies the grouping's pinning map onto every placement it emits; the labelled
// search has to carry the same map through, or downstream loses the PGD slot assignment.
TEST(AdjacencyGuidedPlacement, PlacementCarriesTheGroupingPinningMap) {
    GroupingInfo grouping = make_line_grouping("Pinned", 2);
    grouping.mesh_node_to_asic_position[0] =
        tt::tt_metal::ASICPosition{tt::tt_metal::TrayID{1}, tt::tt_metal::ASICLocation{7}};
    const std::vector<GroupingInfo> groupings = {std::move(grouping)};
    const std::vector<PlacementCandidate> pool = {make_candidate(0, 0, 2)};

    const auto result =
        solve_adjacency_guided_placement(groupings, pool, {{0}}, make_mesh_graph(1, {}), make_chip_line(4));

    ASSERT_TRUE(result.success) << result.error_message;
    ASSERT_EQ(result.placements.size(), 1u);
    EXPECT_EQ(result.placements[0].mesh_node_to_asic_position, groupings[0].mesh_node_to_asic_position);
}

TEST(AdjacencyGuidedPlacement, ReportsGroupingWithNoEnumeratedPlacements) {
    const std::vector<GroupingInfo> groupings = {make_line_grouping("Placed", 2), make_line_grouping("Absent", 2)};
    std::vector<PlacementCandidate> pool;
    append_line_windows(pool, 0, 2, 4);

    const auto result =
        solve_adjacency_guided_placement(groupings, pool, {{1}}, make_mesh_graph(1, {}), make_chip_line(4));

    EXPECT_FALSE(result.success);
    EXPECT_NE(result.error_message.find("Absent"), std::string::npos) << result.error_message;
}

TEST(AdjacencyGuidedPlacement, RejectsCandidateReferencingUnknownChip) {
    const std::vector<GroupingInfo> groupings = {make_line_grouping("S2", 2)};
    std::vector<PlacementCandidate> pool = {make_candidate(0, 0, 2)};
    pool[0].result.target_to_global[1] = chip(99);

    const auto result =
        solve_adjacency_guided_placement(groupings, pool, {{0}}, make_mesh_graph(1, {}), make_chip_line(4));

    EXPECT_FALSE(result.success);
    EXPECT_NE(result.error_message.find("not a node of the physical graph"), std::string::npos) << result.error_message;
}

TEST(AdjacencyGuidedPlacement, RejectsFailedMappingResultInThePool) {
    const std::vector<GroupingInfo> groupings = {make_line_grouping("S2", 2)};
    std::vector<PlacementCandidate> pool = {make_candidate(0, 0, 2)};
    pool[0].result.success = false;

    const auto result =
        solve_adjacency_guided_placement(groupings, pool, {{0}}, make_mesh_graph(1, {}), make_chip_line(4));

    EXPECT_FALSE(result.success);
    EXPECT_NE(result.error_message.find("failed MappingResult"), std::string::npos) << result.error_message;
}

TEST(AdjacencyGuidedPlacement, RejectsOutOfRangeGroupingIndex) {
    const std::vector<GroupingInfo> groupings = {make_line_grouping("S2", 2)};
    const std::vector<PlacementCandidate> pool = {make_candidate(/*grouping_index=*/3, 0, 2)};

    const auto result =
        solve_adjacency_guided_placement(groupings, pool, {{0}}, make_mesh_graph(1, {}), make_chip_line(4));

    EXPECT_FALSE(result.success);
    EXPECT_NE(result.error_message.find("names grouping 3"), std::string::npos) << result.error_message;
}

TEST(AdjacencyGuidedPlacement, NodeBudgetStopsTheSearchAndSaysSo) {
    constexpr std::uint64_t kChipCount = 8;
    const std::vector<GroupingInfo> groupings = {make_line_grouping("S2", 2), make_line_grouping("S4", 4)};
    std::vector<PlacementCandidate> pool;
    append_line_windows(pool, 0, 2, kChipCount);
    append_line_windows(pool, 1, 4, kChipCount);

    const auto result = solve_adjacency_guided_placement(
        groupings,
        pool,
        {{0}, {1}, {0}},
        make_mesh_graph(3, {{0, 1}, {1, 2}}),
        make_chip_line(kChipCount),
        /*node_budget=*/1);

    EXPECT_FALSE(result.success);
    EXPECT_TRUE(result.budget_exhausted);
    EXPECT_NE(result.error_message.find("node budget"), std::string::npos) << result.error_message;
}

TEST(AdjacencyGuidedPlacement, NoLogicalMeshesSucceedsTrivially) {
    const auto result = solve_adjacency_guided_placement({}, {}, {}, make_mesh_graph(0, {}), make_chip_line(4));

    EXPECT_TRUE(result.success);
    EXPECT_TRUE(result.placements.empty());
}

}  // namespace
}  // namespace tt::tt_fabric::fabric_router_tests
