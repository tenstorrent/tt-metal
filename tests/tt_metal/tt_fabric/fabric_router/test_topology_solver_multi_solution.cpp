// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Tests for solve_topology_mapping_n, solve_topology_mapping_all, solve_topology_mapping_next.
// Uses int node IDs so the graphs can be specified inline without any fabric machinery.

#include <gtest/gtest.h>

#include <map>
#include <set>
#include <vector>

#include <tt-metalium/experimental/fabric/topology_solver.hpp>
#include <fabric/topology_solver_sat_detail.hpp>

namespace tt::tt_fabric {

using detail::ConstraintIndexData;
using detail::DFSSearchEngine;
using detail::GraphIndexData;
using detail::MappingValidator;
using detail::SatSearchEngine;

using IntAdj = AdjacencyGraph<int>;
using IntConstraints = MappingConstraints<int, int>;
using IntAdjMap = typename IntAdj::AdjacencyMap;
using IntResult = MappingResult<int, int>;
using IntResultVec = std::vector<IntResult>;

// ---------------------------------------------------------------------------
// Helper: build a mapping-map from a MappingResult for comparison
// ---------------------------------------------------------------------------
static std::map<int, int> result_map(const IntResult& r) { return r.target_to_global; }

// ---------------------------------------------------------------------------
// Test 1: SingleNode_TwoGlobals_FindsBoth
//
// 1-node target can map to either of the 2 global nodes => 2 solutions.
// ---------------------------------------------------------------------------
TEST(TopologySolverMultiSolutionTest, SingleNode_TwoGlobals_FindsBoth) {
    IntAdj target(IntAdjMap{{0, {}}});
    IntAdj global(IntAdjMap{{10, {}}, {11, {}}});
    IntConstraints constraints;

    const auto results = solve_topology_mapping_all<int, int>(target, global, constraints,
                                                               ConnectionValidationMode::RELAXED,
                                                               /*quiet_mode=*/true);

    EXPECT_EQ(results.size(), 2u) << "Expected exactly 2 solutions for single-node target into 2-global graph";
    for (const auto& r : results) {
        EXPECT_TRUE(r.success);
        EXPECT_EQ(r.target_to_global.size(), 1u);
    }

    // The two solutions should map target 0 to different globals.
    std::set<int> chosen_globals;
    for (const auto& r : results) {
        chosen_globals.insert(r.target_to_global.at(0));
    }
    EXPECT_EQ(chosen_globals.size(), 2u) << "Both globals should be covered";
}

// ---------------------------------------------------------------------------
// Test 2: TwoNodeChain_CountsAllEmbeddings
//
// 2-chain target (0-1) embedded into 4-path global (20-21-22-23).
// Valid embeddings: (0->20,1->21), (0->21,1->20), (0->21,1->22),
//                  (0->22,1->21), (0->22,1->23), (0->23,1->22) => 6 embeddings.
// (direction matters since mapping is injective and edges are undirected)
// ---------------------------------------------------------------------------
TEST(TopologySolverMultiSolutionTest, TwoNodeChain_CountsAllEmbeddings) {
    IntAdj target(IntAdjMap{{0, {1}}, {1, {0}}});
    IntAdj global(IntAdjMap{{20, {21}}, {21, {20, 22}}, {22, {21, 23}}, {23, {22}}});
    IntConstraints constraints;

    const auto results = solve_topology_mapping_all<int, int>(target, global, constraints,
                                                               ConnectionValidationMode::RELAXED,
                                                               /*quiet_mode=*/true);

    // 3 undirected positions × 2 directions = 6 distinct injective embeddings.
    EXPECT_EQ(results.size(), 6u) << "Expected 6 embeddings of a 2-chain into a 4-path";
    for (const auto& r : results) {
        EXPECT_TRUE(r.success);
    }
}

// ---------------------------------------------------------------------------
// Test 3: SolveN_RespectsMaxLimit
//
// 4-clique target embedded into a large global graph with many solutions.
// Request max=2 — must get at most 2 results.
// ---------------------------------------------------------------------------
TEST(TopologySolverMultiSolutionTest, SolveN_RespectsMaxLimit) {
    // 4-clique target
    IntAdj target(IntAdjMap{
        {0, {1, 2, 3}},
        {1, {0, 2, 3}},
        {2, {0, 1, 3}},
        {3, {0, 1, 2}},
    });
    // 6-clique global (many embeddings of 4-clique)
    IntAdj global(IntAdjMap{
        {10, {11, 12, 13, 14, 15}},
        {11, {10, 12, 13, 14, 15}},
        {12, {10, 11, 13, 14, 15}},
        {13, {10, 11, 12, 14, 15}},
        {14, {10, 11, 12, 13, 15}},
        {15, {10, 11, 12, 13, 14}},
    });
    IntConstraints constraints;

    const auto results = solve_topology_mapping_n<int, int>(target, global, constraints,
                                                             /*max_solutions=*/2,
                                                             ConnectionValidationMode::RELAXED,
                                                             /*quiet_mode=*/true);

    EXPECT_LE(results.size(), 2u);
    EXPECT_GE(results.size(), 1u) << "There should be at least 1 valid solution";
    for (const auto& r : results) {
        EXPECT_TRUE(r.success);
    }
}

// ---------------------------------------------------------------------------
// Test 4: SolveAll_NoDuplicates
//
// All returned mappings must be distinct (no two are identical).
// ---------------------------------------------------------------------------
TEST(TopologySolverMultiSolutionTest, SolveAll_NoDuplicates) {
    IntAdj target(IntAdjMap{{0, {1}}, {1, {0}}});
    IntAdj global(IntAdjMap{{30, {31}}, {31, {30, 32}}, {32, {31, 33}}, {33, {32}}});
    IntConstraints constraints;

    const auto results = solve_topology_mapping_all<int, int>(target, global, constraints,
                                                               ConnectionValidationMode::RELAXED,
                                                               /*quiet_mode=*/true);

    // Collect all mappings and check for duplicates.
    std::set<std::map<int, int>> seen;
    for (const auto& r : results) {
        EXPECT_TRUE(r.success);
        const auto m = result_map(r);
        EXPECT_TRUE(seen.find(m) == seen.end()) << "Duplicate mapping found";
        seen.insert(m);
    }
}

// ---------------------------------------------------------------------------
// Test 5: SolveAll_AllValid
//
// Every returned result must pass MappingValidator independently.
// ---------------------------------------------------------------------------
TEST(TopologySolverMultiSolutionTest, SolveAll_AllValid) {
    IntAdj target(IntAdjMap{{0, {1}}, {1, {0}}});
    IntAdj global(IntAdjMap{{40, {41}}, {41, {40, 42}}, {42, {41}}});
    IntConstraints constraints;

    GraphIndexData<int, int> graph_data(target, global);
    ConstraintIndexData<int, int> constraint_data(constraints, graph_data);

    const auto results = solve_topology_mapping_all<int, int>(target, global, constraints,
                                                               ConnectionValidationMode::RELAXED,
                                                               /*quiet_mode=*/true);

    EXPECT_GT(results.size(), 0u);
    for (const auto& r : results) {
        EXPECT_TRUE(r.success);

        // Also validate directly via MappingValidator to ensure build_result is consistent.
        // Rebuild index-level mapping from result.
        std::vector<int> raw(graph_data.n_target, -1);
        for (const auto& [tn, gn] : r.target_to_global) {
            auto ti = graph_data.target_to_idx.find(tn);
            auto gi = graph_data.global_to_idx.find(gn);
            ASSERT_NE(ti, graph_data.target_to_idx.end());
            ASSERT_NE(gi, graph_data.global_to_idx.end());
            raw[ti->second] = static_cast<int>(gi->second);
        }

        std::vector<std::string> warnings;
        const bool valid = MappingValidator<int, int>::validate_mapping(
            raw, graph_data, constraint_data, ConnectionValidationMode::RELAXED, &warnings, /*quiet_mode=*/true);
        EXPECT_TRUE(valid) << "Returned result failed independent validation";
    }
}

// ---------------------------------------------------------------------------
// Test 6: SolveNext_ExcludesGivenMappings
//
// After finding the first solution, call solve_topology_mapping_next to get
// a different one. The two must differ.
// ---------------------------------------------------------------------------
TEST(TopologySolverMultiSolutionTest, SolveNext_ExcludesGivenMappings) {
    IntAdj target(IntAdjMap{{0, {}}});
    IntAdj global(IntAdjMap{{50, {}}, {51, {}}, {52, {}}});
    IntConstraints constraints;

    // Get first solution.
    const auto first = solve_topology_mapping<int, int>(target, global, constraints,
                                                         ConnectionValidationMode::RELAXED,
                                                         /*quiet_mode=*/true);
    ASSERT_TRUE(first.success);

    // Exclude the first solution and request a new one.
    std::vector<std::map<int, int>> excluded{first.target_to_global};
    const auto second = solve_topology_mapping_next<int, int>(
        target, global, constraints, excluded,
        ConnectionValidationMode::RELAXED, /*quiet_mode=*/true);

    EXPECT_TRUE(second.success) << "Should find a second solution";
    EXPECT_NE(result_map(first), result_map(second)) << "Second solution must differ from first";
}

// ---------------------------------------------------------------------------
// Test 7: SolveNext_WhenExhausted_ReturnsFailure
//
// Exclude all possible solutions — the next call must return success=false.
// ---------------------------------------------------------------------------
TEST(TopologySolverMultiSolutionTest, SolveNext_WhenExhausted_ReturnsFailure) {
    // 1-node target, 1-node global: exactly one mapping (0 -> 60).
    IntAdj target(IntAdjMap{{0, {}}});
    IntAdj global(IntAdjMap{{60, {}}});
    IntConstraints constraints;

    const auto first = solve_topology_mapping<int, int>(target, global, constraints,
                                                         ConnectionValidationMode::RELAXED,
                                                         /*quiet_mode=*/true);
    ASSERT_TRUE(first.success);

    // Exclude the only solution.
    std::vector<std::map<int, int>> excluded{first.target_to_global};
    const auto next = solve_topology_mapping_next<int, int>(
        target, global, constraints, excluded,
        ConnectionValidationMode::RELAXED, /*quiet_mode=*/true);

    EXPECT_FALSE(next.success) << "Should return failure when all solutions are excluded";
}

// ---------------------------------------------------------------------------
// Test 8: SolveN_BothEnginesAgree
//
// Run solve_topology_mapping_n with both SAT and DFS engines on the same graph
// and verify that both return the same count of solutions, all of which are valid.
// ---------------------------------------------------------------------------
TEST(TopologySolverMultiSolutionTest, SolveN_BothEnginesAgree) {
    // 2-node chain target into 3-node path global — 4 embeddings (2 positions × 2 directions).
    IntAdj target(IntAdjMap{{0, {1}}, {1, {0}}});
    IntAdj global(IntAdjMap{{70, {71}}, {71, {70, 72}}, {72, {71}}});
    IntConstraints constraints;

    const auto sat_results = solve_topology_mapping_n<int, int>(
        target, global, constraints, /*max_solutions=*/100,
        ConnectionValidationMode::RELAXED, /*quiet_mode=*/true,
        TopologyMappingSolverEngine::Sat);

    const auto dfs_results = solve_topology_mapping_n<int, int>(
        target, global, constraints, /*max_solutions=*/100,
        ConnectionValidationMode::RELAXED, /*quiet_mode=*/true,
        TopologyMappingSolverEngine::Dfs);

    EXPECT_EQ(sat_results.size(), dfs_results.size())
        << "SAT and DFS engines should find the same number of solutions";

    for (const auto& r : sat_results) {
        EXPECT_TRUE(r.success) << "Every SAT-engine result must be valid";
    }
    for (const auto& r : dfs_results) {
        EXPECT_TRUE(r.success) << "Every DFS-engine result must be valid";
    }

    // Verify both sets contain the same set of mappings.
    std::set<std::map<int, int>> sat_set, dfs_set;
    for (const auto& r : sat_results) sat_set.insert(result_map(r));
    for (const auto& r : dfs_results) dfs_set.insert(result_map(r));
    EXPECT_EQ(sat_set, dfs_set) << "SAT and DFS should produce identical solution sets";
}

// ---------------------------------------------------------------------------
// Test 9: EmptyTarget_SolveAll_ReturnsOneTrivialResult
//
// Empty target graph => exactly one trivial (empty) mapping.
// ---------------------------------------------------------------------------
TEST(TopologySolverMultiSolutionTest, EmptyTarget_SolveAll_ReturnsOneTrivialResult) {
    IntAdj target;
    IntAdj global(IntAdjMap{{80, {81}}, {81, {80}}});
    IntConstraints constraints;

    const auto results = solve_topology_mapping_all<int, int>(target, global, constraints,
                                                               ConnectionValidationMode::RELAXED,
                                                               /*quiet_mode=*/true);

    EXPECT_EQ(results.size(), 1u) << "Empty target should yield exactly one trivial mapping";
    if (!results.empty()) {
        EXPECT_TRUE(results[0].success);
        EXPECT_TRUE(results[0].target_to_global.empty());
    }
}

// ---------------------------------------------------------------------------
// Test 10: NoGlobals_SolveN_ReturnsEmpty
//
// Non-empty target but empty global graph — no embedding possible.
// ---------------------------------------------------------------------------
TEST(TopologySolverMultiSolutionTest, NoGlobals_SolveN_ReturnsEmpty) {
    IntAdj target(IntAdjMap{{0, {}}});
    IntAdj global;
    IntConstraints constraints;

    const auto results = solve_topology_mapping_n<int, int>(target, global, constraints,
                                                             /*max_solutions=*/5,
                                                             ConnectionValidationMode::RELAXED,
                                                             /*quiet_mode=*/true);

    EXPECT_TRUE(results.empty()) << "No solutions should exist when global graph is empty";
}

}  // namespace tt::tt_fabric
