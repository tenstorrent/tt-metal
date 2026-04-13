// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cadical.hpp>
#include <map>
#include <vector>

#include <tt-metalium/experimental/fabric/topology_solver.hpp>

namespace tt::tt_fabric {

using detail::ConstraintIndexData;
using detail::DFSSearchEngine;
using detail::GraphIndexData;
using detail::MappingValidator;
using detail::SatSearchEngine;
using detail::topology_sat_decode_hard_solution;
using detail::topology_sat_encode_hard_constraints;
using detail::TopologySatHardEncoding;

using IntAdj = AdjacencyGraph<int>;
using IntConstraints = MappingConstraints<int, int>;
using IntAdjMap = typename IntAdj::AdjacencyMap;

TEST(TopologySatEncoderTest, EmptyTarget_NoVars_Sat) {
    IntAdj target;
    IntAdj global(IntAdjMap{{10, {11}}, {11, {10}}});
    IntConstraints constraints;
    GraphIndexData<int, int> graph_data(target, global);
    ConstraintIndexData<int, int> constraint_data(constraints, graph_data);
    CaDiCaL::Solver solver;
    TopologySatHardEncoding enc;
    ASSERT_TRUE(topology_sat_encode_hard_constraints(solver, graph_data, constraint_data, enc));
    ASSERT_FALSE(enc.trivial_unsat);
    ASSERT_TRUE(enc.allowed_global_idx.empty());
    ASSERT_EQ(solver.solve(), CaDiCaL::SATISFIABLE);
}

TEST(TopologySatEncoderTest, NoGlobalNodes_TrivialUnsat) {
    IntAdj target(IntAdjMap{{0, {}}});
    IntAdj global;
    IntConstraints constraints;
    GraphIndexData<int, int> graph_data(target, global);
    ConstraintIndexData<int, int> constraint_data(constraints, graph_data);
    CaDiCaL::Solver solver;
    TopologySatHardEncoding enc;
    EXPECT_FALSE(topology_sat_encode_hard_constraints(solver, graph_data, constraint_data, enc));
    EXPECT_TRUE(enc.trivial_unsat);
}

TEST(TopologySatEncoderTest, SingleIsolatedNode_MapsAndValidates) {
    IntAdj target(IntAdjMap{{0, {}}});
    IntAdj global(IntAdjMap{{10, {}}});
    IntConstraints constraints;
    GraphIndexData<int, int> graph_data(target, global);
    ConstraintIndexData<int, int> constraint_data(constraints, graph_data);
    CaDiCaL::Solver solver;
    TopologySatHardEncoding enc;
    ASSERT_TRUE(topology_sat_encode_hard_constraints(solver, graph_data, constraint_data, enc));
    ASSERT_EQ(solver.solve(), CaDiCaL::SATISFIABLE);
    std::vector<int> mapping;
    ASSERT_TRUE((topology_sat_decode_hard_solution<int, int>(solver, enc, mapping)));
    ASSERT_EQ(mapping.size(), 1u);
    EXPECT_EQ(mapping[0], 0);
    std::vector<std::string> warnings;
    const bool valid = MappingValidator<int, int>::validate_mapping(
        mapping, graph_data, constraint_data, ConnectionValidationMode::RELAXED, &warnings);
    EXPECT_TRUE(valid);
}

TEST(TopologySatEncoderTest, TwoNodeChain_EmbedsIntoPath_Validates) {
    IntAdj target(IntAdjMap{{0, {1}}, {1, {0}}});
    IntAdj global(IntAdjMap{{10, {11}}, {11, {10, 12}}, {12, {11}}});
    IntConstraints constraints;
    GraphIndexData<int, int> graph_data(target, global);
    ConstraintIndexData<int, int> constraint_data(constraints, graph_data);
    CaDiCaL::Solver solver;
    TopologySatHardEncoding enc;
    ASSERT_TRUE(topology_sat_encode_hard_constraints(solver, graph_data, constraint_data, enc));
    ASSERT_EQ(solver.solve(), CaDiCaL::SATISFIABLE);
    std::vector<int> mapping;
    ASSERT_TRUE((topology_sat_decode_hard_solution<int, int>(solver, enc, mapping)));
    ASSERT_EQ(mapping.size(), 2u);
    std::vector<std::string> warnings;
    const bool valid = MappingValidator<int, int>::validate_mapping(
        mapping, graph_data, constraint_data, ConnectionValidationMode::RELAXED, &warnings);
    EXPECT_TRUE(valid);
}

TEST(TopologySatEncoderTest, TriangleTarget_PathGlobal_Unsat) {
    IntAdj target(IntAdjMap{{0, {1, 2}}, {1, {0, 2}}, {2, {0, 1}}});
    IntAdj global(IntAdjMap{{10, {11}}, {11, {10, 12}}, {12, {11}}});
    IntConstraints constraints;
    GraphIndexData<int, int> graph_data(target, global);
    ConstraintIndexData<int, int> constraint_data(constraints, graph_data);
    CaDiCaL::Solver solver;
    TopologySatHardEncoding enc;
    const bool encoded = topology_sat_encode_hard_constraints(solver, graph_data, constraint_data, enc);
    if (encoded) {
        EXPECT_EQ(solver.solve(), CaDiCaL::UNSATISFIABLE);
    } else {
        EXPECT_TRUE(enc.trivial_unsat) << "Should be trivially UNSAT (arc consistency detects infeasibility)";
    }
}

TEST(TopologySatEncoderTest, SatVsDfs_SmallFixtures_Agree) {
    auto run_both = [](const IntAdj& target, const IntAdj& global) {
        IntConstraints constraints;
        GraphIndexData<int, int> graph_data(target, global);
        ConstraintIndexData<int, int> constraint_data(constraints, graph_data);
        DFSSearchEngine<int, int> dfs;
        const bool dfs_ok = dfs.search(graph_data, constraint_data, ConnectionValidationMode::RELAXED, true);
        SatSearchEngine<int, int> sat;
        const bool sat_ok = sat.search(graph_data, constraint_data, ConnectionValidationMode::RELAXED, true);
        EXPECT_EQ(dfs_ok, sat_ok) << "DFS and SAT should agree on SAT/UNSAT for hard-encoded fixtures";
        if (dfs_ok && sat_ok) {
            std::vector<std::string> w1;
            std::vector<std::string> w2;
            const bool v1 = MappingValidator<int, int>::validate_mapping(
                dfs.get_state().mapping, graph_data, constraint_data, ConnectionValidationMode::RELAXED, &w1, true);
            const bool v2 = MappingValidator<int, int>::validate_mapping(
                sat.get_state().mapping, graph_data, constraint_data, ConnectionValidationMode::RELAXED, &w2, true);
            EXPECT_TRUE(v1);
            EXPECT_TRUE(v2);
        }
    };

    run_both(IntAdj{}, IntAdj(IntAdjMap{{10, {11}}, {11, {10}}}));
    run_both(IntAdj(IntAdjMap{{0, {}}}), IntAdj(IntAdjMap{{10, {}}}));
    run_both(IntAdj(IntAdjMap{{0, {1}}, {1, {0}}}), IntAdj(IntAdjMap{{10, {11}}, {11, {10, 12}}, {12, {11}}}));
    run_both(
        IntAdj(IntAdjMap{{0, {1, 2}}, {1, {0, 2}}, {2, {0, 1}}}),
        IntAdj(IntAdjMap{{10, {11}}, {11, {10, 12}}, {12, {11}}}));
}

TEST(TopologySatEncoderTest, SolveTopologyMapping_RespectsSolverEngineSelector) {
    IntAdj target(IntAdjMap{{0, {1}}, {1, {0}}});
    IntAdj global(IntAdjMap{{10, {11}}, {11, {10, 12}}, {12, {11}}});
    IntConstraints constraints;

    const auto sat_result = solve_topology_mapping<int, int>(
        target, global, constraints, ConnectionValidationMode::RELAXED, true, TopologyMappingSolverEngine::Sat);
    const auto dfs_result = solve_topology_mapping<int, int>(
        target, global, constraints, ConnectionValidationMode::RELAXED, true, TopologyMappingSolverEngine::Dfs);

    EXPECT_TRUE(sat_result.success);
    EXPECT_TRUE(dfs_result.success);
    EXPECT_EQ(sat_result.target_to_global.size(), dfs_result.target_to_global.size());
}

}  // namespace tt::tt_fabric
