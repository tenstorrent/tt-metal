// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Private header — not part of the public API header set.
// Include after <tt-metalium/experimental/fabric/topology_solver.hpp> so that
// TopologySatHardEncoding and the SAT view types are already defined.

#include "topology_solver_sat_solver.hpp"
#include <tt-metalium/experimental/fabric/topology_solver.hpp>

namespace tt::tt_fabric::detail {

// Forward-declared in topology_solver.hpp; defined here.
// Aggregates the CaDiCaL solver instance with its hard-encoding metadata so
// that the template code in topology_solver.tpp can manage SAT state through
// an opaque pointer without including CaDiCaL or topology_solver_sat_solver.hpp.
struct TopologySatSession {
    TopologySatSolver solver;
    // enc is intentionally NOT stored here — it stays as a value member of
    // TopologyMappingEnumerationSession (in the public header) because its
    // type, TopologySatHardEncoding, is already defined in topology_solver.hpp.
};

// Internal SAT function declarations used by private code and unit tests.
// These are implemented in topology_solver_sat.cpp.

bool topology_sat_encode_hard_constraints(
    TopologySatSolver& solver,
    const TopologySatGraphView& graph_data,
    const TopologySatConstraintView& constraint_data,
    TopologySatHardEncoding& enc,
    ConnectionValidationMode validation_mode = ConnectionValidationMode::RELAXED);

bool topology_sat_decode_hard_solution(
    TopologySatSolver& solver, const TopologySatHardEncoding& enc, std::vector<int>& mapping_out);

bool topology_sat_add_blocking_clause_for_mapping(
    TopologySatSolver& solver, TopologySatHardEncoding& enc, const std::vector<int>& raw_mapping, bool unique_shapes);

// Template overload: converts GraphIndexData/ConstraintIndexData to views and delegates.
template <typename TargetNode, typename GlobalNode>
bool topology_sat_encode_hard_constraints(
    TopologySatSolver& solver,
    const GraphIndexData<TargetNode, GlobalNode>& graph_data,
    const ConstraintIndexData<TargetNode, GlobalNode>& constraint_data,
    TopologySatHardEncoding& enc,
    ConnectionValidationMode validation_mode = ConnectionValidationMode::RELAXED) {
    return topology_sat_encode_hard_constraints(
        solver,
        TopologySatGraphView(graph_data),
        TopologySatConstraintView(constraint_data),
        enc,
        validation_mode);
}

}  // namespace tt::tt_fabric::detail
