// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>

namespace tt::tt_fabric {

// ------------------------------------------------------------------
// Low-level control-plane wrappers (exposed individually in nanobind)
// ------------------------------------------------------------------

/// Return the direction in which data should be forwarded from *src* to reach *dst*.
/// Returns std::nullopt if *dst* is not reachable from *src*.
std::optional<RoutingDirection> pipeline_get_forwarding_direction(FabricNodeId src, FabricNodeId dst);

/// Return the chips directly connected to *src* via an ethernet cable in *direction*.
/// Result maps mesh_id (uint32_t) -> list of chip_ids (uint32_t).
std::map<uint32_t, std::vector<uint32_t>> pipeline_get_chip_neighbors(FabricNodeId src, RoutingDirection direction);

// ------------------------------------------------------------------
// Graph layout resolution
// ------------------------------------------------------------------

/// Per-chip info supplied by the Python side: (mesh_id, chip_id, row, col).
using ChipTuple = std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>;

/// Input edge: (src_name, dst_name, is_loopback).
using EdgeInputTuple = std::tuple<std::string, std::string, bool>;

/// Physical coordinates discovered for one directed edge.
struct ResolvedEdge {
    std::string src;
    std::string dst;
    bool is_loopback = false;
    uint32_t exit_row = 0;  ///< chip in src's submesh that sends toward dst
    uint32_t exit_col = 0;
    uint32_t entry_row = 0;  ///< chip in dst's submesh that receives from src
    uint32_t entry_col = 0;
};

/// Result returned to Python after topology-based graph layout resolution.
struct GraphLayoutResult {
    /// Node names in topological pipeline stage order (index == stage_idx).
    std::vector<std::string> stage_order;

    /// Maps each node name to the submesh index (index into the submesh_chips list).
    std::map<std::string, size_t> node_to_submesh;

    /// One entry per input edge (same order), filled with discovered physical coords.
    std::vector<ResolvedEdge> resolved_edges;

    /// Unclaimed chip in stage-0's submesh used as the H2D entry.
    uint32_t h2d_entry_row = 0;
    uint32_t h2d_entry_col = 0;

    /// Unclaimed chip in stage-0's submesh used as the D2H exit.
    uint32_t d2h_exit_row = 0;
    uint32_t d2h_exit_col = 0;
};

/// Auto-discover the physical layout of a pipeline graph.
///
/// @param edges         Graph edges as (src_name, dst_name, is_loopback) tuples.
///                      Non-loopback edges define the DAG; the single loopback edge
///                      (if present) is the return path from the last stage to stage 0.
/// @param submesh_chips For each submesh: list of (mesh_id, chip_id, row, col) chips.
///                      Index in the outer vector is the submesh index.
/// @returns             GraphLayoutResult with physical coords for every edge and
///                      unclaimed H2D/D2H chip coords in stage-0's submesh.
GraphLayoutResult resolve_graph_layout(
    const std::vector<EdgeInputTuple>& edges, const std::vector<std::vector<ChipTuple>>& submesh_chips);

}  // namespace tt::tt_fabric
