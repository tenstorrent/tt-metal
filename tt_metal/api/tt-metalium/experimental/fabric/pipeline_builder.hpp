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
    std::optional<uint32_t> exit_core_slot;
    std::optional<uint32_t> entry_core_slot;
};

/// Result returned to Python after topology-based graph layout resolution.
struct GraphLayoutResult {
    /// Node names in topological pipeline stage order (index == stage_idx).
    std::vector<std::string> stage_order;

    /// Maps each node name to the submesh index (index into the submesh_chips list).
    std::map<std::string, size_t> node_to_submesh;

    /// One entry per input edge (same order), filled with discovered physical coords.
    std::vector<ResolvedEdge> resolved_edges;

    /// Chip and abstract pipeline-core slot used for H2D in stage 0.
    uint32_t h2d_entry_row = 0;
    uint32_t h2d_entry_col = 0;
    std::optional<uint32_t> h2d_core_slot;

    /// Chip and abstract pipeline-core slot used for D2H in stage 0.
    uint32_t d2h_exit_row = 0;
    uint32_t d2h_exit_col = 0;
    std::optional<uint32_t> d2h_core_slot;
};

/// Auto-discover the physical layout of a pipeline graph.
///
/// @param nodes         All node names in the graph, in declaration order.  This is the
///                      authoritative node list: it lets the resolver handle graphs whose
///                      nodes are not all covered by edges — e.g. a single-stage pipeline
///                      with no edges at all.  Every endpoint referenced by @p edges must
///                      appear here; a missing endpoint raises std::runtime_error.
/// @param edges         Graph edges as (src_name, dst_name, is_loopback) tuples.
///                      Non-loopback edges define the DAG; the single loopback edge
///                      (if present) is the return path from the last stage to stage 0.
/// @param submesh_chips For each submesh: list of (mesh_id, chip_id, row, col) chips.
///                      Index in the outer vector is the submesh index.
/// @param node_chip_counts Optional per-node expected chip count (rows*cols of the
///                      node's declared shape).  When supplied for a node, that node
///                      may only be assigned to a submesh with exactly that many chips,
///                      so a stage declared 4x2 cannot be placed on a 1x2 submesh of a
///                      different mesh just because ethernet connectivity allows it.
///                      Nodes absent from the map are unconstrained.  An empty map
///                      disables the shape filter entirely (legacy behavior).
/// @param node_pipeline_core_counts Optional per-node pipeline endpoint capacity on
///                      each chip. When non-empty, every node must be present and the
///                      resolver jointly selects submeshes, links, endpoint chips, and
///                      smallest-free abstract core slots. An empty map preserves the
///                      legacy link-selection and host-placement behavior.
/// @returns             GraphLayoutResult with physical coords for every edge and
///                      H2D/D2H chip coords in stage-0's submesh. Core slots are set only
///                      when node_pipeline_core_counts is supplied.
GraphLayoutResult resolve_graph_layout(
    const std::vector<std::string>& nodes,
    const std::vector<EdgeInputTuple>& edges,
    const std::vector<std::vector<ChipTuple>>& submesh_chips,
    const std::map<std::string, uint32_t>& node_chip_counts = {},
    const std::map<std::string, uint32_t>& node_pipeline_core_counts = {});

}  // namespace tt::tt_fabric
