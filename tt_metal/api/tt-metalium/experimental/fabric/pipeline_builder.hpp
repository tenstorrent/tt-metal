// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
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
///                      Non-loopback edges define the DAG. Loopback edges do not
///                      affect stage ordering, but still require physical links
///                      and endpoint capacity between distinct stages.
/// @param submesh_chips For each submesh: list of (mesh_id, chip_id, row, col) chips.
///                      Index in the outer vector is the submesh index.
/// @param node_chip_counts Optional per-node expected chip count (rows*cols of the
///                      node's declared shape).  When supplied for a node, that node
///                      may only be assigned to a submesh with exactly that many chips,
///                      so a stage declared 4x2 cannot be placed on a 1x2 submesh of a
///                      different mesh just because ethernet connectivity allows it.
///                      Nodes absent from the map are unconstrained.  An empty map
///                      disables the shape filter entirely.
/// @param node_pipeline_core_counts Optional per-node pipeline endpoint capacity on
///                      each chip. Overrides pipeline_core_count for listed nodes.
///                      Other nodes use the uniform capacity if provided, otherwise
///                      1 slot per chip on submeshes with at least 8 chips, or 2 on
///                      smaller submeshes. Defaults follow the actual chosen submesh,
///                      even if the node has no declared chip-count requirement.
///                      The resolver jointly selects submeshes, links, endpoint chips,
///                      and smallest-free abstract core slots.
/// @param pipeline_core_count Optional uniform per-chip capacity for all nodes.
///                      Entries in node_pipeline_core_counts override this default.
///                      This describes available slots, not mandatory utilization.
/// @returns             GraphLayoutResult with physical coords for every edge and
///                      H2D/D2H chip coords in stage-0's submesh, including core slots.
GraphLayoutResult resolve_graph_layout(
    const std::vector<std::string>& nodes,
    const std::vector<EdgeInputTuple>& edges,
    const std::vector<std::vector<ChipTuple>>& submesh_chips,
    const std::map<std::string, uint32_t>& node_chip_counts = {},
    const std::map<std::string, uint32_t>& node_pipeline_core_counts = {},
    std::optional<uint32_t> pipeline_core_count = std::nullopt);

// Implementation details shared by the resolver and CPU placement tests.
namespace detail {

struct InternalChip {
    FabricNodeId fid;
    uint32_t row, col;
};

// Keep every direct Ethernet link so placement can choose capacity-compatible
// endpoints rather than committing to the first discovered link.
struct LinkPair {
    uint32_t exit_row, exit_col;
    uint32_t entry_row, entry_col;
    bool operator==(const LinkPair&) const = default;
};

using ConnectionKey = std::pair<size_t, size_t>;  // (submesh_i, submesh_j)
using DirectLinks = std::map<ConnectionKey, std::vector<LinkPair>>;

// CPU tests can supply discovered links; nullptr uses the control plane.
GraphLayoutResult resolve_graph_layout_with_connections(
    const std::vector<std::string>& nodes,
    const std::vector<EdgeInputTuple>& edges,
    const std::vector<std::vector<ChipTuple>>& chips,
    const std::map<std::string, uint32_t>& stage_chip_counts,
    const std::map<std::string, uint32_t>& stage_pipeline_core_counts,
    std::optional<uint32_t> default_capacity,
    const DirectLinks* links);

// Exact placement and endpoint selection, independent of the control plane.
GraphLayoutResult resolve_pipeline_placement(
    const std::vector<std::string>& stage_order,
    const std::vector<EdgeInputTuple>& edges,
    const DirectLinks& submesh_links,
    const std::map<std::string, uint32_t>& stage_chip_counts,
    const std::vector<std::vector<InternalChip>>& chips,
    const std::map<std::string, uint32_t>& capacity_overrides);

}  // namespace detail

}  // namespace tt::tt_fabric
