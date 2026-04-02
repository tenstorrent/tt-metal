// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// This header provides "Node" terminology aliases for the "Core" coordinate types.
//
// In the tt-metal codebase, "core" has historically been overloaded to mean both:
//   1. A RISC core within a node (e.g., 5 per worker on Wormhole, 12 per cluster on Quasar)
//   2. A NOC endpoint / block / node in the accelerator grid
//
// This overload becomes particularly confusing when discussing Quasar and Gen2 architectures.

// We are introducing a new term: "Node".
//   - "Core" has meaning #1
//   - "Node" has meaning #2
//
// A "node" is a NOC endpoint with an x,y address - a block in the accelerator grid.
// This header provides type aliases for the new "Node" terminology.

#include <tt-metalium/core_coord.hpp>

namespace tt::tt_metal::experimental::metal2_host_api {

// Type aliases: Node terminology for NOC endpoint coordinates
using NodeCoord = tt::tt_metal::CoreCoord;
using NodeRange = tt::tt_metal::CoreRange;
using NodeRangeSet = tt::tt_metal::CoreRangeSet;

// Function aliases: Node terminology equivalents

inline std::vector<NodeCoord> grid_to_nodes(
    uint32_t num_nodes, uint32_t grid_size_x, uint32_t grid_size_y, bool row_wise = false) {
    return tt::tt_metal::grid_to_cores(num_nodes, grid_size_x, grid_size_y, row_wise);
}

inline std::vector<NodeCoord> grid_to_nodes(NodeCoord start, NodeCoord end, bool row_wise = false) {
    return tt::tt_metal::grid_to_cores(start, end, row_wise);
}

inline std::vector<NodeCoord> grid_to_nodes_with_noop(
    uint32_t bbox_x, uint32_t bbox_y, uint32_t grid_size_x, uint32_t grid_size_y, bool row_wise = false) {
    return tt::tt_metal::grid_to_cores_with_noop(bbox_x, bbox_y, grid_size_x, grid_size_y, row_wise);
}

inline std::vector<NodeCoord> grid_to_nodes_with_noop(
    const NodeRangeSet& used_nodes, const NodeRangeSet& all_nodes, bool row_wise = false) {
    return tt::tt_metal::grid_to_cores_with_noop(used_nodes, all_nodes, row_wise);
}

inline std::vector<NodeCoord> noderange_to_nodes(
    const NodeRangeSet& nrs, std::optional<uint32_t> max_nodes = std::nullopt, bool row_wise = false) {
    return tt::tt_metal::corerange_to_cores(nrs, max_nodes, row_wise);
}

inline NodeRangeSet select_from_noderangeset(
    const NodeRangeSet& nrs, uint32_t start_index, uint32_t end_index, bool row_wise = false) {
    return tt::tt_metal::select_from_corerangeset(nrs, start_index, end_index, row_wise);
}

inline std::optional<NodeRange> select_contiguous_range_from_noderangeset(
    const NodeRangeSet& nrs, uint32_t x, uint32_t y) {
    return tt::tt_metal::select_contiguous_range_from_corerangeset(nrs, x, y);
}

}  // namespace tt::tt_metal::experimental::metal2_host_api
