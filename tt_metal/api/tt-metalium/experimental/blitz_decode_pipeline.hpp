// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>

namespace tt::tt_metal::experimental::blitz {

struct BlitzDecodePipelineStage {
    std::size_t stage_index;
    ::tt::tt_metal::distributed::MeshCoordinate entry_node_coord;
    ::tt::tt_metal::distributed::MeshCoordinate exit_node_coord;
};

std::vector<BlitzDecodePipelineStage> generate_blitz_decode_pipeline(bool initialize_loopback = true);

namespace detail {

// Choose one (exit, peer) FabricNodeId pair per hop such that no node is reused across hops.
// `candidates[i]` are the candidate pairs for ring-position hop i; returns the chosen pairs in hop
// order, or std::nullopt if no collision-free assignment exists (any hop empty, or overconstrained).
//
// Exposed for unit testing (pure, CPU-only, no control plane). build_pipeline_from_topology() uses
// this to lay out the inter-mesh ring: per-hop greedy first-fit can strand a mid-chain hop on tight
// rings (few cable pairs per boundary), so this does a backtracking global assignment (system of
// distinct representatives, most-constrained-hop-first) that succeeds whenever a valid layout exists.
std::optional<std::vector<std::pair<::tt::tt_fabric::FabricNodeId, ::tt::tt_fabric::FabricNodeId>>>
assign_non_colliding_hops(
    const std::vector<std::vector<std::pair<::tt::tt_fabric::FabricNodeId, ::tt::tt_fabric::FabricNodeId>>>&
        candidates);

}  // namespace detail

}  // namespace tt::tt_metal::experimental::blitz
