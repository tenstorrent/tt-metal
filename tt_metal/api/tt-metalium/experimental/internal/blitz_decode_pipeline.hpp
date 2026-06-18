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

namespace tt::tt_metal::internal::blitz {

// Legacy API: one entry per real stage, plus a synthetic loopback stage when
// initialize_loopback=true.
struct BlitzDecodePipelineStage {
    std::size_t stage_index;
    ::tt::tt_metal::distributed::MeshCoordinate entry_node_coord;
    ::tt::tt_metal::distributed::MeshCoordinate exit_node_coord;
};

// One host contribution to a resolved logical stage.
struct BlitzDecodeStageHostBinding {
    int rank;
    std::size_t mesh_host_rank;
};

// Exact chip placement for a resolved pipeline endpoint.
// mesh_coord is expressed in canonical stage-mesh coordinates.
struct BlitzDecodeEndpointPlacement {
    BlitzDecodeStageHostBinding host_binding;
    ::tt::tt_metal::distributed::MeshCoordinate mesh_coord;
};

// Resolved allocation for one logical pipeline stage.
struct ResolvedBlitzDecodeStageAllocation {
    std::size_t logical_stage_index;
    std::size_t mesh_id;
    std::vector<BlitzDecodeStageHostBinding> host_bindings;
    BlitzDecodeEndpointPlacement entry_endpoint;
    std::optional<BlitzDecodeEndpointPlacement> exit_endpoint;
};

// Resolved allocation for the whole pipeline.
//
// `stages` is ordered by logical_stage_index.
//
// `loopback_entry_stage_index` and `loopback_entry_endpoint` are populated only
// when initialize_loopback=true.
//
// `host_egress_stage_index` and `host_egress_endpoint` always identify the D2H
// endpoint:
//   - stage 0 for fabric loopback
//   - the last stage otherwise
struct ResolvedBlitzDecodePipelineAllocation {
    bool initialize_loopback;
    std::vector<ResolvedBlitzDecodeStageAllocation> stages;
    std::optional<std::size_t> loopback_entry_stage_index;
    std::optional<BlitzDecodeEndpointPlacement> loopback_entry_endpoint;
    std::size_t host_egress_stage_index;
    BlitzDecodeEndpointPlacement host_egress_endpoint;
};

// Legacy API: unchanged
std::vector<BlitzDecodePipelineStage> generate_blitz_decode_pipeline(bool initialize_loopback = true);

// New API: explicit resolved stage-to-host allocation and endpoint ownership.
ResolvedBlitzDecodePipelineAllocation resolve_blitz_decode_pipeline_allocation(bool initialize_loopback = true);

namespace detail {

// Choose one (exit, peer) FabricNodeId pair per hop such that no node is reused across hops.
// `candidates[i]` are the candidate pairs for ring-position hop i; returns the chosen pairs in hop
// order, or std::nullopt if no collision-free assignment exists (any hop empty, or overconstrained).
//
// Exposed for unit testing (pure, CPU-only, no control plane). build_pipeline_allocation_from_topology()
// uses this to lay out the inter-mesh ring: per-hop greedy first-fit can strand a mid-chain hop on tight
// rings (few cable pairs per boundary), so this does a backtracking global assignment (system of distinct
// representatives, most-constrained-hop-first) that succeeds whenever a valid layout exists.
std::optional<std::vector<std::pair<::tt::tt_fabric::FabricNodeId, ::tt::tt_fabric::FabricNodeId>>>
assign_non_colliding_hops(
    const std::vector<std::vector<std::pair<::tt::tt_fabric::FabricNodeId, ::tt::tt_fabric::FabricNodeId>>>&
        candidates);

}  // namespace detail

}  // namespace tt::tt_metal::internal::blitz
