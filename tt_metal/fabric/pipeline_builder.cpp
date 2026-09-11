// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/fabric/pipeline_builder.hpp>

#include <algorithm>
#include <cstdint>
#include <map>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "tt_metal/impl/context/metal_context.hpp"
#include "tt-metalium/experimental/fabric/control_plane.hpp"

namespace tt::tt_fabric {

// ------------------------------------------------------------------
// Low-level control-plane wrappers
// ------------------------------------------------------------------

std::optional<RoutingDirection> pipeline_get_forwarding_direction(FabricNodeId src, FabricNodeId dst) {
    const auto& cp = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto dir_opt = cp.get_forwarding_direction(src, dst);
    if (!dir_opt) {
        return std::nullopt;
    }
    auto planes = cp.get_active_fabric_eth_routing_planes_in_direction(src, *dir_opt);
    if (planes.empty()) {
        return std::nullopt;
    }
    return dir_opt;
}

std::map<uint32_t, std::vector<uint32_t>> pipeline_get_chip_neighbors(FabricNodeId src, RoutingDirection direction) {
    const auto& cp = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto raw = cp.get_chip_neighbors(src, direction);
    std::map<uint32_t, std::vector<uint32_t>> result;
    for (const auto& [mesh_id, chip_ids] : raw) {
        auto& vec = result[*mesh_id];
        vec.insert(vec.end(), chip_ids.begin(), chip_ids.end());
    }
    return result;
}

// ------------------------------------------------------------------
// Graph layout resolution
// ------------------------------------------------------------------

namespace {

using detail::InternalChip;

/// Discover all direct ethernet links between every ordered pair of submeshes.
/// All valid link pairs are collected (not just the first) to enable deconfliction.
detail::DirectLinks discover_submesh_links(const std::vector<std::vector<InternalChip>>& chips) {
    detail::DirectLinks submesh_links;
    using NeighborKey = std::tuple<uint32_t, uint32_t, RoutingDirection>;
    std::map<NeighborKey, std::map<uint32_t, std::vector<uint32_t>>> neighbor_cache;
    size_t n = chips.size();
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (i == j) {
                continue;
            }
            for (const auto& ca : chips[i]) {
                for (const auto& cb : chips[j]) {
                    auto dir_opt = pipeline_get_forwarding_direction(ca.fid, cb.fid);
                    if (!dir_opt) {
                        continue;
                    }
                    // Many destination chips share this source/direction.
                    auto [cached, inserted] = neighbor_cache.try_emplace(
                        NeighborKey{*ca.fid.mesh_id, ca.fid.chip_id, *dir_opt});
                    if (inserted) {
                        cached->second = pipeline_get_chip_neighbors(ca.fid, *dir_opt);
                    }
                    const auto& neighbors = cached->second;
                    uint32_t b_mesh = *cb.fid.mesh_id;
                    auto it = neighbors.find(b_mesh);
                    if (it == neighbors.end()) {
                        continue;
                    }
                    const auto& nlist = it->second;
                    if (std::find(nlist.begin(), nlist.end(), cb.fid.chip_id) != nlist.end()) {
                        submesh_links[{i, j}].push_back({ca.row, ca.col, cb.row, cb.col});
                    }
                }
            }
        }
    }
    return submesh_links;
}

/// Kahn's topological sort on non-loopback edges. Returns stage names in pipeline order.
std::vector<std::string> topological_sort(
    const std::vector<std::string>& all_stage_names, const std::vector<EdgeInputTuple>& edges) {
    std::map<std::string, int> in_degree;
    std::map<std::string, std::vector<std::string>> downstream_stages;
    for (const auto& stage_name : all_stage_names) {
        in_degree[stage_name] = 0;
    }
    for (const auto& [src_stage, dst_stage, is_loopback] : edges) {
        if (!is_loopback) {
            downstream_stages[src_stage].push_back(dst_stage);
            in_degree[dst_stage]++;
        }
    }
    std::queue<std::string> q;
    for (const auto& [stage_name, deg] : in_degree) {
        if (deg == 0) {
            q.push(stage_name);
        }
    }

    std::vector<std::string> order;
    order.reserve(all_stage_names.size());
    while (!q.empty()) {
        auto stage_name = q.front();
        q.pop();
        order.push_back(stage_name);
        for (const auto& downstream_stage : downstream_stages[stage_name]) {
            if (--in_degree[downstream_stage] == 0) {
                q.push(downstream_stage);
            }
        }
    }
    if (order.size() != all_stage_names.size()) {
        throw std::runtime_error("resolve_graph_layout: cycle detected in non-loopback edges");
    }
    return order;
}

}  // anonymous namespace

GraphLayoutResult resolve_graph_layout(
    const std::vector<std::string>& nodes,
    const std::vector<EdgeInputTuple>& edges,
    const std::vector<std::vector<ChipTuple>>& submesh_chips,
    const std::map<std::string, uint32_t>& node_chip_counts,
    const std::map<std::string, uint32_t>& node_pipeline_core_counts,
    std::optional<uint32_t> pipeline_core_count) {
    return detail::resolve_graph_layout_with_connections(
        nodes, edges, submesh_chips, node_chip_counts, node_pipeline_core_counts, pipeline_core_count, nullptr);
}

GraphLayoutResult detail::resolve_graph_layout_with_connections(
    const std::vector<std::string>& nodes,
    const std::vector<EdgeInputTuple>& edges,
    const std::vector<std::vector<ChipTuple>>& submesh_chips,
    const std::map<std::string, uint32_t>& node_chip_counts,
    const std::map<std::string, uint32_t>& node_pipeline_core_counts,
    std::optional<uint32_t> pipeline_core_count,
    const DirectLinks* direct_links) {
    // The public API retains node-oriented names for Blaze compatibility. Inside
    // the resolver, graph nodes represent pipeline stages.
    const auto& stage_chip_counts = node_chip_counts;
    auto stage_pipeline_core_counts = node_pipeline_core_counts;
    if (pipeline_core_count) {
        if (*pipeline_core_count == 0) {
            throw std::runtime_error("resolve_graph_layout: zero default pipeline-core capacity");
        }
        for (const auto& node : nodes) {
            stage_pipeline_core_counts.try_emplace(node, *pipeline_core_count);
        }
    }
    // Validate all graph inputs before querying the control plane.
    //
    // The explicit node list includes isolated stages and must contain every edge endpoint.
    if (nodes.empty()) {
        throw std::runtime_error("resolve_graph_layout: nodes must not be empty");
    }
    const std::set<std::string> stage_names(nodes.begin(), nodes.end());
    if (stage_names.size() != nodes.size()) {
        throw std::runtime_error("resolve_graph_layout: duplicate stage names");
    }
    for (const auto& [src_stage, dst_stage, is_loopback] : edges) {
        if (!stage_names.contains(src_stage)) {
            throw std::runtime_error(
                "resolve_graph_layout: stage " + src_stage + " not found in the explicit nodes list");
        }
        if (!stage_names.contains(dst_stage)) {
            throw std::runtime_error(
                "resolve_graph_layout: stage " + dst_stage + " not found in the explicit nodes list");
        }
    }

    // Topological sort of non-loopback edges
    auto stage_order = topological_sort(nodes, edges);

    // Validate per-stage constraints.
    for (const auto& [stage, chip_count] : stage_chip_counts) {
        if (!stage_names.contains(stage)) {
            throw std::runtime_error("resolve_graph_layout: chip-count override for unknown stage '" + stage + "'");
        }
    }
    for (const auto& [stage, capacity] : stage_pipeline_core_counts) {
        if (!stage_names.contains(stage)) {
            throw std::runtime_error("resolve_graph_layout: capacity override for unknown stage '" + stage + "'");
        }
        if (capacity == 0) {
            throw std::runtime_error("resolve_graph_layout: stage '" + stage + "' declares zero pipeline-core capacity");
        }
    }

    // Convert chip tuples to internal representation
    size_t num_submeshes = submesh_chips.size();
    if (nodes.size() > num_submeshes) {
        throw std::runtime_error("resolve_graph_layout: no valid submesh assignment found; fewer submeshes than stages");
    }
    std::vector<std::vector<InternalChip>> chips(num_submeshes);
    for (size_t i = 0; i < num_submeshes; ++i) {
        for (const auto& [mesh_id, chip_id, row, col] : submesh_chips[i]) {
            chips[i].push_back({FabricNodeId{MeshId{mesh_id}, chip_id}, row, col});
        }
    }

    // Discover physical connections between all submesh pairs
    const auto submesh_links = direct_links ? *direct_links : discover_submesh_links(chips);

    GraphLayoutResult result = detail::resolve_pipeline_placement(
        stage_order, edges, submesh_links, stage_chip_counts, chips, stage_pipeline_core_counts);
    result.stage_order = stage_order;
    return result;
}

}  // namespace tt::tt_fabric
