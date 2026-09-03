// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/fabric/pipeline_builder.hpp>

#include <algorithm>
#include <functional>
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

struct InternalChip {
    FabricNodeId fid;
    uint32_t row, col;
};

// Physical direct-link info between a pair of submeshes.
// All valid ethernet link pairs are collected so that deconfliction can
// pick an alternative when the first-found pair causes entry == exit on
// the same chip for a forwarding stage.
struct ConnectionInfo {
    struct LinkPair {
        uint32_t exit_row, exit_col;    // chip in submesh i that sends toward j
        uint32_t entry_row, entry_col;  // chip in submesh j that receives from i
    };
    std::vector<LinkPair> links;  // all valid direct ethernet links, first = primary

    // Convenience: whether any link exists (replaces old has-value check).
    bool empty() const { return links.empty(); }

    // Primary link coords (backward-compatible accessors).
    uint32_t exit_row() const { return links[0].exit_row; }
    uint32_t exit_col() const { return links[0].exit_col; }
    uint32_t entry_row() const { return links[0].entry_row; }
    uint32_t entry_col() const { return links[0].entry_col; }
};

using ConnectionKey = std::pair<size_t, size_t>;  // (submesh_i, submesh_j)

/// Discover all direct ethernet links between every ordered pair of submeshes.
/// All valid link pairs are collected (not just the first) to enable deconfliction.
std::map<ConnectionKey, ConnectionInfo> discover_connections(const std::vector<std::vector<InternalChip>>& chips) {
    std::map<ConnectionKey, ConnectionInfo> connections;
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
                    auto neighbors = pipeline_get_chip_neighbors(ca.fid, *dir_opt);
                    uint32_t b_mesh = *cb.fid.mesh_id;
                    auto it = neighbors.find(b_mesh);
                    if (it == neighbors.end()) {
                        continue;
                    }
                    const auto& nlist = it->second;
                    if (std::find(nlist.begin(), nlist.end(), cb.fid.chip_id) != nlist.end()) {
                        connections[{i, j}].links.push_back({ca.row, ca.col, cb.row, cb.col});
                    }
                }
            }
        }
    }
    return connections;
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

/// Backtracking search: assign a submesh index to each stage in topological order.
/// Returns {stage_name -> submesh_index} or throws if no valid assignment exists.
std::map<std::string, size_t> assign_stages_to_submeshes(
    const std::vector<std::string>& stage_order,
    const std::vector<EdgeInputTuple>& edges,
    const std::map<ConnectionKey, ConnectionInfo>& connections,
    size_t num_submeshes,
    const std::map<std::string, uint32_t>& stage_chip_counts,
    const std::vector<std::vector<InternalChip>>& chips,
    const std::function<bool(const std::map<std::string, size_t>&)>& complete_assignment_check = {}) {
    // Build reverse-lookup: destination stage -> upstream stages for non-loopback edges.
    std::map<std::string, std::vector<std::string>> upstream_stages;
    for (const auto& [src_stage, dst_stage, is_loopback] : edges) {
        if (!is_loopback) {
            upstream_stages[dst_stage].push_back(src_stage);
        }
    }

    std::map<std::string, size_t> stage_to_submesh;
    std::set<size_t> used;

    std::function<bool(size_t)> solve = [&](size_t idx) -> bool {
        if (idx == stage_order.size()) {
            // Verify every loopback edge has a direct physical link.
            for (const auto& [src_stage, dst_stage, is_loopback] : edges) {
                if (!is_loopback) {
                    continue;
                }
                // A self-loop is satisfied trivially: a single-stage graph's return path
                // never leaves its submesh, so there is no inter-submesh link to require.
                // discover_connections() skips i == j, so demanding one here would make
                // every single-stage graph unassignable.
                if (src_stage == dst_stage) {
                    continue;
                }
                size_t src_submesh = stage_to_submesh.at(src_stage);
                size_t dst_submesh = stage_to_submesh.at(dst_stage);
                if (!connections.contains({src_submesh, dst_submesh})) {
                    return false;
                }
            }
            return !complete_assignment_check || complete_assignment_check(stage_to_submesh);
        }
        const auto& stage_name = stage_order[idx];

        // Start with every unused submesh. Each upstream stage removes the
        // candidates it cannot reach directly; source stages keep the full set.
        std::set<size_t> candidates;
        for (size_t submesh = 0; submesh < num_submeshes; ++submesh) {
            if (!used.contains(submesh)) {
                candidates.insert(submesh);
            }
        }
        if (auto upstream_it = upstream_stages.find(stage_name); upstream_it != upstream_stages.end()) {
            for (const auto& upstream_stage : upstream_it->second) {
                size_t upstream_submesh = stage_to_submesh.at(upstream_stage);
                std::erase_if(
                    candidates, [&](size_t submesh) { return !connections.contains({upstream_submesh, submesh}); });
            }
        }

        // Shape constraint: a stage may only land on a submesh whose chip count
        // matches the stage's declared shape (rows*cols). Without this, a stage
        // declared 4x2 can be placed on a 1x2 submesh of a different mesh whenever
        // ethernet connectivity allows, silently mis-placing it (e.g. the final-
        // layer / loopback stage landing on a 1x2 instead of its 4x2 mesh).
        auto chip_count_it = stage_chip_counts.find(stage_name);
        if (chip_count_it != stage_chip_counts.end()) {
            std::erase_if(candidates, [&](size_t submesh) { return chips[submesh].size() != chip_count_it->second; });
        }

        for (size_t submesh : candidates) {
            stage_to_submesh[stage_name] = submesh;
            used.insert(submesh);
            if (solve(idx + 1)) {
                return true;
            }
            stage_to_submesh.erase(stage_name);
            used.erase(submesh);
        }
        return false;
    };

    if (!solve(0)) {
        throw std::runtime_error(
            "resolve_graph_layout: no valid submesh assignment found — "
            "physical connectivity (and per-stage shape, if constrained) does not "
            "match the graph topology");
    }
    return stage_to_submesh;
}

using EndpointKey = std::tuple<std::string, uint32_t, uint32_t>;

struct CapacityFailure {
    std::string stage;
    uint32_t row = 0;
    uint32_t col = 0;
    uint32_t required = 0;
    uint32_t capacity = 0;
    std::vector<std::string> roles;

    std::string describe() const {
        std::string result = "stage '" + stage + "', chip (" + std::to_string(row) + "," + std::to_string(col) +
                             "), required slots " + std::to_string(required) + ", declared capacity " +
                             std::to_string(capacity) + ", endpoint roles [";
        for (size_t i = 0; i < roles.size(); ++i) {
            result += (i == 0 ? "" : ", ") + roles[i];
        }
        return result + "]";
    }
};

/// Select links and endpoint chips jointly for one stage-to-submesh assignment.
/// Edges are processed in declaration order, followed by stage-0 H2D and D2H.
/// A chip's current role count is both its smallest free slot and capacity demand.
std::optional<GraphLayoutResult> try_resolve_capacity_layout(
    const std::vector<EdgeInputTuple>& edges,
    const std::map<std::string, size_t>& stage_to_submesh,
    const std::map<ConnectionKey, ConnectionInfo>& connections,
    const std::vector<std::vector<InternalChip>>& chips,
    const std::map<std::string, uint32_t>& capacities,
    const std::string& stage0,
    CapacityFailure& best_failure) {
    GraphLayoutResult result;
    result.resolved_edges.resize(edges.size());
    std::map<EndpointKey, std::vector<std::string>> roles_by_chip;

    auto try_reserve_endpoint_slot = [&](const std::string& stage_name,
                                         uint32_t row,
                                         uint32_t col,
                                         const std::string& role) -> std::optional<uint32_t> {
        auto& roles = roles_by_chip[{stage_name, row, col}];
        const uint32_t capacity = capacities.at(stage_name);
        if (roles.size() >= capacity) {
            CapacityFailure failure{stage_name, row, col, static_cast<uint32_t>(roles.size() + 1), capacity, roles};
            failure.roles.push_back(role);
            if (failure.required >= best_failure.required) {
                best_failure = std::move(failure);
            }
            return std::nullopt;
        }
        const uint32_t slot = static_cast<uint32_t>(roles.size());
        roles.push_back(role);
        return slot;
    };
    auto release_endpoint_slot = [&](const std::string& stage_name, uint32_t row, uint32_t col) {
        auto it = roles_by_chip.find({stage_name, row, col});
        it->second.pop_back();
        if (it->second.empty()) {
            roles_by_chip.erase(it);
        }
    };
    std::function<bool(size_t)> place_edges;
    auto try_place_host_endpoint =
        [&](const std::string& role, uint32_t& row, uint32_t& col, std::optional<uint32_t>& core_slot) -> bool {
        const auto& stage_chips = chips.at(stage_to_submesh.at(stage0));
        // Prefer an unused chip, preserving supplied chip order, then permit folding.
        for (bool already_used : {false, true}) {
            for (const auto& chip : stage_chips) {
                if (roles_by_chip.contains({stage0, chip.row, chip.col}) != already_used) {
                    continue;
                }
                if (auto slot = try_reserve_endpoint_slot(stage0, chip.row, chip.col, role)) {
                    row = chip.row;
                    col = chip.col;
                    core_slot = *slot;
                    return true;
                }
            }
        }
        return false;
    };
    auto place_host_endpoints = [&]() -> bool {
        if (!try_place_host_endpoint("h2d", result.h2d_entry_row, result.h2d_entry_col, result.h2d_core_slot)) {
            return false;
        }
        if (try_place_host_endpoint("d2h", result.d2h_exit_row, result.d2h_exit_col, result.d2h_core_slot)) {
            return true;
        }
        release_endpoint_slot(stage0, result.h2d_entry_row, result.h2d_entry_col);
        return false;
    };

    place_edges = [&](size_t edge_idx) -> bool {
        if (edge_idx == edges.size()) {
            return place_host_endpoints();
        }
        const auto& [src_stage, dst_stage, is_loopback] = edges[edge_idx];
        // A self-loop does not cross a submesh boundary and therefore consumes no
        // fabric endpoint slots. Keep it consistent with legacy resolution, which
        // omits self-loops from resolved_edges.
        if (src_stage == dst_stage) {
            return place_edges(edge_idx + 1);
        }
        const auto& links = connections.at({stage_to_submesh.at(src_stage), stage_to_submesh.at(dst_stage)}).links;
        const std::string edge_role = "edge[" + std::to_string(edge_idx) + "] " + src_stage + "->" + dst_stage;
        for (const auto& link : links) {
            auto exit_slot = try_reserve_endpoint_slot(src_stage, link.exit_row, link.exit_col, edge_role + " exit");
            if (!exit_slot) {
                continue;
            }
            auto entry_slot =
                try_reserve_endpoint_slot(dst_stage, link.entry_row, link.entry_col, edge_role + " entry");
            if (!entry_slot) {
                release_endpoint_slot(src_stage, link.exit_row, link.exit_col);
                continue;
            }
            result.resolved_edges[edge_idx] = {
                src_stage,
                dst_stage,
                is_loopback,
                link.exit_row,
                link.exit_col,
                link.entry_row,
                link.entry_col,
                *exit_slot,
                *entry_slot};
            if (place_edges(edge_idx + 1)) {
                return true;
            }
            release_endpoint_slot(dst_stage, link.entry_row, link.entry_col);
            release_endpoint_slot(src_stage, link.exit_row, link.exit_col);
        }
        return false;
    };

    if (!place_edges(0)) {
        return std::nullopt;
    }
    for (size_t edge_idx = edges.size(); edge_idx-- > 0;) {
        if (std::get<0>(edges[edge_idx]) == std::get<1>(edges[edge_idx])) {
            result.resolved_edges.erase(result.resolved_edges.begin() + edge_idx);
        }
    }
    return result;
}

}  // anonymous namespace

GraphLayoutResult resolve_graph_layout(
    const std::vector<std::string>& nodes,
    const std::vector<EdgeInputTuple>& edges,
    const std::vector<std::vector<ChipTuple>>& submesh_chips,
    const std::map<std::string, uint32_t>& stage_chip_counts,
    const std::map<std::string, uint32_t>& stage_pipeline_core_counts) {
    // ------------------------------------------------------------------
    // 0. Convert chip tuples to internal representation
    // ------------------------------------------------------------------
    size_t num_submeshes = submesh_chips.size();
    std::vector<std::vector<InternalChip>> chips(num_submeshes);
    for (size_t i = 0; i < num_submeshes; ++i) {
        for (const auto& [mesh_id, chip_id, row, col] : submesh_chips[i]) {
            chips[i].push_back({FabricNodeId{MeshId{mesh_id}, chip_id}, row, col});
        }
    }

    // ------------------------------------------------------------------
    // 1. Discover physical connections between all submesh pairs
    // ------------------------------------------------------------------
    auto connections = discover_connections(chips);

    // ------------------------------------------------------------------
    // 2. Validate stage names.
    //
    // The explicit `nodes` list is authoritative, so graphs whose nodes are
    // not all covered by edges (e.g. a single-stage pipeline with no edges)
    // still register every node.  Every endpoint referenced by an edge must
    // appear in `nodes`; an edge referencing an unlisted node is an error.
    // ------------------------------------------------------------------
    const std::vector<std::string>& all_stage_names = nodes;
    {
        std::set<std::string> stage_names(nodes.begin(), nodes.end());
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
    }

    // ------------------------------------------------------------------
    // 3. Topological sort of non-loopback edges
    // ------------------------------------------------------------------
    auto stage_order = topological_sort(all_stage_names, edges);

    // ------------------------------------------------------------------
    // 4. Assign submeshes to stages via backtracking
    // ------------------------------------------------------------------
    if (!stage_pipeline_core_counts.empty()) {
        for (const auto& stage_name : stage_order) {
            auto capacity_it = stage_pipeline_core_counts.find(stage_name);
            if (capacity_it == stage_pipeline_core_counts.end()) {
                throw std::runtime_error(
                    "resolve_graph_layout: stage '" + stage_name + "' has no declared pipeline-core capacity");
            }
            if (capacity_it->second == 0) {
                throw std::runtime_error(
                    "resolve_graph_layout: stage '" + stage_name + "' declares zero pipeline-core capacity");
            }
        }

        std::optional<GraphLayoutResult> capacity_layout;
        CapacityFailure best_failure;
        auto try_capacity_layout_for_assignment = [&](const std::map<std::string, size_t>& candidate) {
            auto attempt = try_resolve_capacity_layout(
                edges, candidate, connections, chips, stage_pipeline_core_counts, stage_order.front(), best_failure);
            if (!attempt) {
                return false;
            }
            capacity_layout = std::move(*attempt);
            return true;
        };

        try {
            auto stage_to_submesh = assign_stages_to_submeshes(
                stage_order,
                edges,
                connections,
                num_submeshes,
                stage_chip_counts,
                chips,
                try_capacity_layout_for_assignment);
            GraphLayoutResult result = std::move(*capacity_layout);
            result.stage_order = std::move(stage_order);
            result.node_to_submesh = std::move(stage_to_submesh);
            return result;
        } catch (const std::runtime_error& error) {
            if (best_failure.required != 0) {
                throw std::runtime_error(
                    std::string(error.what()) + "; pipeline-core capacity exhausted at " + best_failure.describe());
            }
            throw;
        }
    }

    auto stage_to_submesh =
        assign_stages_to_submeshes(stage_order, edges, connections, num_submeshes, stage_chip_counts, chips);

    // ------------------------------------------------------------------
    // 5. Resolve physical coords for every edge
    // ------------------------------------------------------------------
    std::vector<ResolvedEdge> resolved_edges;
    resolved_edges.reserve(edges.size());
    for (const auto& [src_stage, dst_stage, is_loopback] : edges) {
        // Self-loop: no physical hop to resolve (see assign_submeshes). It is left out of
        // resolved_edges deliberately — a single-stage caller reads its entry/exit from
        // h2d_entry_* / d2h_exit_* below, not from a per-edge entry.
        if (src_stage == dst_stage) {
            continue;
        }
        size_t src_submesh = stage_to_submesh.at(src_stage);
        size_t dst_submesh = stage_to_submesh.at(dst_stage);
        auto it = connections.find({src_submesh, dst_submesh});
        if (it == connections.end()) {
            throw std::runtime_error(
                "resolve_graph_layout: no direct ethernet link between submesh " + std::to_string(src_submesh) + " (" +
                src_stage + ") and submesh " + std::to_string(dst_submesh) + " (" + dst_stage + ")");
        }
        const auto& c = it->second;
        resolved_edges.push_back(
            {src_stage, dst_stage, is_loopback, c.exit_row(), c.exit_col(), c.entry_row(), c.entry_col()});
    }

    // ------------------------------------------------------------------
    // 5.5. Deconflict same-chip entry/exit for forwarding stages.
    //
    // A forwarding stage i has both an entry chip (where data arrives from
    // stage i-1) and an exit chip (where data leaves to stage i+1).  If
    // the topology resolver assigned the same physical chip to both roles,
    // two persistent BRISC kernels would be dispatched to the same core,
    // causing the second generic_op to block forever.
    //
    // When this happens, scan the full list of valid ethernet links for the
    // exit edge and pick an alternative link whose exit chip differs from
    // the entry chip.  The corresponding entry chip on the next stage is
    // updated in the same step (they are a physically connected pair).
    // ------------------------------------------------------------------
    for (size_t i = 1; i < stage_order.size(); ++i) {
        size_t curr_sub = stage_to_submesh.at(stage_order[i]);

        // Find the resolved entry edge for this stage (non-loopback, dst == stage_order[i]).
        // Keep the edge itself: in a FORK graph the topological stage_order interleaves the
        // branches, so stage_order[i-1] is NOT this stage's predecessor — the entry edge's
        // own source is (that's what the connection lookup below must use).
        ResolvedEdge* entry_re = nullptr;
        for (auto& re : resolved_edges) {
            if (!re.is_loopback && re.dst == stage_order[i]) {
                entry_re = &re;
                break;
            }
        }
        if (entry_re == nullptr) {
            continue;  // stage 0 — no entry edge
        }
        uint32_t entry_row = entry_re->entry_row;
        uint32_t entry_col = entry_re->entry_col;

        // Find the resolved exit edge for this stage (src == stage_order[i], any kind).
        ResolvedEdge* exit_re = nullptr;
        for (auto& re : resolved_edges) {
            if (re.src == stage_order[i]) {
                exit_re = &re;
                break;
            }
        }
        if (!exit_re) {
            continue;  // no exit edge (shouldn't happen in a pipeline)
        }

        if (exit_re->exit_row == entry_row && exit_re->exit_col == entry_col) {
            // Conflict: find an alternative link for the exit edge.
            size_t next_sub = stage_to_submesh.at(exit_re->dst);
            const auto& exit_links = connections.at({curr_sub, next_sub}).links;
            bool resolved = false;
            for (const auto& lp : exit_links) {
                if (lp.exit_row != entry_row || lp.exit_col != entry_col) {
                    exit_re->exit_row = lp.exit_row;
                    exit_re->exit_col = lp.exit_col;
                    exit_re->entry_row = lp.entry_row;
                    exit_re->entry_col = lp.entry_col;
                    resolved = true;
                    break;
                }
            }
            if (!resolved) {
                // No alternative exit link — try changing the entry edge instead. Use the
                // entry edge's ACTUAL source submesh (not stage_order[i-1], which is the
                // wrong branch in an interleaved fork topological order).
                size_t prev_sub = stage_to_submesh.at(entry_re->src);
                const auto& entry_links = connections.at({prev_sub, curr_sub}).links;
                for (const auto& lp : entry_links) {
                    if (lp.entry_row != exit_re->exit_row || lp.entry_col != exit_re->exit_col) {
                        entry_re->exit_row = lp.exit_row;
                        entry_re->exit_col = lp.exit_col;
                        entry_re->entry_row = lp.entry_row;
                        entry_re->entry_col = lp.entry_col;
                        resolved = true;
                        break;
                    }
                }
                if (!resolved) {
                    throw std::runtime_error(
                        "resolve_graph_layout: stage " + std::to_string(i) + " (" + stage_order[i] +
                        ") has only one chip at both the entry and exit "
                        "boundary — cannot deconflict entry/exit on the same chip");
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // 6. Locate H2D and D2H coords in stage-0's submesh.
    //
    // Preferred: two chips in stage-0's submesh not used by any edge.
    // Fallback (e.g. small submeshes where all chips are edge-boundary):
    //   H2D = stage-0's forward-exit chip (the chip that sends to stage 1)
    //   D2H = stage-0's loopback-entry chip (the chip that receives the return)
    // ------------------------------------------------------------------
    size_t stage0_sub = stage_to_submesh.at(stage_order[0]);
    std::set<std::pair<uint32_t, uint32_t>> used_coords;  // (row, col)
    for (const auto& re : resolved_edges) {
        if (stage_to_submesh.at(re.src) == stage0_sub) {
            used_coords.insert({re.exit_row, re.exit_col});
        }
        if (stage_to_submesh.at(re.dst) == stage0_sub) {
            used_coords.insert({re.entry_row, re.entry_col});
        }
    }

    std::vector<std::pair<uint32_t, uint32_t>> unclaimed;
    unclaimed.reserve(chips[stage0_sub].size());
    for (const auto& [fid, row, col] : chips[stage0_sub]) {
        if (!used_coords.contains({row, col})) {
            unclaimed.push_back({row, col});
        }
    }

    uint32_t h2d_row, h2d_col, d2h_row, d2h_col;
    if (unclaimed.size() >= 2) {
        h2d_row = unclaimed[0].first;
        h2d_col = unclaimed[0].second;
        d2h_row = unclaimed[1].first;
        d2h_col = unclaimed[1].second;
    } else {
        // Fall back: reuse the edge boundary chips of stage 0.
        //   H2D socket sits on the forward-exit chip (stage 0 → stage 1).
        //   D2H socket sits on the loopback-entry chip (last stage → stage 0).
        uint32_t fwd_exit_row = 0, fwd_exit_col = 0;
        uint32_t lb_entry_row = 0, lb_entry_col = 0;
        for (const auto& re : resolved_edges) {
            if (!re.is_loopback && re.src == stage_order[0]) {
                fwd_exit_row = re.exit_row;
                fwd_exit_col = re.exit_col;
            }
            if (re.is_loopback && re.dst == stage_order[0]) {
                lb_entry_row = re.entry_row;
                lb_entry_col = re.entry_col;
            }
        }
        h2d_row = fwd_exit_row;
        h2d_col = fwd_exit_col;
        d2h_row = lb_entry_row;
        d2h_col = lb_entry_col;
    }

    // ------------------------------------------------------------------
    // 7. Build result
    // ------------------------------------------------------------------
    GraphLayoutResult result;
    result.stage_order = std::move(stage_order);
    result.node_to_submesh = std::move(stage_to_submesh);
    result.resolved_edges = std::move(resolved_edges);
    result.h2d_entry_row = h2d_row;
    result.h2d_entry_col = h2d_col;
    result.d2h_exit_row = d2h_row;
    result.d2h_exit_col = d2h_col;
    return result;
}

}  // namespace tt::tt_fabric
