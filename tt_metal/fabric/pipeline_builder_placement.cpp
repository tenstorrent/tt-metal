// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/fabric/pipeline_builder.hpp>

#include <algorithm>
#include <queue>
#include <set>
#include <stdexcept>
#include <tuple>

namespace tt::tt_fabric::detail {
namespace {

// Stage placement state; each candidate gets a separate link search.
struct PlacementSearch {
    const std::vector<std::string>& stage_order;
    const std::vector<EdgeInputTuple>& edges;
    const DirectLinks& submesh_links;
    const std::map<std::string, uint32_t>& stage_chip_counts;
    const std::vector<std::vector<InternalChip>>& chips;
    const std::map<std::string, uint32_t>& capacity_overrides;

    std::map<std::string, size_t> stage_index_by_name{};
    std::vector<std::pair<size_t, size_t>> edge_stage_indices{};
    std::vector<size_t> stage_link_counts{};
    std::vector<size_t> stage_neighbor_counts{};
    size_t last_host_neighbor_stage = 0;
    std::vector<std::vector<size_t>> submesh_neighbors{}, boundary_stages{};

    std::map<std::string, size_t> placement{};
    std::vector<bool> used{};
    GraphLayoutResult result{};
    std::string failure{};

    void prepare_search();
    void prepare_connectivity_checks();
    bool place_stages(size_t stage_index);

    uint32_t core_capacity(const std::string& stage, size_t mesh) const {
        auto it = capacity_overrides.find(stage);
        return it == capacity_overrides.end() ? (chips[mesh].size() >= 8 ? 1u : 2u) : it->second;
    }

    // A connected remaining graph needs its placed boundary and enough unused
    // submeshes in one physical component. Other unused submeshes can be ignored.
    bool remaining_stages_can_connect(size_t depth) const {
        if (boundary_stages[depth].empty()) {
            return true;
        }
        std::vector<bool> available(chips.size()), seen(chips.size());
        for (size_t mesh = 0; mesh < chips.size(); ++mesh) {
            available[mesh] = !used[mesh];
        }
        for (size_t stage : boundary_stages[depth]) {
            available[placement.at(stage_order[stage])] = true;
        }
        const size_t start = placement.at(stage_order[boundary_stages[depth].front()]);
        std::queue<size_t> queue;
        queue.push(start);
        seen[start] = true;
        while (!queue.empty()) {
            const size_t mesh = queue.front();
            queue.pop();
            for (size_t to : submesh_neighbors[mesh]) {
                if (available[to] && !seen[to]) {
                    seen[to] = true;
                    queue.push(to);
                }
            }
        }
        for (size_t stage : boundary_stages[depth]) {
            if (!seen[placement.at(stage_order[stage])]) {
                return false;
            }
        }
        size_t reachable_unused = 0;
        for (size_t mesh = 0; mesh < chips.size(); ++mesh) {
            reachable_unused += !used[mesh] && seen[mesh];
        }
        return reachable_unused >= stage_order.size() - depth;
    }
};

// Reservations and failed states belong to one fixed placement prefix.
struct LinkSearch {
    PlacementSearch& pipeline;
    GraphLayoutResult result{};
    using Endpoint = std::tuple<std::string, uint32_t, uint32_t>;
    std::map<Endpoint, uint32_t> used_slots_by_chip{};
    std::vector<size_t> last_required_edge_exclusive{};
    std::set<std::vector<size_t>> failed_link_states{};
    bool can_assign_host_endpoints = false;
    bool allow_shared_chips = false;

    bool solve();
    bool search_links(size_t edge_index);

    std::optional<uint32_t> reserve_slot(const std::string& stage, uint32_t row, uint32_t col) {
        const auto it = used_slots_by_chip.find({stage, row, col});
        const auto used_slots = it == used_slots_by_chip.end() ? 0u : it->second;
        if (!allow_shared_chips && stage != pipeline.stage_order.front() && used_slots != 0) {
            return std::nullopt;
        }
        const auto limit = pipeline.core_capacity(stage, pipeline.placement.at(stage));
        if (used_slots >= limit) {
            auto& failure = pipeline.failure;
            if (failure.empty()) {
                failure = "stage '" + stage + "', chip (" + std::to_string(row) + "," + std::to_string(col) +
                          ") needs " + std::to_string(uint64_t{used_slots} + 1) + " slots but has capacity " +
                          std::to_string(limit);
            }
            return std::nullopt;
        }
        used_slots_by_chip[{stage, row, col}] = used_slots + 1;
        return used_slots;
    }

    void release_slot(const std::string& stage, uint32_t row, uint32_t col) {
        auto it = used_slots_by_chip.find({stage, row, col});
        if (--it->second == 0) {
            used_slots_by_chip.erase(it);
        }
    }

    bool place_host_endpoint(uint32_t& row, uint32_t& col, std::optional<uint32_t>& slot, bool input) {
        const auto& stage = pipeline.stage_order.front();
        std::optional<std::pair<uint32_t, uint32_t>> preferred;
        for (const auto& edge : result.resolved_edges) {
            if (input && edge.src == stage && !edge.is_loopback) {
                preferred = {edge.exit_row, edge.exit_col};
                break;
            }
            if (!input && edge.dst == stage && edge.is_loopback) {
                preferred = {edge.entry_row, edge.entry_col};
                break;
            }
        }
        // Unused chips first; otherwise keep host traffic on its pipeline boundary
        // (H2D beside forward send, D2H beside loopback receive), then try any chip.
        for (int priority = 0; priority < 3; ++priority) {
            for (const auto& chip : pipeline.chips[pipeline.placement.at(stage)]) {
                const int chip_priority = !used_slots_by_chip.contains({stage, chip.row, chip.col})
                                              ? 0
                                              : (preferred == std::pair{chip.row, chip.col} ? 1 : 2);
                if (chip_priority != priority) {
                    continue;
                }
                if (auto candidate = reserve_slot(stage, chip.row, chip.col)) {
                    row = chip.row;
                    col = chip.col;
                    slot = candidate;
                    return true;
                }
            }
        }
        return false;
    }
};

void PlacementSearch::prepare_search() {
    stage_link_counts.resize(stage_order.size());
    std::vector<std::set<size_t>> stage_neighbors(stage_order.size());
    used.resize(chips.size());
    for (size_t i = 0; i < stage_order.size(); ++i) {
        stage_index_by_name.emplace(stage_order[i], i);
    }
    for (const auto& [src, dst, loopback] : edges) {
        const size_t a = stage_index_by_name.at(src), b = stage_index_by_name.at(dst);
        edge_stage_indices.emplace_back(a, b);
        if (a != b) {
            ++stage_link_counts[a];
            ++stage_link_counts[b];
            stage_neighbors[a].insert(b);
            stage_neighbors[b].insert(a);
        }
        if (a == 0 || b == 0) {
            last_host_neighbor_stage = std::max({last_host_neighbor_stage, a, b});
        }
    }
    for (const auto& neighbors : stage_neighbors) {
        stage_neighbor_counts.push_back(neighbors.size());
    }

    prepare_connectivity_checks();
}

// Precompute the placed/unplaced boundary and when connectivity pruning is valid.
void PlacementSearch::prepare_connectivity_checks() {
    submesh_neighbors.resize(chips.size());
    boundary_stages.resize(stage_order.size());
    for (const auto& [pair, links] : submesh_links) {
        submesh_neighbors[pair.first].push_back(pair.second);
        submesh_neighbors[pair.second].push_back(pair.first);
    }
    for (auto& neighbors : submesh_neighbors) {
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    }

    // Depth zero has no placed boundary to check.
    for (size_t depth = 1; depth < stage_order.size(); ++depth) {
        std::vector<std::vector<size_t>> remaining(stage_order.size());
        for (const auto& [a, b] : edge_stage_indices) {
            if (a != b && std::max(a, b) >= depth) {
                remaining[a].push_back(b);
                remaining[b].push_back(a);
            }
        }
        std::vector<bool> seen(stage_order.size());
        std::queue<size_t> queue;
        queue.push(depth);
        seen[depth] = true;
        while (!queue.empty()) {
            const size_t stage = queue.front();
            queue.pop();
            for (size_t to : remaining[stage]) {
                if (!seen[to]) {
                    seen[to] = true;
                    queue.push(to);
                }
            }
        }
        for (size_t stage = 0; stage < stage_order.size(); ++stage) {
            // Disconnected logical components need not share a physical component.
            // An empty boundary disables this necessary check for the prefix.
            if ((stage >= depth || !remaining[stage].empty()) && !seen[stage]) {
                boundary_stages[depth].clear();
                break;
            }
            if (stage < depth && !remaining[stage].empty()) {
                boundary_stages[depth].push_back(stage);
            }
        }
    }
}

// Reject prefixes whose edges cannot fit simultaneously. Successful temporary
// links are discarded: future stages may require different choices for them.
// Placement remains ordinary DFS; worst-case search is still exponential.
bool PlacementSearch::place_stages(size_t stage_index) {
    const auto& name = stage_order[stage_index];
    const auto shape = stage_chip_counts.find(name);
    const bool last_stage = stage_index + 1 == stage_order.size();
    // Link search checks chip-specific conflicts and reports host-only capacity errors.
    const size_t required =
        stage_link_counts[stage_index] + (stage_index == 0 && stage_link_counts[stage_index] != 0 ? 2 : 0);
    for (size_t mesh = 0; mesh < chips.size(); ++mesh) {
        if (used[mesh] || (shape != stage_chip_counts.end() && shape->second != chips[mesh].size())) {
            continue;
        }
        if (required > uint64_t{core_capacity(name, mesh)} * chips[mesh].size()) {
            continue;
        }
        // Distinct logical neighbors need distinct physical submeshes.
        if (stage_neighbor_counts[stage_index] > submesh_neighbors[mesh].size()) {
            continue;
        }
        bool connected = true;
        for (const auto& [a, b] : edge_stage_indices) {
            if (a == b) {
                continue;
            }
            if (a == stage_index && b < stage_index) {
                connected &= submesh_links.contains({mesh, placement.at(stage_order[b])});
            }
            if (b == stage_index && a < stage_index) {
                connected &= submesh_links.contains({placement.at(stage_order[a]), mesh});
            }
            if (!connected) {
                break;
            }
        }
        if (!connected) {
            continue;
        }
        placement[name] = mesh;
        used[mesh] = true;
        bool fits = last_stage || remaining_stages_can_connect(stage_index + 1);
        if (fits) {
            LinkSearch links{*this};
            fits = links.solve();
            if (fits && last_stage) {
                result = std::move(links.result);
                return true;
            }
        }
        if (fits && place_stages(stage_index + 1)) {
            return true;
        }
        used[mesh] = false;
        placement.erase(name);
    }
    return false;
}

// Solve all currently placed edges together. Delay host allocation until all
// source-stage links are included, so a temporary host choice cannot block one.
bool LinkSearch::solve() {
    can_assign_host_endpoints = pipeline.placement.size() > pipeline.last_host_neighbor_stage;
    last_required_edge_exclusive.resize(pipeline.stage_order.size());
    for (size_t edge = 0; edge < pipeline.edge_stage_indices.size(); ++edge) {
        const auto [a, b] = pipeline.edge_stage_indices[edge];
        if (a != b && a < pipeline.placement.size() && b < pipeline.placement.size()) {
            last_required_edge_exclusive[a] = last_required_edge_exclusive[b] = edge + 1;
        }
    }
    // Keep stage-zero occupancy in the cache until the host check below.
    if (can_assign_host_endpoints) {
        last_required_edge_exclusive[0] = pipeline.edges.size() + 1;
    }
    // Prefer a complete link assignment with separate forwarding chips. If none
    // fits this placement, retry with the configured capacities, including sharing.
    if (search_links(0)) {
        return true;
    }
    failed_link_states.clear();
    allow_shared_chips = true;
    return search_links(0);
}

bool LinkSearch::search_links(size_t edge_index) {
    if (edge_index == pipeline.edges.size()) {
        if (!can_assign_host_endpoints) {
            return true;
        }
        if (!place_host_endpoint(result.h2d_entry_row, result.h2d_entry_col, result.h2d_core_slot, true)) {
            return false;
        }
        if (place_host_endpoint(result.d2h_exit_row, result.d2h_exit_col, result.d2h_core_slot, false)) {
            return true;
        }
        release_slot(pipeline.stage_order.front(), result.h2d_entry_row, result.h2d_entry_col);
        return false;
    }
    const auto& [src, dst, loopback] = pipeline.edges[edge_index];
    const auto& placement = pipeline.placement;
    if (src == dst || !placement.contains(src) || !placement.contains(dst)) {
        return search_links(edge_index + 1);
    }
    // Placement is fixed. Only occupancy on stages with remaining links
    // (or pending host endpoints) can affect whether the suffix is feasible.
    std::vector<size_t> key{edge_index};
    for (const auto& [endpoint, used_slots] : used_slots_by_chip) {
        const auto& [stage, row, col] = endpoint;
        const size_t id = pipeline.stage_index_by_name.at(stage);
        if (last_required_edge_exclusive[id] > edge_index) {
            key.insert(key.end(), {id, row, col, used_slots});
        }
    }
    if (failed_link_states.contains(key)) {
        return false;
    }
    for (const auto& link : pipeline.submesh_links.at({placement.at(src), placement.at(dst)})) {
        auto exit_slot = reserve_slot(src, link.exit_row, link.exit_col);
        if (!exit_slot) {
            continue;
        }
        auto entry_slot = reserve_slot(dst, link.entry_row, link.entry_col);
        if (entry_slot) {
            result.resolved_edges.push_back(
                {src,
                 dst,
                 loopback,
                 link.exit_row,
                 link.exit_col,
                 link.entry_row,
                 link.entry_col,
                 exit_slot,
                 entry_slot});
            if (search_links(edge_index + 1)) {
                return true;
            }
            result.resolved_edges.pop_back();
            release_slot(dst, link.entry_row, link.entry_col);
        }
        release_slot(src, link.exit_row, link.exit_col);
    }
    failed_link_states.insert(std::move(key));
    return false;
}

}  // namespace

GraphLayoutResult resolve_pipeline_placement(
    const std::vector<std::string>& stage_order,
    const std::vector<EdgeInputTuple>& edges,
    const DirectLinks& submesh_links,
    const std::map<std::string, uint32_t>& stage_chip_counts,
    const std::vector<std::vector<InternalChip>>& chips,
    const std::map<std::string, uint32_t>& capacity_overrides) {
    PlacementSearch search{stage_order, edges, submesh_links, stage_chip_counts, chips, capacity_overrides};
    search.prepare_search();
    if (!search.place_stages(0)) {
        std::string error =
            "resolve_graph_layout: no valid submesh assignment found; exact placement/link search exhausted: "
            "no assignment satisfies connectivity, shape, and pipeline-core capacity constraints";
        if (!search.failure.empty()) {
            error += "; example capacity conflict: " + search.failure;
        }
        throw std::runtime_error(error);
    }
    search.result.node_to_submesh = search.placement;
    return std::move(search.result);
}

}  // namespace tt::tt_fabric::detail
