// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/fabric/topology_mapper_utils.hpp>

#include <algorithm>
#include <chrono>
#include <functional>
#include <limits>
#include <unordered_set>

#include <tt-logger/tt-logger.hpp>
#include <fmt/format.h>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include "tt_metal/fabric/physical_system_descriptor.hpp"
#include "tt_metal/impl/context/metal_context.hpp"
#include <llrt/tt_cluster.hpp>

namespace tt::tt_metal::experimental::tt_fabric {

// NOTE: This mapping algorithm uses nested lambdas and deep control flow for
// pruning and search. Refactoring would be non-trivial and risks regressions,
// so we suppress the cognitive-complexity check for this function.
// NOLINTBEGIN(readability-function-cognitive-complexity)
TopologyMappingResult map_mesh_to_physical(
    MeshId mesh_id,
    const LogicalAdjacencyMap& logical_adjacency,
    const PhysicalAdjacencyMap& physical_adjacency,
    const std::map<FabricNodeId, MeshHostRankId>& node_to_host_rank,
    const std::map<tt::tt_metal::AsicID, MeshHostRankId>& asic_to_host_rank,
    const TopologyMappingConfig& config) {
    TopologyMappingResult result;

    const auto& log_adj = logical_adjacency;
    const auto& phys_adj = physical_adjacency;
    const bool strict_mode = config.strict_mode;

    // Build node vectors from adjacency maps
    std::vector<FabricNodeId> log_nodes;
    log_nodes.reserve(log_adj.size());
    for (const auto& p : log_adj) {
        log_nodes.push_back(p.first);
    }

    std::vector<tt::tt_metal::AsicID> phys_nodes;
    phys_nodes.reserve(phys_adj.size());
    for (const auto& p : phys_adj) {
        phys_nodes.push_back(p.first);
    }

    size_t n_log = log_nodes.size();
    size_t n_phys = phys_nodes.size();

    if (n_log > n_phys) {
        result.success = false;
        result.error_message = fmt::format(
            "Logical graph ({} nodes) is larger than physical topology ({} nodes) for mesh {}",
            n_log,
            n_phys,
            mesh_id.get());
        return result;
    }

    // Handle empty graph case
    if (n_log == 0) {
        result.success = true;
        return result;
    }

    // Build index mappings
    std::map<FabricNodeId, size_t> log_to_idx;
    for (size_t i = 0; i < n_log; ++i) {
        log_to_idx[log_nodes[i]] = i;
    }

    std::map<tt::tt_metal::AsicID, size_t> phys_to_idx;
    for (size_t i = 0; i < n_phys; ++i) {
        phys_to_idx[phys_nodes[i]] = i;
    }

    // Build connection count maps for STRICT mode validation
    // log_conn_count[i][j] = number of channels from logical node i to logical node j
    std::vector<std::map<size_t, size_t>> log_conn_count(n_log);
    for (size_t i = 0; i < n_log; ++i) {
        for (const auto& neigh : log_adj.at(log_nodes[i])) {
            size_t idx = log_to_idx.at(neigh);
            log_conn_count[i][idx]++;
        }
    }

    // Build deduplicated adjacency index vectors
    std::vector<std::vector<size_t>> log_adj_idx(n_log);
    for (size_t i = 0; i < n_log; ++i) {
        std::unordered_set<size_t> seen_indices;
        for (const auto& neigh : log_adj.at(log_nodes[i])) {
            size_t idx = log_to_idx.at(neigh);
            if (seen_indices.insert(idx).second) {
                log_adj_idx[i].push_back(idx);
            }
        }
        std::sort(log_adj_idx[i].begin(), log_adj_idx[i].end());
    }

    // Build connection count maps for physical topology
    std::vector<std::map<size_t, size_t>> phys_conn_count(n_phys);
    for (size_t i = 0; i < n_phys; ++i) {
        for (const auto& neigh : phys_adj.at(phys_nodes[i])) {
            auto it = phys_to_idx.find(neigh);
            if (it != phys_to_idx.end()) {
                size_t idx = it->second;
                phys_conn_count[i][idx]++;
            }
        }
    }

    std::vector<std::vector<size_t>> phys_adj_idx(n_phys);
    for (size_t i = 0; i < n_phys; ++i) {
        std::unordered_set<size_t> seen_indices;
        for (const auto& neigh : phys_adj.at(phys_nodes[i])) {
            auto it = phys_to_idx.find(neigh);
            if (it != phys_to_idx.end()) {
                size_t idx = it->second;
                if (seen_indices.insert(idx).second) {
                    phys_adj_idx[i].push_back(idx);
                }
            }
        }
        std::sort(phys_adj_idx[i].begin(), phys_adj_idx[i].end());
    }

    // mapping[logical_index] = physical_index
    std::vector<int> mapping(n_log, -1);
    std::vector<bool> used(n_phys, false);

    // Precompute degrees for pruning
    std::vector<size_t> log_deg(n_log, 0);
    for (size_t i = 0; i < n_log; ++i) {
        log_deg[i] = log_adj_idx[i].size();
    }
    std::vector<size_t> phys_deg(n_phys, 0);
    for (size_t j = 0; j < n_phys; ++j) {
        phys_deg[j] = phys_adj_idx[j].size();
    }

    // Emit initial stats for debugging
    auto emit_degree_hist = [](const std::vector<size_t>& degs) {
        std::map<size_t, size_t> hist;
        for (auto d : degs) {
            hist[d]++;
        }
        std::string s = "{";
        bool first = true;
        for (const auto& [d, c] : hist) {
            if (!first) {
                s += ", ";
            }
            first = false;
            s += std::to_string(d) + ":" + std::to_string(c);
        }
        s += "}";
        return s;
    };
    log_info(
        tt::LogFabric,
        "TopologyMapper mapping start (mesh={}): n_log={}, n_phys={}, log_deg_hist={}, phys_deg_hist={}",
        mesh_id.get(),
        n_log,
        n_phys,
        emit_degree_hist(log_deg),
        emit_degree_hist(phys_deg));

    // Candidate restrictions for logical indices pinned by ASIC position (tray, location)
    std::vector<std::vector<size_t>> restricted_phys_indices_for_logical(n_log);
    if (!config.pinnings.empty()) {
        // Validate uniqueness of pins for this mesh and apply
        std::map<FabricNodeId, AsicPosition> first_pinnings;

        for (const auto& [pos, fabric_node] : config.pinnings) {
            if (fabric_node.mesh_id != mesh_id) {
                continue;  // pin for another mesh
            }

            if (!log_to_idx.contains(fabric_node)) {
                result.success = false;
                result.error_message =
                    fmt::format("Pinned fabric node {} not found in logical mesh {}", fabric_node, mesh_id.get());
                return result;
            }

            auto [it, inserted] = first_pinnings.try_emplace(fabric_node, pos);
            if (!inserted) {
                const auto& prev_pos = it->second;
                result.success = false;
                result.error_message = fmt::format(
                    "Fabric node {} in mesh {} is pinned to multiple ASIC positions: (tray {}, loc {}) and (tray "
                    "{}, loc {})",
                    fabric_node,
                    mesh_id.get(),
                    *prev_pos.first,
                    *prev_pos.second,
                    *pos.first,
                    *pos.second);
                return result;
            }

            // Find matching physical indices in this mesh for the pinned ASIC position
            std::vector<size_t> matches;
            for (size_t j = 0; j < n_phys; ++j) {
                auto asic = phys_nodes[j];
                auto pos_it = config.asic_positions.find(asic);
                if (pos_it != config.asic_positions.end()) {
                    auto tray = pos_it->second.first;
                    auto loc = pos_it->second.second;
                    if (tray == pos.first && loc == pos.second) {
                        matches.push_back(j);
                    }
                }
            }

            if (matches.empty()) {
                result.success = false;
                result.error_message = fmt::format(
                    "Pinned ASIC position (tray {}, loc {}) not found among physical ASICs participating in mesh {}",
                    *pos.first,
                    *pos.second,
                    mesh_id.get());
                return result;
            }

            size_t li = log_to_idx.at(fabric_node);
            restricted_phys_indices_for_logical[li] = std::move(matches);
        }

        // Print info about pinnings used for this mesh
        if (!first_pinnings.empty()) {
            std::string pinnings_str;
            bool first = true;
            for (const auto& [fabric_node, pos] : first_pinnings) {
                if (!first) {
                    pinnings_str += ", ";
                }
                first = false;
                pinnings_str += fmt::format(
                    "fabric_node={} (mesh_id={}, chip_id={}) -> ASIC position (tray={}, loc={})",
                    fabric_node,
                    fabric_node.mesh_id.get(),
                    fabric_node.chip_id,
                    *pos.first,
                    *pos.second);
            }
            log_info(
                tt::LogFabric,
                "TopologyMapper: Using {} pinning(s) for mesh {}: [{}]",
                first_pinnings.size(),
                mesh_id.get(),
                pinnings_str);
        }
    }

    // Fast path: if logical graph is a single path (two endpoints with degree 1; all others degree <=2),
    // map it using a linear path-extension DFS over the physical graph
    auto try_fast_path_for_logical_chain = [&]() -> bool {
        std::vector<size_t> endpoints;
        for (size_t i = 0; i < n_log; ++i) {
            if (log_deg[i] == 1) {
                endpoints.push_back(i);
            }
            if (log_deg[i] > 2) {
                return false;
            }
        }
        if (endpoints.size() != 2) {
            return false;
        }

        // Build ordered logical path indices from one endpoint
        std::vector<size_t> log_order;
        log_order.reserve(n_log);
        std::vector<bool> seen(n_log, false);
        size_t curr = endpoints[0];
        size_t prev = n_log;  // sentinel
        for (size_t k = 0; k < n_log; ++k) {
            log_order.push_back(curr);
            seen[curr] = true;
            size_t next_candidate = n_log;
            for (size_t nb : log_adj_idx[curr]) {
                if (nb != prev && !seen[nb]) {
                    next_candidate = nb;
                    break;
                }
            }
            prev = curr;
            curr = next_candidate;
            if (k + 1 < n_log && curr == n_log) {
                return false;  // disconnected
            }
        }

        // Reachability check helper to prevent dead-ends
        auto reachable_unused_count = [&](size_t start_phys) -> size_t {
            std::vector<char> vis(n_phys, 0);
            std::vector<size_t> q;
            q.reserve(n_phys);
            if (used[start_phys]) {
                return 0;
            }
            q.push_back(start_phys);
            vis[start_phys] = 1;
            size_t qi = 0;
            size_t cnt = 0;
            while (qi < q.size()) {
                size_t u = q[qi++];
                cnt++;
                for (size_t v : phys_adj_idx[u]) {
                    if (!vis[v] && !used[v]) {
                        vis[v] = 1;
                        q.push_back(v);
                    }
                }
            }
            return cnt;
        };

        std::function<bool(size_t, size_t)> place = [&](size_t idx_in_path, size_t prev_phys) -> bool {
            if (idx_in_path == n_log) {
                return true;
            }
            size_t li = log_order[idx_in_path];
            if (idx_in_path == 0) {
                // First node: try all candidates
                std::vector<size_t> candidates;
                if (!restricted_phys_indices_for_logical[li].empty()) {
                    candidates = restricted_phys_indices_for_logical[li];
                } else {
                    for (size_t pj = 0; pj < n_phys; ++pj) {
                        candidates.push_back(pj);
                    }
                }
                for (size_t pj : candidates) {
                    if (used[pj]) {
                        continue;
                    }
                    if (phys_deg[pj] < log_deg[li]) {
                        continue;
                    }
                    // Check mesh rank compatibility
                    if (node_to_host_rank.at(log_nodes[li]) != asic_to_host_rank.at(phys_nodes[pj])) {
                        continue;
                    }
                    used[pj] = true;
                    mapping[li] = static_cast<int>(pj);
                    bool ok = place(idx_in_path + 1, pj);
                    if (ok) {
                        return true;
                    }
                    mapping[li] = -1;
                    used[pj] = false;
                }
                return false;
            }  // Next must be an unused neighbor of prev_phys
            size_t remain = n_log - idx_in_path;
            for (size_t pj : phys_adj_idx[prev_phys]) {
                if (used[pj]) {
                    continue;
                }
                if (phys_deg[pj] < log_deg[li]) {
                    continue;
                }
                // Check mesh rank compatibility
                if (node_to_host_rank.at(log_nodes[li]) != asic_to_host_rank.at(phys_nodes[pj])) {
                    continue;
                }
                // Check pinning restrictions if any
                if (!restricted_phys_indices_for_logical[li].empty()) {
                    if (std::find(
                            restricted_phys_indices_for_logical[li].begin(),
                            restricted_phys_indices_for_logical[li].end(),
                            pj) == restricted_phys_indices_for_logical[li].end()) {
                        continue;
                    }
                }
                // Reachability pruning
                size_t reach = reachable_unused_count(pj);
                if (reach < remain) {
                    continue;
                }
                used[pj] = true;
                mapping[li] = static_cast<int>(pj);
                if (place(idx_in_path + 1, pj)) {
                    return true;
                }
                mapping[li] = -1;
                used[pj] = false;
            }
            return false;
        };

        bool ok = place(0, n_phys);
        if (ok) {
            log_info(tt::LogFabric, "Fast-path path-graph mapping succeeded for mesh {}", mesh_id.get());
        } else {
            log_debug(tt::LogFabric, "Fast-path path-graph mapping failed; falling back to general DFS");
        }
        return ok;
    };

    if (try_fast_path_for_logical_chain()) {
        // Validate strict mode constraints for fast-path result
        if (strict_mode) {
            for (size_t i = 0; i < n_log; ++i) {
                size_t phys_i = static_cast<size_t>(mapping[i]);
                for (const auto& [neigh_log_idx, log_count] : log_conn_count[i]) {
                    size_t phys_neigh_idx = static_cast<size_t>(mapping[neigh_log_idx]);
                    auto it = phys_conn_count[phys_i].find(phys_neigh_idx);
                    if (it == phys_conn_count[phys_i].end()) {
                        result.success = false;
                        result.error_message = fmt::format(
                            "Strict mode validation failed: logical edge from node {} to {} exists, "
                            "but physical edge from {} to {} does not",
                            log_nodes[i],
                            log_nodes[neigh_log_idx],
                            phys_nodes[phys_i],
                            phys_nodes[phys_neigh_idx]);
                        return result;
                    }
                    size_t phys_count = it->second;
                    if (phys_count < log_count) {
                        result.success = false;
                        result.error_message = fmt::format(
                            "Strict mode validation failed: logical edge from node {} to {} requires {} channels, "
                            "but physical edge from {} to {} only has {} channels",
                            log_nodes[i],
                            log_nodes[neigh_log_idx],
                            log_count,
                            phys_nodes[phys_i],
                            phys_nodes[phys_neigh_idx],
                            phys_count);
                        return result;
                    }
                }
            }
        }

        // Build result from mapping
        for (size_t i = 0; i < n_log; ++i) {
            if (mapping[i] < 0) {
                result.success = false;
                result.error_message = "Internal error: fast-path produced incomplete mapping";
                return result;
            }
            FabricNodeId fn = log_nodes[i];
            tt::tt_metal::AsicID asic = phys_nodes[static_cast<size_t>(mapping[i])];
            result.fabric_node_to_asic.emplace(fn, asic);
            result.asic_to_fabric_node.emplace(asic, fn);
        }
        result.success = true;
        return result;
    }

    // General DFS with dynamic node selection
    auto select_next_logical = [&](const std::vector<int>& mapping_ref, const std::vector<bool>& used_ref) {
        size_t best_li = n_log;
        size_t best_mapped_neigh = 0;
        size_t best_cand_count = (std::numeric_limits<size_t>::max)();
        size_t best_log_deg = (std::numeric_limits<size_t>::max)();

        for (size_t li = 0; li < n_log; ++li) {
            if (mapping_ref[li] != -1) {
                continue;
            }
            size_t mapped_neigh_count = 0;
            for (size_t v : log_adj_idx[li]) {
                if (mapping_ref[v] != -1) {
                    mapped_neigh_count++;
                }
            }
            // Compute candidate count under current partial assignment
            size_t cand_count = 0;
            for (size_t j = 0; j < n_phys; ++j) {
                if (used_ref[j]) {
                    continue;
                }
                if (phys_deg[j] < log_deg[li]) {
                    continue;
                }
                // Check mesh rank compatibility
                auto log_mesh_rank = node_to_host_rank.at(log_nodes[li]);
                auto phys_mesh_rank = asic_to_host_rank.at(phys_nodes[j]);
                if (log_mesh_rank != phys_mesh_rank) {
                    continue;
                }
                bool ok_local = true;
                for (size_t v : log_adj_idx[li]) {
                    int pj = mapping_ref[v];
                    if (pj == -1) {
                        continue;
                    }
                    // logical edge must be present physically
                    if (!std::binary_search(phys_adj_idx[j].begin(), phys_adj_idx[j].end(), static_cast<size_t>(pj))) {
                        ok_local = false;
                        break;
                    }
                }
                if (!ok_local) {
                    continue;
                }
                cand_count++;
            }
            // Skip logical nodes with zero candidates
            if (cand_count == 0) {
                continue;
            }
            if (best_li == n_log || mapped_neigh_count > best_mapped_neigh ||
                (mapped_neigh_count == best_mapped_neigh &&
                 ((best_mapped_neigh == 0 && log_deg[li] < best_log_deg) ||
                  (best_mapped_neigh != 0 && cand_count < best_cand_count)))) {
                best_li = li;
                best_mapped_neigh = mapped_neigh_count;
                best_cand_count = cand_count;
                best_log_deg = log_deg[li];
            }
        }
        return best_li;
    };

    // Memoization of failed states
    std::unordered_set<std::uint64_t> failed_states;
    auto hash_state = [&](size_t /*pos*/) -> std::uint64_t {
        const std::uint64_t fnv_offset = 1469598103934665603ull;
        const std::uint64_t fnv_prime = 1099511628211ull;
        std::uint64_t h = fnv_offset;
        for (size_t li = 0; li < n_log; ++li) {
            if (mapping[li] != -1) {
                std::uint64_t pairh =
                    (static_cast<std::uint64_t>(li) << 32) ^ static_cast<std::uint64_t>(mapping[li] + 1);
                h ^= pairh;
                h *= fnv_prime;
            }
        }
        return h;
    };

    // Debug counters and timing
    std::size_t dfs_calls = 0;
    auto dfs_start = std::chrono::steady_clock::now();

    std::function<bool(size_t)> dfs = [&](size_t pos) -> bool {
        if (pos == n_log) {
            return true;
        }

        // Periodic progress logging
        dfs_calls++;
        if ((dfs_calls & ((1u << 18) - 1)) == 0) {
            std::size_t assigned = 0;
            for (auto v : mapping) {
                assigned += (v != -1);
            }
            auto now = std::chrono::steady_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - dfs_start).count();
            log_info(
                tt::LogFabric,
                "TopologyMapper DFS progress: calls={}, assigned={}/{}, failed_states={}, elapsed_ms={}",
                dfs_calls,
                assigned,
                n_log,
                failed_states.size(),
                ms);
        }

        std::uint64_t key = hash_state(pos);
        if (failed_states.contains(key)) {
            return false;
        }

        // Select next logical node dynamically
        size_t li = select_next_logical(mapping, used);
        if (li == n_log) {
            return false;
        }

        // Candidate generation
        std::vector<size_t> candidates;
        candidates.reserve(n_phys);

        if (!restricted_phys_indices_for_logical[li].empty()) {
            for (size_t j : restricted_phys_indices_for_logical[li]) {
                if (j < n_phys && !used[j] && phys_deg[j] >= log_deg[li]) {
                    if (node_to_host_rank.at(log_nodes[li]) == asic_to_host_rank.at(phys_nodes[j])) {
                        candidates.push_back(j);
                    }
                }
            }
            if (candidates.empty()) {
                failed_states.insert(key);
                return false;
            }
        } else {
            for (size_t j = 0; j < n_phys; ++j) {
                if (used[j]) {
                    continue;
                }
                if (phys_deg[j] < log_deg[li]) {
                    continue;
                }
                auto log_mesh_rank = node_to_host_rank.at(log_nodes[li]);
                auto phys_mesh_rank = asic_to_host_rank.at(phys_nodes[j]);
                if (log_mesh_rank != phys_mesh_rank) {
                    continue;
                }
                candidates.push_back(j);
            }
            if (candidates.empty()) {
                failed_states.insert(key);
                return false;
            }
        }

        // Order candidates by degree gap
        std::sort(candidates.begin(), candidates.end(), [&](size_t a, size_t b) {
            size_t da =
                (phys_deg[a] >= log_deg[li]) ? (phys_deg[a] - log_deg[li]) : (std::numeric_limits<size_t>::max)();
            size_t db =
                (phys_deg[b] >= log_deg[li]) ? (phys_deg[b] - log_deg[li]) : (std::numeric_limits<size_t>::max)();
            if (da != db) {
                return da < db;
            }
            return a < b;
        });

        for (size_t j : candidates) {
            // Local consistency check
            bool ok = true;
            for (size_t lk = 0; lk < n_log; ++lk) {
                int pk_i = mapping[lk];
                if (pk_i == -1) {
                    continue;
                }
                size_t pk = static_cast<size_t>(pk_i);
                bool log_connected = std::binary_search(log_adj_idx[li].begin(), log_adj_idx[li].end(), lk);
                bool phys_connected = std::binary_search(phys_adj_idx[j].begin(), phys_adj_idx[j].end(), pk);
                if (log_connected && !phys_connected) {
                    ok = false;
                    break;
                }
                // In STRICT mode, also validate connection counts
                if (ok && strict_mode && log_connected) {
                    size_t log_count = log_conn_count[li].at(lk);
                    size_t phys_count = phys_conn_count[j].at(pk);
                    if (phys_count < log_count) {
                        ok = false;
                        break;
                    }
                }
            }
            if (!ok) {
                continue;
            }

            // Forward checking: ensure candidate has enough unused neighbors
            std::vector<size_t> unassigned_neighbors;
            for (size_t v : log_adj_idx[li]) {
                if (mapping[v] == -1) {
                    unassigned_neighbors.push_back(v);
                }
            }
            std::vector<size_t> unused_phys_neighbors;
            for (size_t pj : phys_adj_idx[j]) {
                if (!used[pj]) {
                    unused_phys_neighbors.push_back(pj);
                }
            }
            if (unused_phys_neighbors.size() < unassigned_neighbors.size()) {
                continue;
            }

            // Verify future logical neighbors have viable candidates
            for (size_t v : unassigned_neighbors) {
                bool has_candidate = false;
                for (size_t pj : unused_phys_neighbors) {
                    if (phys_deg[pj] < log_deg[v]) {
                        continue;
                    }
                    if (node_to_host_rank.at(log_nodes[v]) != asic_to_host_rank.at(phys_nodes[pj])) {
                        continue;
                    }
                    bool consistent = true;
                    for (size_t lv = 0; lv < n_log; ++lv) {
                        int pk2_i = mapping[lv];
                        if (pk2_i == -1) {
                            continue;
                        }
                        size_t pk2 = static_cast<size_t>(pk2_i);
                        bool log_conn2 = std::binary_search(log_adj_idx[v].begin(), log_adj_idx[v].end(), lv);
                        bool phys_conn2 = std::binary_search(phys_adj_idx[pj].begin(), phys_adj_idx[pj].end(), pk2);
                        if (log_conn2 && !phys_conn2) {
                            consistent = false;
                            break;
                        }
                        if (consistent && strict_mode && log_conn2) {
                            size_t log_count = log_conn_count[v].at(lv);
                            size_t phys_count = phys_conn_count[pj].at(pk2);
                            if (phys_count < log_count) {
                                consistent = false;
                                break;
                            }
                        }
                    }
                    if (consistent) {
                        has_candidate = true;
                        break;
                    }
                }
                if (!has_candidate) {
                    ok = false;
                    break;
                }
            }
            if (!ok) {
                continue;
            }

            used[j] = true;
            mapping[li] = static_cast<int>(j);
            if (dfs(pos + 1)) {
                return true;
            }
            mapping[li] = -1;
            used[j] = false;
        }

        failed_states.insert(key);
        return false;
    };

    // Start DFS
    size_t assigned_count = 0;
    for (auto v : mapping) {
        if (v != -1) {
            assigned_count++;
        }
    }

    bool found = false;
    if (assigned_count == 0) {
        // Try multiple starting logical nodes
        std::vector<size_t> pinned_nodes;
        for (size_t li = 0; li < n_log; ++li) {
            if (!restricted_phys_indices_for_logical[li].empty()) {
                pinned_nodes.push_back(li);
            }
        }

        auto assign_pinned_nodes = [&]() -> bool {
            for (size_t pinned_li : pinned_nodes) {
                bool assigned = false;
                for (size_t j : restricted_phys_indices_for_logical[pinned_li]) {
                    if (j < n_phys && !used[j] && phys_deg[j] >= log_deg[pinned_li]) {
                        if (node_to_host_rank.at(log_nodes[pinned_li]) == asic_to_host_rank.at(phys_nodes[j])) {
                            used[j] = true;
                            mapping[pinned_li] = static_cast<int>(j);
                            assigned = true;
                            break;
                        }
                    }
                }
                if (!assigned) {
                    return false;
                }
            }
            return true;
        };

        size_t first_logical_node = select_next_logical(mapping, used);

        if (first_logical_node == n_log) {
            if (!pinned_nodes.empty()) {
                if (assign_pinned_nodes()) {
                    size_t current_assigned_count = 0;
                    for (auto v : mapping) {
                        if (v != -1) {
                            current_assigned_count++;
                        }
                    }
                    found = dfs(current_assigned_count);
                }
            }
        } else {
            std::vector<size_t> physical_candidates;
            auto first_logical_mesh_rank = node_to_host_rank.at(log_nodes[first_logical_node]);

            if (!restricted_phys_indices_for_logical[first_logical_node].empty()) {
                for (size_t j : restricted_phys_indices_for_logical[first_logical_node]) {
                    if (j < n_phys && phys_deg[j] >= log_deg[first_logical_node]) {
                        auto phys_mesh_rank = asic_to_host_rank.at(phys_nodes[j]);
                        if (first_logical_mesh_rank == phys_mesh_rank) {
                            physical_candidates.push_back(j);
                        }
                    }
                }
            } else {
                for (size_t j = 0; j < n_phys; ++j) {
                    if (phys_deg[j] < log_deg[first_logical_node]) {
                        continue;
                    }
                    auto phys_mesh_rank = asic_to_host_rank.at(phys_nodes[j]);
                    if (first_logical_mesh_rank != phys_mesh_rank) {
                        continue;
                    }
                    physical_candidates.push_back(j);
                }
            }

            for (size_t first_phys_j : physical_candidates) {
                std::fill(mapping.begin(), mapping.end(), -1);
                std::fill(used.begin(), used.end(), false);
                failed_states.clear();
                dfs_calls = 0;
                dfs_start = std::chrono::steady_clock::now();

                if (!pinned_nodes.empty()) {
                    if (!assign_pinned_nodes()) {
                        continue;
                    }
                }

                if (used[first_phys_j]) {
                    continue;
                }

                used[first_phys_j] = true;
                mapping[first_logical_node] = static_cast<int>(first_phys_j);

                size_t current_assigned_count = 0;
                for (auto v : mapping) {
                    if (v != -1) {
                        current_assigned_count++;
                    }
                }

                found = dfs(current_assigned_count);
                if (found) {
                    break;
                }
            }

            if (!found && !pinned_nodes.empty()) {
                std::fill(mapping.begin(), mapping.end(), -1);
                std::fill(used.begin(), used.end(), false);
                failed_states.clear();
                dfs_calls = 0;
                dfs_start = std::chrono::steady_clock::now();

                if (assign_pinned_nodes()) {
                    size_t current_assigned_count = 0;
                    for (auto v : mapping) {
                        if (v != -1) {
                            current_assigned_count++;
                        }
                    }
                    found = dfs(current_assigned_count);
                }
            }
        }
    } else {
        found = dfs(assigned_count);
    }

    if (!found) {
        result.success = false;
        result.error_message = fmt::format(
            "Could not find valid mapping for mesh {} under the given constraints. "
            "Logical graph may not fit in the physical topology.",
            mesh_id.get());
        return result;
    }

    // Build result from mapping
    for (size_t i = 0; i < n_log; ++i) {
        FabricNodeId fn = log_nodes[i];
        tt::tt_metal::AsicID asic = phys_nodes[static_cast<size_t>(mapping[i])];
        result.fabric_node_to_asic.emplace(fn, asic);
        result.asic_to_fabric_node.emplace(asic, fn);
    }

    // Final validation in strict mode
    if (strict_mode) {
        for (size_t i = 0; i < n_log; ++i) {
            size_t phys_i = static_cast<size_t>(mapping[i]);
            for (const auto& [neigh_log_idx, log_count] : log_conn_count[i]) {
                size_t phys_neigh_idx = static_cast<size_t>(mapping[neigh_log_idx]);
                auto it = phys_conn_count[phys_i].find(phys_neigh_idx);
                if (it == phys_conn_count[phys_i].end()) {
                    result.success = false;
                    result.error_message = fmt::format(
                        "Strict mode validation failed: logical edge from node {} to {} exists, "
                        "but physical edge from {} to {} does not",
                        log_nodes[i],
                        log_nodes[neigh_log_idx],
                        phys_nodes[phys_i],
                        phys_nodes[phys_neigh_idx]);
                    return result;
                }
                size_t phys_count = it->second;
                if (phys_count < log_count) {
                    result.success = false;
                    result.error_message = fmt::format(
                        "Strict mode validation failed: logical edge from node {} to {} requires {} channels, "
                        "but physical edge from {} to {} only has {} channels",
                        log_nodes[i],
                        log_nodes[neigh_log_idx],
                        log_count,
                        phys_nodes[phys_i],
                        phys_nodes[phys_neigh_idx],
                        phys_count);
                    return result;
                }
            }
        }
    }

    result.success = true;
    return result;
}
// NOLINTEND(readability-function-cognitive-complexity)

std::map<MeshId, LogicalAdjacencyMap> build_adjacency_map_logical(const ::tt::tt_fabric::MeshGraph& mesh_graph) {
    std::map<MeshId, LogicalAdjacencyMap> adjacency_map;

    auto get_local_adjacents = [&](FabricNodeId fabric_node_id, MeshId mesh_id) {
        auto adjacent_map = mesh_graph.get_intra_mesh_connectivity()[*mesh_id][fabric_node_id.chip_id];

        std::vector<FabricNodeId> adjacents;
        bool relaxed = mesh_graph.is_intra_mesh_policy_relaxed(mesh_id);
        for (const auto& [neighbor_chip_id, edge] : adjacent_map) {
            // Skip self-connections
            if (neighbor_chip_id == fabric_node_id.chip_id) {
                continue;
            }
            size_t repeat_count = relaxed ? 1 : edge.connected_chip_ids.size();
            for (size_t i = 0; i < repeat_count; ++i) {
                adjacents.push_back(FabricNodeId(mesh_id, neighbor_chip_id));
            }
        }
        return adjacents;
    };

    // Iterate over all mesh IDs from the mesh graph
    for (const auto& mesh_id : mesh_graph.get_all_mesh_ids()) {
        LogicalAdjacencyMap logical_adjacency_map;
        for (const auto& [_, chip_id] : mesh_graph.get_chip_ids(mesh_id)) {
            auto fabric_node_id = FabricNodeId(mesh_id, chip_id);
            logical_adjacency_map[fabric_node_id] = get_local_adjacents(fabric_node_id, mesh_id);
        }
        adjacency_map[mesh_id] = logical_adjacency_map;
    }

    return adjacency_map;
}

std::map<MeshId, PhysicalAdjacencyMap> build_adjacency_map_physical(
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>>& asic_id_to_mesh_rank) {
    std::map<MeshId, PhysicalAdjacencyMap> adjacency_map;

    // Build a set of ASIC IDs for each mesh based on mesh rank mapping
    std::map<MeshId, std::unordered_set<tt::tt_metal::AsicID>> mesh_asic_ids;
    for (const auto& [mesh_id, asic_map] : asic_id_to_mesh_rank) {
        for (const auto& [asic_id, _] : asic_map) {
            mesh_asic_ids[mesh_id].insert(asic_id);
        }
    }

    for (const auto& [mesh_id, mesh_asics] : mesh_asic_ids) {
        auto z_channels = std::unordered_set<uint8_t>{8, 9};
        auto cluster_type = tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type();

        auto get_local_adjacents = [&](tt::tt_metal::AsicID asic_id,
                                       const std::unordered_set<tt::tt_metal::AsicID>& mesh_asics) {
            std::vector<tt::tt_metal::AsicID> adjacents;

            for (const auto& neighbor : physical_system_descriptor.get_asic_neighbors(asic_id)) {
                // Skip self-connections
                if (neighbor == asic_id) {
                    continue;
                }
                // Make sure that the neighbor is in the mesh
                if (mesh_asics.contains(neighbor)) {
                    // Add each neighbor multiple times based on number of ethernet connections
                    auto eth_connections = physical_system_descriptor.get_eth_connections(asic_id, neighbor);
                    for (const auto& eth_connection : eth_connections) {
                        // NOTE: IGNORE Z channels for Blackhole galaxy in intra mesh connectivity for now since
                        // they cause issues with uniform mesh mapping since topology mapper algorithm does not prefer
                        // taking the full connectivity path vs downgrading through z channels for intramesh
                        // connectivity https://github.com/tenstorrent/tt-metal/issues/31846
                        // This is required until we can encode a preference for links.
                        // To support use-cases where a big-mesh must be built over Z Links, a workaround has been added
                        // to allow cross host intramesh connections to be spawned over cross host Z Links.
                        // See: https://github.com/tenstorrent/tt-metal/issues/31846#issuecomment-3644965757
                        bool cross_host_connection =
                            physical_system_descriptor.is_cross_host_eth_link(asic_id, eth_connection.src_chan);
                        TT_ASSERT(
                            cross_host_connection ==
                                physical_system_descriptor.is_cross_host_eth_link(neighbor, eth_connection.dst_chan),
                            "Expected both ethernet endpoints to be mapped as cross host connections.");
                        if (!cross_host_connection && cluster_type == tt::tt_metal::ClusterType::BLACKHOLE_GALAXY &&
                            (z_channels.contains(eth_connection.src_chan) ||
                             z_channels.contains(eth_connection.dst_chan))) {
                            continue;
                        }
                        adjacents.push_back(neighbor);
                    }
                }
            }
            return adjacents;
        };

        PhysicalAdjacencyMap physical_adjacency_map;
        for (const auto& asic_id : mesh_asics) {
            physical_adjacency_map[asic_id] = get_local_adjacents(asic_id, mesh_asics);
        }
        adjacency_map[mesh_id] = physical_adjacency_map;
    }

    return adjacency_map;
}

}  // namespace tt::tt_metal::experimental::tt_fabric
