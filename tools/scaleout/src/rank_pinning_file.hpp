// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstddef>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <yaml-cpp/yaml.h>

// Optional Phase 1 rank pinning YAML. Unlisted ranks stay auto-mapped.
//
//   rank_pinnings:
//     - rank: 0
//       host: host-A
//       env_overrides:
//         TT_VISIBLE_DEVICES: "0,1,2,3"   # optional
//     - mesh_id: 1
//       mesh_host_rank: 0
//       host: host-B
//
// Each entry uses exactly one of rank: or mesh_id:+mesh_host_rank:.

// Either `rank`, or both `mesh_id` and `mesh_host_rank`.
struct RankPinning {
    std::optional<int> rank;
    std::optional<int> mesh_id;
    std::optional<int> mesh_host_rank;
    std::string host;
    std::map<std::string, std::string> env_overrides;
};

struct ResolvedRankPinning {
    int rank = -1;  // global MPI rank, for error messages
    int mesh_id = 0;
    int mesh_host_rank = 0;
    std::string host;
    std::map<std::string, std::string> env_overrides;
};

// Index i is global rank i. Matches extract_rank_bindings primary sort (mesh_id, then mesh_host_rank).
using MeshHostRankOrder = std::vector<std::pair<int, int>>;

inline std::vector<RankPinning> parse_rank_pinning_file(const std::string& path) {
    YAML::Node root;
    try {
        root = YAML::LoadFile(path);
    } catch (const YAML::Exception& e) {
        throw std::runtime_error(fmt::format("Failed to parse rank pinning file {}: {}", path, e.what()));
    }

    if (!root.IsMap() || !root["rank_pinnings"]) {
        throw std::runtime_error(
            fmt::format("Rank pinning file {} must contain a top-level 'rank_pinnings' key", path));
    }
    const YAML::Node& entries = root["rank_pinnings"];
    if (!entries.IsSequence()) {
        throw std::runtime_error(fmt::format("'rank_pinnings' in {} must be a sequence", path));
    }

    static const std::set<std::string> allowed_keys = {"rank", "mesh_id", "mesh_host_rank", "host", "env_overrides"};

    std::vector<RankPinning> pinnings;
    std::set<int> seen_ranks;
    std::set<std::pair<int, int>> seen_mesh_host_ranks;

    for (std::size_t i = 0; i < entries.size(); ++i) {
        const YAML::Node& entry = entries[i];
        if (!entry.IsMap()) {
            throw std::runtime_error(fmt::format("rank_pinnings[{}] in {} must be a map", i, path));
        }

        RankPinning pinning;
        for (const auto& kv : entry) {
            const auto key = kv.first.as<std::string>();
            if (!allowed_keys.contains(key)) {
                throw std::runtime_error(fmt::format(
                    "Unknown key '{}' in rank_pinnings[{}] of {}. Allowed keys: rank, mesh_id, mesh_host_rank, "
                    "host, env_overrides",
                    key,
                    i,
                    path));
            }
        }

        if (entry["rank"]) {
            pinning.rank = entry["rank"].as<int>();
        }
        if (entry["mesh_id"]) {
            pinning.mesh_id = entry["mesh_id"].as<int>();
        }
        if (entry["mesh_host_rank"]) {
            pinning.mesh_host_rank = entry["mesh_host_rank"].as<int>();
        }
        if (entry["host"]) {
            pinning.host = entry["host"].as<std::string>();
        }
        if (entry["env_overrides"]) {
            const YAML::Node& env = entry["env_overrides"];
            if (!env.IsMap()) {
                throw std::runtime_error(
                    fmt::format("'env_overrides' in rank_pinnings[{}] of {} must be a map", i, path));
            }
            for (const auto& kv : env) {
                pinning.env_overrides[kv.first.as<std::string>()] = kv.second.as<std::string>();
            }
        }

        if (pinning.host.empty()) {
            throw std::runtime_error(fmt::format("rank_pinnings[{}] in {} is missing a non-empty 'host'", i, path));
        }

        const bool has_global_rank = pinning.rank.has_value();
        const bool has_mesh_form = pinning.mesh_id.has_value() || pinning.mesh_host_rank.has_value();
        if (has_global_rank && has_mesh_form) {
            throw std::runtime_error(fmt::format(
                "rank_pinnings[{}] in {} sets both 'rank' and 'mesh_id'/'mesh_host_rank'. Use exactly one form",
                i,
                path));
        }
        if (!has_global_rank && !has_mesh_form) {
            throw std::runtime_error(fmt::format(
                "rank_pinnings[{}] in {} must set either 'rank' or both 'mesh_id' and 'mesh_host_rank'", i, path));
        }
        if (has_mesh_form && !(pinning.mesh_id.has_value() && pinning.mesh_host_rank.has_value())) {
            throw std::runtime_error(fmt::format(
                "rank_pinnings[{}] in {} must set both 'mesh_id' and 'mesh_host_rank' when using the mesh form",
                i,
                path));
        }

        if (has_global_rank) {
            if (pinning.rank.value() < 0) {
                throw std::runtime_error(
                    fmt::format("rank_pinnings[{}] in {} has a negative rank ({})", i, path, pinning.rank.value()));
            }
            if (!seen_ranks.insert(pinning.rank.value()).second) {
                throw std::runtime_error(
                    fmt::format("Duplicate pinning for rank {} in {}", pinning.rank.value(), path));
            }
        } else {
            if (pinning.mesh_id.value() < 0 || pinning.mesh_host_rank.value() < 0) {
                throw std::runtime_error(fmt::format(
                    "rank_pinnings[{}] in {} has a negative mesh_id ({}) or mesh_host_rank ({})",
                    i,
                    path,
                    pinning.mesh_id.value(),
                    pinning.mesh_host_rank.value()));
            }
            const auto key = std::make_pair(pinning.mesh_id.value(), pinning.mesh_host_rank.value());
            if (!seen_mesh_host_ranks.insert(key).second) {
                throw std::runtime_error(fmt::format(
                    "Duplicate pinning for mesh_id {} mesh_host_rank {} in {}", key.first, key.second, path));
            }
        }

        pinnings.push_back(std::move(pinning));
    }

    return pinnings;
}

inline std::vector<ResolvedRankPinning> resolve_rank_pinnings(
    const std::vector<RankPinning>& pinnings, const MeshHostRankOrder& rank_order) {
    std::map<std::pair<int, int>, int> mesh_host_rank_to_global_rank;
    for (std::size_t i = 0; i < rank_order.size(); ++i) {
        mesh_host_rank_to_global_rank.emplace(rank_order[i], static_cast<int>(i));
    }

    std::vector<ResolvedRankPinning> resolved;
    std::map<std::pair<int, int>, std::string> claimed;

    for (const auto& pinning : pinnings) {
        ResolvedRankPinning out;
        out.host = pinning.host;
        out.env_overrides = pinning.env_overrides;

        if (pinning.rank.has_value()) {
            const int rank = pinning.rank.value();
            if (static_cast<std::size_t>(rank) >= rank_order.size()) {
                throw std::runtime_error(fmt::format(
                    "Rank pinning file pins rank {}, but the mesh graph descriptor only defines {} rank(s) "
                    "(0..{}). Check the MGD host_topology dims.",
                    rank,
                    rank_order.size(),
                    rank_order.empty() ? 0 : rank_order.size() - 1));
            }
            out.rank = rank;
            out.mesh_id = rank_order[rank].first;
            out.mesh_host_rank = rank_order[rank].second;
        } else {
            out.mesh_id = pinning.mesh_id.value();
            out.mesh_host_rank = pinning.mesh_host_rank.value();
            const auto it = mesh_host_rank_to_global_rank.find({out.mesh_id, out.mesh_host_rank});
            if (it == mesh_host_rank_to_global_rank.end()) {
                throw std::runtime_error(fmt::format(
                    "Rank pinning file pins mesh_id {} mesh_host_rank {}, which the mesh graph descriptor does "
                    "not define",
                    out.mesh_id,
                    out.mesh_host_rank));
            }
            out.rank = it->second;
        }

        const auto key = std::make_pair(out.mesh_id, out.mesh_host_rank);
        auto [it, inserted] = claimed.emplace(key, out.host);
        if (!inserted) {
            throw std::runtime_error(fmt::format(
                "Conflicting pinnings for mesh_id {} mesh_host_rank {} (global rank {}): '{}' and '{}'",
                key.first,
                key.second,
                out.rank,
                it->second,
                out.host));
        }

        resolved.push_back(std::move(out));
    }

    return resolved;
}

inline MeshHostRankOrder build_mesh_host_rank_order(const std::map<int, std::vector<int>>& mesh_host_ranks_per_mesh) {
    MeshHostRankOrder order;
    for (const auto& [mesh_id, host_ranks] : mesh_host_ranks_per_mesh) {
        std::vector<int> sorted_host_ranks = host_ranks;
        std::sort(sorted_host_ranks.begin(), sorted_host_ranks.end());
        for (int host_rank : sorted_host_ranks) {
            order.emplace_back(mesh_id, host_rank);
        }
    }
    return order;
}
