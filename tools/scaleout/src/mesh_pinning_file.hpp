// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <yaml-cpp/yaml.h>

using tt::tt_fabric::MeshHostRankId;
using tt::tt_fabric::MeshId;

struct HostPlacement {
    std::string hostname;
    std::vector<int> tt_visible_devices;
};

// A many-to-many mesh-host-rank pinning group. Any of `mesh_host_ranks` may map to any of
// `host_placements` (all-to-all); selected placements are injective, so distinct logical pairs
// land on distinct hostname + TT_VISIBLE_DEVICES pools. A 1:1 group is the classic exact pin.
struct MeshHostRankPinningGroup {
    std::vector<std::pair<MeshId, MeshHostRankId>> mesh_host_ranks;
    std::vector<HostPlacement> host_placements;
};

namespace mesh_pinning_detail {

inline void reject_unknown_keys(
    const YAML::Node& node,
    const std::set<std::string>& allowed_keys,
    const std::string& context,
    const std::string& path) {
    for (const auto& kv : node) {
        const std::string key = kv.first.as<std::string>();
        if (!allowed_keys.contains(key)) {
            throw std::runtime_error(fmt::format("Unknown key '{}' in {} of {}", key, context, path));
        }
    }
}

inline std::vector<int> parse_visible_devices(
    const YAML::Node& node, const std::string& context, const std::string& path) {
    const std::string value = node.as<std::string>();
    std::vector<int> devices;
    std::size_t begin = 0;
    while (begin <= value.size()) {
        const std::size_t comma = value.find(',', begin);
        const std::size_t end = comma == std::string::npos ? value.size() : comma;
        std::string token = value.substr(begin, end - begin);
        token.erase(
            token.begin(), std::find_if(token.begin(), token.end(), [](unsigned char c) { return !std::isspace(c); }));
        token.erase(
            std::find_if(token.rbegin(), token.rend(), [](unsigned char c) { return !std::isspace(c); }).base(),
            token.end());
        if (token.empty()) {
            throw std::runtime_error(fmt::format("{} in {} contains an empty TT_VISIBLE_DEVICES entry", context, path));
        }
        std::size_t parsed_chars = 0;
        int device = -1;
        try {
            device = std::stoi(token, &parsed_chars);
        } catch (const std::exception&) {
            throw std::runtime_error(
                fmt::format("{} in {} has invalid TT_VISIBLE_DEVICES value '{}'", context, path, value));
        }
        if (parsed_chars != token.size() || device < 0) {
            throw std::runtime_error(
                fmt::format("{} in {} has invalid TT_VISIBLE_DEVICES value '{}'", context, path, value));
        }
        devices.push_back(device);
        if (comma == std::string::npos) {
            break;
        }
        begin = comma + 1;
    }
    std::sort(devices.begin(), devices.end());
    if (std::adjacent_find(devices.begin(), devices.end()) != devices.end()) {
        throw std::runtime_error(
            fmt::format("{} in {} lists a TT_VISIBLE_DEVICES value more than once", context, path));
    }
    return devices;
}

}  // namespace mesh_pinning_detail

inline std::vector<MeshHostRankPinningGroup> parse_mesh_pinning_file(const std::string& path) {
    YAML::Node root;
    try {
        root = YAML::LoadFile(path);
    } catch (const YAML::Exception& e) {
        throw std::runtime_error(fmt::format("Failed to parse mesh pinning file {}: {}", path, e.what()));
    }

    if (!root.IsMap() || !root["mesh_host_pinnings"]) {
        throw std::runtime_error(
            fmt::format("Mesh pinning file {} must contain a top-level 'mesh_host_pinnings' key", path));
    }
    mesh_pinning_detail::reject_unknown_keys(root, {"mesh_host_pinnings"}, "top level", path);
    const YAML::Node& entries = root["mesh_host_pinnings"];
    if (!entries.IsSequence()) {
        throw std::runtime_error(fmt::format("'mesh_host_pinnings' in {} must be a sequence", path));
    }

    std::vector<MeshHostRankPinningGroup> groups;
    std::set<std::pair<MeshId, MeshHostRankId>> seen_mesh_host_ranks;
    std::map<std::string, std::set<int>> used_devices_by_host;
    for (std::size_t group_index = 0; group_index < entries.size(); ++group_index) {
        const YAML::Node& entry = entries[group_index];
        const std::string group_context = fmt::format("mesh_host_pinnings[{}]", group_index);
        if (!entry.IsMap()) {
            throw std::runtime_error(fmt::format("{} in {} must be a map", group_context, path));
        }
        mesh_pinning_detail::reject_unknown_keys(entry, {"mesh_host_ranks", "host_placements"}, group_context, path);
        if (!entry["mesh_host_ranks"] || !entry["mesh_host_ranks"].IsSequence() ||
            entry["mesh_host_ranks"].size() == 0) {
            throw std::runtime_error(
                fmt::format("{} in {} must have a non-empty mesh_host_ranks sequence", group_context, path));
        }
        if (!entry["host_placements"] || !entry["host_placements"].IsSequence() ||
            entry["host_placements"].size() == 0) {
            throw std::runtime_error(
                fmt::format("{} in {} must have a non-empty host_placements sequence", group_context, path));
        }

        MeshHostRankPinningGroup group;
        const YAML::Node& logical_entries = entry["mesh_host_ranks"];
        for (std::size_t logical_index = 0; logical_index < logical_entries.size(); ++logical_index) {
            const YAML::Node& logical = logical_entries[logical_index];
            const std::string context = fmt::format("{}.mesh_host_ranks[{}]", group_context, logical_index);
            if (!logical.IsMap()) {
                throw std::runtime_error(fmt::format("{} in {} must be a map", context, path));
            }
            mesh_pinning_detail::reject_unknown_keys(logical, {"mesh_id", "mesh_host_rank"}, context, path);
            if (!logical["mesh_id"] || !logical["mesh_host_rank"]) {
                throw std::runtime_error(fmt::format("{} in {} requires mesh_id and mesh_host_rank", context, path));
            }
            const int mesh_id = logical["mesh_id"].as<int>();
            const int mesh_host_rank = logical["mesh_host_rank"].as<int>();
            if (mesh_id < 0 || mesh_host_rank < 0) {
                throw std::runtime_error(fmt::format("{} in {} requires non-negative IDs", context, path));
            }
            const std::pair<MeshId, MeshHostRankId> logical_pair{
                MeshId{static_cast<uint32_t>(mesh_id)}, MeshHostRankId{static_cast<uint32_t>(mesh_host_rank)}};
            if (!seen_mesh_host_ranks.insert(logical_pair).second) {
                throw std::runtime_error(
                    fmt::format("Duplicate mesh_id {} mesh_host_rank {} in {}", mesh_id, mesh_host_rank, path));
            }
            group.mesh_host_ranks.push_back(logical_pair);
        }

        const YAML::Node& placement_entries = entry["host_placements"];
        for (std::size_t placement_index = 0; placement_index < placement_entries.size(); ++placement_index) {
            const YAML::Node& placement = placement_entries[placement_index];
            const std::string context = fmt::format("{}.host_placements[{}]", group_context, placement_index);
            if (!placement.IsMap()) {
                throw std::runtime_error(fmt::format("{} in {} must be a map", context, path));
            }
            mesh_pinning_detail::reject_unknown_keys(placement, {"hostname", "TT_VISIBLE_DEVICES"}, context, path);
            if (!placement["hostname"] || !placement["TT_VISIBLE_DEVICES"]) {
                throw std::runtime_error(
                    fmt::format("{} in {} requires hostname and TT_VISIBLE_DEVICES", context, path));
            }
            HostPlacement host_placement;
            host_placement.hostname = placement["hostname"].as<std::string>();
            if (host_placement.hostname.empty()) {
                throw std::runtime_error(fmt::format("{} in {} has an empty hostname", context, path));
            }
            host_placement.tt_visible_devices =
                mesh_pinning_detail::parse_visible_devices(placement["TT_VISIBLE_DEVICES"], context, path);
            auto& used_devices = used_devices_by_host[host_placement.hostname];
            for (const int device : host_placement.tt_visible_devices) {
                if (!used_devices.insert(device).second) {
                    throw std::runtime_error(fmt::format(
                        "TT_VISIBLE_DEVICES {} on host '{}' is used by more than one placement in {}",
                        device,
                        host_placement.hostname,
                        path));
                }
            }
            group.host_placements.push_back(std::move(host_placement));
        }

        if (group.host_placements.size() < group.mesh_host_ranks.size()) {
            throw std::runtime_error(fmt::format(
                "{} in {} has {} logical mesh host ranks but only {} host placements",
                group_context,
                path,
                group.mesh_host_ranks.size(),
                group.host_placements.size()));
        }
        groups.push_back(std::move(group));
    }
    return groups;
}
