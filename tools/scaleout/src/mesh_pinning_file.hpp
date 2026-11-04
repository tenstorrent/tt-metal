// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cctype>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <yaml-cpp/yaml.h>

#include <tt-metalium/experimental/fabric/fabric_types.hpp>

using tt::tt_fabric::MeshHostRankId;
using tt::tt_fabric::MeshId;

// One entry of the --mesh-pinning-file: the logical (mesh_id, mesh_host_rank) must run on `hostname`
// using exactly the devices in `tt_visible_devices` (sorted, unique, non-negative).
//
//   mesh_host_rank_pinnings:
//     - mesh_id: 0
//       mesh_host_rank: 0
//       hostname: nodeA
//       TT_VISIBLE_DEVICES: "0,1,2,3"
//
// Logical host ranks absent from the file stay automatic.
struct MeshHostRankPinning {
    MeshId mesh_id{0};
    MeshHostRankId mesh_host_rank{0};
    std::string hostname;
    std::vector<int> tt_visible_devices;
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

inline std::vector<MeshHostRankPinning> parse_mesh_pinning_file(const std::string& path) {
    YAML::Node root;
    try {
        root = YAML::LoadFile(path);
    } catch (const YAML::Exception& e) {
        throw std::runtime_error(fmt::format("Failed to parse mesh pinning file {}: {}", path, e.what()));
    }

    if (!root.IsMap() || !root["mesh_host_rank_pinnings"]) {
        throw std::runtime_error(
            fmt::format("Mesh pinning file {} must contain a top-level 'mesh_host_rank_pinnings' key", path));
    }
    mesh_pinning_detail::reject_unknown_keys(root, {"mesh_host_rank_pinnings"}, "top level", path);
    const YAML::Node& entries = root["mesh_host_rank_pinnings"];
    if (!entries.IsSequence()) {
        throw std::runtime_error(fmt::format("'mesh_host_rank_pinnings' in {} must be a sequence", path));
    }

    std::vector<MeshHostRankPinning> pinnings;
    std::set<std::pair<MeshId, MeshHostRankId>> seen_mesh_host_ranks;
    std::map<std::string, std::set<int>> used_devices_by_host;
    for (std::size_t index = 0; index < entries.size(); ++index) {
        const YAML::Node& entry = entries[index];
        const std::string context = fmt::format("mesh_host_rank_pinnings[{}]", index);
        if (!entry.IsMap()) {
            throw std::runtime_error(fmt::format("{} in {} must be a map", context, path));
        }
        mesh_pinning_detail::reject_unknown_keys(
            entry, {"mesh_id", "mesh_host_rank", "hostname", "TT_VISIBLE_DEVICES"}, context, path);
        if (!entry["mesh_id"] || !entry["mesh_host_rank"] || !entry["hostname"] || !entry["TT_VISIBLE_DEVICES"]) {
            throw std::runtime_error(fmt::format(
                "{} in {} requires mesh_id, mesh_host_rank, hostname and TT_VISIBLE_DEVICES", context, path));
        }

        const int mesh_id = entry["mesh_id"].as<int>();
        const int mesh_host_rank = entry["mesh_host_rank"].as<int>();
        if (mesh_id < 0 || mesh_host_rank < 0) {
            throw std::runtime_error(fmt::format("{} in {} requires non-negative IDs", context, path));
        }

        MeshHostRankPinning pinning;
        pinning.mesh_id = MeshId{static_cast<uint32_t>(mesh_id)};
        pinning.mesh_host_rank = MeshHostRankId{static_cast<uint32_t>(mesh_host_rank)};
        if (!seen_mesh_host_ranks.insert({pinning.mesh_id, pinning.mesh_host_rank}).second) {
            throw std::runtime_error(
                fmt::format("Duplicate mesh_id {} mesh_host_rank {} in {}", mesh_id, mesh_host_rank, path));
        }

        pinning.hostname = entry["hostname"].as<std::string>();
        if (pinning.hostname.empty()) {
            throw std::runtime_error(fmt::format("{} in {} has an empty hostname", context, path));
        }
        pinning.tt_visible_devices =
            mesh_pinning_detail::parse_visible_devices(entry["TT_VISIBLE_DEVICES"], context, path);
        auto& used_devices = used_devices_by_host[pinning.hostname];
        for (const int device : pinning.tt_visible_devices) {
            if (!used_devices.insert(device).second) {
                throw std::runtime_error(fmt::format(
                    "TT_VISIBLE_DEVICES {} on host '{}' is used by more than one pinning in {}",
                    device,
                    pinning.hostname,
                    path));
            }
        }
        pinnings.push_back(std::move(pinning));
    }
    return pinnings;
}
