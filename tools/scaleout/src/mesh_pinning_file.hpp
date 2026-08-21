// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <yaml-cpp/yaml.h>

// Optional Phase 1 mesh pinning YAML. Unlisted meshes stay auto-mapped.
//
//   mesh_pinnings:
//     - mesh_id: 0
//       host: host-A
//       TT_VISIBLE_DEVICES: "0,1,2,3"  # optional

struct MeshPinning {
    int mesh_id = -1;
    std::string host;
    std::optional<std::string> tt_visible_devices;
};

inline std::vector<MeshPinning> parse_mesh_pinning_file(const std::string& path) {
    YAML::Node root;
    try {
        root = YAML::LoadFile(path);
    } catch (const YAML::Exception& e) {
        throw std::runtime_error(fmt::format("Failed to parse mesh pinning file {}: {}", path, e.what()));
    }

    if (!root.IsMap() || !root["mesh_pinnings"]) {
        throw std::runtime_error(
            fmt::format("Mesh pinning file {} must contain a top-level 'mesh_pinnings' key", path));
    }
    const YAML::Node& entries = root["mesh_pinnings"];
    if (!entries.IsSequence()) {
        throw std::runtime_error(fmt::format("'mesh_pinnings' in {} must be a sequence", path));
    }

    static const std::set<std::string> allowed_keys = {"mesh_id", "host", "TT_VISIBLE_DEVICES"};

    std::vector<MeshPinning> pinnings;
    std::set<int> seen_mesh_ids;

    for (std::size_t i = 0; i < entries.size(); ++i) {
        const YAML::Node& entry = entries[i];
        if (!entry.IsMap()) {
            throw std::runtime_error(fmt::format("mesh_pinnings[{}] in {} must be a map", i, path));
        }

        MeshPinning pinning;
        for (const auto& kv : entry) {
            const auto key = kv.first.as<std::string>();
            if (!allowed_keys.contains(key)) {
                throw std::runtime_error(fmt::format(
                    "Unknown key '{}' in mesh_pinnings[{}] of {}. Allowed keys: mesh_id, host, TT_VISIBLE_DEVICES",
                    key,
                    i,
                    path));
            }
        }

        if (!entry["host"]) {
            throw std::runtime_error(fmt::format("mesh_pinnings[{}] in {} is missing key 'host'", i, path));
        }
        if (!entry["mesh_id"]) {
            throw std::runtime_error(fmt::format("mesh_pinnings[{}] in {} is missing key 'mesh_id'", i, path));
        }

        pinning.mesh_id = entry["mesh_id"].as<int>();
        pinning.host = entry["host"].as<std::string>();
        if (entry["TT_VISIBLE_DEVICES"]) {
            pinning.tt_visible_devices = entry["TT_VISIBLE_DEVICES"].as<std::string>();
        }

        if (pinning.host.empty()) {
            throw std::runtime_error(fmt::format("mesh_pinnings[{}] in {} has an empty 'host'", i, path));
        }
        if (pinning.mesh_id < 0) {
            throw std::runtime_error(
                fmt::format("mesh_pinnings[{}] in {} has a negative mesh_id ({})", i, path, pinning.mesh_id));
        }
        if (!seen_mesh_ids.insert(pinning.mesh_id).second) {
            throw std::runtime_error(fmt::format("Duplicate pinning for mesh_id {} in {}", pinning.mesh_id, path));
        }

        pinnings.push_back(std::move(pinning));
    }

    return pinnings;
}
