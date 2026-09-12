// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "channel_trimming_io.hpp"

#include <tt_stl/assert.hpp>
#include <tt_stl/fmt.hpp>

#include <tt-metalium/distributed_context.hpp>

namespace tt::tt_fabric {

ChipId parse_chip_key(const std::string& key) {
    constexpr auto prefix = std::string_view("chip_");
    TT_FATAL(
        key.size() > prefix.size() && key.substr(0, prefix.size()) == prefix,
        "Invalid chip key in trimming profile: '{}'",
        key);
    return static_cast<ChipId>(std::stoi(key.substr(prefix.size())));
}

chan_id_t parse_eth_channel_key(const std::string& key) {
    constexpr auto prefix = std::string_view("eth_channel_");
    TT_FATAL(
        key.size() > prefix.size() && key.substr(0, prefix.size()) == prefix,
        "Invalid eth channel key in trimming profile: '{}'",
        key);
    return static_cast<chan_id_t>(std::stoi(key.substr(prefix.size())));
}

uint16_t parse_hex_bitfield(const std::string& str) { return static_cast<uint16_t>(std::stoul(str, nullptr, 16)); }

std::vector<std::string> collect_map_keys(const YAML::Node& map_node) {
    std::vector<std::string> keys;
    keys.reserve(map_node.size());
    for (auto it = map_node.begin(); it != map_node.end(); ++it) {
        if (it->first.IsScalar()) {
            keys.push_back(it->first.as<std::string>());
        }
    }
    return keys;
}

int get_channel_trimming_capture_rank() {
    namespace multihost = tt::tt_metal::distributed::multihost;
    if (!multihost::DistributedContext::is_initialized()) {
        return 0;
    }
    return *multihost::DistributedContext::get_current_world()->rank();
}

std::filesystem::path get_channel_trimming_capture_dir(const std::string& logs_dir) {
    return std::filesystem::path(logs_dir) / "generated" / "reports" / "channel_trimming_capture";
}

std::filesystem::path get_channel_trimming_capture_path(const std::string& logs_dir) {
    return get_channel_trimming_capture_dir(logs_dir) /
           fmt::format("rank_{}.yaml", get_channel_trimming_capture_rank());
}

std::filesystem::path resolve_channel_trimming_profile_path(const std::string& profile_path) {
    if (std::filesystem::is_regular_file(profile_path)) {
        return profile_path;
    }
    TT_FATAL(
        std::filesystem::is_directory(profile_path),
        "Channel trimming profile path '{}' is neither a regular file nor a directory",
        profile_path);
    const int world_rank = get_channel_trimming_capture_rank();
    auto rank_file = std::filesystem::path(profile_path) / fmt::format("rank_{}.yaml", world_rank);
    TT_FATAL(
        std::filesystem::exists(rank_file),
        "Channel trimming profile directory '{}' has no capture for rank {} (expected '{}'). "
        "Capture and replay must use the same world size and mesh-to-rank binding.",
        profile_path,
        world_rank,
        rank_file);
    return rank_file;
}

}  // namespace tt::tt_fabric
