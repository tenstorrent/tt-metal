// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "channel_trimming_io.hpp"

#include <tt_stl/assert.hpp>

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
    for (auto it = map_node.begin(); it != map_node.end(); ++it) {
        if (it->first.IsScalar()) {
            keys.push_back(it->first.as<std::string>());
        }
    }
    return keys;
}

}  // namespace tt::tt_fabric
