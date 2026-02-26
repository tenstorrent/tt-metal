// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <umd/device/types/cluster_descriptor_types.hpp>  // ChipId
#include <hostdevcommon/fabric_common.h>                  // chan_id_t
#include <yaml-cpp/yaml.h>

namespace tt::tt_fabric {

// Parse "chip_N" → N
ChipId parse_chip_key(const std::string& key);

// Parse "eth_channel_N" → N
chan_id_t parse_eth_channel_key(const std::string& key);

// Parse hex string like "0x001F" → uint16_t
uint16_t parse_hex_bitfield(const std::string& str);

// Collect all scalar keys from a YAML map node into a vector.
// Must be done before any operator[] lookups on the map's children, because
// yaml-cpp's operator[] mutates the underlying node (inserting null entries)
// which corrupts in-progress iteration.
std::vector<std::string> collect_map_keys(const YAML::Node& map_node);

}  // namespace tt::tt_fabric
