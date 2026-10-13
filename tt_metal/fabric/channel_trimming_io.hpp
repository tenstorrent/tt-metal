// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
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

// --- Multi-rank capture file conventions -------------------------------------
//
// Capture files are per-rank: each rank exports only its local meshes and there is no
// cross-rank aggregation at teardown. Files live in a dedicated directory, suffixed by
// world rank:
//
//   <logs_dir>/generated/reports/channel_trimming_capture/rank_<N>.yaml
//
// Import is symmetric: a rank reads back only its own file (override lookups are
// per-local-router, so full-system visibility is never required). TT_METAL_FABRIC_TRIMMING_PROFILE
// therefore accepts either a regular file (legacy single-rank) or a directory (multi-rank).

// World rank used to disambiguate per-rank capture files; 0 when the distributed
// context is not initialized (single-process runs).
int get_channel_trimming_capture_rank();

// <logs_dir>/generated/reports/channel_trimming_capture/
std::filesystem::path get_channel_trimming_capture_dir(const std::string& logs_dir);

// Capture file for this process's rank: <capture_dir>/rank_<N>.yaml
std::filesystem::path get_channel_trimming_capture_path(const std::string& logs_dir);

// Resolve a user-supplied trimming profile path:
//   - regular file → returned as-is (legacy single-rank behavior);
//   - directory    → <dir>/rank_<N>.yaml for this process's rank; TT_FATAL if missing.
// Capture and replay must use the same world size and mesh→rank binding.
std::filesystem::path resolve_channel_trimming_profile_path(const std::string& profile_path);

}  // namespace tt::tt_fabric
