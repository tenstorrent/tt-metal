// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <tuple>
#include <string>
#include <utility>

namespace tt::tt_fabric {

/**
 * Serializes std::vector<std::tuple<std::pair<uint32_t, std::string>, std::pair<uint32_t, std::string>>> to FlatBuffers
 * bytes
 */
std::vector<uint8_t> serialize_connections_table_to_bytes(
    const std::vector<std::tuple<std::pair<uint32_t, std::string>, std::pair<uint32_t, std::string>>>& connections);

/**
 * Deserializes FlatBuffers bytes to std::vector<std::tuple<std::pair<uint32_t, std::string>, std::pair<uint32_t,
 * std::string>>>
 */
std::vector<std::tuple<std::pair<uint32_t, std::string>, std::pair<uint32_t, std::string>>>
deserialize_connections_table_from_bytes(const std::vector<uint8_t>& data);

}  // namespace tt::tt_fabric
