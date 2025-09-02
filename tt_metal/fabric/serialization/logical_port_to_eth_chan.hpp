// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <string>
#include <vector>
#include <tt-metalium/multi_mesh_types.hpp>

namespace tt::tt_fabric {

/**
 * Serializes std::map<std::string, EthChanDescriptor> to FlatBuffers bytes
 */
std::vector<uint8_t> serialize_logical_port_to_eth_chan_to_bytes(
    const std::map<std::string, EthChanDescriptor>& logical_port_to_eth_chan);

/**
 * Deserializes FlatBuffers bytes to std::map<std::string, EthChanDescriptor>
 */
std::map<std::string, EthChanDescriptor> deserialize_logical_port_to_eth_chan_from_bytes(
    const std::vector<uint8_t>& data);

}  // namespace tt::tt_fabric
