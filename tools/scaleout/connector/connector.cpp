// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "connector.hpp"

#include <cassert>
#include <vector>

namespace tt::scaleout_tools {

namespace connection_mappers {

std::vector<AsicChannelConnection> linear_mapping(
    const std::vector<AsicChannel>& start_channels, const std::vector<AsicChannel>& end_channels) {
    if (start_channels.size() != end_channels.size()) {
        throw std::runtime_error("Start and end channels must have the same size for linear mapping");
    }
    std::vector<AsicChannelConnection> port_mapping;
    port_mapping.reserve(start_channels.size());
    for (size_t i = 0; i < start_channels.size(); i++) {
        port_mapping.emplace_back(start_channels[i], end_channels[i]);
    }
    return port_mapping;
}

std::vector<AsicChannelConnection> cross_connect_mapping(
    const std::vector<AsicChannel>& start_channels, const std::vector<AsicChannel>& end_channels) {
    if (start_channels.size() != end_channels.size()) {
        throw std::runtime_error("Start and end channels must have the same size for cross connect mapping");
    }
    if (start_channels.size() % 2 != 0) {
        throw std::runtime_error("Cross connect mapping must have an even number of channels");
    }
    std::vector<AsicChannelConnection> port_mapping;
    port_mapping.reserve(start_channels.size());

    const size_t half_size = start_channels.size() / 2;
    for (size_t i = 0; i < half_size; i++) {
        port_mapping.emplace_back(start_channels[i], end_channels[half_size + i]);
    }
    for (size_t i = 0; i < half_size; i++) {
        port_mapping.emplace_back(start_channels[half_size + i], end_channels[i]);
    }
    return port_mapping;
}

}  // namespace connection_mappers

std::vector<AsicChannelConnection> get_asic_channel_connections(
    PortType port_type, const std::vector<AsicChannel>& start_channels, const std::vector<AsicChannel>& end_channels) {
    // Validate and dispatch based on port type
    switch (port_type) {
        case PortType::WARP100:
            if (start_channels.size() != 2 || end_channels.size() != 2) {
                throw std::runtime_error("WARP100 connections must have exactly 2 channels");
            }
            return connection_mappers::linear_mapping(start_channels, end_channels);

        case PortType::WARP400:
            if (start_channels.size() != 4 || end_channels.size() != 4) {
                throw std::runtime_error("WARP400 connections must have exactly 4 channels");
            }
            return connection_mappers::cross_connect_mapping(start_channels, end_channels);

        case PortType::QSFP_DD:
        case PortType::LINKING_BOARD_1:
        case PortType::LINKING_BOARD_2:
        case PortType::LINKING_BOARD_3:
        case PortType::TRACE: return connection_mappers::linear_mapping(start_channels, end_channels);
    }
    throw std::runtime_error("Invalid port type");
}

}  // namespace tt::scaleout_tools
