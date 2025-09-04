// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <board/board.hpp>

namespace tt::scaleout_tools {

// Type alias for connection pairs
using AsicChannelConnection = std::pair<AsicChannel, AsicChannel>;

// Main function to get ASIC channel connections based on port type
std::vector<AsicChannelConnection> get_asic_channel_connections(
    PortType port_type, const std::vector<AsicChannel>& start_channels, const std::vector<AsicChannel>& end_channels);

}  // namespace tt::scaleout_tools
