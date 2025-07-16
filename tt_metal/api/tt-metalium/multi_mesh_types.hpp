// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <tt-metalium/fabric_types.hpp>

namespace tt::tt_fabric {

// Fully specifies the location of an ethernet channel using a unique board ID and channel ID
// The board ID is programmed by SPI-ROM Firmware and is queried by reading L1.
struct EthChanDescriptor {
    uint64_t board_id = 0;  // Unique board identifier
    uint32_t chan_id = 0;   // Eth Channel ID

    bool operator==(const EthChanDescriptor& other) const {
        return board_id == other.board_id && chan_id == other.chan_id;
    }

    bool operator<(const EthChanDescriptor& other) const {
        if (board_id != other.board_id) {
            return board_id < other.board_id;
        }
        return chan_id < other.chan_id;
    }
};

// The Control Plane running on each host populates this table to track all intermesh links
// on the Mesh it manages. All hosts exchange their local table with all other hosts to map
// their intermesh links to remote meshes.
struct IntermeshLinkTable {
    // Local mesh ID
    MeshId local_mesh_id = MeshId{0};
    HostRankId local_host_rank_id = HostRankId{0};
    // Maps local eth channel to remote eth channel
    std::map<EthChanDescriptor, EthChanDescriptor> intermesh_links;
};

}  // namespace tt::tt_fabric
