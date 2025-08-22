// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <tt-metalium/fabric_types.hpp>
#include <umd/device/tt_cluster_descriptor.h>

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
    MeshHostRankId local_host_rank_id = MeshHostRankId{0};
    // Maps local eth channel to remote eth channel
    std::map<EthChanDescriptor, EthChanDescriptor> intermesh_links;
};

struct EthConnectivityDescriptor {
    std::string host_name = "";
    std::map<tt::tt_fabric::EthChanDescriptor, tt::tt_fabric::EthChanDescriptor> local_eth_connections = {};
    std::map<tt::tt_fabric::EthChanDescriptor, std::pair<std::string, tt::tt_fabric::EthChanDescriptor>>
        remote_eth_connections = {};
};

struct ASICDescriptor {
    uint64_t unique_id;
    uint32_t tray_id;
    uint32_t n_id;
    BoardType board_type;

    bool operator==(const ASICDescriptor& other) const {
        return unique_id == other.unique_id && tray_id == other.tray_id;
    }
};

struct SystemDescriptor {
    std::unordered_map<std::string, std::vector<ASICDescriptor>> asic_ids;
    std::vector<EthConnectivityDescriptor> eth_connectivity_descs;
};

}  // namespace tt::tt_fabric
