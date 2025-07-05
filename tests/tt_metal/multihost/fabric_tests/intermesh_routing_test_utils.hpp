// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <map>

#include <tt-metalium/device.hpp>
#include "umd/device/types/cluster_descriptor_types.h"
#include "tests/tt_metal/tt_fabric/common/fabric_fixture.hpp"

namespace tt::tt_fabric {
namespace fabric_router_tests {

namespace multihost_utils {

void RandomizedInterMeshUnicast(BaseFabricFixture* fixture);

void InterMeshLineMcast(
    BaseFabricFixture* fixture,
    FabricNodeId mcast_sender_node,
    FabricNodeId mcast_start_node,
    const std::vector<McastRoutingInfo>& mcast_routing_info,
    const std::vector<FabricNodeId>& mcast_group_node_ids);

std::map<FabricNodeId, chip_id_t> get_physical_chip_mapping_from_eth_coords_mapping(
    const std::vector<std::vector<eth_coord_t>>& mesh_graph_eth_coords, uint32_t local_mesh_id);

}  // namespace multihost_utils

}  // namespace fabric_router_tests
}  // namespace tt::tt_fabric
