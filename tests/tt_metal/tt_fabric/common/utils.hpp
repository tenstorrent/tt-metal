// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "fabric_fixture.hpp"
#include <tt-metalium/device_pool.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/test_utils/env_vars.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include "impl/context/metal_context.hpp"

namespace tt::tt_fabric {
namespace fabric_router_tests {

bool find_device_with_neighbor_in_multi_direction(
    BaseFabricFixture* fixture,
    FabricNodeId& src_fabric_node_id,
    std::unordered_map<RoutingDirection, std::vector<FabricNodeId>>& dst_fabric_node_ids_by_dir,
    chip_id_t& src_physical_device_id,
    std::unordered_map<RoutingDirection, std::vector<chip_id_t>>& dst_physical_device_ids_by_dir,
    const std::unordered_map<RoutingDirection, uint32_t>& mcast_hops,
    std::optional<RoutingDirection> incoming_direction = std::nullopt);

bool find_device_with_neighbor_in_direction(
    BaseFabricFixture* fixture,
    FabricNodeId& src_fabric_node_id,
    FabricNodeId& dst_fabric_node_id,
    chip_id_t& src_physical_device_id,
    chip_id_t& dst_physical_device_id,
    RoutingDirection direction);

std::map<FabricNodeId, chip_id_t> get_physical_chip_mapping_from_eth_coords_mapping(
    const std::vector<std::vector<eth_coord_t>>& mesh_graph_eth_coords);

}  // namespace fabric_router_tests
}  // namespace tt::tt_fabric
