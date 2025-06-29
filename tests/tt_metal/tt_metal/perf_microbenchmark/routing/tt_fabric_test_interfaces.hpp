// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <unordered_map>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_graph.hpp>
#include <tt-metalium/fabric_edm_packet_header.hpp>
#include <random>

namespace tt {
namespace tt_fabric {
class FabricNodeId;
}  // namespace tt_fabric
}  // namespace tt

namespace tt::tt_fabric {
namespace fabric_tests {

using MeshCoordinate = tt::tt_metal::distributed::MeshCoordinate;

class IDeviceInfoProvider {
public:
    virtual ~IDeviceInfoProvider() = default;
    virtual FabricNodeId get_fabric_node_id(chip_id_t physical_chip_id) const = 0;
    virtual FabricNodeId get_fabric_node_id(const MeshCoordinate& device_coord) const = 0;
    virtual MeshCoordinate get_device_coord(const FabricNodeId& node_id) const = 0;
    virtual uint32_t get_worker_noc_encoding(const MeshCoordinate& device_coord, CoreCoord logical_core) const = 0;
    virtual uint32_t get_worker_noc_encoding(const FabricNodeId& node_id, CoreCoord logical_core) const = 0;
    virtual CoreCoord get_worker_grid_size() const = 0;
    virtual uint32_t get_worker_id(const FabricNodeId& node_id, CoreCoord logical_core) const = 0;
    virtual std::vector<FabricNodeId> get_all_node_ids() const = 0;
    virtual uint32_t get_l1_unreserved_base(const FabricNodeId& node_id) const = 0;
    virtual uint32_t get_l1_unreserved_size(const FabricNodeId& node_id) const = 0;
    virtual uint32_t get_l1_alignment() const = 0;
    virtual uint32_t get_max_payload_size_bytes() const = 0;
    virtual bool is_2d_fabric() const = 0;
    virtual bool use_dynamic_routing() const = 0;
};

class IRouteManager {
public:
    virtual ~IRouteManager() = default;
    virtual std::vector<FabricNodeId> get_dst_node_ids_from_hops(
        FabricNodeId src_node_id,
        std::unordered_map<RoutingDirection, uint32_t>& hops,
        ChipSendType chip_send_type) const = 0;
    virtual std::unordered_map<RoutingDirection, uint32_t> get_hops_to_chip(
        FabricNodeId src_node_id, FabricNodeId dst_node_id) const = 0;
    virtual bool are_devices_linear(const std::vector<FabricNodeId>& node_ids) const = 0;
    virtual std::vector<std::pair<FabricNodeId, FabricNodeId>> get_all_to_all_unicast_pairs() const = 0;
    virtual std::vector<std::pair<FabricNodeId, FabricNodeId>> get_full_device_random_pairs(
        std::mt19937& gen) const = 0;
    virtual std::unordered_map<RoutingDirection, uint32_t> get_full_mcast_hops(
        const FabricNodeId& src_node_id) const = 0;
    virtual std::vector<std::unordered_map<RoutingDirection, uint32_t>> split_multicast_hops(
        const std::unordered_map<RoutingDirection, uint32_t>& hops) const = 0;
    virtual FabricNodeId get_random_unicast_destination(FabricNodeId src_node_id, std::mt19937& gen) const = 0;
    virtual RoutingDirection get_forwarding_direction(
        const FabricNodeId& src_node_id, const FabricNodeId& dst_node_id) const = 0;
    virtual RoutingDirection get_forwarding_direction(
        const std::unordered_map<RoutingDirection, uint32_t>& hops) const = 0;
    virtual std::vector<uint32_t> get_forwarding_link_indices_in_direction(
        const FabricNodeId& src_node_id, const RoutingDirection& direction) const = 0;
    virtual std::vector<uint32_t> get_forwarding_link_indices_in_direction(
        const FabricNodeId& src_node_id, const FabricNodeId& dst_node_id, const RoutingDirection& direction) const = 0;
    virtual FabricNodeId get_neighbor_node_id(
        const FabricNodeId& src_node_id, const RoutingDirection& direction) const = 0;
};

}  // namespace fabric_tests
}  // namespace tt::tt_fabric
