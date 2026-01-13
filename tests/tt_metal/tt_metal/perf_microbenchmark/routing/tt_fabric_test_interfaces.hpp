// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <optional>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include "fabric/fabric_edm_packet_header.hpp"
#include <random>

namespace tt::tt_fabric {
class FabricNodeId;
}  // namespace tt::tt_fabric

namespace tt::tt_fabric::fabric_tests {

using MeshCoordinate = tt::tt_metal::distributed::MeshCoordinate;

enum class HighLevelTrafficPattern;  // Forward declaration

class IDeviceInfoProvider {
public:
    virtual ~IDeviceInfoProvider() = default;
    virtual FabricNodeId get_fabric_node_id(ChipId physical_chip_id) const = 0;
    virtual FabricNodeId get_fabric_node_id(const MeshCoordinate& device_coord) const = 0;
    virtual FabricNodeId get_fabric_node_id(MeshId mesh_id, const MeshCoordinate& device_coord) const = 0;
    virtual MeshCoordinate get_device_coord(const FabricNodeId& node_id) const = 0;
    virtual uint32_t get_worker_noc_encoding(CoreCoord logical_core) const = 0;
    virtual CoreCoord get_virtual_core_from_logical_core(CoreCoord logical_core) const = 0;
    virtual CoreCoord get_worker_grid_size() const = 0;
    virtual uint32_t get_worker_id(const FabricNodeId& node_id, CoreCoord logical_core) const = 0;
    virtual std::vector<FabricNodeId> get_local_node_ids() const = 0;
    virtual std::vector<FabricNodeId> get_global_node_ids() const = 0;
    virtual bool is_local_fabric_node_id(const FabricNodeId& id) const = 0;
    virtual uint32_t get_l1_unreserved_base() const = 0;
    virtual uint32_t get_l1_unreserved_size() const = 0;
    virtual uint32_t get_l1_alignment() const = 0;
    virtual uint32_t get_max_payload_size_bytes() const = 0;
    virtual bool is_2D_routing_enabled() const = 0;
    virtual uint32_t get_device_frequency_mhz(const FabricNodeId& device_id) const = 0;
    virtual bool is_multi_mesh() const = 0;
    virtual std::unordered_map<MeshId, std::unordered_set<MeshId>> get_mesh_adjacency_map() const = 0;

    // Data reading helpers
    virtual std::unordered_map<CoreCoord, std::vector<uint32_t>> read_buffer_from_cores(
        const MeshCoordinate& device_coord,
        const std::vector<CoreCoord>& cores,
        uint32_t address,
        uint32_t size_bytes) const = 0;
    virtual void zero_out_buffer_on_cores(
        const MeshCoordinate& device_coord,
        const std::vector<CoreCoord>& cores,
        uint32_t address,
        uint32_t size_bytes) const = 0;
    virtual void write_data_to_core(
        const MeshCoordinate& device_coord,
        const CoreCoord& cores,
        uint32_t local_args_address,
        const std::vector<uint32_t>& args) const = 0;
};

class IRouteManager {
public:
    virtual ~IRouteManager() = default;
    virtual MeshShape get_mesh_shape() const = 0;
    virtual uint32_t get_num_mesh_dims() const = 0;
    virtual bool wrap_around_mesh(FabricNodeId node) const = 0;
    virtual std::vector<FabricNodeId> get_dst_node_ids_from_hops(
        FabricNodeId src_node_id,
        std::unordered_map<RoutingDirection, uint32_t>& hops,
        ChipSendType chip_send_type) const = 0;
    virtual std::unordered_map<RoutingDirection, uint32_t> get_hops_to_chip(
        FabricNodeId src_node_id, FabricNodeId dst_node_id) const = 0;
    virtual bool are_devices_linear(const std::vector<FabricNodeId>& node_ids) const = 0;
    virtual std::vector<std::pair<FabricNodeId, FabricNodeId>> get_all_to_all_unicast_pairs() const = 0;
    virtual std::vector<std::pair<FabricNodeId, FabricNodeId>> get_all_to_one_unicast_pairs(
        uint32_t device_idx) const = 0;
    virtual std::vector<std::pair<FabricNodeId, FabricNodeId>> get_full_device_random_pairs(
        std::mt19937& gen) const = 0;
    virtual std::unordered_map<RoutingDirection, uint32_t> get_full_mcast_hops(
        const FabricNodeId& src_node_id) const = 0;
    virtual std::unordered_map<RoutingDirection, uint32_t> get_unidirectional_linear_mcast_hops(
        const FabricNodeId& src_node_id, uint32_t dim) const = 0;
    virtual std::vector<std::pair<FabricNodeId, FabricNodeId>> get_neighbor_exchange_pairs() const = 0;
    virtual std::optional<std::pair<FabricNodeId, FabricNodeId>> get_wrap_around_mesh_ring_neighbors(
        const FabricNodeId& src_node, const std::vector<FabricNodeId>& devices) const = 0;
    virtual std::unordered_map<RoutingDirection, uint32_t> get_wrap_around_mesh_full_or_half_ring_mcast_hops(
        const FabricNodeId& src_node_id,
        const FabricNodeId& dst_node_forward_id,
        const FabricNodeId& dst_node_backward_id,
        HighLevelTrafficPattern pattern_type) const = 0;
    virtual std::unordered_map<RoutingDirection, uint32_t> get_full_or_half_ring_mcast_hops(
        const FabricNodeId& src_node_id, HighLevelTrafficPattern pattern_type, uint32_t dim) const = 0;
    virtual std::vector<std::unordered_map<RoutingDirection, uint32_t>> split_multicast_hops(
        const std::unordered_map<RoutingDirection, uint32_t>& hops) const = 0;
    virtual FabricNodeId get_random_unicast_destination(FabricNodeId src_node_id, std::mt19937& gen) const = 0;
    virtual RoutingDirection get_forwarding_direction(
        const FabricNodeId& src_node_id, const FabricNodeId& dst_node_id) const = 0;
    virtual RoutingDirection get_forwarding_direction(
        const std::unordered_map<RoutingDirection, uint32_t>& hops) const = 0;
    virtual std::vector<uint32_t> get_forwarding_link_indices_in_direction(
        const FabricNodeId& src_node_id, const RoutingDirection& direction) const = 0;
    virtual FabricNodeId get_mcast_start_node_id(
        const FabricNodeId& src_node_id, const std::unordered_map<RoutingDirection, uint32_t>& hops) const = 0;
    virtual std::pair<std::unordered_map<RoutingDirection, uint32_t>, uint32_t> get_sync_hops_and_val(
        const FabricNodeId& src_device, const std::vector<FabricNodeId>& devices) const = 0;
    virtual std::vector<uint32_t> get_forwarding_link_indices_in_direction(
        const FabricNodeId& src_node_id, const FabricNodeId& dst_node_id, const RoutingDirection& direction) const = 0;
    virtual std::optional<FabricNodeId> get_neighbor_node_id_or_nullopt(
        const FabricNodeId& src_node_id, const RoutingDirection& direction) const = 0;
    virtual FabricNodeId get_neighbor_node_id(
        const FabricNodeId& src_node_id, const RoutingDirection& direction) const = 0;
    virtual std::unordered_map<RoutingDirection, uint32_t> get_hops_to_nearest_neighbors(
        const FabricNodeId& src_node_id) const = 0;
    virtual bool validate_num_links_supported(uint32_t num_links) const = 0;
    virtual void validate_single_hop(const std::unordered_map<RoutingDirection, uint32_t>& hops) const = 0;
};

class IDistributedContextManager {
public:
    virtual ~IDistributedContextManager() = default;

private:
    virtual uint32_t get_randomized_master_seed() const = 0;
    virtual void barrier() const = 0;
};

}  // namespace tt::tt_fabric::fabric_tests
