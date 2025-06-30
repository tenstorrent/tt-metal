// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <optional>
#include <random>

#include <tt-metalium/mesh_graph.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/device_pool.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/distributed.hpp>
#include "tt_metal/test_utils/env_vars.hpp"

#include "tt_metal/fabric/fabric_context.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_fabric_test_interfaces.hpp"
#include "tt_fabric_test_common_types.hpp"

using MeshDevice = tt::tt_metal::distributed::MeshDevice;
using MeshCoordinate = tt::tt_metal::distributed::MeshCoordinate;
using MeshShape = tt::tt_metal::distributed::MeshShape;
using MeshWorkload = tt::tt_metal::distributed::MeshWorkload;
using MeshCoordinateRange = tt::tt_metal::distributed::MeshCoordinateRange;

using Topology = tt::tt_fabric::Topology;

namespace tt::tt_fabric {
namespace fabric_tests {

struct pair_hash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        // Simple mixing
        return h1 ^ (h2 << 1);
    }
};

class TestFixture : public IDeviceInfoProvider, public IRouteManager {
    static constexpr uint32_t ROW_DIM = 0;
    static constexpr uint32_t COL_DIM = 1;

    // mapping to convert coords to directions
    static constexpr uint32_t EW_DIM = 1;
    static constexpr uint32_t NS_DIM = 0;

    static const std::unordered_map<std::pair<Topology, RoutingType>, tt::tt_metal::FabricConfig, pair_hash>
        topology_to_fabric_config_map;

public:
    void init() {
        control_plane_ptr_ = &tt::tt_metal::MetalContext::instance().get_control_plane();
        const auto user_meshes = control_plane_ptr_->get_user_physical_mesh_ids();
        TT_FATAL(
            user_meshes.size() == 1,
            "Only expected a single user mesh for a single host, but got: {}",
            user_meshes.size());

        mesh_shape_ = control_plane_ptr_->get_physical_mesh_shape(user_meshes[0]);
        const auto coordinates = MeshCoordinateRange(mesh_shape_);
        for (const auto& coord : coordinates) {
            available_device_coordinates_.push_back(coord);
        }

        // TODO: available node ids should be able to capture the node ids for other meshes as well
        const auto mesh_id = user_meshes[0];
        for (auto i = 0; i < available_device_coordinates_.size(); i++) {
            available_node_ids_.emplace_back(FabricNodeId(mesh_id, i));
        }

        current_fabric_config_ = tt::tt_metal::FabricConfig::DISABLED;
    }

    std::vector<MeshCoordinate> get_available_device_coordinates() const { return this->available_device_coordinates_; }

    void open_devices(Topology topology, RoutingType routing_type) {
        auto it = topology_to_fabric_config_map.find({topology, routing_type});
        TT_FATAL(
            it != topology_to_fabric_config_map.end(),
            "Unsupported topology: {} with routing type: {}",
            topology,
            routing_type);
        auto new_fabric_config = it->second;
        if (new_fabric_config != current_fabric_config_) {
            if (are_devices_open_) {
                log_info(tt::LogTest, "Closing devices and switching to new fabric config: {}", new_fabric_config);
                close_devices();
            }
            open_devices_internal(new_fabric_config);

            topology_ = topology;
            routing_type_ = routing_type;
        } else {
            log_info(tt::LogTest, "Reusing existing device setup with fabric config: {}", current_fabric_config_);
        }
    }

    void setup_workload() {
        // create a new mesh workload for every run
        mesh_workload_ = std::make_unique<MeshWorkload>();
    }

    void enqueue_program(const MeshCoordinate& mesh_coord, tt::tt_metal::Program program) {
        MeshCoordinateRange device(mesh_coord, mesh_coord);
        tt::tt_metal::distributed::AddProgramToMeshWorkload(*mesh_workload_, std::move(program), device);
    }

    void run_programs() {
        tt::tt_metal::distributed::EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), *mesh_workload_, true);
    }

    void wait_for_programs() { tt::tt_metal::distributed::Finish(mesh_device_->mesh_command_queue()); }

    void close_devices() {
        mesh_device_->close();
        tt::tt_metal::detail::SetFabricConfig(tt::tt_metal::FabricConfig::DISABLED);

        mesh_coordinate_to_node_id_.clear();
        node_id_to_mesh_coordinate_.clear();
        current_fabric_config_ = tt::tt_metal::FabricConfig::DISABLED;
        are_devices_open_ = false;
    }

    // ======================================================================================
    // IDeviceInfoProvider methods
    // ======================================================================================
    FabricNodeId get_fabric_node_id(const chip_id_t physical_chip_id) const override {
        return control_plane_ptr_->get_fabric_node_id_from_physical_chip_id(physical_chip_id);
    }

    FabricNodeId get_fabric_node_id(const MeshCoordinate& device_coord) const override {
        return mesh_coordinate_to_node_id_.at(device_coord);
    }

    MeshCoordinate get_device_coord(const FabricNodeId& node_id) const override {
        auto it = node_id_to_mesh_coordinate_.find(node_id);
        TT_FATAL(it != node_id_to_mesh_coordinate_.end(), "Unknown node id: {} for querying mesh coord", node_id);

        return it->second;
    }

    uint32_t get_worker_noc_encoding(const MeshCoordinate& device_coord, const CoreCoord logical_core) const override {
        auto* device = mesh_device_->get_device(device_coord);
        const auto virtual_core = device->worker_core_from_logical_core(logical_core);
        return tt_metal::MetalContext::instance().hal().noc_xy_encoding(virtual_core.x, virtual_core.y);
    }

    uint32_t get_worker_noc_encoding(const FabricNodeId& node_id, const CoreCoord logical_core) const override {
        const auto& device_coord = get_device_coord(node_id);
        return get_worker_noc_encoding(device_coord, logical_core);
    }

    CoreCoord get_worker_grid_size() const override { return mesh_device_->compute_with_storage_grid_size(); }

    uint32_t get_worker_id(const FabricNodeId& node_id, CoreCoord logical_core) const override {
        return (*node_id.mesh_id << 12) | (node_id.chip_id << 8) | (logical_core.x << 4) | (logical_core.y);
    }

    std::vector<FabricNodeId> get_all_node_ids() const override { return available_node_ids_; }

    uint32_t get_l1_unreserved_base(const FabricNodeId& node_id) const override {
        const auto& device_coord = get_device_coord(node_id);
        auto* device = mesh_device_->get_device(device_coord);
        return device->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);
    }

    uint32_t get_l1_unreserved_size(const FabricNodeId& node_id) const override {
        return tt::tt_metal::hal::get_max_worker_l1_unreserved_size();
    }

    uint32_t get_l1_alignment() const override { return tt::tt_metal::hal::get_l1_alignment(); }

    uint32_t get_max_payload_size_bytes() const override {
        return control_plane_ptr_->get_fabric_context().get_fabric_max_payload_size_bytes();
    }

    bool is_2d_fabric() const override { return topology_ == Topology::Mesh; }

    bool use_dynamic_routing() const override { return routing_type_ == RoutingType::Dynamic; }

    // ======================================================================================
    // IRouteManager methods
    // ======================================================================================
    // TODO: instead of parsing ChipSendType, this should only care about unicast/mcast
    // or capturing every device in the path or not
    std::vector<FabricNodeId> get_dst_node_ids_from_hops(
        FabricNodeId src_node,
        std::unordered_map<RoutingDirection, uint32_t>& hops,
        ChipSendType chip_send_type) const override {
        std::vector<FabricNodeId> dst_nodes;
        const MeshCoordinate& src_coord = get_device_coord(src_node);
        auto displacements = convert_hops_to_displacement(hops);

        for (const auto& displacement : displacements) {
            // Ignore zero-length displacements that can occur for some directions in the hops map
            if (displacement == MeshCoordinate::zero_coordinate(displacement.dims())) {
                continue;
            }

            const auto dst_coord = get_coord_from_displacement(src_coord, displacement);

            if (chip_send_type == ChipSendType::CHIP_UNICAST) {
                // For unicast, we only care about the final destination of each displacement vector.
                dst_nodes.push_back(get_fabric_node_id(dst_coord));
            } else if (chip_send_type == ChipSendType::CHIP_MULTICAST) {
                // For multicast, we care about all nodes along the path.
                const auto coords_in_path = get_coords_from_range(src_coord, dst_coord);
                for (const auto& coord : coords_in_path) {
                    if (coord == src_coord) {
                        continue;  // Don't include the source itself
                    }
                    dst_nodes.push_back(get_fabric_node_id(coord));
                }
            }
        }
        return dst_nodes;
    }

    bool are_devices_linear(const std::vector<FabricNodeId>& node_ids) const override {
        if (node_ids.size() <= 1) {
            return true;
        }

        auto first_coord = node_id_to_mesh_coordinate_.at(node_ids[0]);
        bool all_same_row = true;
        bool all_same_col = true;

        for (size_t i = 1; i < node_ids.size(); ++i) {
            auto next_coord = node_id_to_mesh_coordinate_.at(node_ids[i]);
            if (next_coord[COL_DIM] != first_coord[COL_DIM]) {
                all_same_col = false;
            }
            if (next_coord[ROW_DIM] != first_coord[ROW_DIM]) {
                all_same_row = false;
            }
        }
        return all_same_row || all_same_col;
    }

    std::unordered_map<RoutingDirection, uint32_t> get_hops_to_chip(
        FabricNodeId src_node_id, FabricNodeId dst_node_id) const override {
        const auto& src_coord = get_device_coord(src_node_id);
        const auto& dst_coord = get_device_coord(dst_node_id);

        const auto displacement = get_displacement(src_coord, dst_coord);
        return get_hops_from_displacement(displacement);
    }

    std::vector<std::pair<FabricNodeId, FabricNodeId>> get_full_device_random_pairs(std::mt19937& gen) const override {
        auto unpaired = get_all_node_ids();

        if (unpaired.size() % 2 != 0) {
            log_warning(
                tt::LogTest,
                "full_device_random_pairing pattern requires an even number of devices, but found {}. One device will "
                "be left unpaired.",
                unpaired.size());
        }

        std::shuffle(unpaired.begin(), unpaired.end(), gen);
        std::vector<std::pair<FabricNodeId, FabricNodeId>> pairs;
        pairs.reserve(unpaired.size() / 2);

        while (unpaired.size() >= 2) {
            FabricNodeId src_node = unpaired.back();
            unpaired.pop_back();

            std::vector<size_t> valid_dst_indices;
            for (size_t i = 0; i < unpaired.size(); ++i) {
                const auto& dst_node = unpaired[i];
                bool is_valid_pair =
                    (this->topology_ != Topology::Linear) || this->are_devices_linear({src_node, dst_node});
                if (is_valid_pair) {
                    valid_dst_indices.push_back(i);
                }
            }

            if (valid_dst_indices.empty()) {
                log_warning(
                    tt::LogTest,
                    "Could not find a valid partner for device {}. Unable to create a full random pairing.",
                    src_node);
                log_warning(tt::LogTest, "Exiting early with {} pairs", pairs.size());
                break;
            }

            std::shuffle(valid_dst_indices.begin(), valid_dst_indices.end(), gen);
            size_t picked_idx = valid_dst_indices.front();
            FabricNodeId dst_node = unpaired[picked_idx];
            pairs.push_back({src_node, dst_node});

            std::swap(unpaired[picked_idx], unpaired.back());
            unpaired.pop_back();
        }

        return pairs;
    }

    std::vector<std::pair<FabricNodeId, FabricNodeId>> get_all_to_all_unicast_pairs() const override {
        const auto device_ids = get_all_node_ids();
        std::vector<std::pair<FabricNodeId, FabricNodeId>> pairs;
        pairs.reserve(device_ids.size() * (device_ids.size() - 1));
        for (const auto& src_node : device_ids) {
            for (const auto& dst_node : device_ids) {
                if (src_node == dst_node) {
                    continue;
                }
                if (this->topology_ == Topology::Linear) {
                    if (!this->are_devices_linear({src_node, dst_node})) {
                        continue;
                    }
                }
                pairs.push_back({src_node, dst_node});
            }
        }
        return pairs;
    }

    std::unordered_map<RoutingDirection, uint32_t> get_full_mcast_hops(const FabricNodeId& src_node_id) const override {
        std::unordered_map<RoutingDirection, uint32_t> hops;
        for (const auto& direction : FabricContext::routing_directions) {
            hops[direction] = 0;
        }

        const auto src_coord = get_device_coord(src_node_id);
        hops[RoutingDirection::N] = src_coord[NS_DIM];
        hops[RoutingDirection::S] = mesh_shape_[NS_DIM] - src_coord[NS_DIM] - 1;
        hops[RoutingDirection::E] = mesh_shape_[EW_DIM] - src_coord[EW_DIM] - 1;
        hops[RoutingDirection::W] = src_coord[EW_DIM];

        return hops;
    }

    std::vector<std::unordered_map<RoutingDirection, uint32_t>> split_multicast_hops(
        const std::unordered_map<RoutingDirection, uint32_t>& hops) const override {
        std::vector<std::unordered_map<RoutingDirection, uint32_t>> split_hops;
        if (this->topology_ == Topology::Linear) {
            split_hops.reserve(4);
            for (const auto& [dir, hop_count] : hops) {
                if (hop_count > 0) {
                    split_hops.push_back({{dir, hop_count}});
                }
            }
        } else {
            TT_THROW("Unsupported topology: {} for split_multicast_hops", this->topology_);
        }
        return split_hops;
    }

    FabricNodeId get_random_unicast_destination(FabricNodeId src_node_id, std::mt19937& gen) const override {
        auto all_devices = this->get_all_node_ids();
        std::vector<FabricNodeId> possible_dsts;
        possible_dsts.reserve(all_devices.size());
        for (const auto& dev : all_devices) {
            if (dev == src_node_id) {
                continue;
            }
            if (this->topology_ == Topology::Linear && !this->are_devices_linear({src_node_id, dev})) {
                continue;
            }
            possible_dsts.push_back(dev);
        }

        if (possible_dsts.empty()) {
            TT_THROW(
                "Cannot pick a random unicast destination for sender on device {}: no valid partner devices available.",
                src_node_id);
        }

        std::shuffle(possible_dsts.begin(), possible_dsts.end(), gen);
        return possible_dsts[0];
    }

    MeshShape get_mesh_shape() const { return mesh_shape_; }

    RoutingDirection get_forwarding_direction(
        const FabricNodeId& src_node_id, const FabricNodeId& dst_node_id) const override {
        auto forwarding_direction = control_plane_ptr_->get_forwarding_direction(src_node_id, dst_node_id);
        TT_FATAL(
            forwarding_direction.has_value(), "No forwarding direction found for {} -> {}", src_node_id, dst_node_id);
        return forwarding_direction.value();
    }

    RoutingDirection get_forwarding_direction(
        const std::unordered_map<RoutingDirection, uint32_t>& hops) const override {
        if (topology_ == Topology::Linear) {
            // for 1d, we expect hops only in one direction to be non-zero
            for (const auto& [direction, hop] : hops) {
                if (hop != 0) {
                    return direction;
                }
            }
        } else if (topology_ == Topology::Mesh) {
            // TODO: update this logic once 2D mcast is supported
            // for now we return the first direction that is non-zero
            // for 2D, since we use dimension order routing, lookup the directions in the order of N, E, S, W
            for (const auto& direction : FabricContext::routing_directions) {
                if (hops.count(direction) > 0 && hops.at(direction) != 0) {
                    return direction;
                }
            }
        } else {
            TT_THROW("Unsupported topology: {} for get_forwarding_direction", topology_);
        }

        TT_THROW("Failed to find a forwarding direction for hops: {}", hops);
    }

    std::vector<uint32_t> get_forwarding_link_indices_in_direction(
        const FabricNodeId& src_node_id,
        const FabricNodeId& dst_node_id,
        const RoutingDirection& direction) const override {
        return tt::tt_fabric::get_forwarding_link_indices_in_direction(src_node_id, dst_node_id, direction);
    }

    FabricNodeId get_neighbor_node_id(
        const FabricNodeId& src_node_id, const RoutingDirection& direction) const override {
        const auto& neighbors = control_plane_ptr_->get_chip_neighbors(src_node_id, direction);
        TT_FATAL(neighbors.size() == 1, "Expected only neighbor mesh for {} in direction: {}", src_node_id, direction);
        TT_FATAL(
            neighbors.begin()->second.size() >= 1,
            "Expected at least 1 neighbor chip for {} in direction: {}",
            src_node_id,
            direction);
        return FabricNodeId(neighbors.begin()->first, neighbors.begin()->second[0]);
    }

    std::vector<uint32_t> get_forwarding_link_indices_in_direction(
        const FabricNodeId& src_node_id, const RoutingDirection& direction) const override {
        const auto& neighbor_node_id = get_neighbor_node_id(src_node_id, direction);
        return this->get_forwarding_link_indices_in_direction(src_node_id, neighbor_node_id, direction);
    }

private:
    ControlPlane* control_plane_ptr_;
    Topology topology_;
    RoutingType routing_type_;
    MeshShape mesh_shape_;
    tt::tt_metal::FabricConfig current_fabric_config_;
    std::vector<MeshCoordinate> available_device_coordinates_;
    std::vector<FabricNodeId> available_node_ids_;
    std::shared_ptr<MeshDevice> mesh_device_;
    std::unordered_map<MeshCoordinate, FabricNodeId> mesh_coordinate_to_node_id_;
    std::unordered_map<FabricNodeId, MeshCoordinate> node_id_to_mesh_coordinate_;
    std::shared_ptr<MeshWorkload> mesh_workload_;
    bool are_devices_open_ = false;

    void open_devices_internal(tt::tt_metal::FabricConfig fabric_config) {
        tt::tt_metal::detail::SetFabricConfig(fabric_config);
        mesh_device_ = MeshDevice::create(mesh_shape_);

        for (const auto& coord : available_device_coordinates_) {
            auto* device = mesh_device_->get_device(coord);
            const auto fabric_node_id = control_plane_ptr_->get_fabric_node_id_from_physical_chip_id(device->id());
            mesh_coordinate_to_node_id_.emplace(coord, fabric_node_id);
            node_id_to_mesh_coordinate_.emplace(fabric_node_id, coord);
        }

        current_fabric_config_ = fabric_config;
        are_devices_open_ = true;
    }

    MeshCoordinate get_displacement(const MeshCoordinate& src_coords, const MeshCoordinate& dst_coords) const {
        TT_FATAL(
            src_coords.dims() == dst_coords.dims(),
            "Cannot find distance from coords with different dimensions: {} != {}",
            src_coords.dims(),
            dst_coords.dims());

        std::vector<uint32_t> coords(src_coords.dims(), 0);
        for (size_t i = 0; i < src_coords.dims(); ++i) {
            coords[i] = dst_coords[i] - src_coords[i];
        }
        return MeshCoordinate(coords);
    }

    MeshCoordinate get_coord_from_displacement(
        const MeshCoordinate& src_coords, const MeshCoordinate& displacement) const {
        std::vector<uint32_t> coords(src_coords.dims(), 0);
        for (size_t i = 0; i < src_coords.dims(); ++i) {
            coords[i] = src_coords[i] + displacement[i];
        }
        return MeshCoordinate(coords);
    }

    MeshCoordinateRange get_coords_from_range(
        const MeshCoordinate& src_coords, const MeshCoordinate& dst_coords) const {
        const auto start = std::min(src_coords, dst_coords);
        const auto end = std::max(src_coords, dst_coords);
        return MeshCoordinateRange(start, end);
    }

    std::unordered_map<RoutingDirection, uint32_t> get_hops_from_displacement(
        const MeshCoordinate& displacement) const {
        std::unordered_map<RoutingDirection, uint32_t> hops;
        for (const auto& direction : FabricContext::routing_directions) {
            hops[direction] = 0;
        }

        if (displacement[EW_DIM] >= mesh_shape_[EW_DIM]) {
            // wrapped around, negative distance
            hops[RoutingDirection::W] = std::numeric_limits<uint32_t>::max() - displacement[EW_DIM] + 1;
        } else {
            hops[RoutingDirection::E] = displacement[EW_DIM];
        }

        // positive y is south direction in ctrl plane
        if (displacement[NS_DIM] >= mesh_shape_[NS_DIM]) {
            // wrapped around, negative distance
            hops[RoutingDirection::N] = std::numeric_limits<uint32_t>::max() - displacement[NS_DIM] + 1;
        } else {
            hops[RoutingDirection::S] = displacement[NS_DIM];
        }

        return hops;
    }

    MeshCoordinate get_displacement_from_hops(const std::vector<std::pair<RoutingDirection, uint32_t>>& hops) const {
        std::vector<uint32_t> displacement(mesh_shape_.dims(), 0);
        for (const auto& [direction, hop] : hops) {
            switch (direction) {
                case RoutingDirection::N: displacement[NS_DIM] -= hop; break;
                case RoutingDirection::S: displacement[NS_DIM] += hop; break;
                case RoutingDirection::E: displacement[EW_DIM] += hop; break;
                case RoutingDirection::W: displacement[EW_DIM] -= hop; break;
                default: break;
            }
        }
        return MeshCoordinate(displacement);
    }

    // this depends on the fabric topology
    // for 1D, we handle each direction separately
    // for 2D, we look at the combination of directions
    std::vector<MeshCoordinate> convert_hops_to_displacement(
        std::unordered_map<RoutingDirection, uint32_t>& hops) const {
        std::vector<MeshCoordinate> displacements;

        if (topology_ == Topology::Linear) {
            displacements.reserve(4);
            displacements.push_back(get_displacement_from_hops({{RoutingDirection::N, hops[RoutingDirection::N]}}));
            displacements.push_back(get_displacement_from_hops({{RoutingDirection::S, hops[RoutingDirection::S]}}));
            displacements.push_back(get_displacement_from_hops({{RoutingDirection::E, hops[RoutingDirection::E]}}));
            displacements.push_back(get_displacement_from_hops({{RoutingDirection::W, hops[RoutingDirection::W]}}));
        } else if (topology_ == Topology::Mesh) {
            displacements.reserve(4);
            displacements.push_back(get_displacement_from_hops(
                {{RoutingDirection::N, hops[RoutingDirection::N]}, {RoutingDirection::E, hops[RoutingDirection::E]}}));
            displacements.push_back(get_displacement_from_hops(
                {{RoutingDirection::N, hops[RoutingDirection::N]}, {RoutingDirection::W, hops[RoutingDirection::W]}}));
            displacements.push_back(get_displacement_from_hops(
                {{RoutingDirection::S, hops[RoutingDirection::S]}, {RoutingDirection::E, hops[RoutingDirection::E]}}));
            displacements.push_back(get_displacement_from_hops(
                {{RoutingDirection::S, hops[RoutingDirection::S]}, {RoutingDirection::W, hops[RoutingDirection::W]}}));
        } else {
            TT_FATAL(false, "Unsupported topology: {}", topology_);
        }
        return displacements;
    }
};

}  // namespace fabric_tests
}  // namespace tt::tt_fabric
