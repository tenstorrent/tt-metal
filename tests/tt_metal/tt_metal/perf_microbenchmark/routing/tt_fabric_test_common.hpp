// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <optional>

#include <tt-metalium/mesh_graph.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/device_pool.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/distributed.hpp>
#include "tt_metal/test_utils/env_vars.hpp"

#include "tt_metal/fabric/fabric_context.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_fabric_test_config.hpp"
#include "tt_fabric_test_interfaces.hpp"

using MeshDevice = tt::tt_metal::distributed::MeshDevice;
using MeshCoordinate = tt::tt_metal::distributed::MeshCoordinate;
using MeshShape = tt::tt_metal::distributed::MeshShape;
using MeshWorkload = tt::tt_metal::distributed::MeshWorkload;
using MeshCoordinateRange = tt::tt_metal::distributed::MeshCoordinateRange;

namespace tt::tt_fabric {
namespace fabric_tests {

class TestFixture : public IDeviceInfoProvider, public IRouteManager {
    // mapping to convert coords to directions
    static constexpr uint32_t EW_DIM = 1;
    static constexpr uint32_t NS_DIM = 0;

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
    }

    std::vector<MeshCoordinate> get_available_device_coordinates() const { return this->available_device_coordinates_; }

    void open_devices(tt::tt_metal::FabricConfig fabric_config) {
        tt::tt_metal::detail::InitializeFabricConfig(fabric_config);
        mesh_device_ = MeshDevice::create(mesh_shape_);

        for (const auto& coord : available_device_coordinates_) {
            auto* device = mesh_device_->get_device(coord);
            const auto fabric_node_id = control_plane_ptr_->get_fabric_node_id_from_physical_chip_id(device->id());
            mesh_coordinate_to_node_id_.emplace(coord, fabric_node_id);
            node_id_to_mesh_coordinate_.emplace(fabric_node_id, coord);
        }

        mesh_workload_ = std::make_unique<MeshWorkload>();
    }

    void enqueue_program(const MeshCoordinate& mesh_coord, tt::tt_metal::Program& program) {
        MeshCoordinateRange device(mesh_coord, mesh_coord);
        tt::tt_metal::distributed::AddProgramToMeshWorkload(*mesh_workload_, std::move(program), device);
    }

    void run_programs() {
        tt::tt_metal::distributed::EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), *mesh_workload_, true);
    }

    void wait_for_programs() { tt::tt_metal::distributed::Finish(mesh_device_->mesh_command_queue()); }

    void close_devices() {
        mesh_device_->close();
        tt::tt_metal::detail::InitializeFabricConfig(tt::tt_metal::FabricConfig::DISABLED);
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

    virtual uint32_t get_worker_id(const FabricNodeId& node_id, CoreCoord logical_core) const override {
        return (*node_id.mesh_id << 12) | (node_id.chip_id << 8) | (logical_core.x << 4) | (logical_core.y);
    }

    std::vector<FabricNodeId> get_all_node_ids() const override { return available_node_ids_; }

    // ======================================================================================
    // IRouteManager methods
    // ======================================================================================
    // TODO: instead of parsing ChipSendType, this should only care about unicast/mcast
    // or capturing every device in the path or not
    std::vector<FabricNodeId> get_dst_node_ids_from_hops(
        FabricNodeId src_node,
        const std::unordered_map<RoutingDirection, uint32_t>& hops,
        ChipSendType chip_send_type) const override {
        // TODO: get src mesh coord
        // TODO: add a private method to get dst coord for each increment from the src coord

        return {};
    }

    bool are_devices_linear(const std::vector<FabricNodeId>& node_ids) const override {
        /*
        if (node_ids.size() <= 1) {
            return true;
        }

        // TODO: validation for node ids

        auto first_coord = node_id_to_mesh_coordinate_.at(node_ids[0]);
        bool all_same_row = true;
        bool all_same_col = true;

        for (size_t i = 1; i < node_ids.size(); ++i) {
            auto next_coord = node_id_to_mesh_coordinate_.at(node_ids[i]);
            if (next_coord.x != first_coord.x) {
                all_same_col = false;
            }
            if (next_coord.y != first_coord.y) {
                all_same_row = false;
            }
        }
        return all_same_row || all_same_col;
        */
        return true;
    }

    std::unordered_map<RoutingDirection, uint32_t> get_hops_to_chip(
        FabricNodeId src_node_id, FabricNodeId dst_node_id) const override {
        const auto& src_coord = get_device_coord(src_node_id);
        const auto& dst_coord = get_device_coord(dst_node_id);

        const auto distance = get_distance(src_coord, dst_coord);
        return get_hops_from_distance(distance);
    }

    std::vector<chip_id_t> get_chips_from_hops(
        chip_id_t src_chip,
        const std::unordered_map<RoutingDirection, uint32_t>& hops,
        ChipSendType chip_send_type) const {
        std::vector<chip_id_t> dst_chips;
        /*
        if (chip_send_type == ChipSendType::CHIP_UNICAST) {
            dst_chips.push_back(get_chip_from_hops_map(src_chip, hops));
        } else if (chip_send_type == ChipSendType::CHIP_MULTICAST) {
            for (const auto& direction : FabricContext::routing_directions) {
                if (hops.at(direction) > 0) {
                    for (uint32_t hop = 1; hop <= hops.at(direction); hop++) {
                        dst_chips.push_back(get_chip_from_hop_count(src_chip, direction, hop));
                    }
                }
            }
        }*/
        return dst_chips;
    }

private:
    ControlPlane* control_plane_ptr_;
    MeshShape mesh_shape_;
    std::vector<MeshCoordinate> available_device_coordinates_;
    std::vector<FabricNodeId> available_node_ids_;
    std::shared_ptr<MeshDevice> mesh_device_;
    std::unordered_map<MeshCoordinate, FabricNodeId> mesh_coordinate_to_node_id_;
    std::unordered_map<FabricNodeId, MeshCoordinate> node_id_to_mesh_coordinate_;
    std::shared_ptr<MeshWorkload> mesh_workload_;

    MeshCoordinate get_distance(const MeshCoordinate& src_coords, const MeshCoordinate& dst_coords) const {
        TT_FATAL(
            src_coords.dims() == dst_coords.dims(),
            "Cannot find distance from coords with different dimensions: {} != {}",
            src_coords.dims(),
            dst_coords.dims());

        std::vector<uint32_t> coords(src_coords.dims());
        for (size_t i = 0; i < src_coords.dims(); ++i) {
            coords[i] = dst_coords[i] - src_coords[i];
        }
        return MeshCoordinate(coords);
    }

    std::unordered_map<RoutingDirection, uint32_t> get_hops_from_distance(const MeshCoordinate& distance) const {
        std::unordered_map<RoutingDirection, uint32_t> hops;
        for (const auto& direction : FabricContext::routing_directions) {
            hops[direction] = 0;
        }

        if (distance[EW_DIM] >= mesh_shape_[EW_DIM]) {
            // wrapped around, negative distance
            hops[RoutingDirection::W] = std::numeric_limits<uint32_t>::max() - distance[EW_DIM] + 1;
        } else {
            hops[RoutingDirection::E] = distance[EW_DIM];
        }

        // positive y is south direction in ctrl plane
        if (distance[NS_DIM] >= mesh_shape_[NS_DIM]) {
            // wrapped around, negative distance
            hops[RoutingDirection::N] = std::numeric_limits<uint32_t>::max() - distance[NS_DIM] + 1;
        } else {
            hops[RoutingDirection::S] = distance[NS_DIM];
        }

        return hops;
    }

    chip_id_t get_chip_from_hop_count(chip_id_t src_chip, RoutingDirection direction, uint32_t hop_count) const {
        /*
        tt::MeshCoordinate dst_coord = this->mesh_device_->get_device_coord(src_chip);
        switch (direction) {
            case RoutingDirection::N: dst_coord.y -= hop_count; break;
            case RoutingDirection::S: dst_coord.y += hop_count; break;
            case RoutingDirection::E: dst_coord.x += hop_count; break;
            case RoutingDirection::W: dst_coord.x -= hop_count; break;
            default: break;
        }
        return this->mesh_device_->get_chip_id(dst_coord); */
        return src_chip;
    }

    chip_id_t get_chip_from_hops_map(
        chip_id_t src_chip, const std::unordered_map<RoutingDirection, uint32_t>& hops) const {
        /*
    tt::MeshCoordinate dst_coord = this->mesh_device_->get_device_coord(src_chip);
    dst_coord.x += hops.at(RoutingDirection::E);
    dst_coord.x -= hops.at(RoutingDirection::W);
    dst_coord.y += hops.at(RoutingDirection::S);
    dst_coord.y -= hops.at(RoutingDirection::N);
    return this->mesh_device_->get_chip_id(dst_coord); */
        return src_chip;
    }
};

}  // namespace fabric_tests
}  // namespace tt::tt_fabric
