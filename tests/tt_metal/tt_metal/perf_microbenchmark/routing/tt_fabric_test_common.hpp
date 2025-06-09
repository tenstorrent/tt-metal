// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <vector>
#include <unordered_map>
#include <algorithm>

#include <tt-metalium/mesh_graph.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/device_pool.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/test_utils/env_vars.hpp"

#include "tt_metal/fabric/fabric_context.hpp"
#include "impl/context/metal_context.hpp"

namespace tt::tt_fabric {
namespace fabric_tests {

class TestContextInterface {};

struct TestFabricFixture {
    tt::ARCH arch_;
    std::vector<chip_id_t> physical_chip_ids_;
    std::map<chip_id_t, tt::tt_metal::IDevice*> devices_map_;
    bool slow_dispatch_;

    void setup_devices() {
        slow_dispatch_ = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch_) {
            tt::log_info(tt::LogTest, "Running fabric tests with slow dispatch");
        } else {
            tt::log_info(tt::LogTest, "Running fabric tests with fast dispatch");
        }

        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());

        size_t chip_id_offset = 0;
        if (tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() == tt::ClusterType::TG) {
            chip_id_offset = 4;
        }
        const auto num_devices = tt::tt_metal::GetNumAvailableDevices();
        this->physical_chip_ids_.resize(num_devices);
        std::iota(this->physical_chip_ids_.begin(), this->physical_chip_ids_.end(), chip_id_offset);

        // todo: check if and where we need to configure the routing tables
    }

    void open_devices(tt::tt_metal::FabricConfig fabric_config) {
        tt::tt_metal::detail::InitializeFabricConfig(fabric_config);
        this->devices_map_ = tt::tt_metal::detail::CreateDevices(this->physical_chip_ids_);
    }

    std::vector<chip_id_t> get_available_chip_ids() const { return this->physical_chip_ids_; }

    tt::tt_metal::IDevice* get_device_handle(chip_id_t physical_chip_id) const {
        if (this->devices_map_.find(physical_chip_id) == this->devices_map_.end()) {
            tt::log_fatal(tt::LogTest, "Unknown physical chip id: {}", physical_chip_id);
            throw std::runtime_error("Unexpected physical chip id for device handle lookup");
        }
        return this->devices_map_.at(physical_chip_id);
    }

    void run_program_non_blocking(tt::tt_metal::IDevice* device, tt::tt_metal::Program& program) {
        if (this->slow_dispatch_) {
            tt::tt_metal::detail::LaunchProgram(device, program, false);
        } else {
            tt::tt_metal::CommandQueue& cq = device->command_queue();
            tt::tt_metal::EnqueueProgram(cq, program, false);
        }
    }

    void wait_for_program_done(tt::tt_metal::IDevice* device, tt::tt_metal::Program& program) {
        if (this->slow_dispatch_) {
            // Wait for the program to finish
            tt::tt_metal::detail::WaitProgramDone(device, program);
        } else {
            // Wait for all programs on cq to finish
            tt::tt_metal::CommandQueue& cq = device->command_queue();
            tt::tt_metal::Finish(cq);
        }
    }

    void close_devices() {
        tt::tt_metal::detail::CloseDevices(this->devices_map_);
        tt::tt_metal::detail::InitializeFabricConfig(tt::tt_metal::FabricConfig::DISABLED);
    }
};

struct TestPhysCoords {
public:
    TestPhysCoords(uint32_t x, uint32_t y);
    TestPhysCoords(const std::tuple<uint32_t, uint32_t>& mesh_indices);
    bool operator==(const TestPhysCoords& other) const;
    std::unordered_map<RoutingDirection, uint32_t> get_hops_to_dst(const TestPhysCoords& dst_coords) const;
    TestPhysCoords get_dst_from_hops(std::unordered_map<RoutingDirection, uint32_t>& hops) const;
    TestPhysCoords get_dst_from_hops(const RoutingDirection direction, const uint32_t hops) const;

private:
    void validate_hops(std::unordered_map<RoutingDirection, uint32_t>& hops) const;
    void update_coords(const RoutingDirection direction, const uint32_t hops);

    uint32_t x_;
    uint32_t y_;
};

inline void TestPhysCoords::validate_hops(std::unordered_map<RoutingDirection, uint32_t>& hops) const {
    // TODO: should move to mesh
    if ((hops[RoutingDirection::N] > 0 && hops[RoutingDirection::S] > 0) ||
        (hops[RoutingDirection::E] > 0 && hops[RoutingDirection::W] > 0)) {
        tt::log_fatal(tt::LogTest, "Hops in only one of the opposite directions should be set, got both");
        throw std::runtime_error("Unexpected hops");
    }
}

inline TestPhysCoords::TestPhysCoords(uint32_t x, uint32_t y) : x_(x), y_(y) {}

inline TestPhysCoords::TestPhysCoords(const std::tuple<uint32_t, uint32_t>& mesh_indices) {
    // note the flipped order when converting row/col idx to x,y
    this->x_ = std::get<1>(mesh_indices);
    this->y_ = std::get<0>(mesh_indices);
}

inline bool TestPhysCoords::operator==(const TestPhysCoords& other) const {
    return (this->x_ == other.x_ && this->y_ == other.y_);
}

inline std::unordered_map<RoutingDirection, uint32_t> TestPhysCoords::get_hops_to_dst(
    const TestPhysCoords& dst_coords) const {
    int x_dist = dst_coords.x_ - this->x_;
    int y_dist = dst_coords.y_ - this->y_;

    std::unordered_map<RoutingDirection, uint32_t> hops;
    for (const auto& direction : FabricContext::routing_directions) {
        hops[direction] = 0;
    }

    if (y_dist < 0) {
        hops[RoutingDirection::N] = std::abs(y_dist);
    } else if (y_dist > 0) {
        hops[RoutingDirection::S] = y_dist;
    }

    if (x_dist > 0) {
        hops[RoutingDirection::E] = x_dist;
    } else if (x_dist < 0) {
        hops[RoutingDirection::W] = std::abs(x_dist);
    }

    return hops;
}

inline void TestPhysCoords::update_coords(const RoutingDirection direction, const uint32_t hops) {
    switch (direction) {
        case RoutingDirection::N: this->y_ -= hops; break;
        case RoutingDirection::S: this->y_ += hops; break;
        case RoutingDirection::E: this->x_ += hops; break;
        case RoutingDirection::W: this->x_ -= hops; break;
        default: throw std::runtime_error("Unexpected direction");
    }
}

inline TestPhysCoords TestPhysCoords::get_dst_from_hops(const RoutingDirection direction, const uint32_t hops) const {
    TestPhysCoords dst_coords(this->x_, this->y_);
    dst_coords.update_coords(direction, hops);
    return dst_coords;
}

inline TestPhysCoords TestPhysCoords::get_dst_from_hops(std::unordered_map<RoutingDirection, uint32_t>& hops) const {
    this->validate_hops(hops);  // TODO: should be done in mesh

    TestPhysCoords dst_coords(this->x_, this->y_);
    for (const auto& direction : FabricContext::routing_directions) {
        dst_coords.update_coords(direction, hops[direction]);
    }

    return dst_coords;
}

struct TestPhysicalMesh {
    static constexpr uint8_t NUM_DIMS = 2;
    static constexpr uint8_t ROW_IDX = 0;
    static constexpr uint8_t COL_IDX = 1;

public:
    TestPhysicalMesh(ControlPlane* control_plane_ptr, const MeshId mesh_id);
    std::unordered_map<RoutingDirection, uint32_t> get_hops_to_chip(
        const chip_id_t src_phys_chip_id, const chip_id_t dst_phys_chip_id) const;
    std::vector<chip_id_t> get_chips_from_hops(
        const chip_id_t src_phys_chip_id,
        std::unordered_map<RoutingDirection, uint32_t> hops,
        const ChipSendType chip_send_type) const;
    void print_mesh() const;

private:
    void validate_physical_chip_id(const chip_id_t physical_chip_id) const;
    void set_mesh_dims_and_size(const std::array<uint32_t, NUM_DIMS>& dims);
    chip_id_t get_chip_from_coords(const TestPhysCoords& coords) const;
    chip_id_t get_chip_from_hops(
        const chip_id_t src_phys_chip_id, const RoutingDirection direction, const uint32_t hops) const;
    chip_id_t get_chip_from_hops(
        const chip_id_t src_phys_chip_id, std::unordered_map<RoutingDirection, uint32_t>& hops) const;

    MeshId mesh_id_;
    std::vector<std::vector<chip_id_t>> physical_chip_view_;
    std::unordered_map<chip_id_t, TestPhysCoords> physical_chip_coords_;
    std::array<uint32_t, NUM_DIMS> mesh_dims_;
    std::array<uint32_t, NUM_DIMS> mesh_dims_size_;  // size along each dim
};

inline void TestPhysicalMesh::validate_physical_chip_id(const chip_id_t physical_chip_id) const {
    if (this->physical_chip_coords_.find(physical_chip_id) == this->physical_chip_coords_.end()) {
        tt::log_fatal(tt::LogTest, "Unknown chip id: {} for mesh id: {}", physical_chip_id, this->mesh_id_);
        throw std::runtime_error("Unexpected chip id");
    }
}

inline void TestPhysicalMesh::set_mesh_dims_and_size(const std::array<uint32_t, NUM_DIMS>& dims) {
    this->mesh_dims_[ROW_IDX] = dims[ROW_IDX];
    this->mesh_dims_[COL_IDX] = dims[COL_IDX];

    // note the reversed order
    this->mesh_dims_size_[ROW_IDX] = dims[COL_IDX];
    this->mesh_dims_size_[COL_IDX] = dims[ROW_IDX];
}

inline chip_id_t TestPhysicalMesh::get_chip_from_coords(const TestPhysCoords& target_coords) const {
    auto it =
        std::find_if(this->physical_chip_coords_.begin(), this->physical_chip_coords_.end(), [&](const auto& pair) {
            return pair.second == target_coords;
        });
    if (it == this->physical_chip_coords_.end()) {
        tt::log_fatal(tt::LogTest, "Unknown chip coords for translation from coords to chip");
        throw std::runtime_error("Unexpected physical chip coords");
    }

    return it->first;
}

inline chip_id_t TestPhysicalMesh::get_chip_from_hops(
    const chip_id_t src_phys_chip_id, const RoutingDirection direction, const uint32_t hops) const {
    // private method -> chip id already validated
    const auto& src_chip_coords = this->physical_chip_coords_.at(src_phys_chip_id);
    const auto dst_chip_coords = src_chip_coords.get_dst_from_hops(direction, hops);
    return this->get_chip_from_coords(dst_chip_coords);
}

inline chip_id_t TestPhysicalMesh::get_chip_from_hops(
    const chip_id_t src_phys_chip_id, std::unordered_map<RoutingDirection, uint32_t>& hops) const {
    // private method -> chip id already validated
    const auto& src_chip_coords = this->physical_chip_coords_.at(src_phys_chip_id);
    const auto dst_chip_coords = src_chip_coords.get_dst_from_hops(hops);
    return this->get_chip_from_coords(dst_chip_coords);
}

inline TestPhysicalMesh::TestPhysicalMesh(ControlPlane* control_plane_ptr, const MeshId mesh_id) : mesh_id_(mesh_id) {
    const auto& mesh_shape = control_plane_ptr->get_physical_mesh_shape(mesh_id);
    const auto num_rows = mesh_shape[0];
    const auto num_cols = mesh_shape[1];

    this->set_mesh_dims_and_size({num_rows, num_cols});

    this->physical_chip_view_.resize(num_rows, std::vector<chip_id_t>(num_cols));
    chip_id_t logical_chip_id = 0;
    for (uint32_t i = 0; i < num_rows; i++) {
        for (uint32_t j = 0; j < num_cols; j++) {
            FabricNodeId fabric_node_id{mesh_id, logical_chip_id};
            chip_id_t physical_chip_id = control_plane_ptr->get_physical_chip_id_from_fabric_node_id(fabric_node_id);
            this->physical_chip_view_[i][j] = physical_chip_id;
            this->physical_chip_coords_.emplace(physical_chip_id, std::make_tuple(i, j));
            logical_chip_id++;
        }
    }
}

inline std::unordered_map<RoutingDirection, uint32_t> TestPhysicalMesh::get_hops_to_chip(
    chip_id_t src_phys_chip_id, chip_id_t dst_phys_chip_id) const {
    this->validate_physical_chip_id(src_phys_chip_id);
    this->validate_physical_chip_id(dst_phys_chip_id);

    const auto& src_coords = this->physical_chip_coords_.at(src_phys_chip_id);
    const auto& dst_coords = this->physical_chip_coords_.at(dst_phys_chip_id);
    return src_coords.get_hops_to_dst(dst_coords);
}

inline std::vector<chip_id_t> TestPhysicalMesh::get_chips_from_hops(
    chip_id_t src_phys_chip_id,
    std::unordered_map<RoutingDirection, uint32_t> hops,
    ChipSendType chip_send_type) const {
    this->validate_physical_chip_id(src_phys_chip_id);

    std::vector<chip_id_t> dst_phys_chip_ids;
    if (chip_send_type == ChipSendType::CHIP_MULTICAST) {
        // TODO: once 2D mcast is supported, update the logic to return the range of chips
        for (const auto& direction : FabricContext::routing_directions) {
            for (uint32_t hop = 1; hop < hops[direction]; hop++) {
                dst_phys_chip_ids.push_back(this->get_chip_from_hops(src_phys_chip_id, direction, hop));
            }
        }
    } else if (chip_send_type == ChipSendType::CHIP_UNICAST) {
        dst_phys_chip_ids = {this->get_chip_from_hops(src_phys_chip_id, hops)};
    } else {
        tt::log_fatal(tt::LogTest, "Unknown chip send type: {} for getting chips from hops", chip_send_type);
        throw std::runtime_error("Unexpected chip send type");
    }

    return dst_phys_chip_ids;
}

inline void TestPhysicalMesh::print_mesh() const {
    for (const auto& row : this->physical_chip_view_) {
        tt::log_info(tt::LogTest, "{}", row);
    }
}

struct TestPhysicalMeshes {
public:
    void setup_physical_meshes();
    void print_meshes() const;
    std::vector<chip_id_t> get_other_chips_on_same_row(chip_id_t physical_chip_id);
    std::vector<chip_id_t> get_other_chips_on_same_col(chip_id_t physical_chip_id);
    std::unordered_map<RoutingDirection, uint32_t> get_hops_to_chip(
        const chip_id_t src_phys_chip_id, const chip_id_t dst_phys_chip_id) const;
    std::vector<chip_id_t> get_chips_from_hops(
        const chip_id_t src_phys_chip_id,
        std::unordered_map<RoutingDirection, uint32_t> hops,
        const ChipSendType chip_send_type) const;

private:
    void validate_mesh_id(const MeshId mesh_id) const;

    std::unordered_map<MeshId, TestPhysicalMesh> physical_meshes_;
    tt::tt_fabric::ControlPlane* control_plane_ptr_;
};

inline void TestPhysicalMeshes::validate_mesh_id(const MeshId mesh_id) const {
    // TODO: take in a string param for debug/log strings
    if (this->physical_meshes_.find(mesh_id) == this->physical_meshes_.end()) {
        tt::log_fatal(tt::LogTest, "Unknown mesh id: {}", mesh_id);
        throw std::runtime_error("Unexpected mesh id");
    }
}

inline void TestPhysicalMeshes::setup_physical_meshes() {
    this->control_plane_ptr_ = tt::tt_metal::MetalContext::instance().get_cluster().get_control_plane();
    const auto user_meshes = this->control_plane_ptr_->get_user_physical_mesh_ids();

    for (const auto& mesh_id : user_meshes) {
        this->physical_meshes_.insert(std::make_pair(mesh_id, TestPhysicalMesh(this->control_plane_ptr_, mesh_id)));
    }
}

inline void TestPhysicalMeshes::print_meshes() const {
    tt::log_info(tt::LogTest, "Printing physical meshes, (total: {})", this->physical_meshes_.size());
    for (const auto& [mesh_id, mesh] : this->physical_meshes_) {
        tt::log_info(tt::LogTest, "Mesh id: {}", mesh_id);
        mesh.print_mesh();
    }
}

/*
inline std::vector<chip_id_t> TestPhysicalMeshes::get_other_chips_on_same_row(chip_id_t physical_chip_id) {
    const auto& fabric_node_id = this->control_plane_->get_fabric_node_id_from_physical_chip_id(physical_chip_id);
    const auto mesh_id = fabric_node_id.mesh_id;

    this->validate_mesh_id(mesh_id);

    const auto [chip_row, chip_col] = this->get_chip_location_from_physical_id(mesh_id, physical_chip_id);
    std::vector<chip_id_t> chips = this->physical_chip_view_[mesh_id][chip_row];
    chips.erase(std::remove(chips.begin(), chips.end(), physical_chip_id), chips.end());

    return chips;
}

inline std::vector<chip_id_t> TestPhysicalMeshes::get_other_chips_on_same_col(chip_id_t physical_chip_id) {
    const auto& fabric_node_id = this->control_plane_->get_fabric_node_id_from_physical_chip_id(physical_chip_id);
    const auto mesh_id = fabric_node_id.mesh_id;

    this->validate_mesh_id(mesh_id);

    const auto [chip_row, chip_col] = this->get_chip_location_from_physical_id(mesh_id, physical_chip_id);
    std::vector<chip_id_t> chips;
    for (uint32_t i = 0; i < this->physical_mesh_dims_[mesh_id][ROW_IDX]; i++) {
        chips.push_back(this->physical_chip_view_[mesh_id][i][chip_col]);
    }
    chips.erase(std::remove(chips.begin(), chips.end(), physical_chip_id), chips.end());

    return chips;
}
*/

inline std::unordered_map<RoutingDirection, uint32_t> TestPhysicalMeshes::get_hops_to_chip(
    const chip_id_t src_phys_chip_id, const chip_id_t dst_phys_chip_id) const {
    const auto& src_mesh_id =
        this->control_plane_ptr_->get_fabric_node_id_from_physical_chip_id(src_phys_chip_id).mesh_id;
    const auto& dst_mesh_id =
        this->control_plane_ptr_->get_fabric_node_id_from_physical_chip_id(dst_phys_chip_id).mesh_id;

    // TODO: enable inter-mesh hop counts
    if (src_mesh_id != dst_mesh_id) {
        tt::log_fatal(tt::LogTest, "Inter-mesh hops not supported yet");
        throw std::runtime_error("Unexpected hops request b/w meshes");
    }

    this->validate_mesh_id(src_mesh_id);

    // TODO: enable when inter-mesh hops is enabled
    // this->validate_mesh_id(dst_mesh_id);

    return this->physical_meshes_.at(src_mesh_id).get_hops_to_chip(src_phys_chip_id, dst_phys_chip_id);
}

inline std::vector<chip_id_t> TestPhysicalMeshes::get_chips_from_hops(
    const chip_id_t src_phys_chip_id,
    std::unordered_map<RoutingDirection, uint32_t> hops,
    const ChipSendType chip_send_type) const {
    const auto& src_mesh_id =
        this->control_plane_ptr_->get_fabric_node_id_from_physical_chip_id(src_phys_chip_id).mesh_id;
    this->validate_mesh_id(src_mesh_id);

    return this->physical_meshes_.at(src_mesh_id).get_chips_from_hops(src_phys_chip_id, hops, chip_send_type);
}

}  // namespace fabric_tests
}  // namespace tt::tt_fabric
