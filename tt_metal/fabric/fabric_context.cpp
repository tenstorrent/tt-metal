// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <unordered_map>
#include <vector>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/fabric_edm_types.hpp>
#include <tt-metalium/fabric_types.hpp>
#include <tt-metalium/assert.hpp>
#include <tt-metalium/erisc_datamover_builder.hpp>
#include <magic_enum/magic_enum.hpp>
#include <umd/device/types/cluster_descriptor_types.h>  // chip_id_t
#include "tt_metal/fabric/fabric_context.hpp"
#include "impl/context/metal_context.hpp"

namespace tt::tt_fabric {

bool FabricContext::check_for_wrap_around_mesh() const {
    if (tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() == tt::ClusterType::TG) {
        // skip wrapping around mesh for TG since the corner chips connected to the gateway will be
        // using that link to route dispatch or any other traffic
        return false;
    }

    auto* control_plane = tt::tt_metal::MetalContext::instance().get_cluster().get_control_plane();
    auto mesh_id = control_plane->get_user_physical_mesh_ids()[0];

    // we can wrap around mesh if the corner chip (logical chip 0) as exactly 2 connections
    const uint32_t corner_chip_id = 0;
    uint32_t corner_chip_connections = 0;
    for (const auto& direction : FabricContext::routing_directions) {
        if (!control_plane->get_intra_chip_neighbors(mesh_id, corner_chip_id, direction).empty()) {
            corner_chip_connections++;
        }
    }

    return (corner_chip_connections == 2);
}

tt::tt_fabric::Topology FabricContext::get_topology() const {
    switch (this->fabric_config_) {
        case tt::tt_metal::FabricConfig::FABRIC_1D: return tt::tt_fabric::Topology::Linear;
        case tt::tt_metal::FabricConfig::FABRIC_1D_RING: return tt::tt_fabric::Topology::Ring;
        case tt::tt_metal::FabricConfig::FABRIC_2D_PUSH: return tt::tt_fabric::Topology::Mesh;
        case tt::tt_metal::FabricConfig::FABRIC_2D: return tt::tt_fabric::Topology::Mesh;
        case tt::tt_metal::FabricConfig::DISABLED:
        case tt::tt_metal::FabricConfig::CUSTOM:
            TT_THROW("Unsupported fabric config: {}", magic_enum::enum_name(this->fabric_config_));
    }
    return tt::tt_fabric::Topology::Linear;
}

uint32_t FabricContext::get_channel_buffer_size_bytes() const {
    if (this->topology_ == Topology::Mesh) {
        return tt::tt_fabric::FabricEriscDatamoverBuilder::default_mesh_packet_payload_size_bytes +
               sizeof(tt::tt_fabric::LowLatencyMeshPacketHeader);
    } else {
        return tt::tt_fabric::FabricEriscDatamoverBuilder::default_packet_payload_size_bytes +
               sizeof(tt::tt_fabric::PacketHeader);
    }
}

FabricContext::FabricContext(tt::tt_metal::FabricConfig fabric_config) {
    TT_FATAL(
        fabric_config != tt::tt_metal::FabricConfig::DISABLED,
        "Trying to initialize fabric context for disabled fabric config");

    this->fabric_config_ = fabric_config;

    this->wrap_around_mesh_ = this->check_for_wrap_around_mesh();
    this->topology_ = this->get_topology();
    this->channel_buffer_size_bytes_ = this->get_channel_buffer_size_bytes();

    if (is_tt_fabric_config(this->fabric_config_)) {
        this->router_config_ = std::make_unique<tt::tt_fabric::FabricEriscDatamoverConfig>(
            this->channel_buffer_size_bytes_, this->topology_);
    } else {
        this->router_config_ = nullptr;
    }
}

void FabricContext::set_num_fabric_initialized_routers(chip_id_t chip_id, size_t num_routers) {
    auto it = this->num_initialized_routers_.find(chip_id);
    TT_FATAL(
        it == this->num_initialized_routers_.end(),
        "Error, tried to num initialized routers again for the same device");
    this->num_initialized_routers_[chip_id] = num_routers;
}

uint32_t FabricContext::get_num_fabric_initialized_routers(chip_id_t chip_id) const {
    auto it = this->num_initialized_routers_.find(chip_id);
    TT_FATAL(
        it != this->num_initialized_routers_.end(), "Error, querying num initialized routers for an unknown device");
    return it->second;
}

void FabricContext::set_fabric_master_router_chan(chip_id_t chip_id, chan_id_t chan_id) {
    auto it = this->master_router_chans_.find(chip_id);
    TT_FATAL(
        it == this->master_router_chans_.end(), "Error, tried to set master router channel again for the same device");
    this->master_router_chans_[chip_id] = chan_id;
}

chan_id_t FabricContext::get_fabric_master_router_chan(chip_id_t chip_id) const {
    auto it = this->master_router_chans_.find(chip_id);
    TT_FATAL(it != this->master_router_chans_.end(), "Error, querying master router channel for an unknown device");
    return it->second;
}

}  // namespace tt::tt_fabric
