// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <unordered_map>
#include <vector>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/fabric_edm_types.hpp>
#include <tt-metalium/fabric_types.hpp>
#include <tt-metalium/assert.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/erisc_datamover_builder.hpp>
#include <magic_enum/magic_enum.hpp>
#include <umd/device/types/cluster_descriptor_types.h>  // chip_id_t
#include "tt_metal/fabric/fabric_context.hpp"
#include "impl/context/metal_context.hpp"

namespace tt::tt_fabric {

std::unordered_map<MeshId, bool> FabricContext::check_for_wrap_around_mesh() const {
    std::unordered_map<MeshId, bool> wrap_around_mesh;

    auto& control_plane= tt::tt_metal::MetalContext::instance().get_control_plane();
    auto mesh_ids = control_plane.get_user_physical_mesh_ids();
    for (const auto& mesh_id : mesh_ids) {
        if (tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() == tt::ClusterType::TG) {
            // skip wrapping around mesh for TG since the corner chips connected to the gateway will be
            // using that link to route dispatch or any other traffic
            wrap_around_mesh[mesh_id] = false;
            continue;
        }
        // we can wrap around mesh if the corner chip (logical chip 0) has exactly 2 connections
        const uint32_t corner_chip_id = 0;
        uint32_t corner_chip_connections = 0;
        for (const auto& direction : FabricContext::routing_directions) {
            if (!control_plane.get_intra_chip_neighbors(FabricNodeId(mesh_id, corner_chip_id), direction).empty()) {
                corner_chip_connections++;
            }
        }

        wrap_around_mesh[mesh_id] = (corner_chip_connections == 2);
    }
    return wrap_around_mesh;
}

tt::tt_fabric::Topology FabricContext::get_topology_from_config(tt::tt_fabric::FabricConfig fabric_config) {
    switch (fabric_config) {
        case tt::tt_fabric::FabricConfig::FABRIC_1D: return tt::tt_fabric::Topology::Linear;
        case tt::tt_fabric::FabricConfig::FABRIC_1D_RING: return tt::tt_fabric::Topology::Ring;
        case tt::tt_fabric::FabricConfig::FABRIC_2D: return tt::tt_fabric::Topology::Mesh;
        case tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS: return tt::tt_fabric::Topology::Torus;
        case tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC: return tt::tt_fabric::Topology::Mesh;
        case tt::tt_fabric::FabricConfig::DISABLED:
        case tt::tt_fabric::FabricConfig::CUSTOM:
            TT_THROW("Unsupported fabric config: {}", magic_enum::enum_name(fabric_config));
    }
    return tt::tt_fabric::Topology::Linear;
}

size_t FabricContext::get_packet_header_size_bytes() const {
    if (this->topology_ == Topology::Mesh) {
        return (this->fabric_config_ == tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC)
                   ? sizeof(tt::tt_fabric::MeshPacketHeader)
                   : sizeof(tt::tt_fabric::LowLatencyMeshPacketHeader);
    } else {
        return sizeof(tt::tt_fabric::PacketHeader);
    }
}

size_t FabricContext::get_max_payload_size_bytes() const {
    if (this->topology_ == Topology::Mesh) {
        return tt::tt_fabric::FabricEriscDatamoverBuilder::default_mesh_packet_payload_size_bytes;
    } else {
        return tt::tt_fabric::FabricEriscDatamoverBuilder::default_packet_payload_size_bytes;
    }
}

std::unique_ptr<tt::tt_fabric::FabricEriscDatamoverConfig> FabricContext::get_edm_config_options(
    tt::tt_fabric::FabricEriscDatamoverType edm_type, tt::tt_fabric::FabricEriscDatamoverAxis edm_axis) {
    auto edm_buffer_config = tt::tt_fabric::FabricRouterBufferConfig{
        .enable_dateline_sender_extra_buffer_slots = true,
        .enable_dateline_receiver_extra_buffer_slots = true,
        .enable_dateline_upstream_sender_extra_buffer_slots = true,
        .enable_dateline_upstream_receiver_extra_buffer_slots = true,
        .enable_dateline_upstream_adjacent_sender_extra_buffer_slots =
            edm_axis != tt::tt_fabric::FabricEriscDatamoverAxis::Short,
    };
    auto edm_options = tt::tt_fabric::FabricEriscDatamoverOptions{
        .edm_type = edm_type,
        .edm_axis = edm_axis,
        .edm_buffer_config = edm_buffer_config,
    };

    return std::make_unique<tt::tt_fabric::FabricEriscDatamoverConfig>(
        this->channel_buffer_size_bytes_, this->topology_, edm_options);
}

FabricContext::FabricContext(tt::tt_fabric::FabricConfig fabric_config) {
    TT_FATAL(
        fabric_config != tt::tt_fabric::FabricConfig::DISABLED,
        "Trying to initialize fabric context for disabled fabric config");

    this->fabric_config_ = fabric_config;

    this->wrap_around_mesh_ = this->check_for_wrap_around_mesh();
    this->topology_ = this->get_topology_from_config(fabric_config);

    this->packet_header_size_bytes_ = this->get_packet_header_size_bytes();
    this->max_payload_size_bytes_ = this->get_max_payload_size_bytes();
    this->channel_buffer_size_bytes_ = this->packet_header_size_bytes_ + this->max_payload_size_bytes_;

    auto short_axis = static_cast<std::size_t>(tt::tt_fabric::FabricEriscDatamoverAxis::Short);
    auto long_axis = static_cast<std::size_t>(tt::tt_fabric::FabricEriscDatamoverAxis::Long);

    // default router config don't care about the axis, since there's no optimization to it.
    this->router_config_ = get_edm_config_options(
        tt::tt_fabric::FabricEriscDatamoverType::Default, tt::tt_fabric::FabricEriscDatamoverAxis::Short);

    // dateline edm router
    this->dateline_router_config_[short_axis] = get_edm_config_options(
        tt::tt_fabric::FabricEriscDatamoverType::Dateline, tt::tt_fabric::FabricEriscDatamoverAxis::Short);
    this->dateline_router_config_[long_axis] = get_edm_config_options(
        tt::tt_fabric::FabricEriscDatamoverType::Dateline, tt::tt_fabric::FabricEriscDatamoverAxis::Long);

    // dateline upstream edm router
    this->dateline_upstream_router_config_[short_axis] = get_edm_config_options(
        tt::tt_fabric::FabricEriscDatamoverType::DatelineUpstream, tt::tt_fabric::FabricEriscDatamoverAxis::Short);
    this->dateline_upstream_router_config_[long_axis] = get_edm_config_options(
        tt::tt_fabric::FabricEriscDatamoverType::DatelineUpstream, tt::tt_fabric::FabricEriscDatamoverAxis::Long);

    // dateline upstream adjacent edm router
    this->dateline_upstream_adjcent_router_config_[short_axis] = get_edm_config_options(
        tt::tt_fabric::FabricEriscDatamoverType::DatelineUpstreamAdjacentDevice,
        tt::tt_fabric::FabricEriscDatamoverAxis::Short);
    this->dateline_upstream_adjcent_router_config_[long_axis] = get_edm_config_options(
        tt::tt_fabric::FabricEriscDatamoverType::DatelineUpstreamAdjacentDevice,
        tt::tt_fabric::FabricEriscDatamoverAxis::Long);

    // dateline upstream adjacent upstream edm router
    this->dateline_upstream_adjcent_upstream_router_config_[short_axis] = get_edm_config_options(
        tt::tt_fabric::FabricEriscDatamoverType::DatelineUpstreamAdjacentDeviceUpstream,
        tt::tt_fabric::FabricEriscDatamoverAxis::Short);
    this->dateline_upstream_adjcent_upstream_router_config_[long_axis] = get_edm_config_options(
        tt::tt_fabric::FabricEriscDatamoverType::DatelineUpstreamAdjacentDeviceUpstream,
        tt::tt_fabric::FabricEriscDatamoverAxis::Long);

    this->num_devices = tt::tt_metal::GetNumAvailableDevices();
    auto num_pcie_devices = tt::tt_metal::GetNumPCIeDevices();
    if (this->num_devices != 4 && num_pcie_devices == 4) {
        // adding TG's 4 dispatch devices
        this->num_devices += num_pcie_devices;
    }
    this->master_router_chans_.resize(num_devices, UNINITIALIZED_MASTER_ROUTER_CHAN);
    this->num_initialized_routers_.resize(num_devices, UNINITIALIZED_ROUTERS);

    set_routing_mode(this->topology_, this->fabric_config_);
}

bool FabricContext::is_wrap_around_mesh(MeshId mesh_id) const {
    auto it = this->wrap_around_mesh_.find(mesh_id);
    TT_FATAL(it != this->wrap_around_mesh_.end(), "Querying wrap around mesh for an unknown mesh id");
    return it->second;
}

tt::tt_fabric::Topology FabricContext::get_fabric_topology() const { return this->topology_; }

size_t FabricContext::get_fabric_packet_header_size_bytes() const { return this->packet_header_size_bytes_; }

size_t FabricContext::get_fabric_max_payload_size_bytes() const { return this->max_payload_size_bytes_; }

size_t FabricContext::get_fabric_channel_buffer_size_bytes() const { return this->channel_buffer_size_bytes_; }

tt::tt_fabric::FabricEriscDatamoverConfig& FabricContext::get_fabric_router_config(
    tt::tt_fabric::FabricEriscDatamoverType fabric_edm_type,
    tt::tt_fabric::FabricEriscDatamoverAxis fabric_edm_axis) const {
    auto axis_index = static_cast<std::size_t>(fabric_edm_axis);
    switch (fabric_edm_type) {
        case tt::tt_fabric::FabricEriscDatamoverType::Default:
            TT_FATAL(this->router_config_ != nullptr, "Error, fabric router config is uninitialized");
            return *this->router_config_.get();
            break;
        case tt::tt_fabric::FabricEriscDatamoverType::Dateline:
            TT_FATAL(
                this->dateline_router_config_[axis_index] != nullptr,
                "Error, fabric dateline router config is uninitialized");
            return *this->dateline_router_config_[axis_index].get();
            break;
        case tt::tt_fabric::FabricEriscDatamoverType::DatelineUpstream:
            TT_FATAL(
                this->dateline_upstream_router_config_[axis_index] != nullptr,
                "Error, fabric dateline upstream router config is uninitialized");
            return *this->dateline_upstream_router_config_[axis_index].get();
            break;
        case tt::tt_fabric::FabricEriscDatamoverType::DatelineUpstreamAdjacentDevice:
            TT_FATAL(
                this->dateline_upstream_adjcent_router_config_[axis_index] != nullptr,
                "Error, fabric dateline upstream adjacent device router config is uninitialized");
            return *this->dateline_upstream_adjcent_router_config_[axis_index].get();
            break;
        case tt::tt_fabric::FabricEriscDatamoverType::DatelineUpstreamAdjacentDeviceUpstream:
            TT_FATAL(
                this->dateline_upstream_adjcent_upstream_router_config_[axis_index] != nullptr,
                "Error, fabric dateline upstream adjacent device upstream router config is uninitialized");
            return *this->dateline_upstream_adjcent_upstream_router_config_[axis_index].get();
            break;
        default: TT_FATAL(false, "Error, invalid fabric edm type");
    }
};

void FabricContext::set_num_fabric_initialized_routers(chip_id_t chip_id, size_t num_routers) {
    TT_FATAL(chip_id < num_devices, "Device ID {} exceeds maximum supported devices {}", chip_id, num_devices);
    TT_FATAL(
        this->num_initialized_routers_[chip_id] == UNINITIALIZED_ROUTERS,
        "Error, tried to set num initialized routers again for device {}",
        chip_id);
    this->num_initialized_routers_[chip_id] = num_routers;
}

uint32_t FabricContext::get_num_fabric_initialized_routers(chip_id_t chip_id) const {
    TT_FATAL(chip_id < num_devices, "Device ID {} exceeds maximum supported devices {}", chip_id, num_devices);
    TT_FATAL(
        this->num_initialized_routers_[chip_id] != UNINITIALIZED_ROUTERS,
        "Error, querying num initialized routers for an unknown device {}",
        chip_id);
    return this->num_initialized_routers_[chip_id];
}

void FabricContext::set_fabric_master_router_chan(chip_id_t chip_id, chan_id_t chan_id) {
    TT_FATAL(chip_id < num_devices, "Device ID {} exceeds maximum supported devices {}", chip_id, num_devices);
    TT_FATAL(
        this->master_router_chans_[chip_id] == UNINITIALIZED_MASTER_ROUTER_CHAN,
        "Error, tried to set master router channel again for the same device {}",
        chip_id);
    this->master_router_chans_[chip_id] = chan_id;
}

chan_id_t FabricContext::get_fabric_master_router_chan(chip_id_t chip_id) const {
    TT_FATAL(chip_id < num_devices, "Device ID {} exceeds maximum supported devices {}", chip_id, num_devices);
    TT_FATAL(
        this->master_router_chans_[chip_id] != UNINITIALIZED_MASTER_ROUTER_CHAN,
        "Error, querying master router channel for an unknown device {}",
        chip_id);
    return this->master_router_chans_[chip_id];
}

std::vector<size_t> FabricContext::get_fabric_router_addresses_to_clear() const {
    return {this->router_config_->edm_local_sync_address};
}

std::pair<uint32_t, uint32_t> FabricContext::get_fabric_router_sync_address_and_status() const {
    return std::make_pair(this->router_config_->edm_status_address, tt::tt_fabric::EDMStatus::LOCAL_HANDSHAKE_COMPLETE);
}

std::optional<std::pair<uint32_t, tt::tt_fabric::EDMStatus>> FabricContext::get_fabric_router_ready_address_and_signal()
    const {
    return std::make_pair(this->router_config_->edm_status_address, tt::tt_fabric::EDMStatus::READY_FOR_TRAFFIC);
}

std::pair<uint32_t, uint32_t> FabricContext::get_fabric_router_termination_address_and_signal() const {
    return std::make_pair(
        this->router_config_->termination_signal_address, tt::tt_fabric::TerminationSignal::IMMEDIATELY_TERMINATE);
}

}  // namespace tt::tt_fabric
