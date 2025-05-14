// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/fabric_edm_types.hpp>
#include <tt-metalium/fabric_types.hpp>
#include <tt-metalium/mesh_graph.hpp>                   // FabricType
#include <umd/device/types/cluster_descriptor_types.h>  // chip_id_t
#include <tt-metalium/erisc_datamover_builder.hpp>
#include <vector>
#include "tt_metal/fabric/fabric_host_utils.hpp"

namespace tt::tt_fabric {

class FabricContext {
public:
    static constexpr auto routing_directions = {
        RoutingDirection::N, RoutingDirection::S, RoutingDirection::E, RoutingDirection::W};

    explicit FabricContext(tt::tt_metal::FabricConfig fabric_config);
    ~FabricContext() = default;

    bool is_wrap_around_mesh() const { return this->wrap_around_mesh_; }
    tt::tt_fabric::Topology get_fabric_topology() const { return this->topology_; }
    size_t get_fabric_channel_buffer_size_bytes() const { return this->channel_buffer_size_bytes_; }

    tt::tt_fabric::FabricEriscDatamoverConfig* get_fabric_router_config() const {
        TT_FATAL(this->router_config_ != nullptr, "Error, fabric router config is uninitialized");
        return this->router_config_.get();
    };

    void set_num_fabric_initialized_routers(chip_id_t chip_id, size_t num_routers);
    uint32_t get_num_fabric_initialized_routers(chip_id_t chip_id) const;

    void set_fabric_master_router_chan(chip_id_t chip_id, chan_id_t chan_id);
    chan_id_t get_fabric_master_router_chan(chip_id_t chip_id) const;

    std::vector<size_t> get_fabric_router_addresses_to_clear() const {
        if (is_tt_fabric_config(this->fabric_config_)) {
            return {this->router_config_->edm_local_sync_address};
        } else {
            return {tt::tt_metal::hal::get_erisc_l1_unreserved_base()};
        }
    }

    std::pair<uint32_t, uint32_t> get_fabric_router_sync_address_and_status(chip_id_t chip_id) const {
        if (is_tt_fabric_config(this->fabric_config_)) {
            return std::make_pair(
                this->router_config_->edm_status_address, tt::tt_fabric::EDMStatus::LOCAL_HANDSHAKE_COMPLETE);
        } else {
            return std::make_pair(
                tt::tt_metal::hal::get_erisc_l1_unreserved_base(), get_num_fabric_initialized_routers(chip_id));
        }
    }

    std::optional<std::pair<uint32_t, tt::tt_fabric::EDMStatus>> get_fabric_router_ready_address_and_signal() const {
        if (is_tt_fabric_config(this->fabric_config_)) {
            return std::make_pair(
                this->router_config_->edm_status_address, tt::tt_fabric::EDMStatus::READY_FOR_TRAFFIC);
        } else {
            return std::nullopt;
        }
    }

    std::pair<uint32_t, uint32_t> get_fabric_router_termination_address_and_signal() const {
        if (is_tt_fabric_config(this->fabric_config_)) {
            return std::make_pair(
                this->router_config_->termination_signal_address,
                tt::tt_fabric::TerminationSignal::IMMEDIATELY_TERMINATE);
        } else {
            return std::make_pair(tt::tt_metal::hal::get_erisc_l1_unreserved_base(), 0);
        }
    }

private:
    bool check_for_wrap_around_mesh() const;
    tt::tt_fabric::Topology get_topology() const;
    uint32_t get_channel_buffer_size_bytes() const;

    bool initialized_ = false;
    tt::tt_metal::FabricConfig fabric_config_{};

    bool wrap_around_mesh_ = false;
    tt::tt_fabric::Topology topology_{};
    size_t channel_buffer_size_bytes_ = 0;
    std::unique_ptr<tt::tt_fabric::FabricEriscDatamoverConfig> router_config_ = nullptr;
    std::unordered_map<chip_id_t, chan_id_t> master_router_chans_{};
    std::unordered_map<chip_id_t, uint32_t> num_initialized_routers_{};
};

}  // namespace tt::tt_fabric
