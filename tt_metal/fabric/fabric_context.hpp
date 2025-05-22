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

    bool is_wrap_around_mesh(mesh_id_t mesh_id) const;

    tt::tt_fabric::Topology get_fabric_topology() const;

    size_t get_fabric_packet_header_size_bytes() const;
    size_t get_fabric_max_payload_size_bytes() const;
    size_t get_fabric_channel_buffer_size_bytes() const;

    tt::tt_fabric::FabricEriscDatamoverConfig& get_fabric_router_config(bool is_dateline = false) const;

    void set_num_fabric_initialized_routers(chip_id_t chip_id, size_t num_routers);
    uint32_t get_num_fabric_initialized_routers(chip_id_t chip_id) const;

    void set_fabric_master_router_chan(chip_id_t chip_id, chan_id_t chan_id);
    chan_id_t get_fabric_master_router_chan(chip_id_t chip_id) const;

    std::vector<size_t> get_fabric_router_addresses_to_clear() const;

    std::pair<uint32_t, uint32_t> get_fabric_router_sync_address_and_status() const;

    std::optional<std::pair<uint32_t, tt::tt_fabric::EDMStatus>> get_fabric_router_ready_address_and_signal() const;

    std::pair<uint32_t, uint32_t> get_fabric_router_termination_address_and_signal() const;

private:
    std::unordered_map<mesh_id_t, bool> check_for_wrap_around_mesh() const;
    tt::tt_fabric::Topology get_topology() const;
    size_t get_packet_header_size_bytes() const;
    size_t get_max_payload_size_bytes() const;

    bool initialized_ = false;
    tt::tt_metal::FabricConfig fabric_config_{};

    std::unordered_map<mesh_id_t, bool> wrap_around_mesh_;
    tt::tt_fabric::Topology topology_{};
    size_t packet_header_size_bytes_ = 0;
    size_t max_payload_size_bytes_ = 0;
    size_t channel_buffer_size_bytes_ = 0;
    std::unique_ptr<tt::tt_fabric::FabricEriscDatamoverConfig> router_config_ = nullptr;
    std::unique_ptr<tt::tt_fabric::FabricEriscDatamoverConfig> dateline_router_config_ = nullptr;
    std::unordered_map<chip_id_t, chan_id_t> master_router_chans_{};
    std::unordered_map<chip_id_t, uint32_t> num_initialized_routers_{};
};

}  // namespace tt::tt_fabric
