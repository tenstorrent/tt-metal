// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/fabric_edm_types.hpp>
#include <tt-metalium/fabric_types.hpp>
#include <tt-metalium/mesh_graph.hpp>                   // FabricType
#include <umd/device/types/cluster_descriptor_types.h>  // chip_id_t
#include "erisc_datamover_builder.hpp"
#include <vector>
#include <limits>
#include "tt_metal/fabric/fabric_host_utils.hpp"
#include "tt_metal/fabric/fabric_tensix_builder.hpp"

namespace tt::tt_fabric {

class FabricContext {
public:
    static constexpr auto routing_directions = {
        RoutingDirection::N, RoutingDirection::S, RoutingDirection::E, RoutingDirection::W};

    explicit FabricContext(tt::tt_fabric::FabricConfig fabric_config);
    ~FabricContext() = default;

    bool is_wrap_around_mesh(MeshId mesh_id) const;

    static tt::tt_fabric::Topology get_topology_from_config(tt::tt_fabric::FabricConfig fabric_config);

    static bool is_2D_topology(tt::tt_fabric::Topology topology);
    static bool is_dynamic_routing_config(tt::tt_fabric::FabricConfig fabric_config);

    tt::tt_fabric::Topology get_fabric_topology() const;
    bool is_2D_routing_enabled() const;
    bool is_dynamic_routing_enabled() const;

    bool need_deadlock_avoidance_support(eth_chan_directions direction) const;

    size_t get_fabric_packet_header_size_bytes() const;
    size_t get_fabric_max_payload_size_bytes() const;
    size_t get_fabric_channel_buffer_size_bytes() const;

    tt::tt_fabric::FabricEriscDatamoverConfig& get_fabric_router_config(
        tt::tt_fabric::FabricEriscDatamoverType fabric_edm_type = tt::tt_fabric::FabricEriscDatamoverType::Default,
        tt::tt_fabric::FabricEriscDatamoverAxis fabric_edm_axis = tt::tt_fabric::FabricEriscDatamoverAxis::Short,
        tt::tt_fabric::FabricTensixConfig fabric_tensix_config = tt::tt_fabric::FabricTensixConfig::DISABLED,
        eth_chan_directions direction = eth_chan_directions::EAST) const;

    // Get fabric tensix config for mux configuration
    tt::tt_fabric::FabricTensixDatamoverConfig& get_tensix_config() const;

    // Initialize fabric tensix config (call after routing tables are configured)
    void initialize_tensix_config();

    void set_num_fabric_initialized_routers(chip_id_t chip_id, size_t num_routers);
    uint32_t get_num_fabric_initialized_routers(chip_id_t chip_id) const;

    void set_fabric_master_router_chan(chip_id_t chip_id, chan_id_t chan_id);
    chan_id_t get_fabric_master_router_chan(chip_id_t chip_id) const;

    std::vector<size_t> get_fabric_router_addresses_to_clear() const;

    std::pair<uint32_t, uint32_t> get_fabric_router_sync_address_and_status() const;

    std::optional<std::pair<uint32_t, tt::tt_fabric::EDMStatus>> get_fabric_router_ready_address_and_signal() const;

    std::pair<uint32_t, uint32_t> get_fabric_router_termination_address_and_signal() const;

private:
    std::unordered_map<MeshId, bool> check_for_wrap_around_mesh() const;
    tt::tt_fabric::Topology get_topology() const;
    size_t get_packet_header_size_bytes() const;
    size_t get_max_payload_size_bytes() const;
    std::unique_ptr<tt::tt_fabric::FabricEriscDatamoverConfig> get_edm_config_options(
        tt::tt_fabric::FabricEriscDatamoverType edm_type,
        tt::tt_fabric::FabricEriscDatamoverAxis edm_axis,
        tt::tt_fabric::FabricTensixConfig fabric_tensix_config = tt::tt_fabric::FabricTensixConfig::DISABLED,
        eth_chan_directions direction = eth_chan_directions::EAST);

    bool initialized_ = false;
    tt::tt_fabric::FabricConfig fabric_config_{};
    tt::tt_fabric::Topology topology_{};

    bool is_2D_routing_enabled_ = false;
    bool is_dynamic_routing_enabled_ = false;

    std::unordered_map<MeshId, bool> wrap_around_mesh_{};

    size_t packet_header_size_bytes_ = 0;
    size_t max_payload_size_bytes_ = 0;
    size_t channel_buffer_size_bytes_ = 0;
    std::unique_ptr<tt::tt_fabric::FabricEriscDatamoverConfig> router_config_ = nullptr;
    // these edm types will have different optimizations based on the number of devices along the axis.
    std::array<std::unique_ptr<tt::tt_fabric::FabricEriscDatamoverConfig>, 2> dateline_router_config_ = {};
    std::array<std::unique_ptr<tt::tt_fabric::FabricEriscDatamoverConfig>, 2> dateline_upstream_router_config_ = {};
    std::array<std::unique_ptr<tt::tt_fabric::FabricEriscDatamoverConfig>, 2> dateline_upstream_adjcent_router_config_ =
        {};

    std::array<std::unique_ptr<tt::tt_fabric::FabricEriscDatamoverConfig>, eth_chan_directions::COUNT>
        router_with_mux_config_ = {};  // for E W N S.

    // Tensix config for fabric mux configuration (same for all devices)
    std::unique_ptr<tt::tt_fabric::FabricTensixDatamoverConfig> tensix_config_;

    // Using vectors. Use Device IDs as indices
    size_t num_devices = 0;
    static constexpr chan_id_t UNINITIALIZED_MASTER_ROUTER_CHAN = std::numeric_limits<chan_id_t>::max();
    static constexpr uint32_t UNINITIALIZED_ROUTERS = std::numeric_limits<uint32_t>::max();
    // Use vector instead of unordered_map to be thread safe
    std::vector<chan_id_t> master_router_chans_;
    std::vector<uint32_t> num_initialized_routers_;
};

}  // namespace tt::tt_fabric
