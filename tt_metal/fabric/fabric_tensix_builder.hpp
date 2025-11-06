// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <optional>
#include <vector>
#include <memory>
#include <unordered_map>

#include <tt-metalium/device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/fabric.hpp>
#include "llrt/core_descriptor.hpp"
#include "tt_metal/fabric/erisc_datamover_builder.hpp"
#include "tt_metal/fabric/fabric_tensix_builder_impl.hpp"
#include "core_coord.hpp"

namespace tt::tt_fabric {

// Core type enum for fabric tensix datamover (identifies MUX vs RELAY cores)
enum class FabricTensixCoreType : uint8_t {
    MUX = 0,   // BRISC - runs MUX kernel
    RELAY = 1  // NCRISC - runs Relay kernel (UDM mode only)
};

}  // namespace tt::tt_fabric

// Hash function for FabricTensixCoreType to use as unordered_map key
namespace std {
template <>
struct hash<tt::tt_fabric::FabricTensixCoreType> {
    size_t operator()(const tt::tt_fabric::FabricTensixCoreType& id) const { return static_cast<size_t>(id); }
};
}  // namespace std

namespace tt::tt_fabric {

class FabricTensixDatamoverConfig {
public:
    FabricTensixDatamoverConfig();

    // Getters for core and channel configuration
    size_t get_num_configs_per_core() const { return num_configs_per_core_; }
    size_t get_num_riscs_per_core() const { return num_used_riscs_per_tensix_; }
    uint8_t get_num_buffers_per_channel() const { return num_buffers_per_channel_; }
    size_t get_buffer_size_bytes_full_size_channel() const { return buffer_size_bytes_full_size_channel_; }

    // Get base L1 address for a core type
    size_t get_base_l1_address(FabricTensixCoreType core_id) const;

    // Get NOC coordinates for ethernet channel (requires device)
    std::pair<uint32_t, uint32_t> get_noc_xy(tt::tt_metal::IDevice* device, uint32_t eth_chan_id) const;

    // Get channel base address for mux channel ID
    size_t get_channels_base_address(FabricTensixCoreType core_id, uint8_t tensix_channel_id) const;

    // Get the core type for a given ethernet channel on a specific device
    FabricTensixCoreType get_core_id_for_channel(ChipId device_id, uint32_t eth_chan_id) const;

    // Get the core for a given ethernet channel on a specific device
    CoreCoord get_core_for_channel(ChipId device_id, uint32_t eth_chan_id) const;

    // Get the config for a specific core type (returns base config pointer)
    std::shared_ptr<FabricTensixDatamoverBaseConfig> get_config(FabricTensixCoreType core_id) const;

    // Check if a core type is active (has channels)
    bool is_core_id_active(FabricTensixCoreType core_id) const;

    // Get translated fabric mux cores
    const std::unordered_set<CoreCoord>& get_translated_fabric_or_dispatch_mux_cores() const {
        return translated_fabric_or_dispatch_mux_cores_;
    }

    const std::unordered_set<CoreCoord>& get_translated_fabric_mux_cores() const {
        return translated_fabric_mux_cores_;
    }

    const std::unordered_set<CoreCoord>& get_translated_dispatch_mux_cores() const {
        return translated_dispatch_mux_cores_;
    }

    // Wrapper APIs for config access - takes channel_id and core_id
    // Callers must explicitly specify which core type (MUX or RELAY) they want to access
    size_t get_local_flow_control_semaphore_address(uint32_t channel_id, FabricTensixCoreType core_id) const;

    size_t get_connection_semaphore_address(uint32_t channel_id, FabricTensixCoreType core_id) const;

    size_t get_worker_conn_info_base_address(uint32_t channel_id, FabricTensixCoreType core_id) const;

    size_t get_buffer_index_semaphore_address(uint32_t channel_id, FabricTensixCoreType core_id) const;

    size_t get_channel_credits_stream_id(uint32_t channel_id, FabricTensixCoreType core_id) const;

    std::pair<uint32_t, uint32_t> get_termination_address_and_signal(FabricTensixCoreType core_id) const;

private:
    std::vector<CoreCoord> logical_fabric_mux_cores_;
    std::vector<CoreCoord> logical_dispatch_mux_cores_;
    std::unordered_set<CoreCoord> translated_fabric_mux_cores_;
    std::unordered_set<CoreCoord> translated_dispatch_mux_cores_;
    std::unordered_set<CoreCoord> translated_fabric_or_dispatch_mux_cores_;

    // based on the number of channels used, get the number of risc needed per tensix
    size_t num_used_riscs_per_tensix_{};

    // Configuration parameters
    size_t num_configs_per_core_{};
    size_t num_channels_for_mux_{};  // Number of channels for MUX configuration
    size_t num_buffers_per_channel_{};
    size_t buffer_size_bytes_full_size_channel_{};

    // Base L1 addresses for each RISC ID, [risc id] -> [base addr] mapping
    std::unordered_map<FabricTensixCoreType, size_t> base_l1_addresses_;

    // [device_id][eth chan] -> [core index] mapping for round-robin assignment
    std::unordered_map<ChipId, std::unordered_map<size_t, size_t>> eth_chan_to_core_index_;

    // [device_id][eth chan] -> [core type] mapping
    std::unordered_map<ChipId, std::unordered_map<size_t, FabricTensixCoreType>> eth_chan_to_core_id_;

    // Configs per core type, [core type] -> [config] mapping
    // In MUX mode: only MUX has config
    // In UDM mode: both MUX and RELAY have configs
    std::unordered_map<FabricTensixCoreType, std::shared_ptr<FabricTensixDatamoverBaseConfig>> configs_;

    // Helper methods for initialization
    bool initialize_channel_mappings();
    void calculate_buffer_allocations();
    void create_configs();  // Creates mode-aware configs based on FabricTensixConfig

    // Helper methods for config creation
    std::shared_ptr<FabricTensixDatamoverMuxConfig> create_mux_config(FabricTensixCoreType core_id);
    std::shared_ptr<FabricTensixDatamoverRelayConfig> create_relay_config(FabricTensixCoreType core_id);
};

/**
 * FabricTensixDatamoverBuilder
 * - Top-level builder that orchestrates mux and relay builders based on FabricTensixConfig mode
 * - MUX mode: creates only mux builder
 * - UDM mode: creates both mux and relay builders
 */
class FabricTensixDatamoverBuilder {
public:
    // Static builder method called from topology to construct a tensix builder
    static FabricTensixDatamoverBuilder build(
        tt::tt_metal::IDevice* device,
        tt::tt_metal::Program& program,
        tt::tt_fabric::FabricNodeId local_fabric_node_id,
        tt::tt_fabric::FabricNodeId remote_fabric_node_id,
        uint32_t ethernet_channel_id,
        eth_chan_directions direction);

    // Create and compile the kernel(s) based on mode
    void create_and_compile(tt::tt_metal::IDevice* device, tt::tt_metal::Program& program);

    // Build connection to fabric channel - returns connection specs for EDMs to connect to this mux
    tt::tt_fabric::SenderWorkerAdapterSpec build_connection_to_fabric_channel(uint32_t channel_id) const;

    // Getters - delegate to mux builder (primary builder)
    const CoreCoord& get_logical_core() const;
    tt::tt_fabric::FabricNodeId get_local_fabric_node_id() const;
    tt::tt_fabric::FabricNodeId get_remote_fabric_node_id() const;
    uint32_t get_ethernet_channel_id() const;
    FabricTensixCoreType get_core_id() const;
    uint32_t get_noc_x() const;
    uint32_t get_noc_y() const;
    eth_chan_directions get_direction() const;

    void append_upstream_routers_noc_xy(uint32_t noc_x, uint32_t noc_y);

private:
    // Private constructor - use build() factory method
    FabricTensixDatamoverBuilder(
        std::unique_ptr<FabricTensixDatamoverMuxBuilder> mux_builder,
        std::unique_ptr<FabricTensixDatamoverRelayBuilder> relay_builder,
        const CoreCoord& logical_core,
        tt::tt_fabric::FabricNodeId local_fabric_node_id,
        tt::tt_fabric::FabricNodeId remote_fabric_node_id,
        uint32_t ethernet_channel_id,
        uint32_t noc_x,
        uint32_t noc_y,
        eth_chan_directions direction);

    // Helper function to create builders based on core type
    template <typename BuilderType, typename ConfigType>
    static std::unique_ptr<BuilderType> create_builder(
        FabricTensixCoreType core_type,
        const FabricTensixDatamoverConfig& tensix_config,
        const CoreCoord& my_core_logical,
        tt::tt_fabric::FabricNodeId local_fabric_node_id,
        tt::tt_fabric::FabricNodeId remote_fabric_node_id,
        uint32_t ethernet_channel_id,
        uint32_t link_idx,
        uint32_t noc_x,
        uint32_t noc_y,
        eth_chan_directions direction);

    // Sub-builders based on mode
    std::unique_ptr<FabricTensixDatamoverMuxBuilder> mux_builder_;      // Always created
    std::unique_ptr<FabricTensixDatamoverRelayBuilder> relay_builder_;  // Only in UDM mode

    // Common properties shared by both mux and relay builders
    CoreCoord logical_core_;
    tt::tt_fabric::FabricNodeId local_fabric_node_id_;
    tt::tt_fabric::FabricNodeId remote_fabric_node_id_;
    uint32_t ethernet_channel_id_;
    uint32_t noc_x_;
    uint32_t noc_y_;
    eth_chan_directions direction_;
};

}  // namespace tt::tt_fabric
