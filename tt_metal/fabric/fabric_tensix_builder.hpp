// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
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
#include <tt-metalium/core_descriptor.hpp>
#include "tt_metal/fabric/erisc_datamover_builder.hpp"
#include "core_coord.hpp"

namespace tt::tt_fabric {

class FabricTensixDatamoverConfig {
public:
    FabricTensixDatamoverConfig();

    // Getters for core and channel configuration
    size_t get_num_configs_per_core() const { return num_configs_per_core_; }
    size_t get_num_riscs_per_core() const { return num_used_riscs_per_tensix_; }
    uint8_t get_num_buffers_per_channel() const { return num_buffers_per_channel_; }
    size_t get_buffer_size_bytes_full_size_channel() const { return buffer_size_bytes_full_size_channel_; }

    // Get base L1 address for a RISC ID
    size_t get_base_l1_address(size_t risc_id) const;

    // Get NOC coordinates for ethernet channel (requires device)
    std::pair<uint32_t, uint32_t> get_noc_xy(tt::tt_metal::IDevice* device, uint32_t eth_chan_id) const;

    // Get channel base address for mux channel ID
    size_t get_channels_base_address(size_t risc_id, uint8_t tensix_channel_id) const;

    // Get the RISC ID for a given ethernet channel on a specific device
    size_t get_risc_id_for_channel(chip_id_t device_id, uint32_t eth_chan_id) const;

    // Get the core for a given ethernet channel on a specific device
    CoreCoord get_core_for_channel(chip_id_t device_id, uint32_t eth_chan_id) const;

    // Get the mux config for a specific RISC ID
    std::shared_ptr<tt::tt_fabric::FabricMuxConfig> get_mux_config(size_t risc_id) const;

    // Check if a RISC ID is active (has channels)
    bool is_risc_id_active(size_t risc_id) const;

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

    // Wrapper APIs for mux config access - these takes device_id, eth_chan_id and channel_id (channels inside a mux)
    size_t get_local_flow_control_semaphore_address(
        chip_id_t device_id, uint32_t eth_chan_id, uint32_t channel_id) const;
    size_t get_connection_semaphore_address(chip_id_t device_id, uint32_t eth_chan_id, uint32_t channel_id) const;
    size_t get_worker_conn_info_base_address(chip_id_t device_id, uint32_t eth_chan_id, uint32_t channel_id) const;
    size_t get_buffer_index_semaphore_address(chip_id_t device_id, uint32_t eth_chan_id, uint32_t channel_id) const;
    size_t get_channel_credits_stream_id(chip_id_t device_id, uint32_t eth_chan_id, uint32_t channel_id) const;
    std::pair<uint32_t, uint32_t> get_termination_address_and_signal(chip_id_t device_id, uint32_t eth_chan_id) const;

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
    size_t num_channels_{};
    size_t num_buffers_per_channel_{};
    size_t buffer_size_bytes_full_size_channel_{};

    // Base L1 addresses for each RISC ID, [risc id] -> [base addr] mapping
    std::unordered_map<size_t, size_t> base_l1_addresses_;

    // [device_id][eth chan] -> [core index] mapping for round-robin assignment
    std::unordered_map<chip_id_t, std::unordered_map<size_t, size_t>> eth_chan_to_core_index_;

    // [device_id][eth chan] -> [risc id] mapping
    std::unordered_map<chip_id_t, std::unordered_map<size_t, size_t>> eth_chan_to_risc_id_;

    // Mux configs per RISC ID, [risc id] -> [mux config] mapping
    std::unordered_map<size_t, std::shared_ptr<tt::tt_fabric::FabricMuxConfig>> mux_configs_;

    // Helper methods for initialization
    void initialize_channel_mappings();
    void calculate_buffer_allocations();
    void create_mux_configs();
};

/**
 * FabricTensixDatamoverBuilder
 * - Builds mux kernels on fabric tensix cores for worker → mux → fabric router routing.
 * - Build the connections between Fabric routers, fabric router -> mux -> downstream fabric router.
 */
class FabricTensixDatamoverBuilder {
public:
    // Constructor for fabric tensix datamover builder
    FabricTensixDatamoverBuilder(
        const CoreCoord& my_core_logical,
        tt::tt_fabric::FabricNodeId local_fabric_node_id,
        tt::tt_fabric::FabricNodeId remote_fabric_node_id,
        uint32_t ethernet_channel_id,
        uint32_t link_idx,
        size_t risc_id,
        uint32_t noc_x,
        uint32_t noc_y,
        std::shared_ptr<tt::tt_fabric::FabricMuxConfig> fabric_mux_config,
        eth_chan_directions direction);

    // Static builder method called from topology to construct a tensix builder
    static FabricTensixDatamoverBuilder build(
        tt::tt_metal::IDevice* device,
        tt::tt_metal::Program& program,
        tt::tt_fabric::FabricNodeId local_fabric_node_id,
        tt::tt_fabric::FabricNodeId remote_fabric_node_id,
        uint32_t ethernet_channel_id,
        eth_chan_directions direction);

    // Create and compile the mux kernel
    void create_and_compile(tt::tt_metal::IDevice* device, tt::tt_metal::Program& program);

    // Build connection to fabric channel - returns connection specs for EDMs to connect to this mux
    tt::tt_fabric::SenderWorkerAdapterSpec build_connection_to_fabric_channel(uint32_t channel_id) const;

    // Getters
    const CoreCoord& get_logical_core() const { return my_core_logical_; }
    tt::tt_fabric::FabricNodeId get_local_fabric_node_id() const { return local_fabric_node_id_; }
    tt::tt_fabric::FabricNodeId get_remote_fabric_node_id() const { return remote_fabric_node_id_; }
    uint32_t get_ethernet_channel_id() const { return ethernet_channel_id_; }
    size_t get_risc_id() const { return risc_id_; }
    uint32_t get_noc_x() const { return noc_x_; }
    uint32_t get_noc_y() const { return noc_y_; }
    eth_chan_directions get_direction() const { return direction_; }

private:
    // Core and fabric configuration
    CoreCoord my_core_logical_;
    tt::tt_fabric::FabricNodeId local_fabric_node_id_;
    tt::tt_fabric::FabricNodeId remote_fabric_node_id_;
    uint32_t ethernet_channel_id_;
    uint32_t link_idx_;

    // RISC and NOC configuration
    size_t risc_id_;
    uint32_t noc_x_;
    uint32_t noc_y_;

    // Mux configuration
    std::shared_ptr<tt::tt_fabric::FabricMuxConfig> fabric_mux_config_;

    // Direction for routing
    eth_chan_directions direction_;

    // Channel connection liveness check disable array
    mutable std::array<bool, FabricEriscDatamoverConfig::num_sender_channels>
        channel_connection_liveness_check_disable_array_{};

    // Helper methods for kernel compilation
    std::vector<uint32_t> get_compile_time_args(tt::tt_metal::IDevice* device) const;
    std::vector<uint32_t> get_runtime_args(tt::tt_metal::Program& program) const;
};

}  // namespace tt::tt_fabric
