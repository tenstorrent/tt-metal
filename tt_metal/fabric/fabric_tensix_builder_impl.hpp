// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <optional>
#include <vector>
#include <memory>
#include <unordered_map>
#include <array>

#include <tt-metalium/device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/fabric.hpp>
#include "llrt/core_descriptor.hpp"
#include "tt_metal/fabric/erisc_datamover_builder.hpp"
#include "core_coord.hpp"

namespace tt::tt_fabric {

// Forward declarations
enum class FabricMuxChannelType : uint8_t;
enum class FabricTensixRiscId : uint8_t;

/**
 * FabricTensixDatamoverBaseConfig
 * - Base class for mux and relay kernel configurations
 * - Forked from FabricMuxConfig to provide common functionality
 * - Manages memory layout, buffer allocation, and channel configuration
 */
class FabricTensixDatamoverBaseConfig {
protected:
    // Nested struct for memory management
    struct MemoryRegion {
        size_t base_address;
        size_t unit_size;
        size_t num_units;

        MemoryRegion() = default;
        MemoryRegion(size_t base, size_t unit_sz, size_t count);

        size_t get_address(size_t offset = 0) const;
        size_t get_end_address() const;
        size_t get_total_size() const;
    };

public:
    virtual ~FabricTensixDatamoverBaseConfig() = default;

    // Common getters
    uint8_t get_num_channels(FabricMuxChannelType channel_type) const;
    uint8_t get_num_buffers(FabricMuxChannelType channel_type) const;
    size_t get_buffer_size_bytes(FabricMuxChannelType channel_type) const;
    size_t get_status_address() const;
    size_t get_termination_signal_address() const;
    size_t get_channel_credits_stream_id(FabricMuxChannelType channel_type, uint8_t channel_id) const;
    size_t get_channel_base_address(FabricMuxChannelType channel_type, uint8_t channel_id) const;
    size_t get_connection_info_address(FabricMuxChannelType channel_type, uint8_t channel_id) const;
    size_t get_connection_handshake_address(FabricMuxChannelType channel_type, uint8_t channel_id) const;
    size_t get_flow_control_address(FabricMuxChannelType channel_type, uint8_t channel_id) const;
    size_t get_buffer_index_address(FabricMuxChannelType channel_type, uint8_t channel_id) const;
    size_t get_memory_map_end_address() const;

    // Configuration setters
    void set_num_full_size_channel_iters(size_t new_val);
    void set_num_iters_between_teardown_checks(size_t new_val);
    void set_wait_for_fabric_endpoint_ready(bool wait_for_ready);
    void set_fabric_endpoint_channel_num_buffers(size_t num_buffers);
    void set_fabric_endpoint_status_address(size_t address);

    // Returns vector of pairs of base addresses and size to clear
    std::vector<std::pair<size_t, size_t>> get_memory_regions_to_clear() const;

    // Pure virtual methods to be implemented by derived classes
    virtual std::vector<uint32_t> get_compile_time_args(
        const tt::tt_fabric::FabricEriscDatamoverConfig& fabric_router_config) const = 0;

    virtual std::vector<uint32_t> get_run_time_args(
        const FabricNodeId& src_fabric_node_id,
        const FabricNodeId& dst_fabric_node_id,
        uint32_t link_idx,
        tt::tt_metal::Program& program,
        const CoreCoord& logical_core) const;

protected:
    static constexpr uint8_t default_num_buffers = 8;
    static constexpr size_t default_num_full_size_channel_iters = 1;
    static constexpr size_t default_num_iters_between_teardown_checks = 32;

    // Constructor for derived classes
    FabricTensixDatamoverBaseConfig(
        uint8_t num_full_size_channels,
        uint8_t num_header_only_channels,
        uint8_t num_buffers_full_size_channel,
        uint8_t num_buffers_header_only_channel,
        size_t buffer_size_bytes_full_size_channel,
        size_t base_l1_address,
        CoreType core_type = CoreType::WORKER);

    // Helper methods
    void validate_channel_id(FabricMuxChannelType channel_type, uint8_t channel_id) const;
    uint8_t get_channel_global_offset(FabricMuxChannelType channel_type, uint8_t channel_id) const;
    void append_default_stream_ids_to_ct_args(std::vector<uint32_t>& ct_args) const;
    void append_default_persistent_channel_flags_to_ct_args(std::vector<uint32_t>& ct_args) const;
    std::vector<uint32_t> get_compile_time_main_args(
        const tt::tt_fabric::FabricEriscDatamoverConfig& fabric_router_config) const;

    // Configuration parameters
    CoreType core_type_ = CoreType::WORKER;
    uint8_t core_type_index_ = 0;
    uint8_t noc_aligned_address_size_bytes_ = 0;

    uint8_t num_full_size_channels_ = 0;
    uint8_t num_header_only_channels_ = 0;
    uint8_t num_buffers_full_size_channel_ = 0;
    uint8_t num_buffers_header_only_channel_ = 0;
    size_t buffer_size_bytes_full_size_channel_ = 0;
    size_t buffer_size_bytes_header_only_channel_ = 0;

    size_t full_size_channel_size_bytes_ = 0;
    size_t header_only_channel_size_bytes_ = 0;

    size_t num_full_size_channel_iters_ = default_num_full_size_channel_iters;
    size_t num_iters_between_teardown_checks_ = default_num_iters_between_teardown_checks;
    mutable bool wait_for_fabric_endpoint_ready_ = false;
    mutable size_t fabric_endpoint_channel_num_buffers_ = 0;
    mutable size_t fabric_endpoint_status_address_ = 0;

    // Memory regions
    MemoryRegion status_region_{};
    MemoryRegion local_fabric_router_status_region_{};
    MemoryRegion termination_signal_region_{};
    MemoryRegion connection_info_region_{};
    MemoryRegion connection_handshake_region_{};
    MemoryRegion flow_control_region_{};
    MemoryRegion buffer_index_region_{};
    MemoryRegion full_size_channels_region_{};
    MemoryRegion header_only_channels_region_{};

    size_t memory_map_end_address_ = 0;
};

/**
 * FabricTensixDatamoverMuxConfig
 * - Configuration for MUX kernel on BRISC
 * - Handles worker → mux → fabric router routing
 */
class FabricTensixDatamoverMuxConfig : public FabricTensixDatamoverBaseConfig {
public:
    FabricTensixDatamoverMuxConfig(
        uint8_t num_full_size_channels,
        uint8_t num_header_only_channels,
        uint8_t num_buffers_full_size_channel,
        uint8_t num_buffers_header_only_channel,
        size_t buffer_size_bytes_full_size_channel,
        size_t base_l1_address,
        CoreType core_type = CoreType::WORKER);

    std::vector<uint32_t> get_compile_time_args(
        const tt::tt_fabric::FabricEriscDatamoverConfig& fabric_router_config) const override;
};

/**
 * FabricTensixDatamoverRelayConfig
 * - Configuration for relay kernel on NCRISC
 * - Handles upstream fabric router → relay → local chip routing
 * - Permanently uses only one local channel (simplified compared to mux)
 */
class FabricTensixDatamoverRelayConfig : public FabricTensixDatamoverBaseConfig {
public:
    FabricTensixDatamoverRelayConfig(
        uint8_t num_buffers_per_channel,
        size_t buffer_size_bytes,
        size_t base_l1_address,
        CoreType core_type = CoreType::WORKER);

    std::vector<uint32_t> get_compile_time_args(
        const tt::tt_fabric::FabricEriscDatamoverConfig& fabric_router_config) const override;
};

/**
 * FabricTensixDatamoverMuxBuilder
 * - Builds mux kernels on BRISC for worker → mux → fabric router routing
 */
class FabricTensixDatamoverMuxBuilder {
public:
    FabricTensixDatamoverMuxBuilder(
        const CoreCoord& my_core_logical,
        tt::tt_fabric::FabricNodeId local_fabric_node_id,
        tt::tt_fabric::FabricNodeId remote_fabric_node_id,
        uint32_t ethernet_channel_id,
        uint32_t link_idx,
        FabricTensixRiscId risc_id,
        uint32_t noc_x,
        uint32_t noc_y,
        std::shared_ptr<FabricTensixDatamoverMuxConfig> config,
        eth_chan_directions direction);

    void create_and_compile(tt::tt_metal::IDevice* device, tt::tt_metal::Program& program);

    // Build connection to fabric channel - returns connection specs for EDMs to connect
    tt::tt_fabric::SenderWorkerAdapterSpec build_connection_to_fabric_channel(uint32_t channel_id) const;

    // Getters
    const CoreCoord& get_logical_core() const { return my_core_logical_; }
    tt::tt_fabric::FabricNodeId get_local_fabric_node_id() const { return local_fabric_node_id_; }
    tt::tt_fabric::FabricNodeId get_remote_fabric_node_id() const { return remote_fabric_node_id_; }
    uint32_t get_ethernet_channel_id() const { return ethernet_channel_id_; }
    FabricTensixRiscId get_risc_id() const { return risc_id_; }
    uint32_t get_noc_x() const { return noc_x_; }
    uint32_t get_noc_y() const { return noc_y_; }
    eth_chan_directions get_direction() const { return direction_; }

    void append_upstream_routers_noc_xy(uint32_t noc_x, uint32_t noc_y);

private:
    const char* get_kernel_file_path() const;
    std::vector<uint32_t> get_compile_time_args(tt::tt_metal::IDevice* device) const;
    std::vector<uint32_t> get_runtime_args(tt::tt_metal::Program& program) const;

    // Core and fabric configuration
    CoreCoord my_core_logical_;
    tt::tt_fabric::FabricNodeId local_fabric_node_id_;
    tt::tt_fabric::FabricNodeId remote_fabric_node_id_;
    uint32_t ethernet_channel_id_;
    uint32_t link_idx_;

    // RISC and NOC configuration
    FabricTensixRiscId risc_id_;
    uint32_t noc_x_;
    uint32_t noc_y_;

    // Config
    std::shared_ptr<FabricTensixDatamoverMuxConfig> config_;

    // Direction for routing
    eth_chan_directions direction_;

    // Channel connection liveness check disable array
    mutable std::array<bool, builder_config::num_sender_channels> channel_connection_liveness_check_disable_array_{};

    // Upstream router coordinates for sync
    std::vector<uint32_t> upstream_routers_noc_x_;
    std::vector<uint32_t> upstream_routers_noc_y_;
};

/**
 * FabricTensixDatamoverRelayBuilder
 * - Builds relay kernels on NCRISC for upstream fabric router → relay → downstream fabric router routing
 */
class FabricTensixDatamoverRelayBuilder {
public:
    FabricTensixDatamoverRelayBuilder(
        const CoreCoord& my_core_logical,
        tt::tt_fabric::FabricNodeId local_fabric_node_id,
        tt::tt_fabric::FabricNodeId remote_fabric_node_id,
        uint32_t ethernet_channel_id,
        uint32_t link_idx,
        FabricTensixRiscId risc_id,
        uint32_t noc_x,
        uint32_t noc_y,
        std::shared_ptr<FabricTensixDatamoverRelayConfig> config,
        eth_chan_directions direction);

    void create_and_compile(tt::tt_metal::IDevice* device, tt::tt_metal::Program& program);

    // Build connection to fabric channel - returns connection specs for EDMs to connect
    tt::tt_fabric::SenderWorkerAdapterSpec build_connection_to_fabric_channel(uint32_t channel_id) const;

    // Getters
    const CoreCoord& get_logical_core() const { return my_core_logical_; }
    tt::tt_fabric::FabricNodeId get_local_fabric_node_id() const { return local_fabric_node_id_; }
    tt::tt_fabric::FabricNodeId get_remote_fabric_node_id() const { return remote_fabric_node_id_; }
    uint32_t get_ethernet_channel_id() const { return ethernet_channel_id_; }
    FabricTensixRiscId get_risc_id() const { return risc_id_; }
    uint32_t get_noc_x() const { return noc_x_; }
    uint32_t get_noc_y() const { return noc_y_; }
    eth_chan_directions get_direction() const { return direction_; }

    void append_upstream_routers_noc_xy(uint32_t noc_x, uint32_t noc_y);

private:
    const char* get_kernel_file_path() const;
    std::vector<uint32_t> get_compile_time_args(tt::tt_metal::IDevice* device) const;
    std::vector<uint32_t> get_runtime_args(tt::tt_metal::Program& program) const;

    // Core and fabric configuration
    CoreCoord my_core_logical_;
    tt::tt_fabric::FabricNodeId local_fabric_node_id_;
    tt::tt_fabric::FabricNodeId remote_fabric_node_id_;
    uint32_t ethernet_channel_id_;
    uint32_t link_idx_;

    // RISC and NOC configuration
    FabricTensixRiscId risc_id_;
    uint32_t noc_x_;
    uint32_t noc_y_;

    // Config
    std::shared_ptr<FabricTensixDatamoverRelayConfig> config_;

    // Direction for routing
    eth_chan_directions direction_;

    // Channel connection liveness check disable array
    mutable std::array<bool, builder_config::num_sender_channels> channel_connection_liveness_check_disable_array_{};

    // Upstream router coordinate for sync
    uint32_t router_noc_x_;
    uint32_t router_noc_y_;
};

}  // namespace tt::tt_fabric
