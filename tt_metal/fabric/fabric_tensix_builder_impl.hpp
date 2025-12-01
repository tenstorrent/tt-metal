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
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include "llrt/core_descriptor.hpp"
#include "tt_metal/fabric/erisc_datamover_builder.hpp"
#include "core_coord.hpp"

namespace tt::tt_fabric {

// Forward declarations
enum class FabricMuxChannelType : uint8_t;
enum class FabricTensixCoreType : uint8_t;

/**
 * UDM Mode Channel Assignments
 * - Defines which channels are reserved for specific purposes in UDM mode
 * - Channel 0: Worker channel (for local chip traffic)
 * - Channel 1: Relay channel (relay connects to this mux channel)
 * - Channel 2: Inter-mux channel (for forwarding traffic between muxes)
 */
enum class UdmMuxChannelId : uint32_t {
    WORKER_CHANNEL_BASE = 0,  // Local worker traffic
    RELAY_CHANNEL = 1,        // Relay connects to mux on this channel
    INTER_MUX_CHANNEL = 2,    // Inter-mux forwarding channel
    NUM_CHANNELS = 3
};

/**
 * UDM Mode Relay Channel Assignments
 * - Relay has only one channel for upstream fabric router traffic
 */
enum class UdmRelayChannelId : uint32_t {
    ROUTER_CHANNEL = 0,  // Upstream fabric router connects to relay on this channel
    NUM_CHANNELS = 1
};

/**
 * NoC Assignment for mux and relay
 */
enum class UdmNoCSelection : uint32_t {
    mux_noc = 0,
    relay_noc = 1  // this should be the same as edm_to_local_chip_noc, need to have it in some common files between
                   // host and device.
};

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
    size_t get_channel_credits_stream_id(FabricMuxChannelType channel_type, uint32_t channel_id) const;
    size_t get_channel_base_address(FabricMuxChannelType channel_type, uint32_t channel_id) const;
    size_t get_connection_info_address(FabricMuxChannelType channel_type, uint32_t channel_id) const;
    size_t get_connection_handshake_address(FabricMuxChannelType channel_type, uint32_t channel_id) const;
    size_t get_flow_control_address(FabricMuxChannelType channel_type, uint32_t channel_id) const;
    size_t get_buffer_index_address(FabricMuxChannelType channel_type, uint32_t channel_id) const;
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
    virtual std::vector<uint32_t> get_compile_time_args() const = 0;

    virtual std::vector<uint32_t> get_run_time_args(
        const FabricNodeId& src_fabric_node_id,
        const FabricNodeId& dst_fabric_node_id,
        uint32_t link_idx,
        tt::tt_metal::Program& program,
        const CoreCoord& logical_core,
        const std::vector<bool>& sender_channel_is_traffic_injection_channel_array) const;

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

    // Memory regions (sequentially allocated in L1)
    // Each region manages a specific type of control or data structure used by mux/relay kernels

    // Kernel status tracking (STARTED → READY_FOR_TRAFFIC → TERMINATED)
    MemoryRegion status_region_{};

    // Local copy of fabric router status read from local fabric router through NoC
    // Used for synchronization: mux waits for router status before opening connection
    MemoryRegion local_fabric_router_status_region_{};

    // Termination signals from host or other cores (e.g., mux signals relay during teardown)
    MemoryRegion termination_signal_region_{};

    // EDMChannelWorkerLocationInfo (per channel) - stores connection metadata for each channel:
    //   - worker_semaphore_address: Remote L1 address to update worker's read counter
    //   - worker_teardown_semaphore_address: Remote L1 address to acknowledge teardown
    //   - worker_xy: NOC coordinates (x,y) of connected worker/client
    //   - edm_read_counter: Local tracking of packets read by the local kernel
    // Remote clients populate this when establishing connection; local kernel reads it to get client NOC coords
    MemoryRegion connection_info_region_{};

    // Connection liveness/handshake region (per channel)
    // Remote clients write to this address to signal connection state changes:
    //   - Write 1 (open_connection_value): Client requests connection establishment
    //   - Write 2 (close_connection_request_value): Client requests connection teardown
    // Local polls these addresses to detect when clients want to connect/disconnect
    MemoryRegion connection_handshake_region_{};

    // RESERVED/UNUSED: Flow control region (per channel)
    // This region is allocated but NOT currently used by mux/relay kernels
    // Instead, mux/relay use STREAM REGISTERS for credit-based flow control:
    //   - Each channel has a stream register (my_channel_free_slots_stream_id)
    //   - Local reads: get_ptr_val(stream_id) to check available slots locally
    //   - Local decrements: increment_local_update_ptr_val(stream_id, 1) when sending packet
    //   - Remote client increments: writes to stream register via NoC when consuming packet
    // The flow_control_region_ is passed to constructors but never stored/used (legacy/reserved)
    MemoryRegion flow_control_region_{};

    // Buffer index synchronization region (per channel) - used for connection lifecycle synchronization
    // This stores the write pointer (wrptr) for each channel's buffer slots, used ONLY during:
    //   1. CONNECTION OPEN (client → local): Remote client reads this address to get starting buffer slot
    //      - Client performs NoC read from buffer_index_region_ to sync with local's wrptr
    //      - This allows client to resume from correct slot (e.g., after reconnection)
    //   2. CONNECTION CLOSE (client → local): Remote client writes final wrptr back to this address
    //      - Client performs NoC write of its buffer_slot_write_counter to buffer_index_region_
    //      - This tells local where client stopped for proper teardown/cleanup
    // During normal operation: Local tracks wrptr/rdptr internally using local_write_counter/local_read_counter
    //   - These are NOT stored in buffer_index_region_ during packet forwarding
    //   - buffer_index_region_ is only accessed during connection establishment/teardown
    MemoryRegion buffer_index_region_{};

    // Full-size channel buffer storage (header + payload data)
    // Each channel has num_buffers slots of buffer_size_bytes each
    MemoryRegion full_size_channels_region_{};

    // Header-only channel buffer storage (header without payload, for control/small messages)
    // Each channel has num_buffers slots of sizeof(PacketHeader) each
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

    std::vector<uint32_t> get_compile_time_args() const override;
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

    std::vector<uint32_t> get_compile_time_args() const override;

    size_t get_mux_relay_flow_control_semaphore_address() const {
        return mux_relay_flow_control_semaphore_region_.get_address();
    }
    size_t get_mux_relay_teardown_semaphore_address() const {
        return mux_relay_teardown_semaphore_region_.get_address();
    }
    size_t get_mux_relay_buffer_index_semaphore_address() const {
        return mux_relay_buffer_index_semaphore_region_.get_address();
    }

    // Get relay termination signal address (for mux to write during teardown)
    size_t get_relay_termination_signal_address() const { return termination_signal_region_.get_address(); }

private:
    // ==================================================================================
    // Relay-specific memory regions for relay → mux connection
    // The relay acts as a CLIENT connecting to the mux (both kernels run on same core)
    // ==================================================================================

    // Flow control semaphore/counter (relay → mux direction)
    // - Stored in RELAY's L1 memory
    // - Relay writes this address to mux's connection_info.worker_semaphore_address during connection setup
    // - Relay reads from this address to check how many packets mux has consumed
    // - Mux updates this counter when it consumes a packet from relay
    // - This acts as a read-counter: tracks how many packets mux has processed from relay's channel
    MemoryRegion mux_relay_flow_control_semaphore_region_{};

    // Teardown semaphore (relay → mux direction)
    // - Stored in RELAY's L1 memory
    // - Relay writes this address to mux's connection_info.worker_teardown_semaphore_address during setup
    // - Relay writes teardown request, then polls this address waiting for acknowledgment
    // - Mux writes (via NoC) acknowledgment value when it completes teardown processing
    MemoryRegion mux_relay_teardown_semaphore_region_{};

    // Buffer index synchronization (relay → mux direction) - CURRENTLY UNUSED
    // - Stored in RELAY's L1 memory
    // - Unused: Passed to relay's adapter constructor but never stored or accessed
    // - The relay only uses mux's buffer_index_region_ (edm_copy_of_wr_counter_addr) for sync
    MemoryRegion mux_relay_buffer_index_semaphore_region_{};
};

/**
 * FabricTensixDatamoverMuxBuilder
 * - Builds mux kernels on BRISC for worker → mux → fabric router routing
 */
class FabricTensixDatamoverMuxBuilder : public FabricDatamoverBuilderBase {
    friend class FabricTensixDatamoverBuilder;

public:
    FabricTensixDatamoverMuxBuilder(
        const CoreCoord& my_core_logical,
        tt::tt_fabric::FabricNodeId local_fabric_node_id,
        tt::tt_fabric::FabricNodeId remote_fabric_node_id,
        uint32_t ethernet_channel_id,
        uint32_t link_idx,
        FabricTensixCoreType core_id,
        uint32_t noc_x,
        uint32_t noc_y,
        std::shared_ptr<FabricTensixDatamoverMuxConfig> config,
        eth_chan_directions direction);

    void create_and_compile(tt::tt_metal::Program& program);

    // Build connection to fabric channel - returns connection specs for EDMs to connect
    tt::tt_fabric::SenderWorkerAdapterSpec build_connection_to_fabric_channel(uint32_t channel_id) const override;

    // Getters
    const CoreCoord& get_logical_core() const { return my_core_logical_; }
    tt::tt_fabric::FabricNodeId get_local_fabric_node_id() const { return local_fabric_node_id_; }
    tt::tt_fabric::FabricNodeId get_remote_fabric_node_id() const { return remote_fabric_node_id_; }
    uint32_t get_ethernet_channel_id() const { return ethernet_channel_id_; }
    FabricTensixCoreType get_core_id() const { return core_id_; }

    void append_upstream_routers_noc_xy(uint32_t noc_x, uint32_t noc_y);

private:
    const char* get_kernel_file_path() const;
    std::vector<uint32_t> get_compile_time_args() const;
    std::vector<uint32_t> get_runtime_args(tt::tt_metal::Program& program) const;

    // Helper methods for get_compile_time_args
    std::vector<uint32_t> get_channel_stream_ids(uint8_t num_full_size_channels) const;
    std::vector<uint32_t> get_persistent_channels_flags(uint8_t num_full_size_channels) const;

    // Core and fabric configuration
    CoreCoord my_core_logical_;
    tt::tt_fabric::FabricNodeId local_fabric_node_id_;
    tt::tt_fabric::FabricNodeId remote_fabric_node_id_;
    uint32_t ethernet_channel_id_;
    uint32_t link_idx_;

    // RISC and NOC configuration
    FabricTensixCoreType core_id_;

    std::shared_ptr<FabricTensixDatamoverMuxConfig> config_;

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
class FabricTensixDatamoverRelayBuilder : public FabricDatamoverBuilderBase {
    friend class FabricTensixDatamoverBuilder;

public:
    FabricTensixDatamoverRelayBuilder(
        const CoreCoord& my_core_logical,
        tt::tt_fabric::FabricNodeId local_fabric_node_id,
        tt::tt_fabric::FabricNodeId remote_fabric_node_id,
        uint32_t ethernet_channel_id,
        uint32_t link_idx,
        FabricTensixCoreType core_id,
        uint32_t noc_x,
        uint32_t noc_y,
        std::shared_ptr<FabricTensixDatamoverRelayConfig> config,
        eth_chan_directions direction);

    void create_and_compile(tt::tt_metal::Program& program);

    // Build connection to fabric channel - returns connection specs for EDMs to connect
    tt::tt_fabric::SenderWorkerAdapterSpec build_connection_to_fabric_channel(uint32_t channel_id) const override;

    // Getters
    const CoreCoord& get_logical_core() const { return my_core_logical_; }
    tt::tt_fabric::FabricNodeId get_local_fabric_node_id() const { return local_fabric_node_id_; }
    tt::tt_fabric::FabricNodeId get_remote_fabric_node_id() const { return remote_fabric_node_id_; }
    uint32_t get_ethernet_channel_id() const { return ethernet_channel_id_; }
    FabricTensixCoreType get_core_id() const { return core_id_; }

    void append_router_noc_xy(uint32_t noc_x, uint32_t noc_y);

private:
    const char* get_kernel_file_path() const;
    std::vector<uint32_t> get_compile_time_args() const;
    std::vector<uint32_t> get_runtime_args(tt::tt_metal::Program&) const;

    // Core and fabric configuration
    CoreCoord my_core_logical_;
    tt::tt_fabric::FabricNodeId local_fabric_node_id_;
    tt::tt_fabric::FabricNodeId remote_fabric_node_id_;
    uint32_t ethernet_channel_id_;
    uint32_t link_idx_;

    // RISC and NOC configuration
    FabricTensixCoreType core_id_;

    // Config
    std::shared_ptr<FabricTensixDatamoverRelayConfig> config_;

    // Channel connection liveness check disable array
    mutable std::array<bool, builder_config::num_sender_channels> channel_connection_liveness_check_disable_array_{};

    // Router coordinate for sync (relay connects to one local router)
    uint32_t router_noc_x_ = 0;
    uint32_t router_noc_y_ = 0;
};

}  // namespace tt::tt_fabric
