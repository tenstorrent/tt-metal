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
enum class FabricTensixCoreType : uint32_t;

// Shared struct for mux/relay connection info (used by both mux→mux and relay→mux connections)
struct MuxConnectionInfo {
    uint32_t active;
    uint32_t noc_x;
    uint32_t noc_y;
    size_t buffer_base_addr;
    size_t connection_handshake_addr;
    size_t worker_location_info_addr;
    size_t buffer_index_addr;
    size_t flow_control_semaphore_addr;  // In sender's L1 (relay/mux)
    size_t teardown_semaphore_addr;      // In sender's L1 (relay/mux)
    size_t buffer_index_semaphore_addr;  // In sender's L1 (relay/mux)
    size_t stream_id;                    // Receiver mux's stream ID for credit tracking
};

enum class ChannelTypes : uint32_t {
    WORKER_CHANNEL = 0,
    ROUTER_CHANNEL = 1,
    RELAY_TO_MUX_CHANNEL = 2,
    MUX_TO_MUX_CHANNEL = 3,
    NUM_CHANNEL_TYPES = 4
};

enum class UdmMuxRelayToMuxChannelId : uint32_t {
    LOCAL_RELAY_CHANNEL = 0,          // Relay connects to mux on this channel
    EAST_OR_NORTH_RELAY_CHANNEL = 1,  // Upstream East or North Relay connects to mux on this channel
    WEST_OR_SOUTH_RELAY_CHANNEL = 2,  // Upstream West or South Relay connects to mux on this channel
    NUM_CHANNELS = 3
};

enum class UdmMuxInterMuxChannelId : uint32_t {
    // ==================================================================================
    // Inter-Mux channels
    // Used for forwarding local fabric requests to the correct router endpoint
    // Each Mux on a direction will have different upstream Mux connect to self
    // For East mux:
    //   Channel 0: Accept connection from West Mux
    //   Channel 1: Accept connection from North Mux
    //   Channel 2: Accept connection from South Mux
    // For West mux:
    //   Channel 0: Accept connection from East Mux
    //   Channel 1: Accept connection from North Mux
    //   Channel 2: Accept connection from South Mux
    // For North mux:
    //   Channel 0: Accept connection from East Mux
    //   Channel 1: Accept connection from West Mux
    //   Channel 2: Accept connection from South Mux
    // For South mux:
    //   Channel 0: Accept connection from East Mux
    //   Channel 1: Accept connection from West Mux
    //   Channel 2: Accept connection from North Mux
    // ==================================================================================
    EAST_OR_WEST_MUX_CHANNEL = 0,    // Upstream East or West Mux connects to mux on this channel
    WEST_OR_NORTH_MUX_CHANNEL = 1,   // Upstream West or North Mux connects to mux on this channel
    NORTH_OR_SOUTH_MUX_CHANNEL = 2,  // Upstream North or South Mux connects to mux on this channel
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
 * Generic channel configuration for base config
 * Used to pass channel type-agnostic configuration to FabricTensixDatamoverBaseConfig
 */
struct ChannelTypeConfig {
    size_t num_channels;
    size_t num_buffers_per_channel;
    size_t buffer_size_bytes;

    ChannelTypeConfig(size_t num_ch = 0, size_t num_buf = 0, size_t buf_size = 0) :
        num_channels(num_ch), num_buffers_per_channel(num_buf), buffer_size_bytes(buf_size) {}
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

    // Common getters - accept ChannelTypes to select the correct channel type configuration
    size_t get_num_channels(ChannelTypes channel_type) const;
    size_t get_total_num_channels() const;  // Returns total channels across all types
    size_t get_num_buffers(ChannelTypes channel_type) const;
    size_t get_buffer_size_bytes(ChannelTypes channel_type) const;
    const std::map<ChannelTypes, ChannelTypeConfig>& get_channel_configs() const;
    size_t get_status_address() const;
    size_t get_termination_signal_address() const;
    virtual size_t get_channel_credits_stream_id(ChannelTypes channel_type, uint32_t channel_id) const;
    size_t get_channel_base_address(ChannelTypes channel_type, uint32_t channel_id) const;
    size_t get_connection_info_address(ChannelTypes channel_type, uint32_t channel_id) const;
    size_t get_connection_handshake_address(ChannelTypes channel_type, uint32_t channel_id) const;
    size_t get_flow_control_address(ChannelTypes channel_type, uint32_t channel_id) const;
    size_t get_buffer_index_address(ChannelTypes channel_type, uint32_t channel_id) const;
    size_t get_memory_map_end_address() const;

    // Configuration setters
    void set_num_full_size_channel_iters(size_t new_val);
    void set_num_iters_between_teardown_checks(size_t new_val);
    void set_wait_for_fabric_endpoint_ready(bool wait_for_ready);
    void set_fabric_endpoint_channel_num_buffers(size_t num_buffers);
    void set_fabric_endpoint_status_address(size_t address);

    // Returns vector of pairs of base addresses and size to clear
    std::vector<std::pair<size_t, size_t>> get_memory_regions_to_clear() const;

    virtual std::vector<uint32_t> get_run_time_args(
        const FabricNodeId& src_fabric_node_id,
        const FabricNodeId& dst_fabric_node_id,
        uint32_t link_idx,
        tt::tt_metal::Program& program,
        const CoreCoord& logical_core) const;

protected:
    static constexpr size_t default_num_buffers = 8;
    static constexpr size_t default_num_full_size_channel_iters = 1;
    static constexpr size_t default_num_iters_between_teardown_checks = 32;

    // Constructor for derived classes - accepts map of channel type configurations
    // Uses std::map (not unordered_map) to maintain sorted order by ChannelTypes enum
    FabricTensixDatamoverBaseConfig(
        const std::map<ChannelTypes, ChannelTypeConfig>& channel_configs,
        size_t base_l1_address,
        size_t l1_end_address);

    // Helper methods
    void validate_channel_id(ChannelTypes channel_type, size_t channel_id) const;
    size_t get_channel_global_offset(ChannelTypes channel_type, size_t channel_id) const;
    void append_default_persistent_channel_flags_to_ct_args(std::vector<uint32_t>& ct_args) const;

    // Configuration parameters
    size_t core_type_index_ = 0;
    size_t noc_aligned_address_size_bytes_ = 0;

    // Channel type configurations (sorted by ChannelTypes enum)
    std::map<ChannelTypes, ChannelTypeConfig> channel_configs_;

    // Total number of channels across all channel types (cached for efficiency)
    size_t num_total_channels_ = 0;

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

    // EDMChannelWorkerLocationInfo (per channel type) - stores connection metadata for each channel:
    //   - worker_semaphore_address: Remote L1 address to update worker's read counter
    //   - worker_teardown_semaphore_address: Remote L1 address to acknowledge teardown
    //   - worker_xy: NOC coordinates (x,y) of connected worker/client
    //   - edm_read_counter: Local tracking of packets read by the local kernel
    // Remote clients populate this when establishing connection; local kernel reads it to get client NOC coords
    std::map<ChannelTypes, MemoryRegion> connection_info_regions_;

    // Connection liveness/handshake region (per channel type)
    // Remote clients write to this address to signal connection state changes:
    //   - Write 1 (open_connection_value): Client requests connection establishment
    //   - Write 2 (close_connection_request_value): Client requests connection teardown
    // Local polls these addresses to detect when clients want to connect/disconnect
    std::map<ChannelTypes, MemoryRegion> connection_handshake_regions_;

    // RESERVED/UNUSED: Flow control region (per channel type)
    // This region is allocated but NOT currently used by mux/relay kernels
    // Instead, mux/relay use STREAM REGISTERS for credit-based flow control:
    //   - Each channel has a stream register (my_channel_free_slots_stream_id)
    //   - Local reads: get_ptr_val(stream_id) to check available slots locally
    //   - Local decrements: increment_local_update_ptr_val(stream_id, 1) when sending packet
    //   - Remote client increments: writes to stream register via NoC when consuming packet
    // The flow_control_regions_ is passed to constructors but never stored/used (legacy/reserved)
    std::map<ChannelTypes, MemoryRegion> flow_control_regions_;

    // Buffer index synchronization region (per channel type) - used for connection lifecycle synchronization
    // This stores the write pointer (wrptr) for each channel's buffer slots, used ONLY during:
    //   1. CONNECTION OPEN (client → local): Remote client reads this address to get starting buffer slot
    //      - Client performs NoC read from buffer_index_regions_ to sync with local's wrptr
    //      - This allows client to resume from correct slot (e.g., after reconnection)
    //   2. CONNECTION CLOSE (client → local): Remote client writes final wrptr back to this address
    //      - Client performs NoC write of its buffer_slot_write_counter to buffer_index_regions_
    //      - This tells local where client stopped for proper teardown/cleanup
    // During normal operation: Local tracks wrptr/rdptr internally using local_write_counter/local_read_counter
    //   - These are NOT stored in buffer_index_regions_ during packet forwarding
    //   - buffer_index_regions_ is only accessed during connection establishment/teardown
    std::map<ChannelTypes, MemoryRegion> buffer_index_regions_;

    // Channel buffer storage regions (one per channel type, sorted by ChannelTypes enum)
    // Each channel type has its own buffer region with type-specific configuration
    std::map<ChannelTypes, MemoryRegion> channel_buffer_regions_;

    size_t memory_map_end_address_ = 0;
};

/**
 * FabricTensixDatamoverMuxConfig
 * - Configuration for MUX kernel on BRISC
 * - Handles worker → mux → fabric router routing
 */
class FabricTensixDatamoverMuxConfig : public FabricTensixDatamoverBaseConfig {
public:
    // Constructor accepting channel type configuration map (sorted by ChannelTypes enum)
    FabricTensixDatamoverMuxConfig(
        const std::map<ChannelTypes, ChannelTypeConfig>& channel_type_configs,
        size_t base_l1_address,
        size_t l1_end_address);

    std::vector<uint32_t> get_compile_time_args(
        const FabricNodeId& fabric_node_id, routing_plane_id_t routing_plane_id, eth_chan_directions direction) const;

private:
    // Number of downstream mux connections (all directions except self = 3)
    static constexpr uint32_t NUM_DOWNSTREAM_MUX_CONNECTIONS = 3;

    // Channel storage size for storing channel arrays in L1
    static constexpr size_t channel_storage_size_ = 4096;  // 4KB for channel storage

    // ==================================================================================
    // Mux-specific: Support for inter-mux connections (mux → downstream mux)
    // Each mux can connect to 3 other muxes (all directions except itself)
    // ==================================================================================

    // Helper to collect downstream mux connection info (uses shared MuxConnectionInfo struct)
    MuxConnectionInfo get_mux_connection_info(
        const std::pair<uint32_t, uint32_t>* noc_coords,
        uint32_t downstream_mux_channel_id,
        uint32_t connection_region_idx,
        uint32_t stream_id) const;

    // Helper to collect all downstream mux connection infos
    // Returns empty vector for legacy MUX mode (no downstream mux connections)
    std::vector<MuxConnectionInfo> get_all_mux_connection_infos(
        const FabricNodeId& fabric_node_id, routing_plane_id_t routing_plane_id, eth_chan_directions direction) const;

    // Flow control semaphores (mux → downstream mux direction) for each downstream mux connection
    // - Stored in current MUX's L1 memory
    // - Current mux reads from these addresses to check how many packets each downstream mux has consumed
    // - Downstream mux updates its counter when it consumes a packet from current mux's channel
    // - Acts as read-counters: track how many packets each downstream mux has processed from current mux
    std::array<MemoryRegion, NUM_DOWNSTREAM_MUX_CONNECTIONS> downstream_mux_flow_control_semaphore_regions_{};

    // Teardown semaphores (mux → downstream mux direction) for each downstream mux connection
    // - Stored in current MUX's L1 memory
    // - Current mux writes teardown request, then polls these addresses waiting for acknowledgment
    // - Downstream mux writes (via NoC) acknowledgment value when it completes teardown processing
    std::array<MemoryRegion, NUM_DOWNSTREAM_MUX_CONNECTIONS> downstream_mux_teardown_semaphore_regions_{};

    // Buffer index synchronization (mux → downstream mux direction) for each downstream mux connection
    // - Stored in current MUX's L1 memory
    std::array<MemoryRegion, NUM_DOWNSTREAM_MUX_CONNECTIONS> downstream_mux_buffer_index_semaphore_regions_{};

    // Channel storage region: L1 memory for storing channel objects and interfaces
    // Used by the kernel to place channel buffers and interfaces instead of global memory
    MemoryRegion channel_storage_region_{};

public:
    // Getter for channel storage address
    size_t get_channel_storage_base_address() const { return channel_storage_region_.get_address(); }
    size_t get_channel_storage_size() const { return channel_storage_region_.get_total_size(); }
};

/**
 * FabricTensixDatamoverRelayConfig
 * - Configuration for relay kernel on NCRISC
 * - Handles upstream fabric router → relay → local chip routing
 * - Permanently uses only one local channel (simplified compared to mux)
 */
class FabricTensixDatamoverRelayConfig : public FabricTensixDatamoverBaseConfig {
public:
    // Constructor accepting channel type configuration map (sorted by ChannelTypes enum)
    FabricTensixDatamoverRelayConfig(
        const std::map<ChannelTypes, ChannelTypeConfig>& channel_type_configs,
        size_t base_l1_address,
        size_t l1_end_address);

    // Get compile-time args with fabric node, link, and direction for downstream mux connection info
    std::vector<uint32_t> get_compile_time_args(
        const FabricNodeId& fabric_node_id, routing_plane_id_t routing_plane_id, eth_chan_directions direction) const;

    // Get relay termination signal address (for mux to write during teardown)
    size_t get_relay_termination_signal_address() const { return termination_signal_region_.get_address(); }

    // Get UDM memory pool base address and size
    size_t get_udm_memory_pool_base_address() const { return udm_memory_pool_region_.get_address(); }
    size_t get_udm_memory_pool_size() const { return udm_memory_pool_region_.get_total_size(); }

    // Override get_channel_credits_stream_id to add mux channel offset
    // In UDM mode, relay stream IDs must come after mux stream IDs to avoid collisions
    // since both mux and relay are on the same Tensix core sharing the same stream ID space
    size_t get_channel_credits_stream_id(ChannelTypes channel_type, uint32_t channel_id) const override;

private:
    // ==================================================================================
    // Relay-specific: Support for multiple relay → mux connections
    // The relay acts as a CLIENT connecting to mux(es) (both kernels run on same core)
    // ==================================================================================

    // Helper to collect mux connection info (uses shared MuxConnectionInfo struct)
    MuxConnectionInfo get_mux_connection_info(
        const std::pair<uint32_t, uint32_t>* noc_coords,
        uint32_t mux_channel_id,
        uint32_t mux_idx,
        uint32_t stream_id) const;

    // Number of mux connections: [0]=local, [1]=downstream_en, [2]=downstream_ws
    static constexpr uint32_t NUM_MUX_CONNECTIONS = 3;

    // Helper to collect all mux connection infos
    std::array<MuxConnectionInfo, NUM_MUX_CONNECTIONS> get_all_mux_connection_infos(
        const FabricNodeId& fabric_node_id, routing_plane_id_t routing_plane_id, eth_chan_directions direction) const;

    // Flow control semaphores (relay → mux direction) for each mux connection
    // - Stored in RELAY's L1 memory
    // - Relay reads from these addresses to check how many packets each mux has consumed
    // - Mux updates its counter when it consumes a packet from relay
    // - Acts as read-counters: track how many packets each mux has processed from relay's channel
    std::array<MemoryRegion, NUM_MUX_CONNECTIONS> mux_flow_control_semaphore_regions_{};

    // Teardown semaphores (relay → mux direction) for each mux connection
    // - Stored in RELAY's L1 memory
    // - Relay writes teardown request, then polls these addresses waiting for acknowledgment
    // - Mux writes (via NoC) acknowledgment value when it completes teardown processing
    std::array<MemoryRegion, NUM_MUX_CONNECTIONS> mux_teardown_semaphore_regions_{};

    // Buffer index synchronization (relay → mux direction) - CURRENTLY UNUSED for each mux connection
    // - Stored in RELAY's L1 memory
    // - Unused: Passed to relay's adapter constructor but never stored or accessed
    // - The relay only uses mux's buffer_index_region_ (edm_copy_of_wr_counter_addr) for sync
    std::array<MemoryRegion, NUM_MUX_CONNECTIONS> mux_buffer_index_semaphore_regions_{};

    // Memory pool for storing read response data temporarily
    static constexpr size_t udm_memory_pool_num_slots_ = 8;
    size_t udm_memory_pool_slot_size_ = 0;
    MemoryRegion udm_memory_pool_region_{};
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
        eth_chan_directions direction,
        bool has_fabric_router);

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
    std::vector<uint32_t> get_channel_stream_ids(ChannelTypes channel_type) const;
    std::vector<uint32_t> get_persistent_channels_flags(ChannelTypes channel_type) const;

    // Core and fabric configuration
    CoreCoord my_core_logical_;
    tt::tt_fabric::FabricNodeId local_fabric_node_id_;
    tt::tt_fabric::FabricNodeId remote_fabric_node_id_;
    uint32_t ethernet_channel_id_;
    uint32_t link_idx_;

    // RISC and NOC configuration
    FabricTensixCoreType core_id_;

    std::shared_ptr<FabricTensixDatamoverMuxConfig> config_;

    // Whether this mux has a fabric router to connect to
    // False for missing directions in UDM mode (inter-mux forwarding only)
    bool has_fabric_router_ = true;

    // Channel connection liveness check disable array
    mutable std::array<bool, builder_config::num_max_sender_channels>
        channel_connection_liveness_check_disable_array_{};

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
        eth_chan_directions direction,
        bool /*has_fabric_router*/);

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
    mutable std::array<bool, builder_config::num_max_sender_channels>
        channel_connection_liveness_check_disable_array_{};

    // Router coordinate for sync (relay connects to one local router)
    uint32_t router_noc_x_ = 0;
    uint32_t router_noc_y_ = 0;
};

}  // namespace tt::tt_fabric
