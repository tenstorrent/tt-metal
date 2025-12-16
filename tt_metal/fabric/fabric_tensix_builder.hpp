// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <optional>
#include <set>
#include <vector>
#include <memory>
#include <unordered_map>
#include <limits>

#include <tt-metalium/device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include "llrt/core_descriptor.hpp"
#include "tt_metal/fabric/erisc_datamover_builder.hpp"
#include "tt_metal/fabric/fabric_tensix_builder_impl.hpp"
#include "core_coord.hpp"

namespace tt::tt_fabric {

// Core type enum for fabric tensix datamover (identifies MUX vs RELAY cores)
enum class FabricTensixCoreType : uint32_t {
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
    size_t get_num_buffers_per_channel() const { return num_buffers_per_channel_; }
    size_t get_buffer_size_bytes_full_size_channel() const { return buffer_size_bytes_full_size_channel_; }

    // Get base L1 address for a core type
    size_t get_base_l1_address(FabricTensixCoreType core_id) const;

    // Get NOC coordinates for ethernet channel (requires device)
    std::pair<uint32_t, uint32_t> get_noc_xy(tt::tt_metal::IDevice* device, uint32_t eth_chan_id) const;

    // Get channel base address for mux channel ID
    size_t get_channels_base_address(FabricTensixCoreType core_id, size_t tensix_channel_id) const;

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

    // Get tensix NOC coordinates for a specific fabric node, routing plane, and direction
    // Returns pointer to pair (noc_x, noc_y) if tensix exists, nullptr otherwise
    // This includes both active directions (from eth channels) and missing directions (UDM mode)
    const std::pair<uint32_t, uint32_t>* get_tensix_noc_coords(
        const FabricNodeId& fabric_node_id, routing_plane_id_t routing_plane_id, eth_chan_directions direction) const;

    // Get tensix NOC coordinates for active directions only (directions with eth channels)
    // Returns pointer to pair (noc_x, noc_y) if active tensix exists, nullptr otherwise
    const std::pair<uint32_t, uint32_t>* get_active_tensix_noc_coords(
        const FabricNodeId& fabric_node_id, routing_plane_id_t routing_plane_id, eth_chan_directions direction) const;

    // Get the set of missing (routing_plane_id, direction) pairs for a device
    // Only applicable in UDM mode - returns empty set in MUX mode or if all directions have channels
    // Each pair represents a routing plane that doesn't have a tensix builder in that direction
    std::set<std::pair<routing_plane_id_t, eth_chan_directions>> get_missing_directions(ChipId device_id) const;

    // Check if a device has any missing directions
    bool has_missing_directions(ChipId device_id) const;

    // Get core index for a (routing_plane_id, direction) pair on a device
    // Used for missing direction tensix builders where there's no eth channel
    size_t get_core_index_for_direction(
        ChipId device_id, routing_plane_id_t routing_plane_id, eth_chan_directions direction) const;

    // Get core for a (routing_plane_id, direction) pair on a device
    CoreCoord get_core_for_direction(
        ChipId device_id, routing_plane_id_t routing_plane_id, eth_chan_directions direction) const;

    // Get NOC coordinates for a (routing_plane_id, direction) pair on a device
    std::pair<uint32_t, uint32_t> get_noc_xy_for_direction(
        tt::tt_metal::IDevice* device, routing_plane_id_t routing_plane_id, eth_chan_directions direction) const;

    // UDM mode: Worker assignment info
    struct WorkerTensixInfo {
        CoreCoord tensix_core;     // The tensix mux core assigned to this worker (virtual coordinate)
        uint32_t channel_index{};  // The channel index on that tensix mux core
    };

    // Get worker assignment info (tensix core + channel index) for a specific worker (UDM mode only)
    WorkerTensixInfo get_worker_tensix_info(ChipId device_id, const CoreCoord& worker_coord) const;

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
    size_t space_per_risc_{};  // L1 space allocated per RISC

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

    // Mapping: [fabric_node_id] -> [routing_plane_id] -> [direction (E/W/N/S)] -> (noc_x, noc_y)
    // Contains ALL directions (active + missing in UDM mode)
    std::unordered_map<
        FabricNodeId,
        std::unordered_map<routing_plane_id_t, std::unordered_map<eth_chan_directions, std::pair<uint32_t, uint32_t>>>>
        fabric_tensix_noc_coords_map_;

    // Mapping for ACTIVE directions only (directions with eth channels)
    // [fabric_node_id] -> [routing_plane_id] -> [direction (E/W/N/S)] -> (noc_x, noc_y)
    std::unordered_map<
        FabricNodeId,
        std::unordered_map<routing_plane_id_t, std::unordered_map<eth_chan_directions, std::pair<uint32_t, uint32_t>>>>
        fabric_active_tensix_noc_coords_map_;

    // Channel type configuration maps (sorted by ChannelTypes enum)
    // [channel type] -> [number of channels of that type]
    std::map<ChannelTypes, uint32_t> mux_channel_counts_;
    // [channel type] -> [number of buffers per channel of that type]
    std::map<ChannelTypes, uint32_t> mux_channel_buffer_counts_;

    // Cached min/max ethernet channels across all devices
    size_t min_eth_channels_ = 0;
    size_t max_eth_channels_ = 0;

    // Dispatch link info (same for all devices, calculated once in find_min_max_eth_channels)
    uint32_t dispatch_link_idx_ = 0;
    bool has_dispatch_tunnel_ = false;

    // Number of non-dispatch routing planes (excluding dispatch link if present)
    size_t num_non_dispatch_routing_planes_ = 0;

    // Per-device missing (routing_plane_id, direction) pairs without active eth channels
    // Only populated in UDM mode - for edge devices that don't have neighbors in all 4 directions
    // Each pair represents a routing plane that needs a tensix builder in that direction
    std::unordered_map<ChipId, std::set<std::pair<routing_plane_id_t, eth_chan_directions>>>
        missing_directions_per_device_;

    // [device_id][routing_plane_id][direction] -> core_index mapping for ALL directions (active + missing)
    // This is the authoritative lookup for (routing_plane_id, direction) -> core_index
    // Populated for both active directions (from eth channels) and missing directions (UDM mode)
    std::unordered_map<ChipId, std::unordered_map<routing_plane_id_t, std::unordered_map<eth_chan_directions, size_t>>>
        direction_to_core_index_;

    // [device_id][worker coord] -> [WorkerTensixInfo] mapping for UDM mode
    // Maps each worker to its assigned tensix mux core and channel index
    std::unordered_map<ChipId, std::map<CoreCoord, WorkerTensixInfo>> worker_to_tensix_info_map_;

    // Helper methods for initialization

    /**
     * Computes the minimum and maximum number of non-dispatch active ethernet channels
     * across all active devices in the system.
     *
     * For each device, this function:
     * 1. Gathers active fabric ethernet channels in each routing direction
     * 2. Filters out channels reserved for dispatch tunnels
     * 3. Counts the remaining non-dispatch channels
     *
     * The results are stored in min_eth_channels_ and max_eth_channels_ member variables,
     * which are later used by calculate_buffer_allocations() to determine buffer sizing
     * and channel configuration.
     */
    void find_min_max_eth_channels(const std::vector<tt_metal::IDevice*>& all_active_devices);

    /**
     * Builds per-device channel mappings using real ethernet channel IDs.
     *
     * For each device, creates round-robin mapping of ethernet channels to tensix cores,
     * populating eth_chan_to_core_index_ and eth_chan_to_core_id_ maps.
     */
    void build_per_device_channel_mappings(const std::vector<tt_metal::IDevice*>& all_active_devices);

    /**
     * Builds the fabric_router_noc_coords_map_ to track which routers/tensix exist in each direction
     * for each fabric node and routing plane (link index).
     *
     * For each active device and its ethernet channels, this function records the tensix NOC
     * coordinates in the map, keyed by fabric node ID, routing plane ID, and direction.
     */
    void build_fabric_tensix_noc_coords_map(const std::vector<tt_metal::IDevice*>& all_active_devices);

    bool initialize_channel_mappings();
    void calculate_buffer_allocations();
    void create_configs();  // Creates mode-aware configs based on FabricTensixConfig

    // Helper to track missing directions for UDM mode
    // For edge devices that don't have neighbors in all 4 directions, tracks which (routing_plane_id, direction)
    // pairs are missing and assigns core indices for the missing direction tensix builders
    // Also populates fabric_tensix_noc_coords_map_ for the missing directions
    void track_missing_directions_for_udm(
        tt::tt_metal::IDevice* device,
        const FabricNodeId& fabric_node_id,
        const std::set<std::pair<chan_id_t, eth_chan_directions>>& active_channels,
        size_t& channel_index);

    // Helper methods for config creation
    std::shared_ptr<FabricTensixDatamoverMuxConfig> create_mux_config(FabricTensixCoreType core_id);
    std::shared_ptr<FabricTensixDatamoverRelayConfig> create_relay_config(FabricTensixCoreType core_id);

    // Helper to calculate number of channels for mux (handles both UDM and Legacy modes)
    // Also builds worker_to_tensix_core_map_ in UDM mode
    std::map<ChannelTypes, uint32_t> calculate_mux_channel_counts(
        const std::vector<tt_metal::IDevice*>& all_active_devices);

    // UDM mode helper: builds list of workers sorted by column (y first, then x)
    std::vector<CoreCoord> build_workers_by_column(tt::tt_metal::IDevice* device) const;

    // UDM mode helper: gets unique tensix cores for worker assignment (excludes dispatch routing plane)
    std::vector<CoreCoord> get_tensix_cores_for_workers(tt::tt_metal::IDevice* device) const;

    // UDM mode helper: assigns workers to tensix cores in contiguous chunks
    void assign_workers_to_tensix_cores(
        ChipId device_id,
        const std::vector<CoreCoord>& workers_by_column,
        const std::vector<CoreCoord>& tensix_cores_for_workers,
        uint32_t num_worker_channels);
};

/**
 * FabricTensixDatamoverBuilder
 * - Top-level builder that orchestrates mux and relay builders based on FabricTensixConfig mode
 * - MUX mode: creates only mux builder
 * - UDM mode: creates both mux and relay builders
 */
class FabricTensixDatamoverBuilder : public FabricDatamoverBuilderBase {
public:
    // Static builder method called from topology to construct a tensix builder
    static FabricTensixDatamoverBuilder build(
        tt::tt_metal::IDevice* device,
        tt::tt_metal::Program& program,
        tt::tt_fabric::FabricNodeId local_fabric_node_id,
        tt::tt_fabric::FabricNodeId remote_fabric_node_id,
        uint32_t ethernet_channel_id,
        eth_chan_directions direction,
        std::vector<bool>&& sender_channel_injection_flags);

    // Static builder method for missing directions (UDM mode only)
    // Creates a tensix builder for a direction that doesn't have an active eth channel
    // This is used for edge devices that need tensix builders in all 4 directions for inter-mux communication
    // routing_plane_id specifies which routing plane (link index) this tensix builder belongs to
    static FabricTensixDatamoverBuilder build_for_missing_direction(
        tt::tt_metal::IDevice* device,
        tt::tt_metal::Program& program,
        tt::tt_fabric::FabricNodeId local_fabric_node_id,
        routing_plane_id_t routing_plane_id,
        eth_chan_directions direction);

    // Create and compile the kernel(s) based on mode
    void create_and_compile(tt::tt_metal::Program& program);

    // Build connection to fabric channel - returns connection specs for EDMs to connect to this mux
    tt::tt_fabric::SenderWorkerAdapterSpec build_connection_to_fabric_channel(uint32_t channel_id) const override;

    // Build connection to relay channel - returns connection specs for router to connect to relay (UDM mode only)
    tt::tt_fabric::SenderWorkerAdapterSpec build_connection_to_relay_channel() const;

    // Getters - delegate to mux builder (primary builder)
    uint32_t get_noc_x() const;
    uint32_t get_noc_y() const;
    eth_chan_directions get_direction() const;

    void append_upstream_routers_noc_xy(uint32_t noc_x, uint32_t noc_y);
    void append_relay_router_noc_xy(uint32_t noc_x, uint32_t noc_y);

private:
    // Set injection channel flags for bubble flow control (takes ownership via move)
    // Called internally by build() method
    void set_sender_channel_injection_flags_from_vector(std::vector<bool>&& flags);
    // Private constructor - use build() factory method
    FabricTensixDatamoverBuilder(
        std::unique_ptr<FabricTensixDatamoverMuxBuilder> mux_builder,
        std::unique_ptr<FabricTensixDatamoverRelayBuilder> relay_builder,
        const CoreCoord& logical_core,
        tt::tt_fabric::FabricNodeId local_fabric_node_id,
        tt::tt_fabric::FabricNodeId remote_fabric_node_id,
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
        eth_chan_directions direction,
        bool has_fabric_router);

    // Sub-builders based on mode
    std::unique_ptr<FabricTensixDatamoverMuxBuilder> mux_builder_;      // Always created
    std::unique_ptr<FabricTensixDatamoverRelayBuilder> relay_builder_;  // Only in UDM mode

    // Common properties shared by both mux and relay builders
    CoreCoord logical_core_;
    tt::tt_fabric::FabricNodeId local_fabric_node_id_;
    tt::tt_fabric::FabricNodeId remote_fabric_node_id_;
};

}  // namespace tt::tt_fabric
