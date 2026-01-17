// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric_tensix_builder_impl.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-logger/tt-logger.hpp>

#include "impl/context/metal_context.hpp"
#include "llrt/core_descriptor.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/fabric_host_utils.hpp"
#include "tt_metal/fabric/fabric_router_builder.hpp"
#include "dispatch/kernel_config/relay_mux.hpp"
#include "tt_metal/fabric/builder/fabric_static_sized_channels_allocator.hpp"
#include "tt_metal/fabric/builder/fabric_builder_helpers.hpp"
#include "tt_align.hpp"
#include <bit>
#include <algorithm>
#include <utility>

namespace tt::tt_fabric {

static uint32_t get_downstream_mux_channel_id(eth_chan_directions direction, eth_chan_directions downstream_direction) {
    size_t channel_id;
    size_t inter_mux_base_channel_id = static_cast<uint32_t>(UdmMuxInterMuxChannelId::EAST_OR_WEST_MUX_CHANNEL);
    if (downstream_direction == eth_chan_directions::EAST) {
        // For East mux as downstream:
        //   Channel 4: Accept connection from West Mux (direction=1, channel_id=1+4-1=4)
        //   Channel 5: Accept connection from North Mux (direction=2, channel_id=2+4-1=5)
        //   Channel 6: Accept connection from South Mux (direction=3, channel_id=3+4-1=6)
        channel_id = static_cast<uint32_t>(direction) + inter_mux_base_channel_id - 1;
    } else {
        // For other downstream directions: if upstream < downstream, use as-is; else subtract 1
        // For West mux as downstream:
        //   Channel 4: Accept connection from East Mux (direction=0, channel_id=0+4=4)
        //   Channel 5: Accept connection from North Mux (direction=2, channel_id=2+4-1=5)
        //   Channel 6: Accept connection from South Mux (direction=3, channel_id=3+4-1=6)
        // For North mux as downstream:
        //   Channel 4: Accept connection from East Mux (direction=0, channel_id=0+4=4)
        //   Channel 5: Accept connection from West Mux (direction=1, channel_id=1+4=5)
        //   Channel 6: Accept connection from South Mux (direction=3, channel_id=3+4-1=6)
        // For South mux as downstream:
        //   Channel 4: Accept connection from East Mux (direction=0, channel_id=0+4=4)
        //   Channel 5: Accept connection from West Mux (direction=1, channel_id=1+4=5)
        //   Channel 6: Accept connection from North Mux (direction=3, channel_id=3+4-1=6)
        channel_id = (direction < downstream_direction)
                         ? (static_cast<uint32_t>(direction) + inter_mux_base_channel_id)
                         : (static_cast<uint32_t>(direction) + inter_mux_base_channel_id - 1);
    }

    return channel_id;
}

static uint32_t get_upstream_mux_channel_id(eth_chan_directions direction, eth_chan_directions upstream_direction) {
    // Calculate channel based on upstream direction, skipping the current direction in the enum ordering
    uint32_t upstream_idx = static_cast<uint32_t>(upstream_direction);
    uint32_t current_idx = static_cast<uint32_t>(direction);

    // Base channel for inter-mux connections
    uint32_t base_channel = static_cast<uint32_t>(UdmMuxInterMuxChannelId::EAST_OR_WEST_MUX_CHANNEL);

    // If upstream comes before current in enum order (E=0, W=1, N=2, S=3), use index as-is
    // Otherwise subtract 1 to account for skipping the current direction
    return (upstream_idx < current_idx) ? (base_channel + upstream_idx) : (base_channel + upstream_idx - 1);
}

// ==================================================================================================
// FabricTensixDatamoverBaseConfig Implementation
// ==================================================================================================

// MemoryRegion implementation
FabricTensixDatamoverBaseConfig::MemoryRegion::MemoryRegion(size_t base, size_t unit_sz, size_t count) :
    base_address(base), unit_size(unit_sz), num_units(count) {}

size_t FabricTensixDatamoverBaseConfig::MemoryRegion::get_address(size_t offset) const {
    if (num_units == 0) {
        TT_FATAL(offset == 0, "Offset {} is invalid for empty region (num_units == 0)", offset);
        return base_address;
    }
    TT_FATAL(offset < num_units, "Offset {} exceeds region size {}", offset, num_units);
    return base_address + (offset * unit_size);
}

size_t FabricTensixDatamoverBaseConfig::MemoryRegion::get_end_address() const {
    return base_address + (num_units * unit_size);
}

size_t FabricTensixDatamoverBaseConfig::MemoryRegion::get_total_size() const { return num_units * unit_size; }

// Base Config Constructor
FabricTensixDatamoverBaseConfig::FabricTensixDatamoverBaseConfig(
    const std::map<ChannelTypes, ChannelTypeConfig>& channel_configs, size_t base_l1_address, size_t l1_end_address) :
    channel_configs_(channel_configs) {
    TT_FATAL(!channel_configs_.empty(), "At least one channel type must be configured");

    // Calculate total number of channels across all types (cached as member variable)
    num_total_channels_ = 0;
    for (const auto& [type, config] : channel_configs_) {
        num_total_channels_ += config.num_channels;
    }

    TT_FATAL(num_total_channels_ > 0, "Total number of channels must be greater than 0");

    // Validate buffer sizes
    size_t max_buffer_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    for (const auto& [type, config] : channel_configs_) {
        TT_FATAL(
            config.buffer_size_bytes <= max_buffer_size_bytes,
            "Buffer size bytes should be less than or equal to: {}, but got: {}",
            max_buffer_size_bytes,
            config.buffer_size_bytes);
    }

    noc_aligned_address_size_bytes_ =
        tt::tt_metal::MetalContext::instance().hal().get_alignment(tt::tt_metal::HalMemType::L1);

    // Initialize memory regions sequentially
    size_t current_address = base_l1_address;

    status_region_ = MemoryRegion(current_address, noc_aligned_address_size_bytes_, 1);
    current_address = status_region_.get_end_address();

    local_fabric_router_status_region_ = MemoryRegion(current_address, noc_aligned_address_size_bytes_, 1);
    current_address = local_fabric_router_status_region_.get_end_address();

    termination_signal_region_ = MemoryRegion(current_address, noc_aligned_address_size_bytes_, 1);
    current_address = termination_signal_region_.get_end_address();

    // Allocate per-channel-type regions (organized by ChannelTypes)
    // std::map already maintains sorted order by ChannelTypes enum, no need to sort!
    for (const auto& [type, config] : channel_configs_) {
        // Connection info region: stores EDMChannelWorkerLocationInfo per channel
        connection_info_regions_[type] =
            MemoryRegion(current_address, sizeof(tt::tt_fabric::EDMChannelWorkerLocationInfo), config.num_channels);
        current_address = connection_info_regions_[type].get_end_address();

        // Connection handshake region: liveness/handshake per channel
        connection_handshake_regions_[type] =
            MemoryRegion(current_address, noc_aligned_address_size_bytes_, config.num_channels);
        current_address = connection_handshake_regions_[type].get_end_address();

        // Flow control region: reserved/unused per channel
        flow_control_regions_[type] =
            MemoryRegion(current_address, noc_aligned_address_size_bytes_, config.num_channels);
        current_address = flow_control_regions_[type].get_end_address();

        // Buffer index region: write pointer synchronization per channel
        buffer_index_regions_[type] =
            MemoryRegion(current_address, noc_aligned_address_size_bytes_, config.num_channels);
        current_address = buffer_index_regions_[type].get_end_address();

        // Channel buffer region: actual data buffers per channel
        size_t channel_size_bytes = config.num_buffers_per_channel * config.buffer_size_bytes;
        channel_buffer_regions_[type] = MemoryRegion(current_address, channel_size_bytes, config.num_channels);
        current_address = channel_buffer_regions_[type].get_end_address();
    }

    memory_map_end_address_ = current_address;

    const auto& hal = tt_metal::MetalContext::instance().hal();
    core_type_index_ = hal.get_programmable_core_type_index(tt::tt_metal::HalProgrammableCoreType::TENSIX);

    TT_FATAL(
        memory_map_end_address_ <= l1_end_address,
        "Memory map end address: {} is greater than allocated L1 end address: {}",
        memory_map_end_address_,
        l1_end_address);
}

// Getters
size_t FabricTensixDatamoverBaseConfig::get_num_channels(ChannelTypes channel_type) const {
    // Return the number of channels for the specific channel type
    TT_FATAL(
        channel_configs_.contains(channel_type),
        "Channel type {} not found in channel_configs_",
        static_cast<uint32_t>(channel_type));
    return channel_configs_.at(channel_type).num_channels;
}

size_t FabricTensixDatamoverBaseConfig::get_total_num_channels() const {
    // Return the total number of channels across all channel types
    return num_total_channels_;
}

size_t FabricTensixDatamoverBaseConfig::get_num_buffers(ChannelTypes channel_type) const {
    // Return number of buffers for the specific channel type
    TT_FATAL(
        channel_configs_.contains(channel_type),
        "Channel type {} not found in channel_configs_",
        static_cast<uint32_t>(channel_type));
    return channel_configs_.at(channel_type).num_buffers_per_channel;
}

size_t FabricTensixDatamoverBaseConfig::get_buffer_size_bytes(ChannelTypes channel_type) const {
    // Return buffer size for the specific channel type
    TT_FATAL(
        channel_configs_.contains(channel_type),
        "Channel type {} not found in channel_configs_",
        static_cast<uint32_t>(channel_type));
    return channel_configs_.at(channel_type).buffer_size_bytes;
}

const std::map<ChannelTypes, ChannelTypeConfig>& FabricTensixDatamoverBaseConfig::get_channel_configs() const {
    return channel_configs_;
}

size_t FabricTensixDatamoverBaseConfig::get_status_address() const { return status_region_.get_address(); }

size_t FabricTensixDatamoverBaseConfig::get_termination_signal_address() const {
    return termination_signal_region_.get_address();
}

size_t FabricTensixDatamoverBaseConfig::get_channel_credits_stream_id(
    ChannelTypes channel_type, uint32_t channel_id) const {
    validate_channel_id(channel_type, channel_id);
    return get_channel_global_offset(channel_type, channel_id);
}

size_t FabricTensixDatamoverBaseConfig::get_channel_base_address(ChannelTypes channel_type, uint32_t channel_id) const {
    validate_channel_id(channel_type, channel_id);
    return channel_buffer_regions_.at(channel_type).get_address(channel_id);
}

size_t FabricTensixDatamoverBaseConfig::get_connection_info_address(
    ChannelTypes channel_type, uint32_t channel_id) const {
    validate_channel_id(channel_type, channel_id);
    return connection_info_regions_.at(channel_type).get_address(channel_id);
}

size_t FabricTensixDatamoverBaseConfig::get_connection_handshake_address(
    ChannelTypes channel_type, uint32_t channel_id) const {
    validate_channel_id(channel_type, channel_id);
    return connection_handshake_regions_.at(channel_type).get_address(channel_id);
}

size_t FabricTensixDatamoverBaseConfig::get_flow_control_address(ChannelTypes channel_type, uint32_t channel_id) const {
    validate_channel_id(channel_type, channel_id);
    return flow_control_regions_.at(channel_type).get_address(channel_id);
}

size_t FabricTensixDatamoverBaseConfig::get_buffer_index_address(ChannelTypes channel_type, uint32_t channel_id) const {
    validate_channel_id(channel_type, channel_id);
    return buffer_index_regions_.at(channel_type).get_address(channel_id);
}

size_t FabricTensixDatamoverBaseConfig::get_memory_map_end_address() const { return memory_map_end_address_; }

// Setters
void FabricTensixDatamoverBaseConfig::set_num_full_size_channel_iters(size_t new_val) {
    TT_FATAL(!channel_configs_.empty(), "Cannot set iterations when no channels exist");
    TT_FATAL(new_val > 0, "Number of iterations must be greater than 0");
    num_full_size_channel_iters_ = new_val;
}

void FabricTensixDatamoverBaseConfig::set_num_iters_between_teardown_checks(size_t new_val) {
    TT_FATAL(new_val > 0, "Setting num iters b/w teardown checks to 0 will result in no data being sent");
    num_iters_between_teardown_checks_ = new_val;
}

void FabricTensixDatamoverBaseConfig::set_wait_for_fabric_endpoint_ready(bool wait_for_ready) {
    wait_for_fabric_endpoint_ready_ = wait_for_ready;
}

void FabricTensixDatamoverBaseConfig::set_fabric_endpoint_channel_num_buffers(size_t num_buffers) {
    fabric_endpoint_channel_num_buffers_ = num_buffers;
}

void FabricTensixDatamoverBaseConfig::set_fabric_endpoint_status_address(size_t address) {
    fabric_endpoint_status_address_ = address;
}

std::vector<std::pair<size_t, size_t>> FabricTensixDatamoverBaseConfig::get_memory_regions_to_clear() const {
    std::vector<std::pair<size_t, size_t>> regions;

    // Always clear termination signal region
    regions.push_back({termination_signal_region_.get_address(), termination_signal_region_.get_total_size()});

    // Clear per-channel-type regions
    for (const auto& [type, config] : channel_configs_) {
        regions.push_back(
            {connection_handshake_regions_.at(type).get_address(),
             connection_handshake_regions_.at(type).get_total_size()});
        regions.push_back(
            {flow_control_regions_.at(type).get_address(), flow_control_regions_.at(type).get_total_size()});
        regions.push_back(
            {buffer_index_regions_.at(type).get_address(), buffer_index_regions_.at(type).get_total_size()});
    }

    return regions;
}

std::vector<uint32_t> FabricTensixDatamoverBaseConfig::get_run_time_args(
    const FabricNodeId& src_fabric_node_id,
    const FabricNodeId& dst_fabric_node_id,
    uint32_t link_idx,
    tt::tt_metal::Program& program,
    const CoreCoord& logical_core) const {
    std::vector<uint32_t> args;

    auto regions_to_clear = get_memory_regions_to_clear();
    const auto num_regions_to_clear = regions_to_clear.size();
    args.reserve((num_regions_to_clear * 2) + 1);
    args.push_back(static_cast<uint32_t>(num_regions_to_clear));
    for (const auto& [address, size] : regions_to_clear) {
        args.push_back(static_cast<uint32_t>(address));
        args.push_back(static_cast<uint32_t>(size));
    }

    tt::tt_fabric::append_fabric_connection_rt_args(
        src_fabric_node_id, dst_fabric_node_id, link_idx, program, logical_core, args, CoreType::WORKER);

    return args;
}

// Helper methods
void FabricTensixDatamoverBaseConfig::validate_channel_id(ChannelTypes channel_type, size_t channel_id) const {
    // Validate that the channel_id is valid for this specific channel type
    TT_FATAL(
        channel_configs_.contains(channel_type),
        "Channel type {} not found in channel_configs_",
        static_cast<uint32_t>(channel_type));
    const auto& config = channel_configs_.at(channel_type);

    TT_FATAL(
        channel_id < config.num_channels,
        "Invalid channel id {} for channel type {}. This type has {} channels",
        channel_id,
        static_cast<uint32_t>(channel_type),
        config.num_channels);
}

size_t FabricTensixDatamoverBaseConfig::get_channel_global_offset(ChannelTypes channel_type, size_t channel_id) const {
    // Calculate global offset by summing channels from all channel types before this one
    size_t global_offset = 0;
    for (const auto& [type, config] : channel_configs_) {
        if (type == channel_type) {
            return global_offset + channel_id;
        }
        global_offset += config.num_channels;
    }

    TT_THROW("Channel type {} not found in configuration", static_cast<uint32_t>(channel_type));
}

// ==================================================================================================
// FabricTensixDatamoverMuxConfig Implementation
// ==================================================================================================

FabricTensixDatamoverMuxConfig::FabricTensixDatamoverMuxConfig(
    const std::map<ChannelTypes, ChannelTypeConfig>& channel_type_configs,
    size_t base_l1_address,
    size_t l1_end_address) :
    FabricTensixDatamoverBaseConfig(
        channel_type_configs,  // Pass map directly, already sorted!
        base_l1_address,
        l1_end_address) {
    // Allocate semaphore regions for mux → downstream mux connections (three perpendicular directions)
    // These are in current mux's L1 memory for downstream muxes to write flow control signals back
    size_t current_address = memory_map_end_address_;

    // Allocate 3 sets of semaphore regions (one per downstream mux connection)
    for (uint32_t i = 0; i < NUM_DOWNSTREAM_MUX_CONNECTIONS; i++) {
        downstream_mux_flow_control_semaphore_regions_[i] =
            MemoryRegion(current_address, noc_aligned_address_size_bytes_, 1);
        current_address = downstream_mux_flow_control_semaphore_regions_[i].get_end_address();

        downstream_mux_teardown_semaphore_regions_[i] =
            MemoryRegion(current_address, noc_aligned_address_size_bytes_, 1);
        current_address = downstream_mux_teardown_semaphore_regions_[i].get_end_address();

        downstream_mux_buffer_index_semaphore_regions_[i] =
            MemoryRegion(current_address, noc_aligned_address_size_bytes_, 1);
        current_address = downstream_mux_buffer_index_semaphore_regions_[i].get_end_address();
    }

    // Allocate channel storage region for storing channel arrays in L1 (4KB)
    channel_storage_region_ = MemoryRegion(current_address, channel_storage_size_, 1);
    current_address = channel_storage_region_.get_end_address();

    memory_map_end_address_ = current_address;

    TT_FATAL(
        memory_map_end_address_ <= l1_end_address,
        "Mux memory map end address: {} exceeds allocated L1 end address: {}",
        memory_map_end_address_,
        l1_end_address);
}

MuxConnectionInfo FabricTensixDatamoverMuxConfig::get_mux_connection_info(
    const std::pair<uint32_t, uint32_t>* noc_coords,
    uint32_t downstream_mux_channel_id,
    uint32_t connection_region_idx,
    uint32_t stream_id) const {
    // Get downstream mux config (all muxes share the same config)
    auto channel_type = tt::tt_fabric::ChannelTypes::MUX_TO_MUX_CHANNEL;

    return MuxConnectionInfo{
        .active = noc_coords ? 1u : 0u,
        .noc_x = noc_coords ? noc_coords->first : 0u,
        .noc_y = noc_coords ? noc_coords->second : 0u,
        .buffer_base_addr = get_channel_base_address(channel_type, downstream_mux_channel_id),
        .connection_handshake_addr = get_connection_handshake_address(channel_type, downstream_mux_channel_id),
        .worker_location_info_addr = get_connection_info_address(channel_type, downstream_mux_channel_id),
        .buffer_index_addr = get_buffer_index_address(channel_type, downstream_mux_channel_id),
        .flow_control_semaphore_addr =
            downstream_mux_flow_control_semaphore_regions_[connection_region_idx].get_address(),
        .teardown_semaphore_addr = downstream_mux_teardown_semaphore_regions_[connection_region_idx].get_address(),
        .buffer_index_semaphore_addr =
            downstream_mux_buffer_index_semaphore_regions_[connection_region_idx].get_address(),
        .stream_id = stream_id};
}

std::vector<MuxConnectionInfo> FabricTensixDatamoverMuxConfig::get_all_mux_connection_infos(
    const FabricNodeId& fabric_node_id, routing_plane_id_t routing_plane_id, eth_chan_directions direction) const {
    // In legacy MUX mode, we don't have MUX_TO_MUX_CHANNEL, so return empty vector (size 0)
    if (!channel_configs_.contains(tt::tt_fabric::ChannelTypes::MUX_TO_MUX_CHANNEL)) {
        // Legacy MUX mode - no downstream mux connections
        return std::vector<MuxConnectionInfo>();
    }

    // Z direction muxes don't participate in inter-mux forwarding
    // MUX_TO_MUX channels only support mesh directions (E/W/N/S)
    if (direction == eth_chan_directions::Z) {
        return std::vector<MuxConnectionInfo>();
    }

    // UDM mode - collect downstream mux connection info
    const auto& fabric_context = tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_context();
    const auto& tensix_config = fabric_context.get_builder_context().get_tensix_config();

    // Exclude Z direction - UDM MUX mode only supports mesh directions (E/W/N/S)
    auto downstream_dirs = builder::get_all_other_directions(direction, /*exclude_z=*/true);

    std::vector<MuxConnectionInfo> mux_infos;

    // Collect connection info for each downstream mux
    for (uint32_t i = 0; i < downstream_dirs.size(); i++) {
        auto downstream_dir = downstream_dirs[i];
        const auto* downstream_mux_noc_coord =
            tensix_config.get_tensix_noc_coords(fabric_node_id, routing_plane_id, downstream_dir);

        // Calculate which channel on the downstream mux this mux should connect to
        uint32_t downstream_mux_channel = get_downstream_mux_channel_id(direction, downstream_dir);

        // Get the stream ID for the downstream mux's channel
        uint32_t mux_stream_id =
            get_channel_credits_stream_id(tt::tt_fabric::ChannelTypes::MUX_TO_MUX_CHANNEL, downstream_mux_channel);

        // Connection region index maps to the array index
        uint32_t connection_region_idx = i;

        // Collect connection info for this downstream mux
        mux_infos.push_back(get_mux_connection_info(
            downstream_mux_noc_coord, downstream_mux_channel, connection_region_idx, mux_stream_id));
    }

    return mux_infos;
}

std::vector<uint32_t> FabricTensixDatamoverMuxConfig::get_compile_time_args(
    const FabricNodeId& fabric_node_id, routing_plane_id_t routing_plane_id, eth_chan_directions direction) const {
    const auto& fabric_context = tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_context();
    const auto& fabric_tensix_config = tt::tt_metal::MetalContext::instance().get_fabric_tensix_config();
    const auto& fabric_router_config =
        fabric_context.get_builder_context().get_fabric_router_config(fabric_tensix_config);

    auto* channel_allocator = fabric_router_config.channel_allocator.get();
    auto* const static_channel_allocator =
        dynamic_cast<tt::tt_fabric::FabricStaticSizedChannelsAllocator*>(channel_allocator);
    TT_FATAL(static_channel_allocator != nullptr, "Channel allocator must be a FabricStaticSizedChannelsAllocator.");

    fabric_endpoint_channel_num_buffers_ = static_channel_allocator->get_sender_channel_number_of_slots(0);
    fabric_endpoint_status_address_ = fabric_router_config.edm_status_address;
    wait_for_fabric_endpoint_ready_ = true;

    TT_FATAL(fabric_endpoint_channel_num_buffers_ > 0, "fabric_endpoint_channel_num_buffers_ must be larger than 0");
    TT_FATAL(fabric_endpoint_status_address_ != 0, "fabric_endpoint_status_address_ must not be invalid address 0");

    auto mux_infos = get_all_mux_connection_infos(fabric_node_id, routing_plane_id, direction);

    // Build compile time args - organized by channel type
    uint32_t num_channel_types = static_cast<uint32_t>(channel_configs_.size());

    std::vector<uint32_t> ct_args = {
        static_cast<uint32_t>(num_total_channels_),           // 0: total number of channels across all types
        static_cast<uint32_t>(status_region_.get_address()),  // 1
        static_cast<uint32_t>(termination_signal_region_.get_address()),          // 2
        static_cast<uint32_t>(local_fabric_router_status_region_.get_address()),  // 3
        static_cast<uint32_t>(fabric_endpoint_status_address_),                   // 4
        static_cast<uint32_t>(fabric_endpoint_channel_num_buffers_),              // 5
        static_cast<uint32_t>(num_full_size_channel_iters_),                      // 6
        static_cast<uint32_t>(num_iters_between_teardown_checks_),                // 7
        static_cast<uint32_t>(core_type_index_),                                  // 8
        (uint32_t)wait_for_fabric_endpoint_ready_,                                // 9
        static_cast<uint32_t>(mux_infos.size()),  // 10: number of downstream mux connections (0 for legacy, 3 for UDM)
        num_channel_types                         // 11: number of channel types
    };

    constexpr uint32_t base_channel_id = 0;
    // Append per-channel-type arrays (grouped by channel type, sorted by enum)
    for (const auto& [type, config] : channel_configs_) {
        ct_args.push_back(config.num_channels);
    }
    // Number of buffers per channel (one per channel type)
    for (const auto& [type, config] : channel_configs_) {
        ct_args.push_back(config.num_buffers_per_channel);
    }
    // Buffer size in bytes (one per channel type)
    for (const auto& [type, config] : channel_configs_) {
        ct_args.push_back(static_cast<uint32_t>(config.buffer_size_bytes));
    }
    // Connection info base addresses (one per channel type)
    for (const auto& [type, region] : connection_info_regions_) {
        ct_args.push_back(region.get_address(base_channel_id));
    }
    // Connection handshake base addresses (one per channel type)
    for (const auto& [type, region] : connection_handshake_regions_) {
        ct_args.push_back(region.get_address(base_channel_id));
    }
    // Flow control base addresses (one per channel type)
    for (const auto& [type, region] : flow_control_regions_) {
        ct_args.push_back(region.get_address(base_channel_id));
    }
    // Channel buffer base addresses (one per channel type)
    for (const auto& [type, region] : channel_buffer_regions_) {
        ct_args.push_back(region.get_address(base_channel_id));
    }

    // Append downstream mux connection arrays (similar to relay's mux connection arrays)
    // Active flags array
    for (const auto& info : mux_infos) {
        ct_args.push_back(info.active);
    }
    // NOC X coords array
    for (const auto& info : mux_infos) {
        ct_args.push_back(info.noc_x);
    }
    // NOC Y coords array
    for (const auto& info : mux_infos) {
        ct_args.push_back(info.noc_y);
    }
    // Buffer base addresses array
    for (const auto& info : mux_infos) {
        ct_args.push_back(info.buffer_base_addr);
    }
    // Connection handshake addresses array
    for (const auto& info : mux_infos) {
        ct_args.push_back(info.connection_handshake_addr);
    }
    // Worker location info addresses array
    for (const auto& info : mux_infos) {
        ct_args.push_back(info.worker_location_info_addr);
    }
    // Buffer index addresses array
    for (const auto& info : mux_infos) {
        ct_args.push_back(info.buffer_index_addr);
    }
    // Flow control semaphore addresses array (current mux's L1 memory)
    for (const auto& info : mux_infos) {
        ct_args.push_back(info.flow_control_semaphore_addr);
    }
    // Teardown semaphore addresses array (current mux's L1 memory)
    for (const auto& info : mux_infos) {
        ct_args.push_back(info.teardown_semaphore_addr);
    }
    // Buffer index semaphore addresses array (current mux's L1 memory)
    for (const auto& info : mux_infos) {
        ct_args.push_back(info.buffer_index_semaphore_addr);
    }
    // Stream IDs array (downstream mux's stream IDs)
    for (const auto& info : mux_infos) {
        ct_args.push_back(info.stream_id);
    }

    return ct_args;
}

// ==================================================================================================
// FabricTensixDatamoverRelayConfig Implementation
// ==================================================================================================

FabricTensixDatamoverRelayConfig::FabricTensixDatamoverRelayConfig(
    const std::map<ChannelTypes, ChannelTypeConfig>& channel_type_configs,
    size_t base_l1_address,
    size_t l1_end_address) :
    FabricTensixDatamoverBaseConfig(
        channel_type_configs,  // Pass map directly, already sorted!
        base_l1_address,
        l1_end_address) {
    // Allocate semaphore regions for relay → mux connections (local, downstream_en, downstream_ws)
    // These are in relay's L1 memory for the mux to write flow control signals back to the relay
    size_t current_address = memory_map_end_address_;

    // Allocate 3 sets of semaphore regions (one per mux connection)
    for (uint32_t i = 0; i < NUM_MUX_CONNECTIONS; i++) {
        mux_flow_control_semaphore_regions_[i] = MemoryRegion(current_address, noc_aligned_address_size_bytes_, 1);
        current_address = mux_flow_control_semaphore_regions_[i].get_end_address();

        mux_teardown_semaphore_regions_[i] = MemoryRegion(current_address, noc_aligned_address_size_bytes_, 1);
        current_address = mux_teardown_semaphore_regions_[i].get_end_address();

        mux_buffer_index_semaphore_regions_[i] = MemoryRegion(current_address, noc_aligned_address_size_bytes_, 1);
        current_address = mux_buffer_index_semaphore_regions_[i].get_end_address();
    }

    const auto& fabric_context = tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_context();
    udm_memory_pool_slot_size_ = fabric_context.get_fabric_max_payload_size_bytes();
    const size_t udm_memory_pool_size = udm_memory_pool_slot_size_ * udm_memory_pool_num_slots_;
    udm_memory_pool_region_ = MemoryRegion(current_address, udm_memory_pool_size, 1);
    current_address = udm_memory_pool_region_.get_end_address();
    log_debug(
        tt::LogMetal,
        "udm memory pool has: {} slots, each slot is: {} bytes",
        udm_memory_pool_num_slots_,
        udm_memory_pool_slot_size_);

    // Allocate response pool region - use remaining L1 space
    // RegisteredResponse is 32 bytes per slot
    const size_t available_l1_bytes = l1_end_address - current_address;
    udm_registered_response_num_slots_ = available_l1_bytes / udm_registered_response_slot_size_;
    const size_t udm_registered_response_pool_size =
        udm_registered_response_num_slots_ * udm_registered_response_slot_size_;
    udm_registered_response_pool_region_ = MemoryRegion(current_address, udm_registered_response_pool_size, 1);
    current_address = udm_registered_response_pool_region_.get_end_address();
    log_debug(
        tt::LogMetal,
        "udm registered response pool has: {} slots, each slot is: {} bytes",
        udm_registered_response_num_slots_,
        udm_registered_response_slot_size_);

    memory_map_end_address_ = current_address;

    TT_FATAL(
        memory_map_end_address_ <= l1_end_address,
        "Relay memory map end address: {} exceeds allocated L1 end address: {}",
        memory_map_end_address_,
        l1_end_address);
}

MuxConnectionInfo FabricTensixDatamoverRelayConfig::get_mux_connection_info(
    const std::pair<uint32_t, uint32_t>* noc_coords,
    uint32_t mux_channel_id,
    uint32_t connection_region_idx,
    uint32_t stream_id) const {
    const auto& fabric_context = tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_context();
    const auto& tensix_config = fabric_context.get_builder_context().get_tensix_config();
    auto mux_config = tensix_config.get_config(FabricTensixCoreType::MUX);
    auto channel_type = tt::tt_fabric::ChannelTypes::RELAY_TO_MUX_CHANNEL;

    return MuxConnectionInfo{
        .active = noc_coords ? 1u : 0u,
        .noc_x = noc_coords ? noc_coords->first : 0u,
        .noc_y = noc_coords ? noc_coords->second : 0u,
        .buffer_base_addr = mux_config->get_channel_base_address(channel_type, mux_channel_id),
        .connection_handshake_addr = mux_config->get_connection_handshake_address(channel_type, mux_channel_id),
        .worker_location_info_addr = mux_config->get_connection_info_address(channel_type, mux_channel_id),
        .buffer_index_addr = mux_config->get_buffer_index_address(channel_type, mux_channel_id),
        .flow_control_semaphore_addr = mux_flow_control_semaphore_regions_[connection_region_idx].get_address(),
        .teardown_semaphore_addr = mux_teardown_semaphore_regions_[connection_region_idx].get_address(),
        .buffer_index_semaphore_addr = mux_buffer_index_semaphore_regions_[connection_region_idx].get_address(),
        .stream_id = stream_id};
}

std::array<MuxConnectionInfo, FabricTensixDatamoverRelayConfig::NUM_MUX_CONNECTIONS>
FabricTensixDatamoverRelayConfig::get_all_mux_connection_infos(
    const FabricNodeId& fabric_node_id, routing_plane_id_t routing_plane_id, eth_chan_directions direction) const {
    // Z direction relays don't have mux connections (perpendicular directions not defined for Z)
    if (direction == eth_chan_directions::Z) {
        return std::array<MuxConnectionInfo, NUM_MUX_CONNECTIONS>{};
    }

    const auto& fabric_context = tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_context();
    const auto& tensix_config = fabric_context.get_builder_context().get_tensix_config();

    // Determine directions to check: [0]=local, [1]=perp1, [2]=perp2
    auto [perp_dir1, perp_dir2] = builder::get_perpendicular_directions(direction);
    std::array<eth_chan_directions, NUM_MUX_CONNECTIONS> target_mux_dirs = {direction, perp_dir1, perp_dir2};

    std::array<MuxConnectionInfo, NUM_MUX_CONNECTIONS> mux_infos{};

    for (uint32_t i = 0; i < NUM_MUX_CONNECTIONS; i++) {
        auto target_dir = target_mux_dirs[i];
        const auto* mux_noc_coord = tensix_config.get_tensix_noc_coords(fabric_node_id, routing_plane_id, target_dir);

        // Determine which channel on the target mux we connect to
        uint32_t mux_channel_id;
        if (i == 0) {
            // Local connection
            mux_channel_id = static_cast<uint32_t>(UdmMuxRelayToMuxChannelId::LOCAL_RELAY_CHANNEL);
        } else {
            // Downstream connection (perpendicular)
            // EAST/NORTH relay connects to EAST_OR_NORTH_RELAY_CHANNEL
            // WEST/SOUTH relay connects to WEST_OR_SOUTH_RELAY_CHANNEL
            mux_channel_id = (direction == eth_chan_directions::EAST || direction == eth_chan_directions::NORTH)
                                 ? static_cast<uint32_t>(UdmMuxRelayToMuxChannelId::EAST_OR_NORTH_RELAY_CHANNEL)
                                 : static_cast<uint32_t>(UdmMuxRelayToMuxChannelId::WEST_OR_SOUTH_RELAY_CHANNEL);
        }

        // Get stream ID for the target mux channel
        auto mux_config = tensix_config.get_config(FabricTensixCoreType::MUX);
        uint32_t mux_stream_id = mux_config->get_channel_credits_stream_id(
            tt::tt_fabric::ChannelTypes::RELAY_TO_MUX_CHANNEL, mux_channel_id);

        mux_infos[i] = get_mux_connection_info(mux_noc_coord, mux_channel_id, i, mux_stream_id);
    }

    return mux_infos;
}

size_t FabricTensixDatamoverRelayConfig::get_channel_credits_stream_id(
    ChannelTypes channel_type, uint32_t channel_id) const {
    // Get base stream ID from parent class
    size_t stream_id = FabricTensixDatamoverBaseConfig::get_channel_credits_stream_id(channel_type, channel_id);

    // In UDM mode, relay stream IDs must come after mux stream IDs to avoid collisions
    // Both mux and relay are on the same Tensix core, so they share the same stream ID space
    const auto& fabric_context = tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_context();
    const auto& tensix_config = fabric_context.get_builder_context().get_tensix_config();

    auto mux_config = tensix_config.get_config(FabricTensixCoreType::MUX);
    TT_FATAL(mux_config != nullptr, "Mux config cannot be null");

    // Offset relay stream IDs by the total number of mux channels (across all channel types)
    size_t mux_channel_offset = mux_config->get_total_num_channels();
    stream_id += mux_channel_offset;

    return stream_id;
}

std::vector<uint32_t> FabricTensixDatamoverRelayConfig::get_compile_time_args(
    const FabricNodeId& fabric_node_id, routing_plane_id_t routing_plane_id, eth_chan_directions direction) const {
    const auto& fabric_context = tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_context();
    const auto& tensix_config = fabric_context.get_builder_context().get_tensix_config();

    // Get mux configuration
    auto mux_config = tensix_config.get_config(FabricTensixCoreType::MUX);
    TT_FATAL(mux_config != nullptr, "Mux config must exist for relay to connect to it");

    // Channel IDs and stream IDs
    constexpr uint32_t router_to_relay_channel_id = static_cast<uint32_t>(UdmRelayChannelId::ROUTER_CHANNEL);
    const auto router_to_relay_channel_stream_id =
        get_channel_credits_stream_id(tt::tt_fabric::ChannelTypes::ROUTER_CHANNEL, router_to_relay_channel_id);

    // Collect all mux connection info
    auto mux_infos = get_all_mux_connection_infos(fabric_node_id, routing_plane_id, direction);

    // Build compile time args - organized by channel type
    uint32_t num_channel_types = static_cast<uint32_t>(channel_configs_.size());

    auto channel_type = tt::tt_fabric::ChannelTypes::RELAY_TO_MUX_CHANNEL;

    std::vector<uint32_t> ct_args = {
        static_cast<uint32_t>(status_region_.get_address()),                     // 0: status_address
        static_cast<uint32_t>(termination_signal_region_.get_address()),         // 1: termination_signal_address
        static_cast<uint32_t>(router_to_relay_channel_stream_id),                // 2: channel_stream_id
        static_cast<uint32_t>(num_iters_between_teardown_checks_),               // 3: NUM_ITERS_BETWEEN_TEARDOWN_CHECKS
        static_cast<uint32_t>(mux_config->get_num_buffers(channel_type)),        // 4: mux_num_buffers
        static_cast<uint32_t>(mux_config->get_buffer_size_bytes(channel_type)),  // 5: mux_buffer_size_bytes
        static_cast<uint32_t>(
            local_fabric_router_status_region_.get_address()),  // 6: downstream_mux_status_readback_address
        NUM_MUX_CONNECTIONS,                                    // 7: NUM_MUX_CONNECTIONS
        num_channel_types                                       // 8: number of channel types
    };

    constexpr uint32_t base_channel_id = 0;
    // Append per-channel-type arrays (grouped by channel type, sorted by enum)
    for (const auto& [type, config] : channel_configs_) {
        ct_args.push_back(config.num_channels);
    }
    // Number of buffers per channel (one per channel type)
    for (const auto& [type, config] : channel_configs_) {
        ct_args.push_back(config.num_buffers_per_channel);
    }
    // Buffer size in bytes (one per channel type)
    for (const auto& [type, config] : channel_configs_) {
        ct_args.push_back(static_cast<uint32_t>(config.buffer_size_bytes));
    }
    // Connection info base addresses (one per channel type)
    for (const auto& [type, region] : connection_info_regions_) {
        ct_args.push_back(region.get_address(base_channel_id));
    }
    // Connection handshake base addresses (one per channel type)
    for (const auto& [type, region] : connection_handshake_regions_) {
        ct_args.push_back(region.get_address(base_channel_id));
    }
    // Flow control base addresses (one per channel type)
    for (const auto& [type, region] : flow_control_regions_) {
        ct_args.push_back(region.get_address(base_channel_id));
    }
    // Channel buffer base addresses (one per channel type)
    for (const auto& [type, region] : channel_buffer_regions_) {
        ct_args.push_back(region.get_address(base_channel_id));
    }

    // Append mux connection arrays (args 14-46 = 11 fields * 3 connections)
    // Active flags array
    for (const auto& info : mux_infos) {
        ct_args.push_back(info.active);
    }
    // NOC X coords array
    for (const auto& info : mux_infos) {
        ct_args.push_back(info.noc_x);
    }
    // NOC Y coords array
    for (const auto& info : mux_infos) {
        ct_args.push_back(info.noc_y);
    }
    // Buffer base addresses array
    for (const auto& info : mux_infos) {
        ct_args.push_back(info.buffer_base_addr);
    }
    // Connection handshake addresses array
    for (const auto& info : mux_infos) {
        ct_args.push_back(info.connection_handshake_addr);
    }
    // Worker location info addresses array
    for (const auto& info : mux_infos) {
        ct_args.push_back(info.worker_location_info_addr);
    }
    // Buffer index addresses array
    for (const auto& info : mux_infos) {
        ct_args.push_back(info.buffer_index_addr);
    }
    // Flow control semaphore addresses array (relay's L1 memory)
    for (const auto& info : mux_infos) {
        ct_args.push_back(info.flow_control_semaphore_addr);
    }
    // Teardown semaphore addresses array (relay's L1 memory)
    for (const auto& info : mux_infos) {
        ct_args.push_back(info.teardown_semaphore_addr);
    }
    // Buffer index semaphore addresses array (relay's L1 memory)
    for (const auto& info : mux_infos) {
        ct_args.push_back(info.buffer_index_semaphore_addr);
    }
    // Stream IDs array (mux's stream IDs)
    for (const auto& info : mux_infos) {
        ct_args.push_back(info.stream_id);
    }

    // Final args
    ct_args.push_back(mux_config->get_status_address());                   // 44: local_mux_status_address
    ct_args.push_back(udm_memory_pool_region_.get_address());              // 45: udm_memory_pool_base_address
    ct_args.push_back(static_cast<uint32_t>(udm_memory_pool_slot_size_));  // 46: udm_memory_pool_slot_size
    ct_args.push_back(static_cast<uint32_t>(udm_memory_pool_num_slots_));  // 47: udm_memory_pool_num_slots
    ct_args.push_back(static_cast<uint32_t>(direction));                   // 48: direction

    // Response pool args
    ct_args.push_back(
        udm_registered_response_pool_region_.get_address());  // 49: udm_registered_response_pool_base_address
    ct_args.push_back(
        static_cast<uint32_t>(udm_registered_response_num_slots_));  // 50: udm_registered_response_pool_num_slots

    // Note: router NOC coords and sync address will be added by the builder
    return ct_args;
}

// ==================================================================================================
// FabricTensixDatamoverMuxBuilder Implementation
// ==================================================================================================

FabricTensixDatamoverMuxBuilder::FabricTensixDatamoverMuxBuilder(
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
    bool has_fabric_router) :
    FabricDatamoverBuilderBase(noc_x, noc_y, direction),
    my_core_logical_(my_core_logical),
    local_fabric_node_id_(local_fabric_node_id),
    remote_fabric_node_id_(remote_fabric_node_id),
    ethernet_channel_id_(ethernet_channel_id),
    link_idx_(link_idx),
    core_id_(core_id),
    config_(std::move(config)),
    has_fabric_router_(has_fabric_router) {
    channel_connection_liveness_check_disable_array_.fill(false);
    TT_FATAL(config_ != nullptr, "Config cannot be null");
}

const char* FabricTensixDatamoverMuxBuilder::get_kernel_file_path() const {
    const auto& fabric_tensix_config = tt::tt_metal::MetalContext::instance().get_fabric_tensix_config();
    if (fabric_tensix_config == tt::tt_fabric::FabricTensixConfig::UDM) {
        return "tt_metal/fabric/impl/kernels/edm_fabric/fabric_router_udm_mux_extension.cpp";
    }
    return "tt_metal/fabric/impl/kernels/edm_fabric/fabric_router_mux_extension.cpp";
}

tt::tt_fabric::SenderWorkerAdapterSpec FabricTensixDatamoverMuxBuilder::build_connection_to_fabric_channel(
    uint32_t channel_id) const {
    auto channel_type = tt::tt_fabric::ChannelTypes::ROUTER_CHANNEL;
    // minus one to skip the worker channel
    // passed in channel id: 1, 2, 3 for the downstream connections
    // corrected channel id: 0, 1, 2 for the downstream connections in ROUTER_CHANNEL
    channel_id -= 1;

    // skip the channel liveness check if it is used for upstream connection (persistent)
    channel_connection_liveness_check_disable_array_[channel_id] = true;

    return tt::tt_fabric::SenderWorkerAdapterSpec{
        noc_x_,                                                               // edm_noc_x
        noc_y_,                                                               // edm_noc_y
        config_->get_channel_base_address(channel_type, channel_id),          // edm_buffer_base_addr
        config_->get_num_buffers(channel_type),                               // num_buffers_per_channel
        config_->get_flow_control_address(channel_type, channel_id),          // edm_l1_sem_addr
        config_->get_connection_handshake_address(channel_type, channel_id),  // edm_connection_handshake_addr
        config_->get_connection_info_address(channel_type, channel_id),       // edm_worker_location_info_addr
        config_->get_buffer_size_bytes(channel_type),                         // buffer_size_bytes
        config_->get_buffer_index_address(channel_type, channel_id),          // buffer_index_semaphore_id
        tt::tt_fabric::eth_chan_directions::EAST                              // edm_direction
    };
}

void FabricTensixDatamoverMuxBuilder::append_upstream_routers_noc_xy(uint32_t noc_x, uint32_t noc_y) {
    upstream_routers_noc_x_.push_back(noc_x);
    upstream_routers_noc_y_.push_back(noc_y);
}

void FabricTensixDatamoverMuxBuilder::create_and_compile(tt::tt_metal::Program& program) {
    const auto& fabric_tensix_config = tt::tt_metal::MetalContext::instance().get_fabric_tensix_config();

    // Select processor and NOC based on core type
    tt::tt_metal::DataMovementProcessor processor = (core_id_ == FabricTensixCoreType::MUX)
                                                        ? tt::tt_metal::DataMovementProcessor::RISCV_0
                                                        : tt::tt_metal::DataMovementProcessor::RISCV_1;

    tt::tt_metal::NOC noc;
    if (fabric_tensix_config == tt::tt_fabric::FabricTensixConfig::UDM) {
        // In UDM mode, mux uses NOC selection from UdmNoCSelection enum
        noc = static_cast<tt::tt_metal::NOC>(UdmNoCSelection::mux_noc);
    } else {
        noc = (core_id_ == FabricTensixCoreType::MUX) ? tt::tt_metal::NOC::RISCV_0_default
                                                      : tt::tt_metal::NOC::RISCV_1_default;
    }
    // Create the mux kernel
    auto mux_kernel = tt::tt_metal::CreateKernel(
        program,
        get_kernel_file_path(),
        my_core_logical_,
        tt::tt_metal::DataMovementConfig{
            .processor = processor, .noc = noc, .compile_args = get_compile_time_args(), .defines = {}});

    // Set runtime arguments
    tt::tt_metal::SetRuntimeArgs(program, mux_kernel, my_core_logical_, get_runtime_args(program));
}

std::vector<uint32_t> FabricTensixDatamoverMuxBuilder::get_channel_stream_ids(ChannelTypes channel_type) const {
    const auto& fabric_context = tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_context();
    size_t num_channels = config_->get_num_channels(channel_type);
    std::vector<uint32_t> fabric_stream_ids;

    switch (channel_type) {
        case ChannelTypes::WORKER_CHANNEL:
        case ChannelTypes::RELAY_TO_MUX_CHANNEL:
        case ChannelTypes::MUX_TO_MUX_CHANNEL: {
            // Worker channels: one stream ID per channel
            for (size_t channel = 0; channel < num_channels; channel++) {
                fabric_stream_ids.push_back(config_->get_channel_credits_stream_id(channel_type, channel));
            }
            break;
        }
        case ChannelTypes::ROUTER_CHANNEL: {
            // Router channels: topology-based fabric router stream IDs (only in Legacy MUX mode)
            const auto topology = fabric_context.get_fabric_topology();
            switch (topology) {
                case tt::tt_fabric::Topology::NeighborExchange:
                    TT_THROW("NeighborExchange topology has not been tested in MUX mode");
                    break;
                case tt::tt_fabric::Topology::Linear:
                case tt::tt_fabric::Topology::Ring:
                    fabric_stream_ids = {tt::tt_fabric::StreamRegAssignments::sender_channel_1_free_slots_stream_id};
                    break;
                case tt::tt_fabric::Topology::Mesh:
                case tt::tt_fabric::Topology::Torus:
                    fabric_stream_ids = {
                        tt::tt_fabric::StreamRegAssignments::sender_channel_1_free_slots_stream_id,
                        tt::tt_fabric::StreamRegAssignments::sender_channel_2_free_slots_stream_id,
                        tt::tt_fabric::StreamRegAssignments::sender_channel_3_free_slots_stream_id};
                    break;
                default: TT_THROW("Unknown fabric topology: {}", static_cast<int>(topology)); break;
            }
            break;
        }
        default: TT_THROW("Unknown channel type: {}", static_cast<uint32_t>(channel_type)); break;
    }

    TT_FATAL(
        num_channels == fabric_stream_ids.size(),
        "Channel type {} expects {} channels but got {} stream IDs",
        static_cast<uint32_t>(channel_type),
        num_channels,
        fabric_stream_ids.size());

    return fabric_stream_ids;
}

std::vector<uint32_t> FabricTensixDatamoverMuxBuilder::get_persistent_channels_flags(ChannelTypes channel_type) const {
    const auto& fabric_context = tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_context();
    const auto& tensix_config = fabric_context.get_builder_context().get_tensix_config();
    size_t num_channels = config_->get_num_channels(channel_type);
    std::vector<uint32_t> is_persistent_channels(num_channels, 0);

    switch (channel_type) {
        case ChannelTypes::ROUTER_CHANNEL: {
            // Router channels: check liveness array for persistent connections (e.g., to upstream fabric routers)
            for (size_t i = 0; i < num_channels; i++) {
                if (channel_connection_liveness_check_disable_array_[i]) {
                    is_persistent_channels[i] = 1;
                }
            }
            break;
        }
        case ChannelTypes::RELAY_TO_MUX_CHANNEL: {
            // Relay-to-Mux channels: check if relays exist in perpendicular directions (UDM mode only)
            TT_FATAL(num_channels == 3, "RELAY_TO_MUX_CHANNEL should have exactly 3 channels (got {})", num_channels);

            // Z direction muxes don't have relay connections (perpendicular directions not defined for Z)
            if (direction_ == eth_chan_directions::Z) {
                break;
            }

            // Channel 0: Local relay (not persistent for non-active tensix core)
            const auto* noc_coords =
                tensix_config.get_active_tensix_noc_coords(local_fabric_node_id_, link_idx_, direction_);
            if (noc_coords) {
                is_persistent_channels[static_cast<uint32_t>(UdmMuxRelayToMuxChannelId::LOCAL_RELAY_CHANNEL)] = 1;
            }

            // Channels 1-2: Upstream relays (perpendicular directions)
            auto [upstream_relay_dir1, upstream_relay_dir2] = builder::get_perpendicular_directions(direction_);
            std::array<eth_chan_directions, 2> upstream_relay_dirs = {upstream_relay_dir1, upstream_relay_dir2};
            std::array<UdmMuxRelayToMuxChannelId, 2> upstream_relay_channels = {
                UdmMuxRelayToMuxChannelId::EAST_OR_NORTH_RELAY_CHANNEL,
                UdmMuxRelayToMuxChannelId::WEST_OR_SOUTH_RELAY_CHANNEL};

            for (size_t i = 0; i < upstream_relay_dirs.size(); i++) {
                const auto* noc_coords = tensix_config.get_active_tensix_noc_coords(
                    local_fabric_node_id_, link_idx_, upstream_relay_dirs[i]);
                if (noc_coords) {
                    is_persistent_channels[static_cast<uint32_t>(upstream_relay_channels[i])] = 1;
                }
            }
            break;
        }
        case ChannelTypes::MUX_TO_MUX_CHANNEL: {
            // Mux-to-Mux channels: check if muxes exist in other directions (UDM mode only)
            TT_FATAL(num_channels == 3, "MUX_TO_MUX_CHANNEL should have exactly 3 channels (got {})", num_channels);

            // Z direction muxes don't have inter-mux connections
            // MUX_TO_MUX channels only support mesh directions (E/W/N/S)
            if (direction_ == eth_chan_directions::Z) {
                break;
            }

            // Exclude Z direction - MUX mode only supports mesh directions (E/W/N/S)
            auto upstream_mux_dirs = builder::get_all_other_directions(direction_, /*exclude_z=*/true);
            // for mux to mux channel, we need to check all the tensix cores, since there are mux kernel in the
            // non-active tensix cores as well
            for (auto upstream_dir : upstream_mux_dirs) {
                const auto* noc_coords =
                    tensix_config.get_tensix_noc_coords(local_fabric_node_id_, link_idx_, upstream_dir);
                if (noc_coords) {
                    uint32_t channel_id = get_upstream_mux_channel_id(direction_, upstream_dir);
                    is_persistent_channels[channel_id] = 1;
                }
            }
            break;
        }
        case ChannelTypes::WORKER_CHANNEL: break;  // Worker channels: always non-persistent
        default: TT_THROW("Unknown channel type: {}", static_cast<uint32_t>(channel_type)); break;
    }

    return is_persistent_channels;
}

std::vector<uint32_t> FabricTensixDatamoverMuxBuilder::get_compile_time_args() const {
    const auto& fabric_context = tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_context();
    const auto& builder_context = fabric_context.get_builder_context();
    const auto& fabric_tensix_config = tt::tt_metal::MetalContext::instance().get_fabric_tensix_config();
    const auto& fabric_router_config = builder_context.get_fabric_router_config(fabric_tensix_config);

    // Call config's get_compile_time_args with fabric node, routing plane, and direction
    auto ct_args = config_->get_compile_time_args(local_fabric_node_id_, link_idx_, direction_);

    // Add bubble flow control flag, then upstream routers count and sync address
    // Order must match kernel expectations: bubble_flow_control, num_upstream_routers, sync_address
    ct_args.push_back(fabric_context.is_bubble_flow_control_enabled());
    ct_args.push_back(static_cast<uint32_t>(upstream_routers_noc_x_.size()));
    ct_args.push_back(fabric_router_config.edm_local_tensix_sync_address);

    // Append stream IDs and persistent flags grouped by channel type
    // For each channel type: [stream_ids...], [persistent_flags...]
    for (const auto& [type, config] : config_->get_channel_configs()) {
        auto fabric_stream_ids = get_channel_stream_ids(type);
        ct_args.insert(ct_args.end(), fabric_stream_ids.begin(), fabric_stream_ids.end());
    }

    for (const auto& [type, config] : config_->get_channel_configs()) {
        auto is_persistent_channels = get_persistent_channels_flags(type);
        ct_args.insert(ct_args.end(), is_persistent_channels.begin(), is_persistent_channels.end());
    }

    // In UDM mode, add relay's termination signal address for mux to write during teardown
    if (fabric_tensix_config == tt::tt_fabric::FabricTensixConfig::UDM) {
        const auto& tensix_config = fabric_context.get_builder_context().get_tensix_config();
        auto relay_config = tensix_config.get_config(FabricTensixCoreType::RELAY);
        auto relay_config_typed = std::dynamic_pointer_cast<const FabricTensixDatamoverRelayConfig>(relay_config);
        ct_args.push_back(relay_config_typed->get_relay_termination_signal_address());
    } else if (fabric_tensix_config == tt::tt_fabric::FabricTensixConfig::MUX) {
        // Add sender_channel_is_traffic_injection_channel_array to ct args for MUX mode
        for (bool is_injection : sender_channel_is_traffic_injection_channel_array) {
            ct_args.push_back(is_injection ? 1 : 0);
        }
    }

    // Add direction
    ct_args.push_back(static_cast<uint32_t>(direction_));

    // Only needed in UDM mode
    if (fabric_tensix_config == tt::tt_fabric::FabricTensixConfig::UDM) {
        // Add has_fabric_router flag (1 = has router, 0 = no router / missing direction)
        ct_args.push_back(static_cast<uint32_t>(has_fabric_router_ ? 1 : 0));
        // Add channel storage address (L1 address for storing worker channel arrays)
        ct_args.push_back(config_->get_channel_storage_base_address());
    }

    return ct_args;
}

std::vector<uint32_t> FabricTensixDatamoverMuxBuilder::get_runtime_args(tt::tt_metal::Program& program) const {
    std::vector<uint32_t> runtime_args;
    const auto& fabric_tensix_config = tt::tt_metal::MetalContext::instance().get_fabric_tensix_config();
    if (fabric_tensix_config == tt::tt_fabric::FabricTensixConfig::UDM) {
        TT_FATAL(
            upstream_routers_noc_x_.empty() && upstream_routers_noc_y_.empty(),
            "In UDM mode there should NOT be any upstream routers being set");
    }
    runtime_args.insert(runtime_args.end(), upstream_routers_noc_x_.begin(), upstream_routers_noc_x_.end());
    runtime_args.insert(runtime_args.end(), upstream_routers_noc_y_.begin(), upstream_routers_noc_y_.end());

    auto config_runtime_args =
        config_->get_run_time_args(local_fabric_node_id_, remote_fabric_node_id_, link_idx_, program, my_core_logical_);

    runtime_args.insert(runtime_args.end(), config_runtime_args.begin(), config_runtime_args.end());
    return runtime_args;
}

// ==================================================================================================
// FabricTensixDatamoverRelayBuilder Implementation
// ==================================================================================================

FabricTensixDatamoverRelayBuilder::FabricTensixDatamoverRelayBuilder(
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
    bool /*has_fabric_router*/) :
    FabricDatamoverBuilderBase(noc_x, noc_y, direction),
    my_core_logical_(my_core_logical),
    local_fabric_node_id_(local_fabric_node_id),
    remote_fabric_node_id_(remote_fabric_node_id),
    ethernet_channel_id_(ethernet_channel_id),
    link_idx_(link_idx),
    core_id_(core_id),
    config_(std::move(config)) {
    channel_connection_liveness_check_disable_array_.fill(false);
    TT_FATAL(config_ != nullptr, "Config cannot be null");
}

tt::tt_fabric::SenderWorkerAdapterSpec FabricTensixDatamoverRelayBuilder::build_connection_to_fabric_channel(
    uint32_t channel_id) const {
    auto channel_type = tt::tt_fabric::ChannelTypes::ROUTER_CHANNEL;

    // skip the channel liveness check if it is used for upstream connection (persistent)
    channel_connection_liveness_check_disable_array_[channel_id] = true;

    return tt::tt_fabric::SenderWorkerAdapterSpec{
        noc_x_,                                                               // edm_noc_x
        noc_y_,                                                               // edm_noc_y
        config_->get_channel_base_address(channel_type, channel_id),          // edm_buffer_base_addr
        config_->get_num_buffers(channel_type),                               // num_buffers_per_channel
        config_->get_flow_control_address(channel_type, channel_id),          // edm_l1_sem_addr
        config_->get_connection_handshake_address(channel_type, channel_id),  // edm_connection_handshake_addr
        config_->get_connection_info_address(channel_type, channel_id),       // edm_worker_location_info_addr
        config_->get_buffer_size_bytes(channel_type),                         // buffer_size_bytes
        config_->get_buffer_index_address(channel_type, channel_id),          // buffer_index_semaphore_id
        tt::tt_fabric::eth_chan_directions::EAST                              // edm_direction
    };
}

void FabricTensixDatamoverRelayBuilder::create_and_compile(tt::tt_metal::Program& program) {
    // Select processor and NOC based on core type
    tt::tt_metal::DataMovementProcessor processor = (core_id_ == FabricTensixCoreType::MUX)
                                                        ? tt::tt_metal::DataMovementProcessor::RISCV_0
                                                        : tt::tt_metal::DataMovementProcessor::RISCV_1;

    // In UDM mode, relay uses NOC selection from UdmNoCSelection enum (NOC 1 = edm_to_local_chip_noc)
    tt::tt_metal::NOC noc = static_cast<tt::tt_metal::NOC>(UdmNoCSelection::relay_noc);

    // Create the relay kernel
    auto relay_kernel = tt::tt_metal::CreateKernel(
        program,
        "tt_metal/fabric/impl/kernels/edm_fabric/fabric_router_relay_extension.cpp",
        my_core_logical_,
        tt::tt_metal::DataMovementConfig{
            .processor = processor, .noc = noc, .compile_args = get_compile_time_args(), .defines = {}});

    // Set runtime arguments
    tt::tt_metal::SetRuntimeArgs(program, relay_kernel, my_core_logical_, get_runtime_args(program));
}

std::vector<uint32_t> FabricTensixDatamoverRelayBuilder::get_compile_time_args() const {
    const auto& fabric_context = tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_context();
    const auto& fabric_tensix_config = tt::tt_metal::MetalContext::instance().get_fabric_tensix_config();
    TT_FATAL(
        fabric_tensix_config == tt::tt_fabric::FabricTensixConfig::UDM,
        "Relay builder should only be used in UDM mode");

    // 0-47: All relay channel configuration, mux connection config + arrays (1 + 3x10 fields), and UDM memory pool from
    // config
    auto ct_args = config_->get_compile_time_args(local_fabric_node_id_, link_idx_, direction_);

    // 48-49: Fabric router NOC coordinates
    ct_args.push_back(router_noc_x_);  // 48: router_noc_x
    ct_args.push_back(router_noc_y_);  // 49: router_noc_y

    // 50: fabric_router_sync_address
    const auto& fabric_router_config =
        fabric_context.get_builder_context().get_fabric_router_config(fabric_tensix_config);
    ct_args.push_back(fabric_router_config.edm_local_tensix_sync_address);  // 50: fabric_router_sync_address

    return ct_args;
}

std::vector<uint32_t> FabricTensixDatamoverRelayBuilder::get_runtime_args(tt::tt_metal::Program&) const {
    std::vector<uint32_t> runtime_args;
    // Memory regions to clear at startup
    auto regions_to_clear = config_->get_memory_regions_to_clear();
    runtime_args.push_back(static_cast<uint32_t>(regions_to_clear.size()));
    for (const auto& [address, size] : regions_to_clear) {
        runtime_args.push_back(static_cast<uint32_t>(address));
        runtime_args.push_back(static_cast<uint32_t>(size));
    }

    return runtime_args;
}

void FabricTensixDatamoverRelayBuilder::append_router_noc_xy(uint32_t noc_x, uint32_t noc_y) {
    router_noc_x_ = noc_x;
    router_noc_y_ = noc_y;
}

}  // namespace tt::tt_fabric
