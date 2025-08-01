// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <vector>
#include <tt-metalium/fabric.hpp>
#include <tt-metalium/control_plane.hpp>
#include "impl/context/metal_context.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "umd/device/tt_core_coordinates.h"
#include <enchantum/enchantum.hpp>

namespace tt::tt_fabric {

// Implementation of nested MemoryRegion methods
FabricMuxConfig::MemoryRegion::MemoryRegion(size_t base, size_t unit_sz, size_t count) :
    base_address(base), unit_size(unit_sz), num_units(count) {}

size_t FabricMuxConfig::MemoryRegion::get_address(size_t offset) const {
    TT_FATAL(offset < num_units, "Offset {} exceeds region size {}", offset, num_units);
    return base_address + (offset * unit_size);
}

size_t FabricMuxConfig::MemoryRegion::get_end_address() const { return base_address + (num_units * unit_size); }

size_t FabricMuxConfig::MemoryRegion::get_total_size() const { return num_units * unit_size; }

/*
    Channel layout: Each channel consists of buffers/slots
    ┌──────────────────────────┐
    │ ┌──┐┌──┐┌──┐        ┌──┐ │
    │ └──┘└──┘└──┘        └──┘ │
    └──────────────────────────┘

    Channel organization in L1:
    |  Full size channel  |
    |  Full size channel  |
    |  Full size channel  |
    |         .           |
    |         .           |
    | Header only channel |
    | Header only channel |
    | Header only channel |
    |          .          |
    |          .          |

    Basic operation:
        In each iteration the mux kernel round robins over all the channels, and forwards data over fabric.
        It processes the full size channels first and then the header only channels.

    Configuration parameters:
    -> Number of full size channels
    -> Number of header only channels
    -> Number of buffers/slots in a full size channel
    -> Number of buffers/slots in a header only channel
    -> Buffer size in bytes for a full size channel (for a header only channel its equal to the pre-determined packet
        header size)
    -> Base address where the channels start in the mux's L1
    -> Core Type of the mux. Supports Worker and Idle Ethernet

    Advanced configuration parameters:
    -> Number of full size channel iters
        This determines the number of full size channel iters to run per iter of header only channels.
        By default its set to 1, which indicates that the full size channels and header only channels are processed
        equally. This can be incremented in cases where the full size channels are not big enough compared to the
        buffers on the receiver. In such cases, the receiver can also accumulate credits and send them back in one shot
        instead of sending back one-by-one which may not always be the most efficient.
    -> Number of iters between teardown checks
        This determines how frequently the mux kernel checks for the termination signal. The larger this value, the less
        frequently mux kernel will check for the termination signal. Can be used to optimize performance, but very large
        values can impact teardown times.
*/

FabricMuxConfig::FabricMuxConfig(
    uint8_t num_full_size_channels,
    uint8_t num_header_only_channels,
    uint8_t num_buffers_full_size_channel,
    uint8_t num_buffers_header_only_channel,
    size_t buffer_size_bytes_full_size_channel,
    size_t base_l1_address,
    CoreType core_type) :
    num_full_size_channels_(num_full_size_channels),
    num_header_only_channels_(num_header_only_channels),
    // set to default number of buffers only for compilation purposes, no functional impact
    num_buffers_full_size_channel_(
        num_buffers_full_size_channel == 0 ? default_num_buffers : num_buffers_full_size_channel),
    num_buffers_header_only_channel_(
        num_buffers_header_only_channel == 0 ? default_num_buffers : num_buffers_header_only_channel),
    buffer_size_bytes_full_size_channel_(buffer_size_bytes_full_size_channel),
    core_type_(core_type) {
    TT_FATAL(
        num_full_size_channels_ > 0 || num_header_only_channels_ > 0,
        "At least one type of channel must be configured");

    size_t max_buffer_size_bytes_full_size_channel = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    TT_FATAL(
        buffer_size_bytes_full_size_channel <= max_buffer_size_bytes_full_size_channel,
        "Buffer size bytes for full size channel should be less than or equal to: {}, but got: {}",
        max_buffer_size_bytes_full_size_channel,
        buffer_size_bytes_full_size_channel);

    noc_aligned_address_size_bytes_ =
        tt::tt_metal::MetalContext::instance().hal().get_alignment(tt::tt_metal::HalMemType::L1);

    buffer_size_bytes_header_only_channel_ = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();

    full_size_channel_size_bytes_ = num_buffers_full_size_channel_ * buffer_size_bytes_full_size_channel_;
    header_only_channel_size_bytes_ = num_buffers_header_only_channel_ * buffer_size_bytes_header_only_channel_;

    auto num_total_channels = num_full_size_channels_ + num_header_only_channels_;

    // Initialize memory regions sequentially
    size_t current_address = base_l1_address;

    // Status region (single address)
    status_region_ = MemoryRegion(current_address, noc_aligned_address_size_bytes_, 1);
    current_address = status_region_.get_end_address();

    // Local fabric router status region (single address)
    local_fabric_router_status_region_ = MemoryRegion(current_address, noc_aligned_address_size_bytes_, 1);
    current_address = local_fabric_router_status_region_.get_end_address();

    // Termination signal region (single address)
    termination_signal_region_ = MemoryRegion(current_address, noc_aligned_address_size_bytes_, 1);
    current_address = termination_signal_region_.get_end_address();

    // Connection info region (one entry per channel)
    connection_info_region_ =
        MemoryRegion(current_address, sizeof(tt::tt_fabric::EDMChannelWorkerLocationInfo), num_total_channels);
    current_address = connection_info_region_.get_end_address();

    // Connection handshake region (one entry per channel)
    connection_handshake_region_ = MemoryRegion(current_address, noc_aligned_address_size_bytes_, num_total_channels);
    current_address = connection_handshake_region_.get_end_address();

    // Flow control region (one entry per channel)
    flow_control_region_ = MemoryRegion(current_address, noc_aligned_address_size_bytes_, num_total_channels);
    current_address = flow_control_region_.get_end_address();

    // Buffer index region (one entry per channel)
    buffer_index_region_ = MemoryRegion(current_address, noc_aligned_address_size_bytes_, num_total_channels);
    current_address = buffer_index_region_.get_end_address();

    // Full size channels region
    full_size_channels_region_ = MemoryRegion(current_address, full_size_channel_size_bytes_, num_full_size_channels_);
    current_address = full_size_channels_region_.get_end_address();

    // Header only channels region
    header_only_channels_region_ =
        MemoryRegion(current_address, header_only_channel_size_bytes_, num_header_only_channels_);

    memory_map_end_address_ = header_only_channels_region_.get_end_address();

    const auto& hal = tt_metal::MetalContext::instance().hal();
    tt_metal::HalProgrammableCoreType hal_core_type;
    if (core_type_ == CoreType::WORKER) {
        hal_core_type = tt_metal::HalProgrammableCoreType::TENSIX;
    } else if (core_type_ == CoreType::IDLE_ETH) {
        hal_core_type = tt_metal::HalProgrammableCoreType::IDLE_ETH;
    } else {
        TT_THROW("Fabric Mux does not support core type {}", enchantum::to_string(core_type));
    }

    core_type_index_ = hal.get_programmable_core_type_index(hal_core_type);
    auto l1_end_address = hal.get_dev_addr(hal_core_type, tt_metal::HalL1MemAddrType::BASE) +
                          hal.get_dev_size(hal_core_type, tt_metal::HalL1MemAddrType::BASE);

    // The memory map ends at the end of the last region (header-only channels)
    TT_FATAL(
        memory_map_end_address_ <= l1_end_address,
        "Memory map end address: {} is greater than L1 end address: {}",
        memory_map_end_address_,
        l1_end_address);
}

std::vector<uint32_t> FabricMuxConfig::get_fabric_mux_compile_time_args() const {
    const auto& fabric_router_config =
        tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_context().get_fabric_router_config();
    return std::vector<uint32_t>{
        num_full_size_channels_,
        num_buffers_full_size_channel_,
        buffer_size_bytes_full_size_channel_,
        num_header_only_channels_,
        num_buffers_header_only_channel_,
        status_region_.get_address(),
        termination_signal_region_.get_address(),
        connection_info_region_.get_address(),
        connection_handshake_region_.get_address(),
        flow_control_region_.get_address(),
        full_size_channels_region_.get_address(),
        local_fabric_router_status_region_.get_address(),
        fabric_router_config.edm_status_address,
        fabric_router_config.sender_channels_num_buffers[0],
        num_full_size_channel_iters_,
        num_iters_between_teardown_checks_,
        core_type_index_};
}

std::vector<uint32_t> FabricMuxConfig::get_fabric_mux_run_time_args(
    const FabricNodeId& src_fabric_node_id,
    const FabricNodeId& dst_fabric_node_id,
    uint32_t link_idx,
    tt::tt_metal::Program& mux_program,
    const CoreCoord& mux_logical_core) const {
    std::vector<uint32_t> args;

    auto regions_to_clear = get_memory_regions_to_clear();
    const auto num_regions_to_clear = regions_to_clear.size();
    args.reserve(num_regions_to_clear * 2 + 1);
    args.push_back(static_cast<uint32_t>(num_regions_to_clear));
    for (const auto& [address, size] : regions_to_clear) {
        args.push_back(static_cast<uint32_t>(address));
        args.push_back(static_cast<uint32_t>(size));
    }

    tt::tt_fabric::append_fabric_connection_rt_args(
        src_fabric_node_id, dst_fabric_node_id, link_idx, mux_program, mux_logical_core, args, core_type_);

    return args;
}

uint8_t FabricMuxConfig::get_num_buffers(FabricMuxChannelType channel_type) const {
    return channel_type == FabricMuxChannelType::FULL_SIZE_CHANNEL ? num_buffers_full_size_channel_
                                                                   : num_buffers_header_only_channel_;
}

size_t FabricMuxConfig::get_buffer_size_bytes(FabricMuxChannelType channel_type) const {
    return channel_type == FabricMuxChannelType::FULL_SIZE_CHANNEL ? buffer_size_bytes_full_size_channel_
                                                                   : buffer_size_bytes_header_only_channel_;
}

size_t FabricMuxConfig::get_status_address() const { return status_region_.get_address(); }

size_t FabricMuxConfig::get_termination_signal_address() const { return termination_signal_region_.get_address(); }

size_t FabricMuxConfig::get_channel_credits_stream_id(FabricMuxChannelType channel_type, uint8_t channel_id) const {
    validate_channel_id(channel_type, channel_id);

    return get_channel_global_offset(channel_type, channel_id);
}

size_t FabricMuxConfig::get_channel_base_address(FabricMuxChannelType channel_type, uint8_t channel_id) const {
    validate_channel_id(channel_type, channel_id);

    return channel_type == FabricMuxChannelType::FULL_SIZE_CHANNEL
               ? full_size_channels_region_.get_address(channel_id)
               : header_only_channels_region_.get_address(channel_id);
}

size_t FabricMuxConfig::get_connection_info_address(FabricMuxChannelType channel_type, uint8_t channel_id) const {
    validate_channel_id(channel_type, channel_id);

    return connection_info_region_.get_address(get_channel_global_offset(channel_type, channel_id));
}

size_t FabricMuxConfig::get_connection_handshake_address(FabricMuxChannelType channel_type, uint8_t channel_id) const {
    validate_channel_id(channel_type, channel_id);

    return connection_handshake_region_.get_address(get_channel_global_offset(channel_type, channel_id));
}

size_t FabricMuxConfig::get_flow_control_address(FabricMuxChannelType channel_type, uint8_t channel_id) const {
    validate_channel_id(channel_type, channel_id);

    return flow_control_region_.get_address(get_channel_global_offset(channel_type, channel_id));
}

size_t FabricMuxConfig::get_buffer_index_address(FabricMuxChannelType channel_type, uint8_t channel_id) const {
    validate_channel_id(channel_type, channel_id);

    return buffer_index_region_.get_address(get_channel_global_offset(channel_type, channel_id));
}

void FabricMuxConfig::set_num_full_size_channel_iters(size_t new_val) {
    TT_FATAL(num_full_size_channels_ > 0, "Cannot set iterations when no full size channels exist");
    TT_FATAL(new_val > 0, "Number of iterations must be greater than 0");
    num_full_size_channel_iters_ = new_val;
}

void FabricMuxConfig::set_num_iters_between_teardown_checks(size_t new_val) {
    TT_FATAL(new_val > 0, "Setting num iters b/w teardown checks to 0 will result in no data being sent over fabric");
    num_iters_between_teardown_checks_ = new_val;
}

size_t FabricMuxConfig::get_memory_map_end_address() const { return memory_map_end_address_; }

uint8_t FabricMuxConfig::get_num_channels(FabricMuxChannelType channel_type) const {
    return channel_type == FabricMuxChannelType::FULL_SIZE_CHANNEL ? num_full_size_channels_
                                                                   : num_header_only_channels_;
}

void FabricMuxConfig::validate_channel_id(FabricMuxChannelType channel_type, uint8_t channel_id) const {
    TT_FATAL(
        channel_id < get_num_channels(channel_type),
        "Invalid channel id for channel type: {}. Requested channel id: {} but maximum is {}",
        enchantum::to_string(channel_type),
        channel_id,
        get_num_channels(channel_type));
}

uint8_t FabricMuxConfig::get_channel_global_offset(FabricMuxChannelType channel_type, uint8_t channel_id) const {
    return (channel_type == FabricMuxChannelType::HEADER_ONLY_CHANNEL) ? num_full_size_channels_ + channel_id
                                                                       : channel_id;
}

std::vector<std::pair<size_t, size_t>> FabricMuxConfig::get_memory_regions_to_clear() const {
    return {
        {termination_signal_region_.get_address(), termination_signal_region_.get_total_size()},
        {connection_handshake_region_.get_address(), connection_handshake_region_.get_total_size()},
        {flow_control_region_.get_address(), flow_control_region_.get_total_size()},
        {buffer_index_region_.get_address(), buffer_index_region_.get_total_size()}};
}

}  // namespace tt::tt_fabric
