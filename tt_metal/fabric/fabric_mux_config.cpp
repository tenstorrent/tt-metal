// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstddef>
#include <vector>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt_stl/assert.hpp>
#include "distributed/mesh_device_impl.hpp"
#include "impl/context/metal_env_impl.hpp"
#include "tt_metal/fabric/fabric_mux_program_local_l1.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/fabric_host_utils.hpp"
#include <tt-umd/types/core_coordinates.hpp>
#include <enchantum/enchantum.hpp>
#include "tt_metal/fabric/builder/fabric_static_sized_channels_allocator.hpp"
namespace tt::tt_fabric {

namespace {

tt_metal::HalProgrammableCoreType get_mux_hal_core_type(CoreType core_type) {
    if (core_type == CoreType::WORKER) {
        return tt_metal::HalProgrammableCoreType::TENSIX;
    }
    if (core_type == CoreType::IDLE_ETH) {
        return tt_metal::HalProgrammableCoreType::IDLE_ETH;
    }
    TT_THROW("Fabric Mux does not support core type {}", enchantum::to_string(core_type));
}

size_t compute_physical_l1_end_address(const tt_metal::Hal& hal, tt_metal::HalProgrammableCoreType hal_core_type) {
    return hal.get_dev_addr(hal_core_type, tt_metal::HalL1MemAddrType::BASE) +
           hal.get_dev_size(hal_core_type, tt_metal::HalL1MemAddrType::BASE);
}

// A caller-supplied ceiling only ever tightens the bound; it can never license the map to run past the physical end
// of L1.
size_t compute_mux_l1_end_address(size_t physical_l1_end_address, size_t usable_l1_end_address) {
    return usable_l1_end_address == 0 ? physical_l1_end_address
                                      : std::min<size_t>(usable_l1_end_address, physical_l1_end_address);
}

}  // namespace

// Implementation of nested MemoryRegion methods
FabricMuxConfig::MemoryRegion::MemoryRegion(size_t base, size_t unit_sz, size_t count) :
    base_address(base), unit_size(unit_sz), num_units(count) {}

size_t FabricMuxConfig::MemoryRegion::get_address(size_t offset) const {
    // Special case for empty regions
    if (num_units == 0) {
        TT_FATAL(offset == 0, "Offset {} is invalid for empty region (num_units == 0)", offset);
        return base_address;
    }

    TT_FATAL(offset < num_units, "Offset {} exceeds region size {}", offset, num_units);
    return base_address + (offset * unit_size);
}

size_t FabricMuxConfig::MemoryRegion::get_end_address() const { return base_address + (num_units * unit_size); }

size_t FabricMuxConfig::MemoryRegion::get_total_size() const { return num_units * unit_size; }

size_t FabricMuxConfig::MemoryMap::get_end_address() const { return header_only_channels.get_end_address(); }

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
    const tt::tt_metal::distributed::MeshDevice& mesh_device,
    uint8_t num_full_size_channels,
    uint8_t num_header_only_channels,
    uint8_t num_buffers_full_size_channel,
    uint8_t num_buffers_header_only_channel,
    size_t buffer_size_bytes_full_size_channel,
    size_t base_l1_address,
    CoreType core_type,
    size_t usable_l1_end_address) :
    FabricMuxConfig(
        mesh_device.impl().metal_env().get_control_plane(),
        num_full_size_channels,
        num_header_only_channels,
        num_buffers_full_size_channel,
        num_buffers_header_only_channel,
        buffer_size_bytes_full_size_channel,
        base_l1_address,
        core_type,
        usable_l1_end_address) {}

FabricMuxConfig::FabricMuxConfig(
    const ControlPlane& control_plane,
    uint8_t num_full_size_channels,
    uint8_t num_header_only_channels,
    uint8_t num_buffers_full_size_channel,
    uint8_t num_buffers_header_only_channel,
    size_t buffer_size_bytes_full_size_channel,
    size_t base_l1_address,
    CoreType core_type,
    size_t usable_l1_end_address) :
    core_type_(core_type),
    num_full_size_channels_(num_full_size_channels),
    num_header_only_channels_(num_header_only_channels),
    num_buffers_full_size_channel_(num_buffers_or_default(num_buffers_full_size_channel)),
    num_buffers_header_only_channel_(num_buffers_or_default(num_buffers_header_only_channel)),
    buffer_size_bytes_full_size_channel_(buffer_size_bytes_full_size_channel) {
    TT_FATAL(
        num_full_size_channels_ > 0 || num_header_only_channels_ > 0,
        "At least one type of channel must be configured");

    const auto& fabric_context = control_plane.get_fabric_context();
    const auto& hal = fabric_context.get_hal();

    size_t max_buffer_size_bytes_full_size_channel = fabric_context.get_fabric_channel_buffer_size_bytes();
    TT_FATAL(
        buffer_size_bytes_full_size_channel <= max_buffer_size_bytes_full_size_channel,
        "Buffer size bytes for full size channel should be less than or equal to: {}, but got: {}",
        max_buffer_size_bytes_full_size_channel,
        buffer_size_bytes_full_size_channel);

    noc_aligned_address_size_bytes_ = hal.get_alignment(tt::tt_metal::HalMemType::L1);

    buffer_size_bytes_header_only_channel_ = fabric_context.get_fabric_packet_header_size_bytes();

    // The producer reads and writes each buffer index entry as a whole SenderChannelProducerCursor, so the struct *is*
    // the entry -- require equality, not fit.
    TT_FATAL(
        sizeof(tt::tt_fabric::SenderChannelProducerCursor) == noc_aligned_address_size_bytes_,
        "SenderChannelProducerCursor is {} B but the per-channel buffer index entry is {} B; they must match",
        sizeof(tt::tt_fabric::SenderChannelProducerCursor),
        noc_aligned_address_size_bytes_);

    memory_map_ = lay_out_memory_map(
        base_l1_address,
        noc_aligned_address_size_bytes_,
        num_full_size_channels_,
        num_header_only_channels_,
        num_buffers_full_size_channel_,
        num_buffers_header_only_channel_,
        buffer_size_bytes_full_size_channel_,
        buffer_size_bytes_header_only_channel_);

    const tt_metal::HalProgrammableCoreType hal_core_type = get_mux_hal_core_type(core_type_);
    core_type_index_ = hal.get_programmable_core_type_index(hal_core_type);
    const size_t physical_l1_end_address = compute_physical_l1_end_address(hal, hal_core_type);
    const size_t l1_end_address = compute_mux_l1_end_address(physical_l1_end_address, usable_l1_end_address);

    TT_FATAL(
        get_memory_map_end_address() <= l1_end_address,
        "Memory map end address: {} is greater than L1 end address: {} (physical L1 end: {})",
        get_memory_map_end_address(),
        l1_end_address,
        physical_l1_end_address);

    // The fabric endpoints the mux can connect to. Cached here so the compile-time args need no context lookup; each
    // getter picks its endpoint and fails there if it is unavailable.
    const auto& builder_context = fabric_context.get_builder_context();
    const auto& fabric_router_config = builder_context.get_fabric_router_config();
    fabric_endpoint_status_address_ = fabric_router_config.edm_status_address;
    if (auto* const static_channel_allocator = dynamic_cast<tt::tt_fabric::FabricStaticSizedChannelsAllocator*>(
            fabric_router_config.channel_allocator.get())) {
        fabric_router_channel_num_buffers_ = static_channel_allocator->get_sender_channel_number_of_slots(0);
    }
    connect_via_tensix_ = fabric_context.is_tensix_enabled();
    if (connect_via_tensix_ && builder_context.has_tensix_config()) {
        fabric_tensix_channel_num_buffers_ = builder_context.get_tensix_config().get_num_buffers_per_channel();
    }
}

std::vector<uint32_t> FabricMuxConfig::get_fabric_mux_compile_time_main_args(
    size_t fabric_endpoint_channel_num_buffers, bool wait_for_fabric_endpoint_ready) const {
    TT_FATAL(fabric_endpoint_channel_num_buffers > 0, "fabric_endpoint_channel_num_buffers must be larger than 0");
    TT_FATAL(fabric_endpoint_status_address_ != 0, "fabric_endpoint_status_address_ must not be invalid address 0");
    return std::vector<uint32_t>{
        num_full_size_channels_,
        num_buffers_full_size_channel_,
        static_cast<uint32_t>(buffer_size_bytes_full_size_channel_),
        num_header_only_channels_,
        num_buffers_header_only_channel_,
        static_cast<uint32_t>(memory_map_.status.get_address()),
        static_cast<uint32_t>(memory_map_.termination_signal.get_address()),
        static_cast<uint32_t>(memory_map_.connection_info.get_address()),
        static_cast<uint32_t>(memory_map_.connection_handshake.get_address()),
        static_cast<uint32_t>(memory_map_.flow_control.get_address()),
        static_cast<uint32_t>(memory_map_.full_size_channels.get_address()),
        static_cast<uint32_t>(memory_map_.local_fabric_router_status.get_address()),
        static_cast<uint32_t>(fabric_endpoint_status_address_),
        static_cast<uint32_t>(fabric_endpoint_channel_num_buffers),
        static_cast<uint32_t>(num_full_size_channel_iters_),
        static_cast<uint32_t>(num_iters_between_teardown_checks_),
        core_type_index_,
        (uint32_t)wait_for_fabric_endpoint_ready,
        static_cast<uint32_t>(in_program_local_l1_),
        program_local_l1_cb_index_,
        l1_base_semaphore_id_};
}

std::vector<uint32_t> FabricMuxConfig::get_fabric_mux_compile_time_args() const {
    if (connect_via_tensix_) {
        TT_FATAL(fabric_tensix_channel_num_buffers_ > 0, "Error, fabric tensix config is uninitialized");
    } else {
        TT_FATAL(
            fabric_router_channel_num_buffers_ > 0, "Channel allocator must be a FabricStaticSizedChannelsAllocator.");
    }
    auto ct_args = get_fabric_mux_compile_time_main_args(
        connect_via_tensix_ ? fabric_tensix_channel_num_buffers_ : fabric_router_channel_num_buffers_,
        /*wait_for_fabric_endpoint_ready=*/false);
    append_default_stream_ids_to_ct_args(ct_args);
    return ct_args;
}

std::vector<uint32_t> FabricMuxConfig::get_fabric_mux_compile_time_args_for_relay_mux() const {
    TT_FATAL(fabric_router_channel_num_buffers_ > 0, "Channel allocator must be a FabricStaticSizedChannelsAllocator.");
    // The relay mux waits for the fabric router to be ready before connecting.
    auto ct_args = get_fabric_mux_compile_time_main_args(
        fabric_router_channel_num_buffers_, /*wait_for_fabric_endpoint_ready=*/true);
    append_default_stream_ids_to_ct_args(ct_args);
    return ct_args;
}

template <typename ProgramOrDescriptor>
std::vector<uint32_t> FabricMuxConfig::get_fabric_mux_run_time_args(
    const tt::tt_metal::distributed::MeshDevice& mesh_device,
    const FabricNodeId& src_fabric_node_id,
    const FabricNodeId& dst_fabric_node_id,
    uint32_t link_idx,
    ProgramOrDescriptor& mux_program_or_desc,
    const tt::tt_metal::CoreCoord& mux_logical_core) const {
    return get_fabric_mux_run_time_args<ProgramOrDescriptor>(
        mesh_device.impl().metal_env().get_control_plane(),
        src_fabric_node_id,
        dst_fabric_node_id,
        link_idx,
        mux_program_or_desc,
        mux_logical_core);
}

template <typename ProgramOrDescriptor>
std::vector<uint32_t> FabricMuxConfig::get_fabric_mux_run_time_args(
    const ControlPlane& control_plane,
    const FabricNodeId& src_fabric_node_id,
    const FabricNodeId& dst_fabric_node_id,
    uint32_t link_idx,
    ProgramOrDescriptor& mux_program_or_desc,
    const tt::tt_metal::CoreCoord& mux_logical_core) const {
    std::vector<uint32_t> args;

    auto regions_to_clear = get_memory_regions_to_clear();
    const auto num_regions_to_clear = regions_to_clear.size();
    args.reserve((num_regions_to_clear * 2) + 1);
    args.push_back(static_cast<uint32_t>(num_regions_to_clear));
    for (const auto& [address, size] : regions_to_clear) {
        args.push_back(static_cast<uint32_t>(address));
        args.push_back(static_cast<uint32_t>(size));
    }

    tt::tt_fabric::append_fabric_connection_rt_args<ProgramOrDescriptor>(
        control_plane.get_fabric_context(),
        src_fabric_node_id,
        dst_fabric_node_id,
        link_idx,
        mux_program_or_desc,
        mux_logical_core,
        args,
        core_type_);

    return args;
}

template std::vector<uint32_t> FabricMuxConfig::get_fabric_mux_run_time_args<tt::tt_metal::Program>(
    const tt::tt_metal::distributed::MeshDevice&,
    const FabricNodeId&,
    const FabricNodeId&,
    uint32_t,
    tt::tt_metal::Program&,
    const tt::tt_metal::CoreCoord&) const;

template std::vector<uint32_t> FabricMuxConfig::get_fabric_mux_run_time_args<tt::tt_metal::ProgramDescriptor>(
    const tt::tt_metal::distributed::MeshDevice&,
    const FabricNodeId&,
    const FabricNodeId&,
    uint32_t,
    tt::tt_metal::ProgramDescriptor&,
    const tt::tt_metal::CoreCoord&) const;

template std::vector<uint32_t> FabricMuxConfig::get_fabric_mux_run_time_args<tt::tt_metal::Program>(
    const ControlPlane&,
    const FabricNodeId&,
    const FabricNodeId&,
    uint32_t,
    tt::tt_metal::Program&,
    const tt::tt_metal::CoreCoord&) const;

template std::vector<uint32_t> FabricMuxConfig::get_fabric_mux_run_time_args<tt::tt_metal::ProgramDescriptor>(
    const ControlPlane&,
    const FabricNodeId&,
    const FabricNodeId&,
    uint32_t,
    tt::tt_metal::ProgramDescriptor&,
    const tt::tt_metal::CoreCoord&) const;

uint8_t FabricMuxConfig::get_num_buffers(FabricMuxChannelType channel_type) const {
    return channel_type == FabricMuxChannelType::FULL_SIZE_CHANNEL ? num_buffers_full_size_channel_
                                                                   : num_buffers_header_only_channel_;
}

size_t FabricMuxConfig::get_buffer_size_bytes(FabricMuxChannelType channel_type) const {
    return channel_type == FabricMuxChannelType::FULL_SIZE_CHANNEL ? buffer_size_bytes_full_size_channel_
                                                                   : buffer_size_bytes_header_only_channel_;
}

size_t FabricMuxConfig::get_status_address() const { return memory_map_.status.get_address(); }

size_t FabricMuxConfig::get_termination_signal_address() const { return memory_map_.termination_signal.get_address(); }

size_t FabricMuxConfig::get_channel_credits_stream_id(FabricMuxChannelType channel_type, uint8_t channel_id) const {
    validate_channel_id(channel_type, channel_id);

    return get_channel_global_offset(channel_type, channel_id);
}

size_t FabricMuxConfig::get_channel_base_address(FabricMuxChannelType channel_type, uint8_t channel_id) const {
    validate_channel_id(channel_type, channel_id);

    return channel_type == FabricMuxChannelType::FULL_SIZE_CHANNEL
               ? memory_map_.full_size_channels.get_address(channel_id)
               : memory_map_.header_only_channels.get_address(channel_id);
}

size_t FabricMuxConfig::get_connection_info_address(FabricMuxChannelType channel_type, uint8_t channel_id) const {
    validate_channel_id(channel_type, channel_id);

    return memory_map_.connection_info.get_address(get_channel_global_offset(channel_type, channel_id));
}

size_t FabricMuxConfig::get_connection_handshake_address(FabricMuxChannelType channel_type, uint8_t channel_id) const {
    validate_channel_id(channel_type, channel_id);

    return memory_map_.connection_handshake.get_address(get_channel_global_offset(channel_type, channel_id));
}

size_t FabricMuxConfig::get_flow_control_address(FabricMuxChannelType channel_type, uint8_t channel_id) const {
    validate_channel_id(channel_type, channel_id);

    return memory_map_.flow_control.get_address(get_channel_global_offset(channel_type, channel_id));
}

size_t FabricMuxConfig::get_buffer_index_address(FabricMuxChannelType channel_type, uint8_t channel_id) const {
    validate_channel_id(channel_type, channel_id);

    return memory_map_.buffer_index.get_address(get_channel_global_offset(channel_type, channel_id));
}

void FabricMuxConfig::set_num_full_size_channel_iters(size_t new_val) {
    TT_FATAL(num_full_size_channels_ > 0, "Cannot set iterations when no full size channels exist");
    TT_FATAL(new_val > 0, "Number of iterations must be greater than 0");
    num_full_size_channel_iters_ = new_val;
}

size_t FabricMuxConfig::get_memory_map_end_address() const { return memory_map_.get_end_address(); }

size_t FabricMuxConfig::compute_memory_map_end_address(
    const tt::tt_metal::distributed::MeshDevice& mesh_device,
    uint8_t num_full_size_channels,
    uint8_t num_header_only_channels,
    uint8_t num_buffers_full_size_channel,
    uint8_t num_buffers_header_only_channel,
    size_t buffer_size_bytes_full_size_channel,
    size_t base_l1_address) {
    const auto& fabric_context = mesh_device.impl().metal_env().get_control_plane().get_fabric_context();
    const MemoryMap memory_map = lay_out_memory_map(
        base_l1_address,
        fabric_context.get_hal().get_alignment(tt::tt_metal::HalMemType::L1),
        num_full_size_channels,
        num_header_only_channels,
        num_buffers_or_default(num_buffers_full_size_channel),
        num_buffers_or_default(num_buffers_header_only_channel),
        buffer_size_bytes_full_size_channel,
        fabric_context.get_fabric_packet_header_size_bytes());
    return memory_map.get_end_address();
}

size_t FabricMuxConfig::compute_l1_end_address(
    const tt::tt_metal::distributed::MeshDevice& mesh_device, CoreType core_type, size_t usable_l1_end_address) {
    const auto& hal = mesh_device.impl().metal_env().get_control_plane().get_fabric_context().get_hal();
    return compute_mux_l1_end_address(
        compute_physical_l1_end_address(hal, get_mux_hal_core_type(core_type)), usable_l1_end_address);
}

FabricMuxConfig::MemoryMap FabricMuxConfig::lay_out_memory_map(
    size_t base_l1_address,
    size_t noc_aligned_address_size_bytes,
    uint8_t num_full_size_channels,
    uint8_t num_header_only_channels,
    uint8_t num_buffers_full_size_channel,
    uint8_t num_buffers_header_only_channel,
    size_t buffer_size_bytes_full_size_channel,
    size_t buffer_size_bytes_header_only_channel) {
    const size_t num_total_channels = num_full_size_channels + num_header_only_channels;
    MemoryMap memory_map;
    // Each region starts where the one before it ends
    size_t current_address = base_l1_address;
    const auto next_region = [&](size_t unit_size, size_t num_units) {
        const MemoryRegion region(current_address, unit_size, num_units);
        current_address = region.get_end_address();
        return region;
    };

    // Single-address regions
    memory_map.status = next_region(noc_aligned_address_size_bytes, 1);
    memory_map.local_fabric_router_status = next_region(noc_aligned_address_size_bytes, 1);
    memory_map.termination_signal = next_region(noc_aligned_address_size_bytes, 1);
    // One entry per channel
    memory_map.connection_info = next_region(sizeof(tt::tt_fabric::EDMChannelWorkerLocationInfo), num_total_channels);
    memory_map.connection_handshake = next_region(noc_aligned_address_size_bytes, num_total_channels);
    memory_map.flow_control = next_region(noc_aligned_address_size_bytes, num_total_channels);
    memory_map.buffer_index = next_region(noc_aligned_address_size_bytes, num_total_channels);
    // The channels' buffers
    memory_map.full_size_channels =
        next_region(num_buffers_full_size_channel * buffer_size_bytes_full_size_channel, num_full_size_channels);
    memory_map.header_only_channels =
        next_region(num_buffers_header_only_channel * buffer_size_bytes_header_only_channel, num_header_only_channels);
    return memory_map;
}

uint8_t FabricMuxConfig::num_buffers_or_default(uint8_t num_buffers) {
    return num_buffers == 0 ? default_num_buffers : num_buffers;
}

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

uint32_t FabricMuxConfig::validate_program_local_l1_placement() const {
    TT_FATAL(!in_program_local_l1_, "Fabric mux config is already placed in program-local L1");
    TT_FATAL(
        core_type_ == CoreType::WORKER,
        "Only a worker-core fabric mux can be placed in program-local L1, not a {} mux",
        enchantum::to_string(core_type_));
    TT_FATAL(
        memory_map_.status.base_address == 0,
        "A fabric mux placed in program-local L1 must be configured with base_l1_address = 0 so its addresses are "
        "offsets into the map, but base_l1_address is {}",
        memory_map_.status.base_address);
    return static_cast<uint32_t>(tt::align(get_memory_map_end_address(), noc_aligned_address_size_bytes_));
}

void FabricMuxConfig::place_in_program_local_l1(
    tt::tt_metal::ProgramDescriptor& desc, const tt::tt_metal::CoreRangeSet& mux_cores) {
    const uint32_t map_size = validate_program_local_l1_placement();
    const auto placement = detail::reserve_fabric_mux_program_local_l1(desc, mux_cores, map_size);
    in_program_local_l1_ = true;
    program_local_l1_cb_index_ = placement.cb_index;
    l1_base_semaphore_id_ = placement.l1_base_semaphore_id;
}

void FabricMuxConfig::place_in_program_local_l1(
    tt::tt_metal::Program& program, const tt::tt_metal::CoreRangeSet& mux_cores) {
    const uint32_t map_size = validate_program_local_l1_placement();
    const auto placement = detail::reserve_fabric_mux_program_local_l1(program, mux_cores, map_size);
    in_program_local_l1_ = true;
    program_local_l1_cb_index_ = placement.cb_index;
    l1_base_semaphore_id_ = placement.l1_base_semaphore_id;
}

bool FabricMuxConfig::is_in_program_local_l1() const { return in_program_local_l1_; }

uint32_t FabricMuxConfig::get_l1_base_semaphore_id() const {
    TT_FATAL(in_program_local_l1_, "Fabric mux config is not placed in program-local L1, so it has no base semaphore");
    return l1_base_semaphore_id_;
}

std::vector<std::pair<size_t, size_t>> FabricMuxConfig::get_memory_regions_to_clear() const {
    return {
        {memory_map_.termination_signal.get_address(), memory_map_.termination_signal.get_total_size()},
        {memory_map_.connection_handshake.get_address(), memory_map_.connection_handshake.get_total_size()},
        {memory_map_.flow_control.get_address(), memory_map_.flow_control.get_total_size()},
        {memory_map_.buffer_index.get_address(), memory_map_.buffer_index.get_total_size()}};
}

void FabricMuxConfig::append_default_stream_ids_to_ct_args(std::vector<uint32_t>& ct_args) const {
    // Add stream IDs for all channels (full size + header only)
    // Full size channels first
    for (uint8_t i = 0; i < num_full_size_channels_; i++) {
        ct_args.push_back(get_channel_credits_stream_id(FabricMuxChannelType::FULL_SIZE_CHANNEL, i));
    }
    // Header only channels second
    for (uint8_t i = 0; i < num_header_only_channels_; i++) {
        ct_args.push_back(get_channel_credits_stream_id(FabricMuxChannelType::HEADER_ONLY_CHANNEL, i));
    }
}

void FabricMuxConfig::append_default_persistent_channel_flags_to_ct_args(std::vector<uint32_t>& ct_args) const {
    // Add persistent channel flags - all false by default
    for (uint8_t i = 0; i < num_full_size_channels_; i++) {
        ct_args.push_back(0);
    }
    // Header only channels second
    for (uint8_t i = 0; i < num_header_only_channels_; i++) {
        ct_args.push_back(0);
    }
}

}  // namespace tt::tt_fabric
