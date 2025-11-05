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
#include <tt-metalium/device_pool.hpp>
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/fabric_host_utils.hpp"
#include "dispatch/kernel_config/relay_mux.hpp"
#include "tt_metal/fabric/builder/fabric_static_sized_channels_allocator.hpp"
#include "tt_align.hpp"
#include <bit>
#include <algorithm>
#include <utility>

namespace tt::tt_fabric {

// Helper function
namespace {
bool device_has_dispatch_tunnel(ChipId device_id) {
    auto mmio_device_id = tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(device_id);
    auto tunnels_from_mmio =
        tt::tt_metal::MetalContext::instance().get_cluster().get_devices_controlled_by_mmio_device(mmio_device_id);
    // results are inclusive of the mmio_device_id so they will never be zero
    TT_FATAL(!tunnels_from_mmio.empty(), "must have at least one mmio device");
    return (tunnels_from_mmio.size() - 1) > 0;
}
}  // namespace

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
    uint8_t num_full_size_channels,
    uint8_t num_header_only_channels,
    uint8_t num_buffers_full_size_channel,
    uint8_t num_buffers_header_only_channel,
    size_t buffer_size_bytes_full_size_channel,
    size_t base_l1_address,
    CoreType core_type) :
    core_type_(core_type),
    num_full_size_channels_(num_full_size_channels),
    num_header_only_channels_(num_header_only_channels),
    num_buffers_full_size_channel_(
        num_buffers_full_size_channel == 0 ? default_num_buffers : num_buffers_full_size_channel),
    num_buffers_header_only_channel_(
        num_buffers_header_only_channel == 0 ? default_num_buffers : num_buffers_header_only_channel),
    buffer_size_bytes_full_size_channel_(buffer_size_bytes_full_size_channel) {
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

    status_region_ = MemoryRegion(current_address, noc_aligned_address_size_bytes_, 1);
    current_address = status_region_.get_end_address();

    local_fabric_router_status_region_ = MemoryRegion(current_address, noc_aligned_address_size_bytes_, 1);
    current_address = local_fabric_router_status_region_.get_end_address();

    termination_signal_region_ = MemoryRegion(current_address, noc_aligned_address_size_bytes_, 1);
    current_address = termination_signal_region_.get_end_address();

    connection_info_region_ =
        MemoryRegion(current_address, sizeof(tt::tt_fabric::EDMChannelWorkerLocationInfo), num_total_channels);
    current_address = connection_info_region_.get_end_address();

    connection_handshake_region_ = MemoryRegion(current_address, noc_aligned_address_size_bytes_, num_total_channels);
    current_address = connection_handshake_region_.get_end_address();

    flow_control_region_ = MemoryRegion(current_address, noc_aligned_address_size_bytes_, num_total_channels);
    current_address = flow_control_region_.get_end_address();

    buffer_index_region_ = MemoryRegion(current_address, noc_aligned_address_size_bytes_, num_total_channels);
    current_address = buffer_index_region_.get_end_address();

    full_size_channels_region_ = MemoryRegion(current_address, full_size_channel_size_bytes_, num_full_size_channels_);
    current_address = full_size_channels_region_.get_end_address();

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
        TT_THROW("Fabric Tensix Datamover does not support core type {}", static_cast<int>(core_type));
    }

    core_type_index_ = hal.get_programmable_core_type_index(hal_core_type);
    auto l1_end_address = hal.get_dev_addr(hal_core_type, tt_metal::HalL1MemAddrType::BASE) +
                          hal.get_dev_size(hal_core_type, tt_metal::HalL1MemAddrType::BASE);

    TT_FATAL(
        memory_map_end_address_ <= l1_end_address,
        "Memory map end address: {} is greater than L1 end address: {}",
        memory_map_end_address_,
        l1_end_address);
}

// Getters
uint8_t FabricTensixDatamoverBaseConfig::get_num_channels(FabricMuxChannelType channel_type) const {
    return channel_type == FabricMuxChannelType::FULL_SIZE_CHANNEL ? num_full_size_channels_
                                                                   : num_header_only_channels_;
}

uint8_t FabricTensixDatamoverBaseConfig::get_num_buffers(FabricMuxChannelType channel_type) const {
    return channel_type == FabricMuxChannelType::FULL_SIZE_CHANNEL ? num_buffers_full_size_channel_
                                                                   : num_buffers_header_only_channel_;
}

size_t FabricTensixDatamoverBaseConfig::get_buffer_size_bytes(FabricMuxChannelType channel_type) const {
    return channel_type == FabricMuxChannelType::FULL_SIZE_CHANNEL ? buffer_size_bytes_full_size_channel_
                                                                   : buffer_size_bytes_header_only_channel_;
}

size_t FabricTensixDatamoverBaseConfig::get_status_address() const { return status_region_.get_address(); }

size_t FabricTensixDatamoverBaseConfig::get_termination_signal_address() const {
    return termination_signal_region_.get_address();
}

size_t FabricTensixDatamoverBaseConfig::get_channel_credits_stream_id(
    FabricMuxChannelType channel_type, uint8_t channel_id) const {
    validate_channel_id(channel_type, channel_id);
    return get_channel_global_offset(channel_type, channel_id);
}

size_t FabricTensixDatamoverBaseConfig::get_channel_base_address(
    FabricMuxChannelType channel_type, uint8_t channel_id) const {
    validate_channel_id(channel_type, channel_id);
    return channel_type == FabricMuxChannelType::FULL_SIZE_CHANNEL
               ? full_size_channels_region_.get_address(channel_id)
               : header_only_channels_region_.get_address(channel_id);
}

size_t FabricTensixDatamoverBaseConfig::get_connection_info_address(
    FabricMuxChannelType channel_type, uint8_t channel_id) const {
    validate_channel_id(channel_type, channel_id);
    return connection_info_region_.get_address(get_channel_global_offset(channel_type, channel_id));
}

size_t FabricTensixDatamoverBaseConfig::get_connection_handshake_address(
    FabricMuxChannelType channel_type, uint8_t channel_id) const {
    validate_channel_id(channel_type, channel_id);
    return connection_handshake_region_.get_address(get_channel_global_offset(channel_type, channel_id));
}

size_t FabricTensixDatamoverBaseConfig::get_flow_control_address(
    FabricMuxChannelType channel_type, uint8_t channel_id) const {
    validate_channel_id(channel_type, channel_id);
    return flow_control_region_.get_address(get_channel_global_offset(channel_type, channel_id));
}

size_t FabricTensixDatamoverBaseConfig::get_buffer_index_address(
    FabricMuxChannelType channel_type, uint8_t channel_id) const {
    validate_channel_id(channel_type, channel_id);
    return buffer_index_region_.get_address(get_channel_global_offset(channel_type, channel_id));
}

size_t FabricTensixDatamoverBaseConfig::get_memory_map_end_address() const { return memory_map_end_address_; }

// Setters
void FabricTensixDatamoverBaseConfig::set_num_full_size_channel_iters(size_t new_val) {
    TT_FATAL(num_full_size_channels_ > 0, "Cannot set iterations when no full size channels exist");
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
    return {
        {termination_signal_region_.get_address(), termination_signal_region_.get_total_size()},
        {connection_handshake_region_.get_address(), connection_handshake_region_.get_total_size()},
        {flow_control_region_.get_address(), flow_control_region_.get_total_size()},
        {buffer_index_region_.get_address(), buffer_index_region_.get_total_size()}};
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
        src_fabric_node_id, dst_fabric_node_id, link_idx, program, logical_core, args, core_type_);

    return args;
}

// Helper methods
void FabricTensixDatamoverBaseConfig::validate_channel_id(FabricMuxChannelType channel_type, uint8_t channel_id) const {
    TT_FATAL(
        channel_id < get_num_channels(channel_type),
        "Invalid channel id for channel type. Requested channel id: {} but maximum is {}",
        channel_id,
        get_num_channels(channel_type));
}

uint8_t FabricTensixDatamoverBaseConfig::get_channel_global_offset(
    FabricMuxChannelType channel_type, uint8_t channel_id) const {
    return (channel_type == FabricMuxChannelType::HEADER_ONLY_CHANNEL) ? num_full_size_channels_ + channel_id
                                                                       : channel_id;
}

void FabricTensixDatamoverBaseConfig::append_default_stream_ids_to_ct_args(std::vector<uint32_t>& ct_args) const {
    for (uint8_t i = 0; i < num_full_size_channels_; i++) {
        ct_args.push_back(get_channel_credits_stream_id(FabricMuxChannelType::FULL_SIZE_CHANNEL, i));
    }
    for (uint8_t i = 0; i < num_header_only_channels_; i++) {
        ct_args.push_back(get_channel_credits_stream_id(FabricMuxChannelType::HEADER_ONLY_CHANNEL, i));
    }
}

std::vector<uint32_t> FabricTensixDatamoverBaseConfig::get_compile_time_main_args(
    const tt::tt_fabric::FabricEriscDatamoverConfig& fabric_router_config) const {
    TT_FATAL(fabric_endpoint_channel_num_buffers_ > 0, "fabric_endpoint_channel_num_buffers_ must be larger than 0");
    TT_FATAL(fabric_endpoint_status_address_ != 0, "fabric_endpoint_status_address_ must not be invalid address 0");
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
        fabric_endpoint_status_address_,
        fabric_endpoint_channel_num_buffers_,
        num_full_size_channel_iters_,
        num_iters_between_teardown_checks_,
        core_type_index_,
        (uint32_t)wait_for_fabric_endpoint_ready_};
}

// ==================================================================================================
// FabricTensixDatamoverMuxConfig Implementation
// ==================================================================================================

FabricTensixDatamoverMuxConfig::FabricTensixDatamoverMuxConfig(
    uint8_t num_full_size_channels,
    uint8_t num_header_only_channels,
    uint8_t num_buffers_full_size_channel,
    uint8_t num_buffers_header_only_channel,
    size_t buffer_size_bytes_full_size_channel,
    size_t base_l1_address,
    CoreType core_type) :
    FabricTensixDatamoverBaseConfig(
        num_full_size_channels,
        num_header_only_channels,
        num_buffers_full_size_channel,
        num_buffers_header_only_channel,
        buffer_size_bytes_full_size_channel,
        base_l1_address,
        core_type) {}

std::vector<uint32_t> FabricTensixDatamoverMuxConfig::get_compile_time_args(
    const tt::tt_fabric::FabricEriscDatamoverConfig& fabric_router_config) const {
    auto channel_allocator = fabric_router_config.channel_allocator.get();
    const auto static_channel_allocator =
        dynamic_cast<tt::tt_fabric::FabricStaticSizedChannelsAllocator*>(channel_allocator);
    TT_FATAL(static_channel_allocator != nullptr, "Channel allocator must be a FabricStaticSizedChannelsAllocator.");

    fabric_endpoint_channel_num_buffers_ = static_channel_allocator->get_sender_channel_number_of_slots(0);
    fabric_endpoint_status_address_ = fabric_router_config.edm_status_address;

    auto ct_args = get_compile_time_main_args(fabric_router_config);
    return ct_args;
}

// ==================================================================================================
// FabricTensixDatamoverRelayConfig Implementation
// ==================================================================================================

FabricTensixDatamoverRelayConfig::FabricTensixDatamoverRelayConfig(
    uint8_t num_buffers_per_channel,
    size_t buffer_size_bytes,
    size_t base_l1_address,
    CoreType core_type) :
    FabricTensixDatamoverBaseConfig(
        1,                        // num_full_size_channels - relay uses only 1 channel
        0,                        // num_header_only_channels - no header-only channels for relay
        num_buffers_per_channel,  // num_buffers_full_size_channel
        0,                        // num_buffers_header_only_channel
        buffer_size_bytes,        // buffer_size_bytes_full_size_channel
        base_l1_address,
        core_type) {}

std::vector<uint32_t> FabricTensixDatamoverRelayConfig::get_compile_time_args(
    const tt::tt_fabric::FabricEriscDatamoverConfig& fabric_router_config) const {
    std::vector<uint32_t> ct_args;
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
    eth_chan_directions direction) :
    my_core_logical_(my_core_logical),
    local_fabric_node_id_(local_fabric_node_id),
    remote_fabric_node_id_(remote_fabric_node_id),
    ethernet_channel_id_(ethernet_channel_id),
    link_idx_(link_idx),
    core_id_(core_id),
    noc_x_(noc_x),
    noc_y_(noc_y),
    config_(config),
    direction_(direction) {
    channel_connection_liveness_check_disable_array_.fill(false);
    TT_FATAL(config_ != nullptr, "Config cannot be null");
}

const char* FabricTensixDatamoverMuxBuilder::get_kernel_file_path() const {
    return "tt_metal/fabric/impl/kernels/edm_fabric/fabric_router_mux_extension.cpp";
}

tt::tt_fabric::SenderWorkerAdapterSpec FabricTensixDatamoverMuxBuilder::build_connection_to_fabric_channel(
    uint32_t channel_id) const {
    auto channel_type = tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL;

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

void FabricTensixDatamoverMuxBuilder::create_and_compile(
    tt::tt_metal::IDevice* device, tt::tt_metal::Program& program) {
    // Select processor and NOC based on core type
    tt::tt_metal::DataMovementProcessor processor = (core_id_ == FabricTensixCoreType::MUX)
                                                        ? tt::tt_metal::DataMovementProcessor::RISCV_0
                                                        : tt::tt_metal::DataMovementProcessor::RISCV_1;

    tt::tt_metal::NOC noc = (core_id_ == FabricTensixCoreType::MUX) ? tt::tt_metal::NOC::RISCV_0_default
                                                                    : tt::tt_metal::NOC::RISCV_1_default;

    // Create the mux kernel
    auto mux_kernel = tt::tt_metal::CreateKernel(
        program,
        get_kernel_file_path(),
        my_core_logical_,
        tt::tt_metal::DataMovementConfig{
            .processor = processor, .noc = noc, .compile_args = get_compile_time_args(device), .defines = {}});

    // Set runtime arguments
    tt::tt_metal::SetRuntimeArgs(program, mux_kernel, my_core_logical_, get_runtime_args(program));
}

std::vector<uint32_t> FabricTensixDatamoverMuxBuilder::get_compile_time_args(tt::tt_metal::IDevice* device) const {
    const auto& fabric_context = tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_context();
    const auto& fabric_tensix_config = tt::tt_metal::MetalContext::instance().get_fabric_tensix_config();

    const bool has_dispatch_tunnel = device_has_dispatch_tunnel(device->id());
    uint32_t dispatch_link_idx =
        tt_metal::RelayMux::get_dispatch_link_index(local_fabric_node_id_, remote_fabric_node_id_, device);
    bool is_dispatch_link = has_dispatch_tunnel && link_idx_ == dispatch_link_idx;

    // use normal router config for dispatch link, since it doesn't have tensix extension
    const auto& fabric_router_config = [&]() {
        if (is_dispatch_link) {
            return fabric_context.get_fabric_router_config();
        } else {
            return fabric_context.get_fabric_router_config(
                tt::tt_fabric::FabricEriscDatamoverType::Default,
                tt::tt_fabric::FabricEriscDatamoverAxis::Short,
                fabric_tensix_config);
        }
    }();

    auto channel_allocator_base = fabric_router_config.channel_allocator.get();
    const auto channel_allocator =
        dynamic_cast<tt::tt_fabric::FabricStaticSizedChannelsAllocator*>(channel_allocator_base);
    TT_FATAL(channel_allocator != nullptr, "Channel allocator must be a FabricStaticSizedChannelsAllocator.");

    config_->set_fabric_endpoint_channel_num_buffers(channel_allocator->get_sender_channel_number_of_slots(0));
    config_->set_wait_for_fabric_endpoint_ready(true);
    config_->set_fabric_endpoint_status_address(fabric_router_config.edm_status_address);

    auto ct_args = config_->get_compile_time_args(fabric_router_config);

    // Add number of upstream routers and sync address
    ct_args.push_back(static_cast<uint32_t>(upstream_routers_noc_x_.size()));
    ct_args.push_back(fabric_router_config.edm_local_tensix_sync_address);

    // Get stream IDs based on mode
    const auto topology = fabric_context.get_fabric_topology();
    const bool is_2d_fabric = fabric_context.is_2D_routing_enabled();
    const auto worker_channel = is_2d_fabric ? direction_ : 0;
    const auto& tensix_config = fabric_context.get_tensix_config();
    const auto worker_stream_id =
        tensix_config.get_channel_credits_stream_id(device->id(), ethernet_channel_id_, worker_channel, core_id_);

    std::vector<uint32_t> fabric_stream_ids;
    if (fabric_tensix_config == tt::tt_fabric::FabricTensixConfig::UDM) {
        // UDM mode: MUX only has 1 worker channel
        fabric_stream_ids = {worker_stream_id};
    } else {
        // MUX mode: topology-based channels (includes fabric routers)
        switch (topology) {
            case tt::tt_fabric::Topology::Linear:
            case tt::tt_fabric::Topology::Ring:
                fabric_stream_ids = {
                    worker_stream_id, tt::tt_fabric::StreamRegAssignments::sender_channel_1_free_slots_stream_id};
                break;
            case tt::tt_fabric::Topology::Mesh:
            case tt::tt_fabric::Topology::Torus:
                fabric_stream_ids = {
                    tt::tt_fabric::StreamRegAssignments::sender_channel_1_free_slots_stream_id,
                    tt::tt_fabric::StreamRegAssignments::sender_channel_2_free_slots_stream_id,
                    tt::tt_fabric::StreamRegAssignments::sender_channel_3_free_slots_stream_id,
                    tt::tt_fabric::StreamRegAssignments::sender_channel_4_free_slots_stream_id};
                break;
            default: TT_THROW("Unknown fabric topology: {}", static_cast<int>(topology)); break;
        }
        fabric_stream_ids[worker_channel] = worker_stream_id;
    }

    uint8_t num_full_size_channels = config_->get_num_channels(tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL);
    TT_FATAL(
        num_full_size_channels == fabric_stream_ids.size(),
        "the number of fabric stream ids used must equal to the number of mux channels");
    ct_args.insert(ct_args.end(), fabric_stream_ids.begin(), fabric_stream_ids.end());

    // Add persistent channels flags
    std::vector<uint32_t> is_persistent_channels(num_full_size_channels, 0);
    for (uint8_t i = 0; i < num_full_size_channels; i++) {
        if (channel_connection_liveness_check_disable_array_[i]) {
            is_persistent_channels[i] = 1;
        }
    }
    ct_args.insert(ct_args.end(), is_persistent_channels.begin(), is_persistent_channels.end());

    return ct_args;
}

std::vector<uint32_t> FabricTensixDatamoverMuxBuilder::get_runtime_args(tt::tt_metal::Program& program) const {
    std::vector<uint32_t> runtime_args;
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
    eth_chan_directions direction) :
    my_core_logical_(my_core_logical),
    local_fabric_node_id_(local_fabric_node_id),
    remote_fabric_node_id_(remote_fabric_node_id),
    ethernet_channel_id_(ethernet_channel_id),
    link_idx_(link_idx),
    core_id_(core_id),
    noc_x_(noc_x),
    noc_y_(noc_y),
    config_(config),
    direction_(direction) {
    channel_connection_liveness_check_disable_array_.fill(false);
    TT_FATAL(config_ != nullptr, "Config cannot be null");
}

tt::tt_fabric::SenderWorkerAdapterSpec FabricTensixDatamoverRelayBuilder::build_connection_to_fabric_channel(
    uint32_t channel_id) const {
    auto channel_type = tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL;

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

void FabricTensixDatamoverRelayBuilder::create_and_compile(
    tt::tt_metal::IDevice* device, tt::tt_metal::Program& program) {
    // Select processor and NOC based on core type
    tt::tt_metal::DataMovementProcessor processor = (core_id_ == FabricTensixCoreType::MUX)
                                                        ? tt::tt_metal::DataMovementProcessor::RISCV_0
                                                        : tt::tt_metal::DataMovementProcessor::RISCV_1;

    tt::tt_metal::NOC noc = (core_id_ == FabricTensixCoreType::MUX) ? tt::tt_metal::NOC::RISCV_0_default
                                                                    : tt::tt_metal::NOC::RISCV_1_default;

    // Create the relay kernel
    auto relay_kernel = tt::tt_metal::CreateKernel(
        program,
        "tt_metal/fabric/impl/kernels/edm_fabric/fabric_router_relay_extension.cpp",
        my_core_logical_,
        tt::tt_metal::DataMovementConfig{
            .processor = processor, .noc = noc, .compile_args = get_compile_time_args(device), .defines = {}});

    // Set runtime arguments
    tt::tt_metal::SetRuntimeArgs(program, relay_kernel, my_core_logical_, get_runtime_args(program));
}

std::vector<uint32_t> FabricTensixDatamoverRelayBuilder::get_compile_time_args(tt::tt_metal::IDevice* device) const {
    std::vector<uint32_t> ct_args;
    return ct_args;
}

std::vector<uint32_t> FabricTensixDatamoverRelayBuilder::get_runtime_args(tt::tt_metal::Program& program) const {
    std::vector<uint32_t> runtime_args;
    return runtime_args;
}

}  // namespace tt::tt_fabric
