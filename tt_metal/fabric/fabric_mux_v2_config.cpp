// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <hostdevcommon/fabric_mux_v2_common.h>

#include "impl/context/metal_context.hpp"

namespace tt::tt_fabric {
namespace {

constexpr uint32_t kTensixWorkerStreamRegisterCount = 64;
// Internal forwarder scheduling knob; not exposed on FabricMuxV2Config.
constexpr uint32_t kForwarderServiceBurstSize = 8;
constexpr const char* kFabricMuxV2KernelPath = "tt_metal/fabric/impl/kernels/tt_fabric_mux_v2.cpp";

size_t align_up(size_t value, size_t alignment) {
    TT_FATAL(alignment > 0, "Alignment must be greater than zero");
    const size_t remainder = value % alignment;
    return remainder == 0 ? value : (value + (alignment - remainder));
}

uint32_t to_uint32_checked(size_t value, const char* field_name) {
    TT_FATAL(
        value <= std::numeric_limits<uint32_t>::max(), "Value for {} exceeds uint32_t range: {}", field_name, value);
    return static_cast<uint32_t>(value);
}

bool is_supported_mux_v2_worker_noc(tt::tt_metal::NOC noc) {
    return noc == tt::tt_metal::NOC::RISCV_0_default || noc == tt::tt_metal::NOC::RISCV_1_default;
}

tt::tt_metal::NOC get_manager_noc_from_forwarder_noc(tt::tt_metal::NOC forwarder_noc) {
    TT_FATAL(
        is_supported_mux_v2_worker_noc(forwarder_noc),
        "FabricMuxV2 forwarder_noc must be RISCV_0_default or RISCV_1_default, got {}",
        static_cast<uint32_t>(forwarder_noc));
    return forwarder_noc == tt::tt_metal::NOC::RISCV_0_default ? tt::tt_metal::NOC::RISCV_1_default
                                                               : tt::tt_metal::NOC::RISCV_0_default;
}

struct FabricMuxV2ClientConnectionRtArgs {
    uint32_t mux_x = 0;
    uint32_t mux_y = 0;
    uint32_t logical_channel_id = 0;
    uint32_t num_buffers_per_channel = 0;
    uint32_t channel_buffer_size_bytes = 0;
    uint32_t channel_base_address = 0;
    uint32_t connection_info_address = 0;
    uint32_t connection_handshake_address = 0;
    uint32_t flow_control_sem_id = 0;
    uint32_t teardown_sem_id = 0;
    uint32_t mux_status_address = 0;

    static constexpr size_t kWordCount = 11;

    std::array<uint32_t, kWordCount> serialize() const {
        return {
            mux_x,
            mux_y,
            logical_channel_id,
            num_buffers_per_channel,
            channel_buffer_size_bytes,
            channel_base_address,
            connection_info_address,
            connection_handshake_address,
            flow_control_sem_id,
            teardown_sem_id,
            mux_status_address,
        };
    }
};

static_assert(
    sizeof(FabricMuxV2ClientConnectionRtArgs) == FabricMuxV2ClientConnectionRtArgs::kWordCount * sizeof(uint32_t));

void validate_forwarder_service_burst_size(uint32_t service_burst_size) {
    TT_FATAL(service_burst_size > 0, "FabricMuxV2 forwarder service burst size must be greater than zero");
}

}  // namespace

FabricMuxV2Config::MemoryRegion::MemoryRegion(size_t base, size_t unit_sz, size_t count) :
    base_address(base), unit_size(unit_sz), num_units(count) {}

size_t FabricMuxV2Config::MemoryRegion::get_address(size_t offset) const {
    if (num_units == 0) {
        TT_FATAL(offset == 0, "Offset {} is invalid for empty region", offset);
        return base_address;
    }

    TT_FATAL(offset < num_units, "Offset {} exceeds region size {}", offset, num_units);
    return base_address + (offset * unit_size);
}

size_t FabricMuxV2Config::MemoryRegion::get_end_address() const { return base_address + (unit_size * num_units); }

FabricMuxV2Config::FabricMuxV2Config(
    uint8_t num_channels, uint8_t num_buffers_per_channel, size_t channel_buffer_size_bytes, size_t base_l1_address) :
    num_channels_(num_channels),
    num_buffers_per_channel_(num_buffers_per_channel),
    channel_buffer_size_bytes_(channel_buffer_size_bytes),
    forwarder_service_burst_size_(kForwarderServiceBurstSize),
    trid_ring_capacity_(kTridRingCapacity) {
    TT_FATAL(num_channels_ > 0, "FabricMuxV2Config requires at least one logical channel");
    TT_FATAL(num_buffers_per_channel_ > 0, "FabricMuxV2Config requires at least one buffer per channel");
    validate_forwarder_service_burst_size(forwarder_service_burst_size_);

    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    noc_aligned_address_size_bytes_ = hal.get_alignment(tt::tt_metal::HalMemType::L1);
    per_channel_scalar_region_stride_bytes_ = noc_aligned_address_size_bytes_;

    const size_t max_channel_buffer_size_bytes = get_tt_fabric_channel_buffer_size_bytes();
    TT_FATAL(
        channel_buffer_size_bytes_ <= max_channel_buffer_size_bytes,
        "FabricMuxV2 channel buffer size must be <= {}, got {}",
        max_channel_buffer_size_bytes,
        channel_buffer_size_bytes_);
    TT_FATAL(
        channel_buffer_size_bytes_ % noc_aligned_address_size_bytes_ == 0,
        "FabricMuxV2 channel buffer size must be L1-aligned ({}), got {}",
        noc_aligned_address_size_bytes_,
        channel_buffer_size_bytes_);

    // One stream register per logical channel for forwarder-visible backlog tracking.
    const uint32_t total_reserved_stream_ids = static_cast<uint32_t>(num_channels_);
    TT_FATAL(
        total_reserved_stream_ids <= kTensixWorkerStreamRegisterCount,
        "FabricMuxV2 requires num_channels ({}) <= {}",
        static_cast<uint32_t>(num_channels_),
        kTensixWorkerStreamRegisterCount);

    size_t current_address = align_up(base_l1_address, noc_aligned_address_size_bytes_);

    status_region_ = MemoryRegion(current_address, noc_aligned_address_size_bytes_, 1);
    current_address = status_region_.get_end_address();

    current_address = align_up(current_address, noc_aligned_address_size_bytes_);
    connection_info_region_ =
        MemoryRegion(current_address, sizeof(tt::tt_fabric::EDMChannelWorkerLocationInfo), num_channels_);
    current_address = connection_info_region_.get_end_address();

    current_address = align_up(current_address, noc_aligned_address_size_bytes_);
    connection_handshake_region_ =
        MemoryRegion(current_address, per_channel_scalar_region_stride_bytes_, num_channels_);
    current_address = connection_handshake_region_.get_end_address();

    current_address = align_up(current_address, noc_aligned_address_size_bytes_);
    const size_t shared_ring_region_size_bytes =
        sizeof(FabricMuxV2SharedTridRingHeader) + (trid_ring_capacity_ * sizeof(FabricMuxV2SharedTridRingEntry));
    shared_ring_region_ = MemoryRegion(current_address, shared_ring_region_size_bytes, 1);
    current_address = shared_ring_region_.get_end_address();

    current_address = align_up(current_address, noc_aligned_address_size_bytes_);
    const size_t channel_stride_bytes = static_cast<size_t>(num_buffers_per_channel_) * channel_buffer_size_bytes_;
    channel_region_ = MemoryRegion(current_address, channel_stride_bytes, num_channels_);
    current_address = channel_region_.get_end_address();

    current_address = align_up(current_address, noc_aligned_address_size_bytes_);
    shared_control_region_ = MemoryRegion(current_address, sizeof(FabricMuxV2SharedControlBlock), 1);
    current_address = shared_control_region_.get_end_address();

    // Per-channel L1 scratch for BH spoofed posted credit notifies with flush disabled.
    // Unused on Wormhole (flush is a no-op there) but allocated on both for a uniform map.
    current_address = align_up(current_address, noc_aligned_address_size_bytes_);
    credit_notify_scratch_region_ =
        MemoryRegion(current_address, per_channel_scalar_region_stride_bytes_, num_channels_);
    current_address = credit_notify_scratch_region_.get_end_address();

    memory_map_end_address_ = current_address;

    const size_t l1_end_address =
        hal.get_dev_addr(tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::HalL1MemAddrType::BASE) +
        hal.get_dev_size(tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::HalL1MemAddrType::BASE);
    TT_FATAL(
        memory_map_end_address_ <= l1_end_address,
        "FabricMuxV2 memory map end address {} exceeds worker L1 end address {}",
        memory_map_end_address_,
        l1_end_address);
}

void FabricMuxV2Config::append_client_connection_rt_args(
    const tt::tt_metal::CoreCoord& mux_virtual_core,
    uint8_t logical_channel_id,
    const ClientSemaphores& client_semaphores,
    std::vector<uint32_t>& worker_args) const {
    validate_logical_channel_id(logical_channel_id);

    FabricMuxV2ClientConnectionRtArgs connection_rt_args{};
    connection_rt_args.mux_x = static_cast<uint32_t>(mux_virtual_core.x);
    connection_rt_args.mux_y = static_cast<uint32_t>(mux_virtual_core.y);
    connection_rt_args.logical_channel_id = logical_channel_id;
    connection_rt_args.num_buffers_per_channel = static_cast<uint32_t>(num_buffers_per_channel_);
    connection_rt_args.channel_buffer_size_bytes =
        to_uint32_checked(channel_buffer_size_bytes_, "channel_buffer_size_bytes");
    connection_rt_args.channel_base_address =
        to_uint32_checked(channel_region_.get_address(logical_channel_id), "channel_base_address");
    connection_rt_args.connection_info_address =
        to_uint32_checked(connection_info_region_.get_address(logical_channel_id), "connection_info_address");
    connection_rt_args.connection_handshake_address =
        to_uint32_checked(connection_handshake_region_.get_address(logical_channel_id), "connection_handshake_address");
    connection_rt_args.flow_control_sem_id = client_semaphores.flow_control_sem_id;
    connection_rt_args.teardown_sem_id = client_semaphores.teardown_sem_id;
    connection_rt_args.mux_status_address = to_uint32_checked(status_region_.get_address(), "mux_status_address");

    const auto serialized_rt_args = connection_rt_args.serialize();
    worker_args.insert(worker_args.end(), serialized_rt_args.begin(), serialized_rt_args.end());
}

std::unordered_map<std::string, uint32_t> FabricMuxV2Config::get_fabric_mux_v2_named_compile_time_args() const {
    return {
        {"fabric_mux_v2_num_buffers_per_channel", static_cast<uint32_t>(num_buffers_per_channel_)},
        {"fabric_mux_v2_num_channels", static_cast<uint32_t>(num_channels_)},
        {"fabric_mux_v2_mux_status_address", to_uint32_checked(status_region_.get_address(), "mux_status_address")},
        {"fabric_mux_v2_channel_region_base_address",
         to_uint32_checked(channel_region_.get_address(), "channel_region_base_address")},
        {"fabric_mux_v2_connection_info_region_base_address",
         to_uint32_checked(connection_info_region_.get_address(), "connection_info_region_base_address")},
        {"fabric_mux_v2_connection_handshake_region_base_address",
         to_uint32_checked(connection_handshake_region_.get_address(), "connection_handshake_region_base_address")},
        {"fabric_mux_v2_shared_ring_region_base_address",
         to_uint32_checked(shared_ring_region_.get_address(), "shared_ring_region_base_address")},
        {"fabric_mux_v2_shared_trid_ring_capacity", trid_ring_capacity_},
        {"fabric_mux_v2_shared_control_region_base_address",
         to_uint32_checked(shared_control_region_.get_address(), "shared_control_region_base_address")},
        {"fabric_mux_v2_credit_notify_scratch_region_base_address",
         to_uint32_checked(credit_notify_scratch_region_.get_address(), "credit_notify_scratch_region_base_address")},
        {"fabric_mux_v2_channel_buffer_size_bytes",
         to_uint32_checked(channel_buffer_size_bytes_, "channel_buffer_size_bytes")},
        {"fabric_mux_v2_per_channel_scalar_region_stride_bytes",
         to_uint32_checked(per_channel_scalar_region_stride_bytes_, "per_channel_scalar_region_stride_bytes")},
        {"fabric_mux_v2_forwarder_service_burst_size", forwarder_service_burst_size_}};
}

void add_fabric_mux_v2_to_program(
    tt::tt_metal::Program& program,
    const FabricMuxV2Config& config,
    const tt::tt_metal::CoreCoord& mux_logical_core,
    const std::vector<uint32_t>& downstream_sender_rt_args,
    tt::tt_metal::NOC forwarder_noc) {
    const auto manager_noc = get_manager_noc_from_forwarder_noc(forwarder_noc);
    auto named_compile_args = config.get_fabric_mux_v2_named_compile_time_args();
    const auto forwarder_ready_sem_id = tt::tt_metal::CreateSemaphore(program, mux_logical_core, 0);
    const auto manager_init_done_sem_id = tt::tt_metal::CreateSemaphore(program, mux_logical_core, 0);
    named_compile_args["fabric_mux_v2_forwarder_ready_sem_id"] = forwarder_ready_sem_id;
    named_compile_args["fabric_mux_v2_manager_init_done_sem_id"] = manager_init_done_sem_id;

    auto forwarder_kernel = tt::tt_metal::CreateKernel(
        program,
        kFabricMuxV2KernelPath,
        mux_logical_core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = forwarder_noc,
            .named_compile_args = named_compile_args});
    tt::tt_metal::SetRuntimeArgs(program, forwarder_kernel, mux_logical_core, downstream_sender_rt_args);

    tt::tt_metal::CreateKernel(
        program,
        kFabricMuxV2KernelPath,
        mux_logical_core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = manager_noc,
            .named_compile_args = named_compile_args});
}

void add_fabric_mux_v2_to_program(
    tt::tt_metal::Program& program,
    const FabricMuxV2Config& config,
    const tt::tt_metal::CoreCoord& mux_logical_core,
    const FabricNodeId& src_fabric_node_id,
    const FabricNodeId& dst_fabric_node_id,
    uint32_t link_idx,
    tt::tt_metal::NOC forwarder_noc) {
    std::vector<uint32_t> forwarder_runtime_args;
    append_fabric_connection_rt_args(
        src_fabric_node_id,
        dst_fabric_node_id,
        link_idx,
        program,
        mux_logical_core,
        forwarder_runtime_args,
        CoreType::WORKER);
    add_fabric_mux_v2_to_program(program, config, mux_logical_core, forwarder_runtime_args, forwarder_noc);
}

size_t FabricMuxV2Config::get_memory_map_end_address() const { return memory_map_end_address_; }

void FabricMuxV2Config::validate_logical_channel_id(uint8_t logical_channel_id) const {
    TT_FATAL(
        logical_channel_id < num_channels_,
        "Invalid FabricMuxV2 logical_channel_id {} (num_channels = {})",
        static_cast<uint32_t>(logical_channel_id),
        static_cast<uint32_t>(num_channels_));
}

}  // namespace tt::tt_fabric
