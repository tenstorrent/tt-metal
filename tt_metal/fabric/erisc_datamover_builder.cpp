// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/math.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/assert.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/program_impl.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/hal_exp.hpp>
#include <tt-metalium/erisc_datamover_builder.hpp>
#include <tt-metalium/fabric_edm_packet_header.hpp>

#include <iterator>
#include <vector>
#include <algorithm>
#include <ranges>

using namespace tt::tt_metal::experimental;

namespace tt::tt_fabric {

// The channel structure is as follows:
//              &header->  |----------------| channel_base_address
//                         |    header      |
//             &payload->  |----------------|
//                         |                |
//                         |    payload     |
//                         |                |
//        &channel_sync->  |----------------|
//                         |  channel_sync  |
//                         ------------------
//

FabricEriscDatamoverConfig::FabricEriscDatamoverConfig(
    std::size_t channel_buffer_size_bytes, std::size_t sender_ratio_size, std::size_t receiver_ratio_size) {
    TT_FATAL(
        (receiver_completed_packet_header_cb_address % eth_word_l1_alignment == 0),
        "receiver_completed_packet_header_cb_address must be aligned to 16 bytes");
    TT_FATAL(
        (sender_0_completed_packet_header_cb_address % eth_word_l1_alignment == 0),
        "receiver_completed_packet_header_cb_address must be aligned to 16 bytes");
    TT_FATAL(
        (sender_1_completed_packet_header_cb_address % eth_word_l1_alignment == 0),
        "receiver_completed_packet_header_cb_address must be aligned to 16 bytes");
    TT_FATAL(
        (sender_channel_0_buffer_index_address % eth_word_l1_alignment == 0),
        "receiver_completed_packet_header_cb_address must be aligned to 16 bytes");
    TT_FATAL(
        (sender_channel_0_worker_conn_info_base_address % eth_word_l1_alignment == 0),
        "receiver_completed_packet_header_cb_address must be aligned to 16 bytes");
    TT_FATAL(
        (sender_channel_0_local_flow_control_semaphore_address % eth_word_l1_alignment == 0),
        "receiver_completed_packet_header_cb_address must be aligned to 16 bytes");
    TT_FATAL(
        (sender_channel_0_producer_terminate_connection_address % eth_word_l1_alignment == 0),
        "receiver_completed_packet_header_cb_address must be aligned to 16 bytes");
    TT_FATAL(
        (sender_channel_1_local_flow_control_semaphore_address % eth_word_l1_alignment == 0),
        "receiver_completed_packet_header_cb_address must be aligned to 16 bytes");
    TT_FATAL(
        (sender_channel_1_producer_terminate_connection_address % eth_word_l1_alignment == 0),
        "receiver_completed_packet_header_cb_address must be aligned to 16 bytes");

    TT_FATAL(
        sender_channel_1_buffer_index_address != sender_channel_0_buffer_index_address,
        "FabricEriscDatamoverConfig was constructed with illegal buffer index address");
    const size_t min_buffer_size =
        sizeof(tt::tt_fabric::PacketHeader) + 2 * FabricEriscDatamoverConfig::eth_channel_sync_size;
    TT_FATAL(
        channel_buffer_size_bytes >= min_buffer_size,
        "FabricEriscDatamoverConfig was constructed with `channel_buffer_size_bytes` argument set smaller than minimum "
        "size of {}",
        min_buffer_size);

    constexpr size_t default_pow2_num_sender_buffer_slots = 8;
    constexpr size_t default_pow2_num_receiver_buffer_slots = 16;

    const std::size_t channel_buffer_size_with_channel_sync =
        channel_buffer_size_bytes +
        sizeof(tt::tt_fabric::PacketHeader);  // + 16 // sizeof(tt::tt_fabric::PacketHeader);

    const size_t next_lowest_power_of_2_buffer_slot_count =

        this->channel_buffer_size_bytes = channel_buffer_size_bytes;
    this->channel_buffer_size_bytes_with_channel_sync = channel_buffer_size_with_channel_sync;
    const std::size_t total_ratio_count = 2 * sender_ratio_size + receiver_ratio_size;

    this->sender_0_channel_size_bytes = tt::round_down(
        (available_channel_buffering_space / total_ratio_count) * sender_ratio_size,
        channel_buffer_size_with_channel_sync);
    if constexpr (FabricEriscDatamoverConfig::constrain_to_power_of_2_buffer_slot_counts) {
        this->sender_0_num_buffers = default_pow2_num_sender_buffer_slots;
    } else {
        this->sender_0_num_buffers = this->sender_0_channel_size_bytes / channel_buffer_size_with_channel_sync;
    }
    this->sender_1_channel_size_bytes = tt::round_down(
        (available_channel_buffering_space / total_ratio_count) * sender_ratio_size,
        channel_buffer_size_with_channel_sync);
    if constexpr (FabricEriscDatamoverConfig::constrain_to_power_of_2_buffer_slot_counts) {
        this->sender_1_num_buffers = default_pow2_num_sender_buffer_slots;
    } else {
        this->sender_1_num_buffers = this->sender_1_channel_size_bytes / channel_buffer_size_with_channel_sync;
    }
    this->receiver_channel_size_bytes = tt::round_down(
        (available_channel_buffering_space / total_ratio_count) * receiver_ratio_size,
        channel_buffer_size_with_channel_sync);
    if constexpr (FabricEriscDatamoverConfig::constrain_to_power_of_2_buffer_slot_counts) {
        this->receiver_num_buffers = default_pow2_num_receiver_buffer_slots;
    } else {
        this->receiver_num_buffers = this->receiver_channel_size_bytes / channel_buffer_size_with_channel_sync;
    }

    this->sender_0_channel_base_address = buffer_region_start;
    this->sender_1_channel_base_address = this->sender_0_channel_base_address + this->sender_0_channel_size_bytes;
    this->receiver_channel_base_address = this->sender_1_channel_base_address + this->sender_1_channel_size_bytes;

    log_trace(tt::LogOp, "Sender 0 channel_start: {}", this->sender_0_channel_base_address);
    log_trace(tt::LogOp, "Sender 1 channel_start: {}", this->sender_1_channel_base_address);
    log_trace(tt::LogOp, "Receiver channel_start: {}", this->receiver_channel_base_address);

    static constexpr size_t total_num_channels = 3;  // sender0, sender1, receiver
    const size_t max_channel_buffer_size = (available_channel_buffering_space / total_num_channels) -
                                           FabricEriscDatamoverConfig::eth_channel_sync_size -
                                           sizeof(tt::tt_fabric::PacketHeader);
    TT_FATAL(
        channel_buffer_size_bytes <= max_channel_buffer_size,
        "Specified size of `channel_buffer_size_bytes` was too large. Maximum allowable size is {} B",
        max_channel_buffer_size);
    TT_FATAL(
        this->sender_0_channel_size_bytes > 0,
        "Internal error when computing `sender_0_channel_size_bytes` which was computed to be size 0");
    TT_FATAL(
        this->sender_1_channel_size_bytes > 0,
        "Internal error when computing `sender_1_channel_size_bytes` which was computed to be size 0");
    TT_FATAL(
        this->receiver_channel_size_bytes > 0,
        "Internal error when computing `receiver_channel_size_bytes` which was computed to be size 0");
    TT_FATAL(
        this->sender_0_channel_size_bytes + this->sender_1_channel_size_bytes + this->receiver_channel_size_bytes <=
            this->available_channel_buffering_space,
        "Internal error when computing channel sizes. Total channel size exceeds available space");
    TT_FATAL(
        this->receiver_channel_base_address + this->receiver_channel_size_bytes < this->max_l1_loading_size,
        "Internal error - channel buffers spilled past the end of usable L1 region.");
}

void get_runtime_args_for_edm_termination_infos(
    const std::vector<edm_termination_info_t>& edm_termination_infos, std::vector<uint32_t>& args_out) {
    args_out.reserve(args_out.size() + edm_termination_infos.size() * 4 + 1);
    args_out.push_back(edm_termination_infos.size());
    for (const auto& info : edm_termination_infos) {
        args_out.push_back(info.edm_noc_x);
        args_out.push_back(info.edm_noc_y);
        args_out.push_back(info.distance);
        args_out.push_back(info.termination_addr);
        log_trace(
            tt::LogTest,
            "EDM termination info: x={}, y={}, distance={}, termination_addr={}",
            info.edm_noc_x,
            info.edm_noc_y,
            info.distance,
            info.termination_addr);
    }
}

void append_worker_to_fabric_edm_sender_rt_args(
    const SenderWorkerAdapterSpec& connection,
    size_t sender_worker_flow_control_semaphore_id,
    size_t sender_worker_terminate_semaphore_id,
    size_t sender_worker_buffer_index_semaphore_id,
    std::vector<uint32_t>& args_out) {
    auto edm_noc_xy = tt::tt_fabric::WorkerXY(connection.edm_noc_x, connection.edm_noc_y);
    const std::vector<uint32_t> values = {
        connection.persistent_fabric,
        edm_noc_xy.to_uint32(),
        connection.edm_buffer_base_addr,
        connection.num_buffers_per_channel,
        connection.edm_l1_sem_addr,
        connection.edm_connection_handshake_addr,
        connection.edm_worker_location_info_addr,
        connection.buffer_size_bytes,
        connection.buffer_index_semaphore_id,
        sender_worker_flow_control_semaphore_id,
        sender_worker_terminate_semaphore_id,
        sender_worker_buffer_index_semaphore_id};
    args_out.reserve(args_out.size() + (values.size() / sizeof(size_t)));
    std::ranges::copy(values, std::back_inserter(args_out));
}

size_t log_worker_to_fabric_edm_sender_rt_args(const std::vector<uint32_t>& args, size_t starting_arg_idx) {
    log_trace(tt::LogOp, "Worker to fabric EDM Sender has {} RT Args: {}", args.size(), args);
    log_trace(tt::LogOp, "arg[{}]: edm_noc_xy {}", starting_arg_idx, args[starting_arg_idx++]);
    log_trace(tt::LogOp, "arg[{}]: edm_buffer_base_addr {}", starting_arg_idx, args[starting_arg_idx++]);
    log_trace(tt::LogOp, "arg[{}]: num_buffers_per_channel {}", starting_arg_idx, args[starting_arg_idx++]);
    log_trace(tt::LogOp, "arg[{}]: edm_l1_sem_addr {}", starting_arg_idx, args[starting_arg_idx++]);
    log_trace(tt::LogOp, "arg[{}]: edm_connection_handshake_addr {}", starting_arg_idx, args[starting_arg_idx++]);
    log_trace(tt::LogOp, "arg[{}]: edm_worker_location_info_addr {}", starting_arg_idx, args[starting_arg_idx++]);
    log_trace(tt::LogOp, "arg[{}]: buffer_size_bytes {}", starting_arg_idx, args[starting_arg_idx++]);
    log_trace(tt::LogOp, "arg[{}]: buffer_index_semaphore_id {}", starting_arg_idx, args[starting_arg_idx++]);
    log_trace(
        tt::LogOp, "arg[{}]: sender_worker_flow_control_semaphore_id {}", starting_arg_idx, args[starting_arg_idx++]);
    log_trace(
        tt::LogOp, "arg[{}]: sender_worker_buffer_index_semaphore_id {}", starting_arg_idx, args[starting_arg_idx++]);
    return starting_arg_idx + 10;
}

FabricEriscDatamoverBuilder::FabricEriscDatamoverBuilder(
    const CoreCoord& my_eth_core_logical,
    size_t my_noc_x,
    size_t my_noc_y,
    size_t my_chip_id,
    size_t peer_chip_id,

    std::optional<size_t> receiver_channel_downstream_flow_control_semaphore_id,
    std::optional<size_t> receiver_channel_downstream_teardown_semaphore_id,
    size_t sender_channel_0_flow_control_semaphore_id,
    size_t sender_channel_1_flow_control_semaphore_id,
    size_t sender_channel_0_connection_semaphore_id,
    size_t sender_channel_1_connection_semaphore_id,
    size_t sender_channel_0_buffer_index_semaphore_id,
    size_t sender_channel_1_buffer_index_semaphore_id,

    const FabricEriscDatamoverConfig& config,
    bool enable_persistent_mode,
    bool build_in_worker_connection_mode) :
    my_eth_core_logical(my_eth_core_logical),
    my_noc_x(my_noc_x),
    my_noc_y(my_noc_y),
    config(config),
    my_chip_id(my_chip_id),
    peer_chip_id(peer_chip_id),
    handshake_address(
        tt::round_up(hal::get_erisc_l1_unreserved_base(), FabricEriscDatamoverConfig::eth_channel_sync_size)),
    channel_buffer_size(config.channel_buffer_size_bytes),
    sender_0_num_buffers(config.sender_0_num_buffers),
    sender_1_num_buffers(config.sender_1_num_buffers),
    receiver_num_buffers(config.receiver_num_buffers),

    // this is the receiver channel's local sem for flow controlling with downstream fabric sender
    receiver_channel_downstream_flow_control_semaphore_id(receiver_channel_downstream_flow_control_semaphore_id),
    receiver_channel_downstream_teardown_semaphore_id(receiver_channel_downstream_teardown_semaphore_id),
    sender_channel_0_flow_control_semaphore_id(sender_channel_0_flow_control_semaphore_id),
    sender_channel_1_flow_control_semaphore_id(sender_channel_1_flow_control_semaphore_id),
    sender_channel_0_connection_semaphore_id(sender_channel_0_connection_semaphore_id),
    sender_channel_1_connection_semaphore_id(sender_channel_1_connection_semaphore_id),
    sender_channel_0_buffer_index_semaphore_id(sender_channel_0_buffer_index_semaphore_id),
    sender_channel_1_buffer_index_semaphore_id(sender_channel_1_buffer_index_semaphore_id),

    receiver_channel_local_buffer_index_address(config.receiver_channel_local_buffer_index_address),

    local_sender_channel_0_buffer_address(config.sender_0_channel_base_address),
    local_sender_channel_0_connection_info_addr(config.sender_channel_0_worker_conn_info_base_address),
    local_sender_channel_1_buffer_address(config.sender_1_channel_base_address),
    local_sender_channel_1_connection_info_addr(config.sender_channel_1_worker_conn_info_base_address),
    local_receiver_channel_buffer_address(config.receiver_channel_base_address),

    termination_signal_ptr(config.termination_signal_address),
    edm_status_ptr(config.edm_status_address),
    enable_persistent_mode(enable_persistent_mode),
    build_in_worker_connection_mode(build_in_worker_connection_mode) {}

std::vector<uint32_t> FabricEriscDatamoverBuilder::get_compile_time_args() const {
    const bool is_handshake_master = this->my_chip_id < this->peer_chip_id;
    TT_ASSERT(this->my_chip_id != this->peer_chip_id);
    TT_ASSERT(
        this->sender_0_num_buffers == this->sender_1_num_buffers);  //, "Implementation expects sender_0_num_buffers and
                                                                    // sender_1_num_buffers to be the same for now");
    log_trace(tt::LogTest, "Sender 0 num buffers: {}", this->sender_0_num_buffers);
    log_trace(tt::LogTest, "Sender 0 channel address: {}", this->local_sender_channel_0_buffer_address);
    log_trace(tt::LogTest, "Sender 1 num buffers: {}", this->sender_1_num_buffers);
    log_trace(tt::LogTest, "Sender 1 channel address: {}", this->local_sender_channel_1_buffer_address);
    log_trace(tt::LogTest, "Receiver num buffers: {}", this->receiver_num_buffers);
    log_trace(tt::LogTest, "Receiver channel address: {}", this->local_receiver_channel_buffer_address);

    return std::vector<uint32_t>{
        this->firmware_context_switch_interval,
        is_handshake_master,
        this->handshake_address,
        this->channel_buffer_size,

        this->sender_0_num_buffers,
        this->receiver_num_buffers,

        config.sender_0_channel_base_address,
        config.sender_channel_0_worker_conn_info_base_address,
        config.sender_1_channel_base_address,
        config.sender_channel_1_worker_conn_info_base_address,
        config.receiver_channel_base_address,
        config.receiver_channel_base_address,

        config.sender_0_channel_base_address,
        config.sender_1_channel_base_address,

        this->termination_signal_ptr,
        this->edm_status_ptr,
        this->enable_persistent_mode,

        // fabric counters
        FabricEriscDatamoverConfig::enable_fabric_counters,
        config.receiver_channel_counters_address,
        config.sender_channel_0_counters_address,
        config.sender_channel_1_counters_address,

        // fabric pkt header recording
        FabricEriscDatamoverConfig::enable_fabric_pkt_header_recording,

        config.receiver_completed_packet_header_cb_address,
        FabricEriscDatamoverConfig::receiver_completed_packet_header_cb_size_headers,
        config.sender_0_completed_packet_header_cb_address,
        FabricEriscDatamoverConfig::sender_completed_packet_header_cb_size_headers,
        config.sender_1_completed_packet_header_cb_address,
        FabricEriscDatamoverConfig::sender_completed_packet_header_cb_size_headers};
}

std::vector<uint32_t> FabricEriscDatamoverBuilder::get_runtime_args() const {
    return std::vector<uint32_t>{
        this->sender_channel_0_connection_semaphore_id,
        this->sender_channel_1_connection_semaphore_id,
        this->sender_channel_0_buffer_index_semaphore_id,
        this->downstream_sender_channel_buffer_index_semaphore_id.value_or(-1),
        this->downstream_edm_buffer_base_address != std::nullopt,
        this->downstream_edm_buffer_base_address.value_or(0),
        this->downstream_edm_noc_x.value_or(0),
        this->downstream_edm_noc_y.value_or(0),
        this->downstream_edm_semaphore_address.value_or(-1),
        this->downstream_edm_worker_registration_address.value_or(0),
        this->downstream_edm_worker_location_info_address.value_or(0),
        this->receiver_channel_local_buffer_index_address,
        // this is the receiver channel's local sem for flow controlling with downstream fabric sender
        this->receiver_channel_downstream_flow_control_semaphore_id.value_or(-1),
        this->receiver_channel_downstream_teardown_semaphore_id.value_or(-1),
        this->sender_channel_0_flow_control_semaphore_id,
        this->sender_channel_1_flow_control_semaphore_id};
}

FabricEriscDatamoverBuilder FabricEriscDatamoverBuilder::build(
    tt::tt_metal::IDevice* device,
    tt::tt_metal::Program& program,
    const CoreCoord& ethernet_core,
    chip_id_t local_chip_id,
    chip_id_t peer_chip_id,
    const FabricEriscDatamoverConfig& config,
    bool enable_persistent_mode,
    bool build_in_worker_connection_mode) {
    if (enable_persistent_mode) {
        auto sender_channel_0_buffer_index_semaphore_address = config.sender_channel_0_buffer_index_semaphore_address;
        auto sender_channel_0_flow_control_semaphore_address =
            config.sender_channel_0_local_flow_control_semaphore_address;
        auto sender_channel_0_connection_semaphore_address = config.sender_channel_0_connection_semaphore_address;

        std::optional<size_t> receiver_channel_downstream_flow_control_semaphore_address =
            build_in_worker_connection_mode ? 0
                                            : tt::tt_metal::CreateSemaphore(program, ethernet_core, 0, CoreType::ETH);
        std::optional<size_t> receiver_channel_downstream_terminate_semaphore_address =
            build_in_worker_connection_mode ? 0
                                            : tt::tt_metal::CreateSemaphore(program, ethernet_core, 0, CoreType::ETH);
        auto sender_channel_1_flow_control_semaphore_id =
            build_in_worker_connection_mode ? 0
                                            : tt::tt_metal::CreateSemaphore(program, ethernet_core, 0, CoreType::ETH);
        auto sender_channel_1_connection_semaphore_id =
            build_in_worker_connection_mode ? 0
                                            : tt::tt_metal::CreateSemaphore(program, ethernet_core, 0, CoreType::ETH);
        auto sender_channel_1_buffer_index_semaphore_id =
            build_in_worker_connection_mode ? 0
                                            : tt::tt_metal::CreateSemaphore(program, ethernet_core, 0, CoreType::ETH);

        return FabricEriscDatamoverBuilder(
            ethernet_core,
            device->ethernet_core_from_logical_core(ethernet_core).x,
            device->ethernet_core_from_logical_core(ethernet_core).y,
            local_chip_id,
            peer_chip_id,

            receiver_channel_downstream_flow_control_semaphore_address,
            receiver_channel_downstream_terminate_semaphore_address,
            sender_channel_0_flow_control_semaphore_address,
            sender_channel_1_flow_control_semaphore_id,
            sender_channel_0_connection_semaphore_address,
            sender_channel_1_connection_semaphore_id,
            sender_channel_0_buffer_index_semaphore_address,
            sender_channel_1_buffer_index_semaphore_id,

            config,
            enable_persistent_mode,
            build_in_worker_connection_mode);

    } else {
        std::optional<size_t> receiver_channel_downstream_flow_control_semaphore_id =
            tt::tt_metal::CreateSemaphore(program, ethernet_core, 0, CoreType::ETH);
        std::optional<size_t> receiver_channel_downstream_teardown_semaphore_id =
            tt::tt_metal::CreateSemaphore(program, ethernet_core, 0, CoreType::ETH);
        auto sender_channel_0_flow_control_semaphore_id =
            tt::tt_metal::CreateSemaphore(program, ethernet_core, 0, CoreType::ETH);
        auto sender_channel_1_flow_control_semaphore_id =
            tt::tt_metal::CreateSemaphore(program, ethernet_core, 0, CoreType::ETH);
        auto sender_channel_0_connection_semaphore_id =
            tt::tt_metal::CreateSemaphore(program, ethernet_core, 0, CoreType::ETH);
        auto sender_channel_1_connection_semaphore_id =
            tt::tt_metal::CreateSemaphore(program, ethernet_core, 0, CoreType::ETH);
        auto sender_channel_0_buffer_index_semaphore_id =
            tt::tt_metal::CreateSemaphore(program, ethernet_core, 0, CoreType::ETH);
        auto sender_channel_1_buffer_index_semaphore_id =
            tt::tt_metal::CreateSemaphore(program, ethernet_core, 0, CoreType::ETH);

        return FabricEriscDatamoverBuilder(
            ethernet_core,
            device->ethernet_core_from_logical_core(ethernet_core).x,
            device->ethernet_core_from_logical_core(ethernet_core).y,
            local_chip_id,
            peer_chip_id,

            receiver_channel_downstream_flow_control_semaphore_id,
            receiver_channel_downstream_teardown_semaphore_id,
            sender_channel_0_flow_control_semaphore_id,
            sender_channel_1_flow_control_semaphore_id,
            sender_channel_0_connection_semaphore_id,
            sender_channel_1_connection_semaphore_id,
            sender_channel_0_buffer_index_semaphore_id,
            sender_channel_1_buffer_index_semaphore_id,

            config,
            enable_persistent_mode);
    }
}

SenderWorkerAdapterSpec FabricEriscDatamoverBuilder::build_connection_to_worker_channel() const {
    if (this->enable_persistent_mode) {
        log_trace(tt::LogOp, "Building connection to persistent fabric");
    } else {
        log_trace(tt::LogOp, "Building connection to non-persistent fabric");
    }
    TT_FATAL(
        sender_channel_0_buffer_index_semaphore_id != sender_channel_0_flow_control_semaphore_id,
        "Internal error - sender_channel_0_buffer_index_semaphore_id and sender_channel_0_flow_control_semaphore_id "
        "aliased eachother");
    return SenderWorkerAdapterSpec{
        this->my_noc_x,
        this->my_noc_y,
        this->local_sender_channel_0_buffer_address,
        this->sender_0_num_buffers,
        this->sender_channel_0_flow_control_semaphore_id,
        this->sender_channel_0_connection_semaphore_id,
        this->config.sender_channel_0_worker_conn_info_base_address,
        this->config.channel_buffer_size_bytes,
        this->sender_channel_0_buffer_index_semaphore_id,
        this->enable_persistent_mode};
}

SenderWorkerAdapterSpec FabricEriscDatamoverBuilder::build_connection_to_fabric_channel() const {
    return SenderWorkerAdapterSpec{
        this->my_noc_x,
        this->my_noc_y,
        this->local_sender_channel_1_buffer_address,
        this->sender_1_num_buffers,
        this->sender_channel_1_flow_control_semaphore_id,
        this->sender_channel_1_connection_semaphore_id,
        this->config.sender_channel_1_worker_conn_info_base_address,
        this->config.channel_buffer_size_bytes,
        this->sender_channel_1_buffer_index_semaphore_id,
        false};
}

void FabricEriscDatamoverBuilder::connect_to_downstream_edm(const FabricEriscDatamoverBuilder& downstream_edm) {
    TT_FATAL(
        !this->build_in_worker_connection_mode, "Tried to connect two EDMs to each other in worker connection mode");
    const auto adapter_spec = downstream_edm.build_connection_to_fabric_channel();

    log_trace(
        tt::LogTest, "Connecting to downstream EDM at x={}, y={}", adapter_spec.edm_noc_x, adapter_spec.edm_noc_y);

    this->downstream_edm_noc_x = adapter_spec.edm_noc_x;
    this->downstream_edm_noc_y = adapter_spec.edm_noc_y;
    this->downstream_edm_buffer_base_address = adapter_spec.edm_buffer_base_addr;
    this->downstream_edm_semaphore_address = adapter_spec.edm_l1_sem_addr;
    this->downstream_edm_worker_registration_address = adapter_spec.edm_connection_handshake_addr;
    this->downstream_edm_worker_location_info_address = adapter_spec.edm_worker_location_info_addr;
    this->downstream_sender_channel_buffer_index_semaphore_id = adapter_spec.buffer_index_semaphore_id;
}

void FabricEriscDatamoverBuilder::teardown_from_host(
    tt::tt_metal::IDevice* d, tt::tt_fabric::TerminationSignal termination_signal) const {
    std::vector<uint32_t> val(1, termination_signal);
    d->push_work(
        [&]() {
            tt::tt_metal::detail::WriteToDeviceL1(
                d,
                d->logical_core_from_ethernet_core(CoreCoord(this->my_noc_x, this->my_noc_y)),
                config.termination_signal_address,
                val,
                CoreType::ETH);
        },
        true);
}

void FabricEriscDatamoverBuilder::set_firmware_context_switch_interval(size_t interval) {
    this->firmware_context_switch_interval = interval;
}

}  // namespace tt::tt_fabric
