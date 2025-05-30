// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
#include "dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_utils.h"
#include "tt_metal/api/tt-metalium/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp"

#include <cstddef>
#include <array>
// clang-format on

constexpr uint8_t NUM_FULL_SIZE_CHANNELS = get_compile_time_arg_val(0);
constexpr uint8_t NUM_BUFFERS_FULL_SIZE_CHANNEL = get_compile_time_arg_val(1);
constexpr size_t BUFFER_SIZE_BYTES_FULL_SIZE_CHANNEL = get_compile_time_arg_val(2);
constexpr uint8_t NUM_HEADER_ONLY_CHANNELS = get_compile_time_arg_val(3);
constexpr uint8_t NUM_BUFFERS_HEADER_ONLY_CHANNEL = get_compile_time_arg_val(4);
// header only buffer slot size is the same as the edm packet header size

constexpr size_t status_address = get_compile_time_arg_val(5);
constexpr size_t termination_signal_address = get_compile_time_arg_val(6);
constexpr size_t connection_info_base_address = get_compile_time_arg_val(7);
constexpr size_t connection_handshake_base_address = get_compile_time_arg_val(8);
constexpr size_t sender_flow_control_base_address = get_compile_time_arg_val(9);
constexpr size_t channels_base_l1_address = get_compile_time_arg_val(10);
constexpr size_t local_fabric_router_status_address = get_compile_time_arg_val(11);
constexpr size_t fabric_router_status_address = get_compile_time_arg_val(12);

constexpr uint8_t NUM_EDM_BUFFERS = get_compile_time_arg_val(13);
constexpr size_t NUM_FULL_SIZE_CHANNELS_ITERS = get_compile_time_arg_val(14);
constexpr size_t NUM_ITERS_BETWEEN_TEARDOWN_CHECKS = get_compile_time_arg_val(15);

constexpr ProgrammableCoreType CORE_TYPE = static_cast<ProgrammableCoreType>(get_compile_time_arg_val(16));

constexpr size_t NOC_ALIGN_PADDING_BYTES = 12;

namespace tt::tt_fabric {
using FabricMuxToEdmSender = WorkerToFabricEdmSenderImpl<NUM_EDM_BUFFERS>;
}  // namespace tt::tt_fabric

template <uint8_t NUM_BUFFERS>
void setup_channel(
    tt::tt_fabric::FabricMuxChannelBuffer<NUM_BUFFERS>* channel_ptr,
    tt::tt_fabric::FabricMuxChannelWorkerInterface<NUM_BUFFERS>* worker_interface_ptr,
    bool& channel_connection_established,
    uint8_t channel_id,
    size_t buffer_size_bytes,
    size_t& channel_base_address,
    size_t& connection_info_address,
    size_t& connection_handshake_address,
    size_t& sender_flow_control_address) {
    new (channel_ptr) tt::tt_fabric::FabricMuxChannelBuffer<NUM_BUFFERS>(
        channel_base_address,
        buffer_size_bytes,
        sizeof(PACKET_HEADER_TYPE),
        0, /* unused, eth_transaction_ack_word_addr */
        channel_id);
    channel_base_address += NUM_BUFFERS * buffer_size_bytes;

    auto connection_worker_info_ptr =
        reinterpret_cast<volatile tt::tt_fabric::FabricMuxChannelClientLocationInfo*>(connection_info_address);
    connection_worker_info_ptr->edm_rdptr = 0;
    connection_info_address += sizeof(tt::tt_fabric::FabricMuxChannelClientLocationInfo);

    new (worker_interface_ptr) tt::tt_fabric::FabricMuxChannelWorkerInterface<NUM_BUFFERS>(
        connection_worker_info_ptr,
        reinterpret_cast<volatile tt_l1_ptr uint32_t* const>(sender_flow_control_address),
        reinterpret_cast<volatile tt_l1_ptr uint32_t* const>(connection_handshake_address),
        0 /* unused, sender_sync_noc_cmd_buf */);
    sender_flow_control_address += sizeof(uint32_t) + NOC_ALIGN_PADDING_BYTES;
    connection_handshake_address += sizeof(uint32_t) + NOC_ALIGN_PADDING_BYTES;

    channel_connection_established = false;
}

template <uint8_t NUM_BUFFERS>
void forward_data(
    tt::tt_fabric::FabricMuxChannelBuffer<NUM_BUFFERS>& channel,
    tt::tt_fabric::FabricMuxChannelWorkerInterface<NUM_BUFFERS>& worker_interface,
    tt::tt_fabric::FabricMuxToEdmSender& fabric_connection,
    bool& channel_connection_established) {
    if (worker_interface.has_unsent_payload()) {
        size_t buffer_address = channel.get_buffer_address(worker_interface.local_wrptr.get_buffer_index());
        auto packet_header = reinterpret_cast<PACKET_HEADER_TYPE*>(buffer_address);

        fabric_connection.wait_for_empty_write_slot();
        fabric_connection.send_payload_blocking_from_address(
            (uint32_t)packet_header, packet_header->get_payload_size_including_header());

        auto& local_wrptr = worker_interface.local_wrptr;
        local_wrptr.increment();

        auto& local_rdptr = worker_interface.local_rdptr;
        local_rdptr.increment();

        if (channel_connection_established) {
            worker_interface.template update_worker_copy_of_read_ptr<false>(local_rdptr.get_ptr());
        }

        // not handling/processing acks for now, re-evaluate if needed
    }

    check_worker_connections(worker_interface, channel_connection_established);
}

void kernel_main() {
    auto status_ptr = reinterpret_cast<tt_l1_ptr uint32_t*>(status_address);
    status_ptr[0] = tt::tt_fabric::FabricMuxStatus::STARTED;

    size_t rt_args_idx = 0;
    auto fabric_connection = tt::tt_fabric::FabricMuxToEdmSender::build_from_args<CORE_TYPE>(rt_args_idx);

    std::array<tt::tt_fabric::FabricMuxChannelBuffer<NUM_BUFFERS_FULL_SIZE_CHANNEL>, NUM_FULL_SIZE_CHANNELS>
        full_size_channels;
    std::array<tt::tt_fabric::FabricMuxChannelWorkerInterface<NUM_BUFFERS_FULL_SIZE_CHANNEL>, NUM_FULL_SIZE_CHANNELS>
        full_size_channel_worker_interfaces;
    std::array<bool, NUM_FULL_SIZE_CHANNELS> full_size_channel_connection_established;

    std::array<tt::tt_fabric::FabricMuxChannelBuffer<NUM_BUFFERS_HEADER_ONLY_CHANNEL>, NUM_HEADER_ONLY_CHANNELS>
        header_only_channels;
    std::
        array<tt::tt_fabric::FabricMuxChannelWorkerInterface<NUM_BUFFERS_HEADER_ONLY_CHANNEL>, NUM_HEADER_ONLY_CHANNELS>
            header_only_channel_worker_interfaces;
    std::array<bool, NUM_HEADER_ONLY_CHANNELS> header_only_channel_connection_established;

    size_t channel_base_address = channels_base_l1_address;
    size_t connection_info_address = connection_info_base_address;
    size_t connection_handshake_address = connection_handshake_base_address;
    size_t sender_flow_control_address = sender_flow_control_base_address;

    for (uint8_t i = 0; i < NUM_FULL_SIZE_CHANNELS; i++) {
        setup_channel<NUM_BUFFERS_FULL_SIZE_CHANNEL>(
            &full_size_channels[i],
            &full_size_channel_worker_interfaces[i],
            full_size_channel_connection_established[i],
            i,
            BUFFER_SIZE_BYTES_FULL_SIZE_CHANNEL,
            channel_base_address,
            connection_info_address,
            connection_handshake_address,
            sender_flow_control_address);
    }

    for (uint8_t i = 0; i < NUM_HEADER_ONLY_CHANNELS; i++) {
        setup_channel<NUM_BUFFERS_HEADER_ONLY_CHANNEL>(
            &header_only_channels[i],
            &header_only_channel_worker_interfaces[i],
            header_only_channel_connection_established[i],
            i,
            sizeof(PACKET_HEADER_TYPE),
            channel_base_address,
            connection_info_address,
            connection_handshake_address,
            sender_flow_control_address);
    }

    volatile auto termination_signal_ptr =
        reinterpret_cast<volatile tt::tt_fabric::TerminationSignal*>(termination_signal_address);

    // wait for fabric router to be ready before setting up the connection
    tt::tt_fabric::wait_for_fabric_endpoint_ready(
        fabric_connection.edm_noc_x,
        fabric_connection.edm_noc_y,
        fabric_router_status_address,
        local_fabric_router_status_address);

    fabric_connection.open();

    status_ptr[0] = tt::tt_fabric::FabricMuxStatus::READY_FOR_TRAFFIC;

#if defined(COMPILE_FOR_IDLE_ERISC)
    uint32_t heartbeat = 0;
#endif
    while (!got_immediate_termination_signal(termination_signal_ptr)) {
        bool got_graceful_termination = got_graceful_termination_signal(termination_signal_ptr);
        if (got_graceful_termination) {
            bool all_channels_drained = false;
            for (uint8_t channel_id = 0; channel_id < NUM_FULL_SIZE_CHANNELS; channel_id++) {
                all_channels_drained &= full_size_channel_worker_interfaces[channel_id].has_unsent_payload();
            }
            for (uint8_t channel_id = 0; channel_id < NUM_HEADER_ONLY_CHANNELS; channel_id++) {
                all_channels_drained &= header_only_channel_worker_interfaces[channel_id].has_unsent_payload();
            }

            if (all_channels_drained) {
                return;
            }
        }

        for (size_t i = 0; i < NUM_ITERS_BETWEEN_TEARDOWN_CHECKS; i++) {
            for (size_t iter = 0; iter < NUM_FULL_SIZE_CHANNELS_ITERS; iter++) {
                for (uint8_t channel_id = 0; channel_id < NUM_FULL_SIZE_CHANNELS; channel_id++) {
                    forward_data<NUM_BUFFERS_FULL_SIZE_CHANNEL>(
                        full_size_channels[channel_id],
                        full_size_channel_worker_interfaces[channel_id],
                        fabric_connection,
                        full_size_channel_connection_established[channel_id]);
                }
            }

            for (uint8_t channel_id = 0; channel_id < NUM_HEADER_ONLY_CHANNELS; channel_id++) {
                forward_data<NUM_BUFFERS_HEADER_ONLY_CHANNEL>(
                    header_only_channels[channel_id],
                    header_only_channel_worker_interfaces[channel_id],
                    fabric_connection,
                    header_only_channel_connection_established[channel_id]);
            }
        }
#if defined(COMPILE_FOR_IDLE_ERISC)
        RISC_POST_HEARTBEAT(heartbeat);
#endif
    }

    fabric_connection.close();
    noc_async_write_barrier();
    noc_async_atomic_barrier();

    status_ptr[0] = tt::tt_fabric::FabricMuxStatus::TERMINATED;
}
