// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
#include "dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux.hpp"
#include "tt_metal/api/tt-metalium/fabric_edm_packet_header.hpp"

#include <cstddef>
#include <array>
// clang-format on

constexpr uint8_t NUM_FULL_SIZE_CHANNELS = get_compile_time_arg_val(0);
constexpr uint8_t NUM_BUFFERS_FULL_SIZE_CHANNEL = get_compile_time_arg_val(1);
constexpr size_t BUFFER_SIZE_BYTES_FULL_SIZE_CHANNEL = get_compile_time_arg_val(2);
constexpr uint8_t NUM_HEADER_ONLY_CHANNELS = get_compile_time_arg_val(3);
constexpr uint8_t NUM_BUFFERS_HEADER_ONLY_CHANNEL = get_compile_time_arg_val(4);
// header only buffer slot size is the same as the edm packet header size

constexpr size_t termination_signal_address = get_compile_time_arg_val(5);

// l1 start address for worker location info
constexpr size_t connection_info_base_address = get_compile_time_arg_val(6);

// connection liveness
constexpr size_t connection_handshake_base_address = get_compile_time_arg_val(7);  // need to be cleared from host

// flow control with the sender, sender update the wrptr after pushing
constexpr size_t sender_flow_control_base_address = get_compile_time_arg_val(8);  // need to be cleared from host

// workers read from and write to this address upon connecting/disconnecting
// constexpr size_t sender_buffer_index_base_address = 0x35000;        // need to be cleared from host

// l1 start address for buffers
constexpr size_t channels_base_l1_address = get_compile_time_arg_val(9);

constexpr uint8_t NUM_EDM_BUFFERS = get_compile_time_arg_val(10);
constexpr size_t NUM_FULL_SIZE_CHANNELS_ITERS = get_compile_time_arg_val(11);
constexpr size_t DEFAULT_NUM_ITERS_BETWEEN_TEARDOWN_CHECKS = get_compile_time_arg_val(12);

constexpr size_t NOC_ALIGN_PADDING_BYTES = 12;

namespace tt::tt_fabric {
using FabricMuxToEdmSender = WorkerToFabricEdmSenderImpl<NUM_EDM_BUFFERS>;
}  // namespace tt::tt_fabric

FORCE_INLINE bool got_immediate_termination_signal(volatile tt::tt_fabric::TerminationSignal* termination_signal_ptr) {
    return *termination_signal_ptr == tt::tt_fabric::TerminationSignal::IMMEDIATELY_TERMINATE;
}

FORCE_INLINE bool got_graceful_termination_signal(volatile tt::tt_fabric::TerminationSignal* termination_signal_ptr) {
    return *termination_signal_ptr == tt::tt_fabric::TerminationSignal::GRACEFULLY_TERMINATE;
}

template <uint8_t NUM_BUFFERS, bool FULL_SIZE_CHANNEL>
void forward_data(
    tt::tt_fabric::FabricMuxChannelBuffer<NUM_BUFFERS>& channel,
    tt::tt_fabric::FabricMuxChannelWorkerInterface<NUM_BUFFERS>& worker_interface,
    tt::tt_fabric::FabricMuxToEdmSender& fabric_connection) {
    bool has_data_to_send = worker_interface.has_unsent_payload();
    if (has_data_to_send) {
        size_t buffer_address = channel.get_buffer_address(worker_interface.local_wrptr.get_buffer_index());
        auto packet_header = reinterpret_cast<PACKET_HEADER_TYPE*>(buffer_address);

        fabric_connection.wait_for_empty_write_slot();
        if constexpr (FULL_SIZE_CHANNEL) {
            auto payload_address = buffer_address + sizeof(PACKET_HEADER_TYPE);
            fabric_connection.send_payload_without_header_non_blocking_from_address(
                payload_address, packet_header->get_payload_size_excluding_header());
            fabric_connection.send_payload_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
        } else {
            fabric_connection.send_payload_flush_non_blocking_from_address(
                (uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
        }

        auto& local_wrptr = worker_interface.local_wrptr;
        local_wrptr.increment();

        auto& sender_rdptr = worker_interface.local_rdptr;
        sender_rdptr.increment();
        worker_interface.template update_worker_copy_of_read_ptr<false>(sender_rdptr.get_ptr());

        // not handling/processing acks for now, re-evaluate if needed
    }
}

void kernel_main() {
    size_t rt_args_idx = 0;
    auto fabric_connection =
        tt::tt_fabric::FabricMuxToEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);

    std::array<tt::tt_fabric::FabricMuxChannelBuffer<NUM_BUFFERS_FULL_SIZE_CHANNEL>, NUM_FULL_SIZE_CHANNELS>
        full_size_channels;
    std::array<tt::tt_fabric::FabricMuxChannelWorkerInterface<NUM_BUFFERS_FULL_SIZE_CHANNEL>, NUM_FULL_SIZE_CHANNELS>
        full_size_channel_worker_interfaces;

    std::array<tt::tt_fabric::FabricMuxChannelBuffer<NUM_BUFFERS_HEADER_ONLY_CHANNEL>, NUM_HEADER_ONLY_CHANNELS>
        header_only_channels;
    std::
        array<tt::tt_fabric::FabricMuxChannelWorkerInterface<NUM_BUFFERS_HEADER_ONLY_CHANNEL>, NUM_HEADER_ONLY_CHANNELS>
            header_only_channel_worker_interfaces;

    size_t channel_base_address = channels_base_l1_address;
    size_t connection_info_address = connection_info_base_address;
    size_t connection_handshake_address = connection_handshake_base_address;
    size_t sender_flow_control_address = sender_flow_control_base_address;

    for (uint8_t i = 0; i < NUM_FULL_SIZE_CHANNELS; i++) {
        new (&full_size_channels[i]) tt::tt_fabric::FabricMuxChannelBuffer<NUM_BUFFERS_FULL_SIZE_CHANNEL>(
            channel_base_address,
            BUFFER_SIZE_BYTES_FULL_SIZE_CHANNEL,
            sizeof(PACKET_HEADER_TYPE),
            0, /* unused, eth_transaction_ack_word_addr */
            i);
        channel_base_address += NUM_BUFFERS_FULL_SIZE_CHANNEL * BUFFER_SIZE_BYTES_FULL_SIZE_CHANNEL;

        auto connection_worker_info_ptr =
            reinterpret_cast<volatile tt::tt_fabric::FabricMuxChannelClientLocationInfo*>(connection_info_address);
        connection_worker_info_ptr->edm_rdptr = 0;
        connection_info_address += sizeof(tt::tt_fabric::FabricMuxChannelClientLocationInfo);

        new (&full_size_channel_worker_interfaces[i])
            tt::tt_fabric::FabricMuxChannelWorkerInterface<NUM_BUFFERS_FULL_SIZE_CHANNEL>(
                connection_worker_info_ptr,
                reinterpret_cast<volatile tt_l1_ptr uint32_t* const>(sender_flow_control_address),
                reinterpret_cast<volatile tt_l1_ptr uint32_t* const>(connection_handshake_address),
                0 /* unused, sender_sync_noc_cmd_buf */);
        sender_flow_control_address += sizeof(uint32_t) + NOC_ALIGN_PADDING_BYTES;
        connection_handshake_address += sizeof(uint32_t) + NOC_ALIGN_PADDING_BYTES;
    }

    for (uint8_t i = 0; i < NUM_HEADER_ONLY_CHANNELS; i++) {
        new (&header_only_channels[i]) tt::tt_fabric::FabricMuxChannelBuffer<NUM_BUFFERS_HEADER_ONLY_CHANNEL>(
            channel_base_address,
            sizeof(PACKET_HEADER_TYPE),
            sizeof(PACKET_HEADER_TYPE),
            0, /* unused, eth_transaction_ack_word_addr */
            i);
        channel_base_address += NUM_BUFFERS_HEADER_ONLY_CHANNEL * sizeof(PACKET_HEADER_TYPE);

        auto connection_worker_info_ptr =
            reinterpret_cast<volatile tt::tt_fabric::FabricMuxChannelClientLocationInfo*>(connection_info_address);
        connection_worker_info_ptr->edm_rdptr = 0;
        connection_info_address += sizeof(tt::tt_fabric::FabricMuxChannelClientLocationInfo);

        new (&header_only_channel_worker_interfaces[i])
            tt::tt_fabric::FabricMuxChannelWorkerInterface<NUM_BUFFERS_HEADER_ONLY_CHANNEL>(
                connection_worker_info_ptr,
                reinterpret_cast<volatile tt_l1_ptr uint32_t* const>(sender_flow_control_address),
                reinterpret_cast<volatile tt_l1_ptr uint32_t* const>(connection_handshake_address),
                0 /* unused, sender_sync_noc_cmd_buf */);
        sender_flow_control_address += sizeof(uint32_t) + NOC_ALIGN_PADDING_BYTES;
        connection_handshake_address += sizeof(uint32_t) + NOC_ALIGN_PADDING_BYTES;
    }

    fabric_connection.open();

    volatile auto termination_signal_ptr =
        reinterpret_cast<volatile tt::tt_fabric::TerminationSignal*>(termination_signal_address);

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

        for (size_t i = 0; i < DEFAULT_NUM_ITERS_BETWEEN_TEARDOWN_CHECKS; i++) {
            for (size_t iter = 0; iter < NUM_FULL_SIZE_CHANNELS_ITERS; iter++) {
                for (uint8_t channel_id = 0; channel_id < NUM_FULL_SIZE_CHANNELS; channel_id++) {
                    forward_data<NUM_BUFFERS_FULL_SIZE_CHANNEL, true>(
                        full_size_channels[channel_id],
                        full_size_channel_worker_interfaces[channel_id],
                        fabric_connection);
                }
            }

            for (uint8_t channel_id = 0; channel_id < NUM_HEADER_ONLY_CHANNELS; channel_id++) {
                forward_data<NUM_BUFFERS_HEADER_ONLY_CHANNEL, false>(
                    header_only_channels[channel_id],
                    header_only_channel_worker_interfaces[channel_id],
                    fabric_connection);
            }
        }
    }

    fabric_connection.close();
}
