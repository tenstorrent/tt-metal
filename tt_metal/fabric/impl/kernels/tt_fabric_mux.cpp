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

// full buffers config
constexpr uint8_t NUM_FULL_SIZE_CHANNELS = get_compile_time_arg_val(0);
;
constexpr uint8_t NUM_BUFFERS_FULL_SIZE_CHANNEL = get_compile_time_arg_val(1);
constexpr size_t BUFFER_SIZE_BYTES_FULL_SIZE_CHANNEL = get_compile_time_arg_val(2);

// header only buffers config
constexpr uint8_t NUM_HEADER_ONLY_CHANNELS = get_compile_time_arg_val(3);
constexpr uint8_t NUM_BUFFERS_HEADER_ONLY_CHANNEL = get_compile_time_arg_val(4);
// header only buffer slot size is the same as the edm packet header size

// l1 start address for worker location info
constexpr size_t connection_info_base_address = get_compile_time_arg_val(5);

// connection liveness
constexpr size_t connection_handshake_base_address = get_compile_time_arg_val(6);  // need to be cleared from host

// flow control with the sender, sender update the wrptr after pushing
constexpr size_t sender_flow_control_base_address = get_compile_time_arg_val(7);  // need to be cleared from host

// workers read from and write to this address upon connecting/disconnecting
// constexpr size_t sender_buffer_index_base_address = 0x35000;        // need to be cleared from host

// l1 start address for buffers
constexpr size_t channels_base_l1_address = get_compile_time_arg_val(8);

// TODO: ratio of processing full buffers to header only buffers

constexpr size_t NOC_ALIGN_PADDING_BYTES = 12;

// buffer setup using compile args

void kernel_main() {
    size_t rt_args_idx = 0;

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
    size_t sender_buffer_index_address = sender_buffer_index_base_address;

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

    /*
        while (true) {
            // process from the full channel
            // id++;
            // process from the header only channel
            // id++;

            // check for termination?
        }
    */

    /*
        // mux connection with fabric is uni-directional
        // on mmio chip it connects to the router sending data towards remote chips
        // and on remote chips it connects to the router sending data towards mmio chips
        auto fabric_connection =
                tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);


        // edm connection setup
        fabric_connection.open();

        bool terminated = false;
        // main loop
        while (!terminated) {
            for (uint32_t i = 0; i < num_buffers; i++) {
                if (mux_buffers[i].is_empty()) {
                    continue;
                }

                // get packet address from the buffer
                fabric_connection.wait_for_empty_write_slot();
                fabric_connection.send_payload_without_header_non_blocking_from_address(payload_buffer_address,
       packet_payload_size_bytes); fabric_connection.send_payload_flush_blocking_from_address((uint32_t)packet_header,
       sizeof(PACKET_HEADER_TYPE));

                // update local ptrs

                // update remote ptrs
            }

            // check for termination
        }

        // edm connection teardown
        fabric_connection.close();
    */
}
