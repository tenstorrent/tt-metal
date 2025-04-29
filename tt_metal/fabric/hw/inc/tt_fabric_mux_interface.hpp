// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"

namespace tt::tt_fabric {

template <uint8_t FABRIC_MUX_CHANNEL_NUM_BUFFERS>
using WorkerToFabricMuxSender = WorkerToFabricEdmSenderImpl<FABRIC_MUX_CHANNEL_NUM_BUFFERS>;

template <uint8_t FABRIC_MUX_CHANNEL_NUM_BUFFERS = 0>
auto build_fabric_mux_connection(
    uint8_t fabric_mux_x,
    uint8_t fabric_mux_y,
    uint8_t fabric_mux_num_buffers_per_channel,
    size_t fabric_mux_channel_buffer_size_bytes,
    size_t fabric_mux_channel_base_address,
    size_t fabric_mux_connection_info_address,
    size_t fabric_mux_connection_handshake_address,
    size_t fabric_mux_flow_control_address,
    size_t fabric_mux_buffer_index_address,
    uint32_t local_flow_control_address,
    uint32_t local_teardown_address,
    uint32_t local_buffer_index_address) -> WorkerToFabricMuxSender<FABRIC_MUX_CHANNEL_NUM_BUFFERS> {
    auto local_flow_control_ptr = reinterpret_cast<volatile uint32_t* const>(local_flow_control_address);
    auto local_teardown_ptr = reinterpret_cast<volatile uint32_t* const>(local_teardown_address);
    return WorkerToFabricMuxSender<FABRIC_MUX_CHANNEL_NUM_BUFFERS>(
        true, /* ignored, connected_to_persistent_fabric */
        fabric_mux_x,
        fabric_mux_y,
        fabric_mux_channel_base_address,
        fabric_mux_num_buffers_per_channel,
        fabric_mux_flow_control_address,
        fabric_mux_connection_handshake_address,
        fabric_mux_connection_info_address,
        fabric_mux_channel_buffer_size_bytes,
        fabric_mux_buffer_index_address,
        local_flow_control_ptr,
        local_teardown_ptr,
        local_buffer_index_address,
        write_reg_cmd_buf,
        write_at_cmd_buf);
}

template <uint8_t FABRIC_MUX_CHANNEL_NUM_BUFFERS = 0>
FORCE_INLINE void fabric_mux_client_connect(
    WorkerToFabricMuxSender<FABRIC_MUX_CHANNEL_NUM_BUFFERS>& connection_handle) {
    connection_handle.open();
}

template <uint8_t FABRIC_MUX_CHANNEL_NUM_BUFFERS = 0>
FORCE_INLINE void fabric_mux_client_disconnect(
    WorkerToFabricMuxSender<FABRIC_MUX_CHANNEL_NUM_BUFFERS>& connection_handle) {
    connection_handle.close();
}

// assumes packet header is correctly populated
template <uint8_t FABRIC_MUX_CHANNEL_NUM_BUFFERS = 0>
FORCE_INLINE void fabric_mux_async_write(
    WorkerToFabricMuxSender<FABRIC_MUX_CHANNEL_NUM_BUFFERS>& connection_handle,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header,
    uint32_t source_payload_address,
    uint32_t packet_payload_size_bytes) {
    connection_handle.wait_for_empty_write_slot();
    connection_handle.send_payload_without_header_non_blocking_from_address(
        source_payload_address, packet_payload_size_bytes);
    connection_handle.send_payload_blocking_from_address((uint32_t)packet_header, sizeof(tt::tt_fabric::PacketHeader));
}

template <uint8_t FABRIC_MUX_CHANNEL_NUM_BUFFERS = 0>
FORCE_INLINE void fabric_mux_atomic_inc(
    WorkerToFabricMuxSender<FABRIC_MUX_CHANNEL_NUM_BUFFERS>& connection_handle,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header) {
    connection_handle.wait_for_empty_write_slot();
    connection_handle.send_payload_flush_non_blocking_from_address(
        (uint32_t)packet_header, sizeof(tt::tt_fabric::PacketHeader));
}

}  // namespace tt::tt_fabric
