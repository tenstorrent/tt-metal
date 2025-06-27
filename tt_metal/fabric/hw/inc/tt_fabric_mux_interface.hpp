// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux.hpp"

namespace tt::tt_fabric {

using FabricEndpointStatus = EDMStatus;

template <uint8_t FABRIC_MUX_CHANNEL_NUM_BUFFERS = 0>
WorkerToFabricMuxSender<FABRIC_MUX_CHANNEL_NUM_BUFFERS> build_connection_to_fabric_endpoint(
    uint8_t fabric_mux_x,
    uint8_t fabric_mux_y,
    uint8_t fabric_mux_channel_id,
    uint8_t fabric_mux_num_buffers_per_channel,
    size_t fabric_mux_channel_buffer_size_bytes,
    size_t fabric_mux_channel_base_address,
    size_t fabric_mux_connection_info_address,
    size_t fabric_mux_connection_handshake_address,
    size_t fabric_mux_flow_control_address,
    size_t fabric_mux_buffer_index_address,
    uint32_t local_flow_control_address,
    uint32_t local_teardown_address,
    uint32_t local_buffer_index_address,
    uint32_t direction = 0) {
    auto get_mux_channel_stream_id_from_channel_id = [](uint8_t fabric_mux_channel_id) -> uint32_t {
        return fabric_mux_channel_id;
    };
    auto local_flow_control_ptr = reinterpret_cast<volatile uint32_t* const>(local_flow_control_address);
    auto local_teardown_ptr = reinterpret_cast<volatile uint32_t* const>(local_teardown_address);

    auto mux_channel_credits_stream_id = get_mux_channel_stream_id_from_channel_id(fabric_mux_channel_id);
    return WorkerToFabricMuxSender<FABRIC_MUX_CHANNEL_NUM_BUFFERS>(
        true,      /* ignored, connected_to_persistent_fabric */
        direction, /* ignored, direction */
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
        mux_channel_credits_stream_id,
        StreamId{0},  // my stream id -- As a sender I currently do NOT get acks over stream regs
        write_reg_cmd_buf,
        write_at_cmd_buf);
}

FORCE_INLINE void wait_for_fabric_endpoint_ready(
    uint8_t fabric_ep_x,
    uint8_t fabric_ep_y,
    size_t fabric_ep_status_address,
    uint32_t local_fabric_ep_status_address) {
    uint64_t noc_addr = get_noc_addr(fabric_ep_x, fabric_ep_y, fabric_ep_status_address);
    auto local_fabric_ep_status_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_fabric_ep_status_address);

    local_fabric_ep_status_ptr[0] = tt::tt_fabric::FabricEndpointStatus::TERMINATED;
    while (local_fabric_ep_status_ptr[0] != tt::tt_fabric::FabricEndpointStatus::READY_FOR_TRAFFIC) {
        noc_async_read_one_packet(noc_addr, local_fabric_ep_status_address, 4);
        noc_async_read_barrier();
    }
}

template <uint8_t FABRIC_MUX_CHANNEL_NUM_BUFFERS = 0>
FORCE_INLINE void fabric_client_connect(WorkerToFabricMuxSender<FABRIC_MUX_CHANNEL_NUM_BUFFERS>& connection_handle) {
    connection_handle.open();
}

template <uint8_t FABRIC_MUX_CHANNEL_NUM_BUFFERS = 0>
FORCE_INLINE void fabric_client_disconnect(WorkerToFabricMuxSender<FABRIC_MUX_CHANNEL_NUM_BUFFERS>& connection_handle) {
    connection_handle.close();
}

// assumes packet header is correctly populated
template <uint8_t FABRIC_MUX_CHANNEL_NUM_BUFFERS = 0>
FORCE_INLINE void fabric_async_write(
    WorkerToFabricMuxSender<FABRIC_MUX_CHANNEL_NUM_BUFFERS>& connection_handle,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header,
    uint32_t source_payload_address,
    uint32_t packet_payload_size_bytes) {
    connection_handle.wait_for_empty_write_slot();
    connection_handle.send_payload_without_header_non_blocking_from_address(
        source_payload_address, packet_payload_size_bytes);
    connection_handle.send_payload_flush_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

// assumes packet header is correctly populated
template <uint8_t FABRIC_MUX_CHANNEL_NUM_BUFFERS = 0>
FORCE_INLINE void fabric_atomic_inc(
    WorkerToFabricMuxSender<FABRIC_MUX_CHANNEL_NUM_BUFFERS>& connection_handle,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header) {
    connection_handle.wait_for_empty_write_slot();
    connection_handle.send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

}  // namespace tt::tt_fabric
