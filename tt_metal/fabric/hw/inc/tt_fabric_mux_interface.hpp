// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux.hpp"
#include "tools/profiler/fabric_event_profiler.hpp"

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
        true, /* ignored, connected_to_persistent_fabric */
        fabric_mux_x,
        fabric_mux_y,
        fabric_mux_channel_base_address,
        fabric_mux_num_buffers_per_channel,
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

// Poll the fabric endpoint's status address until it reports READY_FOR_TRAFFIC, or
// until max_poll_iterations elapses.  The bounded default matches
// wait_for_fabric_endpoint_terminated() and prevents a stuck ERISC/MUX from
// permanently wedging the calling Tensix kernel.  On timeout the function falls
// through; the caller (dispatch relay / MUX) will proceed on a best-effort basis
// and host-side reset logic recovers the device if the endpoint never becomes ready.
FORCE_INLINE bool wait_for_fabric_endpoint_ready(
    uint8_t fabric_ep_x,
    uint8_t fabric_ep_y,
    size_t fabric_ep_status_address,
    uint32_t local_fabric_ep_status_address,
    uint32_t max_poll_iterations = 1'000'000) {
    uint64_t noc_addr = get_noc_addr(fabric_ep_x, fabric_ep_y, fabric_ep_status_address);
    auto local_fabric_ep_status_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_fabric_ep_status_address);

    local_fabric_ep_status_ptr[0] = tt::tt_fabric::FabricEndpointStatus::TERMINATED;
    for (uint32_t i = 0; i < max_poll_iterations; ++i) {
        noc_async_read_one_packet(noc_addr, local_fabric_ep_status_address, 4);
        noc_async_read_barrier();
        invalidate_l1_cache();
        if (local_fabric_ep_status_ptr[0] == tt::tt_fabric::FabricEndpointStatus::READY_FOR_TRAFFIC) {
            return true;
        }
    }
    // Fall through on timeout: host-side reset logic recovers. Avoid infinite spin
    // so a stuck peer cannot permanently wedge this kernel.
    return false;
}

template <uint8_t FABRIC_MUX_CHANNEL_NUM_BUFFERS = 0>
FORCE_INLINE void fabric_client_connect(WorkerToFabricMuxSender<FABRIC_MUX_CHANNEL_NUM_BUFFERS>& connection_handle) {
    connection_handle.template open<true>();
}

template <uint8_t FABRIC_MUX_CHANNEL_NUM_BUFFERS = 0>
FORCE_INLINE void fabric_client_connect_start(
    WorkerToFabricMuxSender<FABRIC_MUX_CHANNEL_NUM_BUFFERS>& connection_handle) {
    connection_handle.open_start();
}

template <uint8_t FABRIC_MUX_CHANNEL_NUM_BUFFERS = 0>
FORCE_INLINE void fabric_client_connect_finish(
    WorkerToFabricMuxSender<FABRIC_MUX_CHANNEL_NUM_BUFFERS>& connection_handle) {
    connection_handle.open_finish();
}

template <uint8_t FABRIC_MUX_CHANNEL_NUM_BUFFERS = 0>
FORCE_INLINE void fabric_client_disconnect(WorkerToFabricMuxSender<FABRIC_MUX_CHANNEL_NUM_BUFFERS>& connection_handle) {
    connection_handle.close();
}

FORCE_INLINE void fabric_endpoint_terminate(
    uint8_t fabric_ep_x,
    uint8_t fabric_ep_y,
    size_t fabric_ep_termination_signal_address,
    bool graceful_termination = true) {
    uint64_t noc_addr = get_noc_addr(fabric_ep_x, fabric_ep_y, fabric_ep_termination_signal_address);
    noc_inline_dw_write(
        noc_addr,
        graceful_termination ? tt::tt_fabric::TerminationSignal::GRACEFULLY_TERMINATE
                             : tt::tt_fabric::TerminationSignal::IMMEDIATELY_TERMINATE);
    noc_async_write_barrier();
}

// Poll the mux core's status address until it reports TERMINATED, or until a bounded
// number of iterations elapses.
//
// Must be called after fabric_endpoint_terminate() by the termination master to ensure
// the mux Tensix core has advanced its teardown handshake far enough to publish
// TERMINATED. Note: the mux writes TERMINATED between close_start() and close_finish()
// (see tt_fabric_mux.cpp), so observing TERMINATED here only guarantees that the
// teardown handshake has progressed, NOT that the mux kernel has fully returned.
// Full mux completion is still tracked by normal dispatch completion for that core.
//
// Without this wait, quiesce_devices() / wait_for_completion() can return while the
// mux core is still mid-teardown, causing binary integrity failures when the next
// AllGather reprograms the same cores.
//
// The loop is bounded by kMaxPollIterations so a stuck mux does not hang the writer
// forever; host-side reset logic (RiscFirmwareInitializer::reset_cores,
// teardown_fabric_config) recovers in that case. The default (~1M reads) matches the
// host-side 5s timeout at typical NOC read latencies and is conservative enough that
// a healthy MUX always terminates well within it.
FORCE_INLINE void wait_for_fabric_endpoint_terminated(
    uint8_t fabric_ep_x,
    uint8_t fabric_ep_y,
    size_t fabric_ep_status_address,
    uint32_t local_fabric_ep_status_address,
    uint32_t max_poll_iterations = 1'000'000) {
    uint64_t noc_addr = get_noc_addr(fabric_ep_x, fabric_ep_y, fabric_ep_status_address);
    auto local_fabric_ep_status_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_fabric_ep_status_address);

    local_fabric_ep_status_ptr[0] = tt::tt_fabric::FabricEndpointStatus::READY_FOR_TRAFFIC;
    for (uint32_t i = 0; i < max_poll_iterations; ++i) {
        noc_async_read_one_packet(noc_addr, local_fabric_ep_status_address, 4);
        noc_async_read_barrier();
        invalidate_l1_cache();
        if (local_fabric_ep_status_ptr[0] == tt::tt_fabric::FabricEndpointStatus::TERMINATED) {
            return;
        }
    }
    // Fall through on timeout: host-side ERISC/MUX reset recovers. Avoid infinite spin
    // so a stuck peer cannot permanently wedge the writer kernel.
}

// assumes packet header is correctly populated
template <uint8_t FABRIC_MUX_CHANNEL_NUM_BUFFERS = 0>
FORCE_INLINE void fabric_async_write(
    WorkerToFabricMuxSender<FABRIC_MUX_CHANNEL_NUM_BUFFERS>& connection_handle,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header,
    uint32_t source_payload_address,
    uint32_t packet_payload_size_bytes) {
    connection_handle.wait_for_empty_write_slot();
    RECORD_FABRIC_HEADER(packet_header);
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
    RECORD_FABRIC_HEADER(packet_header);
    connection_handle.send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

}  // namespace tt::tt_fabric
