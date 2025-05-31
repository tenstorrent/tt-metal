// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
#include "dataflow_api.h"
#include "debug/dprint.h"
#include "debug/assert.h"
#include "tt_metal/api/tt-metalium/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric.h" // zero_l1_buf
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp"
// clang-format on

constexpr uint8_t fabric_mux_x = get_compile_time_arg_val(0);
constexpr uint8_t fabric_mux_y = get_compile_time_arg_val(1);
constexpr uint8_t fabric_mux_num_buffers_per_channel = get_compile_time_arg_val(2);
constexpr size_t fabric_mux_channel_buffer_size_bytes = get_compile_time_arg_val(3);
constexpr size_t fabric_mux_channel_base_address = get_compile_time_arg_val(4);
constexpr size_t fabric_mux_connection_info_address = get_compile_time_arg_val(5);
constexpr size_t fabric_mux_connection_handshake_address = get_compile_time_arg_val(6);
constexpr size_t fabric_mux_flow_control_address = get_compile_time_arg_val(7);
constexpr size_t fabric_mux_buffer_index_address = get_compile_time_arg_val(8);
constexpr size_t fabric_mux_status_address = get_compile_time_arg_val(9);

constexpr bool is_master_sender = get_compile_time_arg_val(10);
constexpr bool is_full_size_channel_sender = get_compile_time_arg_val(11);

// Used to derive the mux's stream register ID for flow control credits (mux side)
constexpr uint8_t fabric_mux_channel_id = get_compile_time_arg_val(12);

constexpr uint8_t num_hops = 1;

void kernel_main() {
    uint32_t rt_args_idx = 0;
    uint32_t num_packets = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t packet_payload_size_bytes = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t test_results_size_bytes = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t test_results_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t senders_sync_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t local_fabric_mux_status_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t local_flow_control_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t local_teardown_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t local_buffer_index_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t target_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t packet_header_buffer_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t payload_buffer_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t receiver_noc_xy_encoding = get_arg_val<uint32_t>(rt_args_idx++);

    uint8_t drainer_x, drainer_y;
    uint32_t drainer_status_address, mcast_encoding, num_mcast_dests;
    if constexpr (is_master_sender) {
        drainer_x = static_cast<uint8_t>(get_arg_val<uint32_t>(rt_args_idx++));
        drainer_y = static_cast<uint8_t>(get_arg_val<uint32_t>(rt_args_idx++));
        drainer_status_address = get_arg_val<uint32_t>(rt_args_idx++);
        mcast_encoding = get_arg_val<uint32_t>(rt_args_idx++);
        num_mcast_dests = get_arg_val<uint32_t>(rt_args_idx++);
    }

    auto test_results = reinterpret_cast<tt_l1_ptr uint32_t*>(test_results_address);
    zero_l1_buf(test_results, test_results_size_bytes);
    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_STARTED;

    auto mux_connection_handle = tt::tt_fabric::build_connection_to_fabric_endpoint<fabric_mux_num_buffers_per_channel>(
        fabric_mux_x,
        fabric_mux_y,
        fabric_mux_channel_id,
        fabric_mux_num_buffers_per_channel,
        fabric_mux_channel_buffer_size_bytes,
        fabric_mux_channel_base_address,
        fabric_mux_connection_info_address,
        fabric_mux_connection_handshake_address,
        fabric_mux_flow_control_address,
        fabric_mux_buffer_index_address,
        local_flow_control_address,
        local_teardown_address,
        local_buffer_index_address);

    auto packet_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(packet_header_buffer_address);
    uint64_t noc_dest_address = get_noc_addr_helper(receiver_noc_xy_encoding, target_address);
    packet_header->to_chip_unicast(static_cast<uint8_t>(num_hops));
    if constexpr (is_full_size_channel_sender) {
        packet_header->to_noc_unicast_write(NocUnicastCommandHeader{noc_dest_address}, packet_payload_size_bytes);
    } else {
        packet_header->to_noc_unicast_atomic_inc(
            NocUnicastAtomicIncCommandHeader{noc_dest_address, 1, std::numeric_limits<uint16_t>::max()});
    }

    tt::tt_fabric::wait_for_fabric_endpoint_ready(
        fabric_mux_x, fabric_mux_y, fabric_mux_status_address, local_fabric_mux_status_address);

    if constexpr (is_master_sender) {
        // wait for drainer kernel as well, we need to ensure that the connection b/w mux and drainer
        // is established before we send packets
        tt::tt_fabric::wait_for_fabric_endpoint_ready(
            drainer_x, drainer_y, drainer_status_address, local_fabric_mux_status_address);

        auto senders_sync_ptr = reinterpret_cast<tt_l1_ptr uint32_t*>(senders_sync_address);
        senders_sync_ptr[0] = 1;

        uint64_t mcast_dest_addr = get_noc_addr_helper(mcast_encoding, senders_sync_address);
        noc_async_write_multicast_loopback_src(
            senders_sync_address, mcast_dest_addr, sizeof(uint32_t), num_mcast_dests);
    } else {
        auto senders_sync_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(senders_sync_address);
        while (senders_sync_ptr[0] == 0);
    }

    tt::tt_fabric::fabric_client_connect(mux_connection_handle);

    uint64_t start_timestamp = get_timestamp();
    for (uint32_t packet_id = 0; packet_id < num_packets; packet_id++) {
        if constexpr (is_full_size_channel_sender) {
            tt::tt_fabric::fabric_async_write(
                mux_connection_handle, packet_header, payload_buffer_address, packet_payload_size_bytes);
        } else {
            tt::tt_fabric::fabric_atomic_inc(mux_connection_handle, packet_header);
        }
    }
    uint64_t cycles_elapsed = get_timestamp() - start_timestamp;

    tt::tt_fabric::fabric_client_disconnect(mux_connection_handle);
    noc_async_write_barrier();

    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_PASS;
    test_results[TT_FABRIC_CYCLES_INDEX] = (uint32_t)cycles_elapsed;
    test_results[TT_FABRIC_CYCLES_INDEX + 1] = cycles_elapsed >> 32;
}
