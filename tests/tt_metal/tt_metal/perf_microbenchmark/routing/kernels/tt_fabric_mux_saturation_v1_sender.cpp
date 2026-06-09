// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_traffic_gen.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"

constexpr uint32_t test_results_address = get_compile_time_arg_val(0);
constexpr uint32_t test_results_size_bytes = get_compile_time_arg_val(1);
constexpr uint32_t start_signal_address = get_compile_time_arg_val(2);
constexpr uint32_t ready_count_address = get_compile_time_arg_val(3);
constexpr uint32_t local_fabric_mux_status_address = get_compile_time_arg_val(4);
constexpr uint32_t local_flow_control_address = get_compile_time_arg_val(5);
constexpr uint32_t local_teardown_address = get_compile_time_arg_val(6);
constexpr uint32_t local_buffer_index_address = get_compile_time_arg_val(7);
constexpr bool is_master_sender = get_compile_time_arg_val(8) != 0;
constexpr bool is_2d_fabric = get_compile_time_arg_val(9) != 0;
constexpr uint8_t fabric_mux_x = get_compile_time_arg_val(10);
constexpr uint8_t fabric_mux_y = get_compile_time_arg_val(11);
constexpr uint8_t fabric_mux_num_buffers_per_channel = get_compile_time_arg_val(12);
constexpr size_t fabric_mux_channel_buffer_size_bytes = get_compile_time_arg_val(13);
constexpr size_t fabric_mux_channel_base_address = get_compile_time_arg_val(14);
constexpr size_t fabric_mux_connection_info_address = get_compile_time_arg_val(15);
constexpr size_t fabric_mux_connection_handshake_address = get_compile_time_arg_val(16);
constexpr size_t fabric_mux_flow_control_address = get_compile_time_arg_val(17);
constexpr size_t fabric_mux_buffer_index_address = get_compile_time_arg_val(18);
constexpr size_t fabric_mux_status_address = get_compile_time_arg_val(19);
constexpr uint8_t fabric_mux_channel_id = get_compile_time_arg_val(20);

namespace {

bool reached_start_cycle(uint32_t start_cycle) {
    return static_cast<int32_t>(static_cast<uint32_t>(get_timestamp()) - start_cycle) >= 0;
}

}  // namespace

void kernel_main() {
    size_t arg_idx = 0;
    const uint32_t packet_header_buffer_address = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t payload_buffer_address = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t packet_payload_size_bytes = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_packets = get_arg_val<uint32_t>(arg_idx++);
    uint32_t seed = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_delay_cycles = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t target_noc_x = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
    const uint8_t target_noc_y = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
    const uint32_t target_address = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t linear_num_hops = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
    const uint8_t dst_device_id = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
    const uint16_t dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(arg_idx++));
    const uint32_t master_sender_noc_xy_encoding = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t expected_ready_count = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_peer_senders = get_arg_val<uint32_t>(arg_idx++);
    const size_t peer_sender_noc_xy_encodings_arg_idx = arg_idx;
    arg_idx += num_peer_senders;

    auto start_signal_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(start_signal_address);
    auto ready_count_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_count_address);

    auto test_results = reinterpret_cast<tt_l1_ptr uint32_t*>(test_results_address);
    zero_l1_buf(test_results, test_results_size_bytes);
    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_STARTED;

    auto packet_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(packet_header_buffer_address);
    zero_l1_buf(reinterpret_cast<tt_l1_ptr uint32_t*>(packet_header_buffer_address), sizeof(PACKET_HEADER_TYPE));

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

    if constexpr (is_2d_fabric) {
        fabric_set_unicast_route((HybridMeshPacketHeader*)packet_header, dst_device_id, dst_mesh_id);
    } else {
        fabric_set_unicast_route<false>((LowLatencyPacketHeader*)packet_header, linear_num_hops);
    }
    packet_header->to_noc_unicast_write(
        NocUnicastCommandHeader{get_noc_addr(target_noc_x, target_noc_y, target_address)}, packet_payload_size_bytes);

    tt::tt_fabric::wait_for_fabric_endpoint_ready(
        fabric_mux_x, fabric_mux_y, fabric_mux_status_address, local_fabric_mux_status_address);

    if constexpr (is_master_sender) {
        noc_semaphore_wait(ready_count_ptr, expected_ready_count);
        start_signal_ptr[0] = static_cast<uint32_t>(get_timestamp()) + start_delay_cycles;
    } else {
        const uint64_t master_ready_count_noc_addr =
            get_noc_addr_helper(master_sender_noc_xy_encoding, ready_count_address);
        noc_semaphore_inc(master_ready_count_noc_addr, 1);
    }

    for (uint32_t peer_idx = 0; peer_idx < num_peer_senders; ++peer_idx) {
        const auto peer_noc_xy_encoding = get_arg_val<uint32_t>(peer_sender_noc_xy_encodings_arg_idx + peer_idx);
        if constexpr (is_master_sender) {
            const uint64_t peer_start_signal_noc_addr = get_noc_addr_helper(peer_noc_xy_encoding, start_signal_address);
            noc_async_write(start_signal_address, peer_start_signal_noc_addr, sizeof(uint32_t));
        }
    }

    if constexpr (is_master_sender) {
        noc_async_write_barrier();
    } else {
        while (start_signal_ptr[0] == 0) {
            invalidate_l1_cache();
        }
    }

    const uint32_t start_cycle = start_signal_ptr[0];
    while (!reached_start_cycle(start_cycle)) {
    }

    auto payload_start_ptr = reinterpret_cast<tt_l1_ptr uint32_t*>(payload_buffer_address);

    uint64_t bytes_sent = 0;
    const uint64_t start_timestamp = get_timestamp();
    tt::tt_fabric::fabric_client_connect(mux_connection_handle);

    for (uint32_t packet_idx = 0; packet_idx < num_packets; ++packet_idx) {
        seed = prng_next(seed);
        fill_packet_data(payload_start_ptr, packet_payload_size_bytes / 16, seed);
        tt::tt_fabric::fabric_async_write(
            mux_connection_handle, packet_header, payload_buffer_address, packet_payload_size_bytes);
        bytes_sent += packet_payload_size_bytes;
    }

    tt::tt_fabric::fabric_client_disconnect(mux_connection_handle);
    noc_async_write_barrier();
    const uint64_t cycles_elapsed = get_timestamp() - start_timestamp;

    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_PASS;
    test_results[TT_FABRIC_WORD_CNT_INDEX] = static_cast<uint32_t>(bytes_sent);
    test_results[TT_FABRIC_WORD_CNT_INDEX + 1] = static_cast<uint32_t>(bytes_sent >> 32);
    test_results[TT_FABRIC_CYCLES_INDEX] = static_cast<uint32_t>(cycles_elapsed);
    test_results[TT_FABRIC_CYCLES_INDEX + 1] = static_cast<uint32_t>(cycles_elapsed >> 32);
}
