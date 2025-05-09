// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
#include <limits>
#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tt_metal/api/tt-metalium/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric.h" // zero_l1_buf
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_traffic_gen.hpp"
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

void kernel_main() {
    uint32_t rt_args_idx = 0;
    uint32_t num_open_close_iters = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t num_packets = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t num_credits = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t packet_payload_size_bytes = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t time_seed = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t return_credits_per_packet = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t test_results_size_bytes = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t test_results_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t local_fabric_mux_status_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t local_flow_control_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t local_teardown_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t local_buffer_index_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t base_l1_target_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t credit_handshake_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t packet_header_buffer_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t payload_buffer_address = get_arg_val<uint32_t>(rt_args_idx++); /* unused for this kernel */
    uint32_t num_hops = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t sender_id = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t sender_noc_xy_encoding = get_arg_val<uint32_t>(rt_args_idx++);

    auto test_results = reinterpret_cast<tt_l1_ptr uint32_t*>(test_results_address);
    zero_l1_buf(test_results, test_results_size_bytes);
    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_STARTED;

    auto mux_connection_handle = tt::tt_fabric::build_connection_to_fabric_endpoint<fabric_mux_num_buffers_per_channel>(
        fabric_mux_x,
        fabric_mux_y,
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

    uint64_t noc_dest_addr = get_noc_addr_helper(sender_noc_xy_encoding, credit_handshake_address);
    auto packet_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(packet_header_buffer_address);
    packet_header->to_chip_unicast(static_cast<uint8_t>(num_hops));

    auto base_payload_start_ptr = reinterpret_cast<tt_l1_ptr uint32_t*>(base_l1_target_address);
    auto base_poll_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base_l1_target_address + packet_payload_size_bytes - 4);

    bool match = true;
    uint32_t num_packets_processed = 0;
    uint32_t mismatch_addr = 0;
    uint32_t mismatch_val = 0;
    uint32_t expected_val = 0;

    // need to wait for fabric mux to be ready to accept connections
    tt::tt_fabric::wait_for_fabric_endpoint_ready(
        fabric_mux_x, fabric_mux_y, fabric_mux_status_address, local_fabric_mux_status_address);

    for (uint32_t iter = 0; iter < num_open_close_iters; iter++) {
        tt::tt_fabric::fabric_client_connect(mux_connection_handle);

        packet_header->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{
            noc_dest_addr, static_cast<uint16_t>(return_credits_per_packet), std::numeric_limits<uint16_t>::max()});

        tt_l1_ptr uint32_t* payload_start_ptr = base_payload_start_ptr;
        volatile tt_l1_ptr uint32_t* poll_ptr = base_poll_ptr;
        uint32_t seed = time_seed ^ sender_id ^ (iter + 1);
        uint32_t slot_id = 0;
        uint32_t num_accumulated_credits = 0;
        for (uint32_t packet_id = 0; packet_id < num_packets; packet_id++) {
            seed = prng_next(seed);
            expected_val = seed + (packet_payload_size_bytes / 16) - 1;

            uint32_t ptr_offset = slot_id * (packet_payload_size_bytes / 4);
            payload_start_ptr = base_payload_start_ptr + ptr_offset;
            poll_ptr = base_poll_ptr + ptr_offset;

            // poll on the last word in the payload -> this ensures that the entire payload is written
            while (*poll_ptr != expected_val);

            // check for data correctness
            match = check_packet_data(
                payload_start_ptr, packet_payload_size_bytes / 16, seed, mismatch_addr, mismatch_val, expected_val);
            if (!match) {
                break;
            }

            if (++slot_id == num_credits) {
                slot_id = 0;
            }

            if (++num_accumulated_credits == return_credits_per_packet) {
                num_accumulated_credits = 0;
                tt::tt_fabric::fabric_atomic_inc(mux_connection_handle, packet_header);
            }

            num_packets_processed++;
        }

        // return any unsent credits before disconnecting
        if (num_accumulated_credits > 0) {
            packet_header->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{
                noc_dest_addr, static_cast<uint16_t>(num_accumulated_credits), std::numeric_limits<uint16_t>::max()});
            tt::tt_fabric::fabric_atomic_inc(mux_connection_handle, packet_header);
            num_accumulated_credits = 0;
        }
        noc_async_write_barrier();
        tt::tt_fabric::fabric_client_disconnect(mux_connection_handle);

        if (!match) {
            break;
        }
    }

    if (match) {
        test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_PASS;
    } else {
        test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_DATA_MISMATCH;
        test_results[TT_FABRIC_MISC_INDEX + 12] = mismatch_addr;
        test_results[TT_FABRIC_MISC_INDEX + 13] = mismatch_val;
        test_results[TT_FABRIC_MISC_INDEX + 14] = expected_val;
    }
    test_results[TX_TEST_IDX_NPKT] = num_packets_processed;
}
