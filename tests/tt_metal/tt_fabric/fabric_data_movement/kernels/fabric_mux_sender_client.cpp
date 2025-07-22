// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tt_metal/api/tt-metalium/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric.h" // zero_l1_buf
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_traffic_gen.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp"
// clang-format on

constexpr bool is_2d_fabric = get_compile_time_arg_val(0);
constexpr bool terminate_from_kernel = get_compile_time_arg_val(1);
constexpr bool is_termination_master = get_compile_time_arg_val(2);
constexpr uint8_t fabric_mux_x = get_compile_time_arg_val(3);
constexpr uint8_t fabric_mux_y = get_compile_time_arg_val(4);
constexpr uint8_t fabric_mux_num_buffers_per_channel = get_compile_time_arg_val(5);
constexpr size_t fabric_mux_channel_buffer_size_bytes = get_compile_time_arg_val(6);
constexpr size_t fabric_mux_channel_base_address = get_compile_time_arg_val(7);
constexpr size_t fabric_mux_connection_info_address = get_compile_time_arg_val(8);
constexpr size_t fabric_mux_connection_handshake_address = get_compile_time_arg_val(9);
constexpr size_t fabric_mux_flow_control_address = get_compile_time_arg_val(10);
constexpr size_t fabric_mux_buffer_index_address = get_compile_time_arg_val(11);
constexpr size_t fabric_mux_status_address = get_compile_time_arg_val(12);
constexpr uint8_t fabric_mux_channel_id = get_compile_time_arg_val(13);
constexpr size_t fabric_mux_termination_signal_address = get_compile_time_arg_val(14);

void kernel_main() {
    uint32_t rt_args_idx = 0;
    uint32_t num_open_close_iters = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t num_packets = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t num_credits = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t packet_payload_size_bytes = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t time_seed = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t return_credits_per_packet = get_arg_val<uint32_t>(rt_args_idx++); /* unused for this kernel */
    uint32_t test_results_size_bytes = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t test_results_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t termination_sync_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t local_fabric_mux_status_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t local_flow_control_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t local_teardown_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t local_buffer_index_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t base_l1_target_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t credit_handshake_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t packet_header_buffer_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t payload_buffer_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t num_hops = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t sender_id = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t receiver_noc_xy_encoding = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t num_mux_clients = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t termination_master_noc_xy_encoding = get_arg_val<uint32_t>(rt_args_idx++);

    eth_chan_directions outgoing_direction;
    uint32_t my_device_id, dst_device_id, dst_mesh_id, mesh_ew_dim;
    if constexpr (is_2d_fabric) {
        outgoing_direction = static_cast<eth_chan_directions>(get_arg_val<uint32_t>(rt_args_idx++));
        my_device_id = get_arg_val<uint32_t>(rt_args_idx++);
        dst_device_id = get_arg_val<uint32_t>(rt_args_idx++);
        dst_mesh_id = get_arg_val<uint32_t>(rt_args_idx++);
        mesh_ew_dim = get_arg_val<uint32_t>(rt_args_idx++);
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
    if constexpr (is_2d_fabric) {
        fabric_set_unicast_route(
            (LowLatencyMeshPacketHeader*)packet_header,
            outgoing_direction,
            my_device_id,
            dst_device_id,
            dst_mesh_id,  // Ignored since Low Latency Mesh Fabric is not used for Inter-Mesh Routing
            mesh_ew_dim);
    } else {
        packet_header->to_chip_unicast(static_cast<uint8_t>(num_hops));
    }

    uint64_t local_credit_handshake_noc_address = get_noc_addr(0) + credit_handshake_address;
    auto credit_handshake_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(credit_handshake_address);
    credit_handshake_ptr[0] = num_credits;

    auto payload_start_ptr = reinterpret_cast<tt_l1_ptr uint32_t*>(payload_buffer_address);
    uint64_t base_noc_dest_address = get_noc_addr_helper(receiver_noc_xy_encoding, base_l1_target_address);

    // need to wait for fabric mux to be ready to accept connections
    tt::tt_fabric::wait_for_fabric_endpoint_ready(
        fabric_mux_x, fabric_mux_y, fabric_mux_status_address, local_fabric_mux_status_address);

    for (uint32_t iter = 0; iter < num_open_close_iters; iter++) {
        tt::tt_fabric::fabric_client_connect(mux_connection_handle);

        uint64_t noc_dest_addr = base_noc_dest_address;
        uint32_t seed = time_seed ^ sender_id ^ (iter + 1);
        uint32_t dest_payload_slot_id = 0;
        for (uint32_t packet_id = 0; packet_id < num_packets; packet_id++) {
            // wait until we have atleast 1 credit
            while (credit_handshake_ptr[0] == 0) {
                invalidate_l1_cache();
            }

            seed = prng_next(seed);
            fill_packet_data(payload_start_ptr, packet_payload_size_bytes / 16, seed);

            noc_dest_addr = base_noc_dest_address + (dest_payload_slot_id * packet_payload_size_bytes);
            packet_header->to_noc_unicast_write(NocUnicastCommandHeader{noc_dest_addr}, packet_payload_size_bytes);

            // decrement local credits
            noc_semaphore_inc(local_credit_handshake_noc_address, -1);
            noc_async_atomic_barrier();

            tt::tt_fabric::fabric_async_write(
                mux_connection_handle, packet_header, payload_buffer_address, packet_payload_size_bytes);

            // update the slot id for next packet
            if (++dest_payload_slot_id == num_credits) {
                dest_payload_slot_id = 0;
            }
        }
        noc_async_write_barrier();
        // wait for all credits to be returned before disconnecting
        while (credit_handshake_ptr[0] != num_credits) {
            invalidate_l1_cache();
        }
        tt::tt_fabric::fabric_client_disconnect(mux_connection_handle);
    }

    if constexpr (terminate_from_kernel) {
        if constexpr (is_termination_master) {
            auto* termination_sync_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_sync_address);
            noc_semaphore_wait(termination_sync_ptr, num_mux_clients - 1);
            tt::tt_fabric::fabric_endpoint_terminate(fabric_mux_x, fabric_mux_y, fabric_mux_termination_signal_address);
        } else {
            uint64_t dest_addr = get_noc_addr_helper(termination_master_noc_xy_encoding, termination_sync_address);
            noc_semaphore_inc(dest_addr, 1);
            noc_async_atomic_barrier();
        }
    }

    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_PASS;
}
