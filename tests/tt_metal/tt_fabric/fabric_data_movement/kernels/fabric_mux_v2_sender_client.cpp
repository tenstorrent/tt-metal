// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_traffic_gen.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#if defined(FABRIC_2D)
#include "tt_metal/fabric/hw/inc/mesh/api.h"
#endif
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_v2_sender.hpp"
#include "api/debug/device_print.h"

constexpr uint32_t test_results_address = get_compile_time_arg_val(0);
constexpr uint32_t test_results_size_bytes = get_compile_time_arg_val(1);
constexpr uint32_t receiver_slots_base_address = get_compile_time_arg_val(2);
constexpr uint32_t credit_handshake_address = get_compile_time_arg_val(3);
constexpr bool is_2d_fabric = get_compile_time_arg_val(4) != 0;

void kernel_main() {
    size_t arg_idx = 0;
    const uint32_t packet_header_buffer_address = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t payload_buffer_address = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t packet_payload_size_bytes = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_packets = get_arg_val<uint32_t>(arg_idx++);
    uint32_t seed = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t receiver_noc_x = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
    const uint8_t receiver_noc_y = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
    const uint32_t num_receiver_slots = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t num_hops = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
    const uint8_t dst_device_id = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
    const uint16_t dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(arg_idx++));

    auto sender = tt::tt_fabric::FabricMuxV2Sender<>::build_from_args(arg_idx);

    auto test_results = reinterpret_cast<tt_l1_ptr uint32_t*>(test_results_address);
    zero_l1_buf(test_results, test_results_size_bytes);
    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_STARTED;

    auto packet_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(packet_header_buffer_address);
    zero_l1_buf(reinterpret_cast<tt_l1_ptr uint32_t*>(packet_header_buffer_address), sizeof(PACKET_HEADER_TYPE));

    auto credit_handshake_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(credit_handshake_address);
    credit_handshake_ptr[0] = num_receiver_slots;
    const uint64_t local_credit_handshake_noc_addr = get_noc_addr(0) + credit_handshake_address;

    sender.open();

    uint64_t bytes_sent = 0;
    auto payload_start_ptr = reinterpret_cast<tt_l1_ptr uint32_t*>(payload_buffer_address);
    uint32_t receiver_slot_id = 0;
    for (uint32_t packet_idx = 0; packet_idx < num_packets; ++packet_idx) {
        while (credit_handshake_ptr[0] == 0) {
            invalidate_l1_cache();
        }

        seed = prng_next(seed);
        fill_packet_data(payload_start_ptr, packet_payload_size_bytes / 16, seed);

        const uint64_t dest_noc_addr = get_noc_addr(
            receiver_noc_x,
            receiver_noc_y,
            receiver_slots_base_address + (receiver_slot_id * packet_payload_size_bytes));
        const auto dest_command_header = tt::tt_fabric::NocUnicastCommandHeader{dest_noc_addr};

        noc_semaphore_inc(local_credit_handshake_noc_addr, -1);
        noc_async_atomic_barrier();

#if defined(FABRIC_2D)
        if constexpr (is_2d_fabric) {
            tt::tt_fabric::mesh::experimental::fabric_unicast_noc_unicast_write(
                &sender,
                packet_header,
                dst_device_id,
                dst_mesh_id,
                payload_buffer_address,
                packet_payload_size_bytes,
                dest_command_header);
        } else
#endif
        {
            tt::tt_fabric::linear::experimental::fabric_unicast_noc_unicast_write(
                &sender,
                packet_header,
                payload_buffer_address,
                packet_payload_size_bytes,
                dest_command_header,
                num_hops);
        }
        bytes_sent += packet_payload_size_bytes;

        receiver_slot_id += 1;
        if (receiver_slot_id == num_receiver_slots) {
            receiver_slot_id = 0;
        }
        noc_async_writes_flushed();
    }

    noc_async_write_barrier();
    while (credit_handshake_ptr[0] != num_receiver_slots) {
        invalidate_l1_cache();
    }

    sender.close();

    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_PASS;
    test_results[TT_FABRIC_WORD_CNT_INDEX] = static_cast<uint32_t>(bytes_sent);
    test_results[TT_FABRIC_WORD_CNT_INDEX + 1] = static_cast<uint32_t>(bytes_sent >> 32);
    test_results[TX_TEST_IDX_NPKT] = num_packets;
}
