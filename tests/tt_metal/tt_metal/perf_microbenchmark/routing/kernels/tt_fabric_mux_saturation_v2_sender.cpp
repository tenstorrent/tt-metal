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
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_v2_sender.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"

constexpr uint32_t test_results_address = get_compile_time_arg_val(0);
constexpr uint32_t test_results_size_bytes = get_compile_time_arg_val(1);
constexpr uint32_t start_signal_address = get_compile_time_arg_val(2);
constexpr uint32_t ready_count_address = get_compile_time_arg_val(3);
constexpr bool is_master_sender = get_compile_time_arg_val(4) != 0;
constexpr bool is_2d_fabric = get_compile_time_arg_val(5) != 0;

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

    auto sender = tt::tt_fabric::FabricMuxV2Sender<>::build_from_args(arg_idx);

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

    const uint64_t dest_noc_addr = get_noc_addr(target_noc_x, target_noc_y, target_address);
    const auto dest_command_header = tt::tt_fabric::NocUnicastCommandHeader{dest_noc_addr};
    auto payload_start_ptr = reinterpret_cast<tt_l1_ptr uint32_t*>(payload_buffer_address);

    uint64_t bytes_sent = 0;
    const uint64_t start_timestamp = get_timestamp();
    sender.open();

    for (uint32_t packet_idx = 0; packet_idx < num_packets; ++packet_idx) {
        seed = prng_next(seed);
        fill_packet_data(payload_start_ptr, packet_payload_size_bytes / 16, seed);

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
                linear_num_hops);
        }
        bytes_sent += packet_payload_size_bytes;
    }

    noc_async_write_barrier();
    sender.close();
    const uint64_t cycles_elapsed = get_timestamp() - start_timestamp;

    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_PASS;
    test_results[TT_FABRIC_WORD_CNT_INDEX] = static_cast<uint32_t>(bytes_sent);
    test_results[TT_FABRIC_WORD_CNT_INDEX + 1] = static_cast<uint32_t>(bytes_sent >> 32);
    test_results[TT_FABRIC_CYCLES_INDEX] = static_cast<uint32_t>(cycles_elapsed);
    test_results[TT_FABRIC_CYCLES_INDEX + 1] = static_cast<uint32_t>(cycles_elapsed >> 32);
}
