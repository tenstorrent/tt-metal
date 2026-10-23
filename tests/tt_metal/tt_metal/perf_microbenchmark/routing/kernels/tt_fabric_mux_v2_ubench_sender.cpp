// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_v2_sender.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"

constexpr uint32_t test_results_address = get_compile_time_arg_val(0);
constexpr uint32_t test_results_size_bytes = get_compile_time_arg_val(1);
constexpr uint32_t start_signal_address = get_compile_time_arg_val(2);
constexpr uint32_t ready_count_address = get_compile_time_arg_val(3);
constexpr uint32_t local_poll_scratch_address = get_compile_time_arg_val(4);
constexpr bool is_master_sender = get_compile_time_arg_val(5) != 0;

namespace {

bool reached_start_cycle(uint32_t start_cycle) {
    return static_cast<int32_t>(static_cast<uint32_t>(get_timestamp()) - start_cycle) >= 0;
}

}  // namespace

void kernel_main() {
    size_t arg_idx = 0;
    const uint32_t num_packets = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t packet_payload_size_bytes = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t packet_header_buffer_address = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t dummy_target_address = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t dummy_receiver_noc_xy_encoding = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_delay_cycles = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t drainer_noc_x = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
    const uint8_t drainer_noc_y = static_cast<uint8_t>(get_arg_val<uint32_t>(arg_idx++));
    const uint32_t drainer_status_address = get_arg_val<uint32_t>(arg_idx++);
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

    // All senders rendezvous here so the master only releases a future start cycle
    // after every sender has finished local setup and reached the barrier.
    if constexpr (is_master_sender) {
        noc_semaphore_wait(ready_count_ptr, expected_ready_count);
        tt::tt_fabric::wait_for_fabric_endpoint_ready(
            drainer_noc_x, drainer_noc_y, drainer_status_address, local_poll_scratch_address);
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

    sender.open();

    const uint64_t dummy_noc_dest_address = get_noc_addr_helper(dummy_receiver_noc_xy_encoding, dummy_target_address);
    const auto dummy_command_header = tt::tt_fabric::NocUnicastCommandHeader{dummy_noc_dest_address};
    constexpr uint8_t num_hops = 0;
    packet_header->to_chip_unicast(num_hops);
    packet_header->to_noc_unicast_write(dummy_command_header, packet_payload_size_bytes);
    sender.setup_stateful_send_cmd_bufs</*posted=*/false>();

    uint64_t start_timestamp = get_timestamp();
    uint32_t cached_free_write_slots = 0;

    for (uint32_t packet_idx = 0; packet_idx < num_packets; ++packet_idx) {
        while (cached_free_write_slots == 0) {
            cached_free_write_slots = sender.get_num_free_write_slots();
        }

        sender.send_current_slot_stateful_non_blocking_from_address</*posted=*/false>(
            packet_header_buffer_address, sizeof(PACKET_HEADER_TYPE));
        cached_free_write_slots--;
    }

    sender.close();
    const uint64_t cycles_elapsed = get_timestamp() - start_timestamp;
    const uint64_t bytes_sent = static_cast<uint64_t>(num_packets) * packet_payload_size_bytes;

    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_PASS;
    test_results[TT_FABRIC_WORD_CNT_INDEX] = static_cast<uint32_t>(bytes_sent);
    test_results[TT_FABRIC_WORD_CNT_INDEX + 1] = static_cast<uint32_t>(bytes_sent >> 32);
    test_results[TT_FABRIC_CYCLES_INDEX] = static_cast<uint32_t>(cycles_elapsed);
    test_results[TT_FABRIC_CYCLES_INDEX + 1] = static_cast<uint32_t>(cycles_elapsed >> 32);

    noc_async_write_barrier();
    noc_async_atomic_barrier();

    DEVICE_PRINT("Sender exiting\n");
}
