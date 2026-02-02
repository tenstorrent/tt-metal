// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/common/kernel_utils.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/test_udm_utils.hpp"
#include "tt_metal/fabric/fabric_edm_packet_header.hpp"
#include <type_traits>

constexpr uint32_t test_results_addr_arg = get_compile_time_arg_val(0);
constexpr uint32_t test_results_size_bytes = get_compile_time_arg_val(1);
tt_l1_ptr uint32_t* const test_results = reinterpret_cast<tt_l1_ptr uint32_t*>(test_results_addr_arg);
constexpr uint32_t notification_mailbox_address = get_compile_time_arg_val(2);
uint32_t target_address = get_compile_time_arg_val(3);
constexpr NocSendType noc_send_type = static_cast<NocSendType>(get_compile_time_arg_val(4));
constexpr uint16_t packet_payload_size_bytes = static_cast<uint16_t>(get_compile_time_arg_val(5));
constexpr uint32_t num_packets = get_compile_time_arg_val(6);
constexpr uint32_t time_seed_init = get_compile_time_arg_val(7);
constexpr uint32_t req_notification_size_bytes = get_compile_time_arg_val(8);
constexpr uint32_t dst_dev_id = get_compile_time_arg_val(9);
constexpr uint32_t dst_mesh_id = get_compile_time_arg_val(10);

/*
 * This test kernel is a kernel to test the functionality that will be implemented in a fabric relay kernel.
 * The relay kernel would handle read/write acknowledgement and related functionality, so  the building
 * blocks can be tested without requiring full integration/deployment into fabric
 */
void kernel_main() {
    // Per-core sender coordinates from runtime args
    uint32_t arg_index = 0;
    uint32_t noc_x_start = get_arg_val<uint32_t>(arg_index++);
    uint32_t noc_y_start = get_arg_val<uint32_t>(arg_index);

    uint32_t time_seed = time_seed_init;

    zero_l1_buf(test_results, test_results_size_bytes);
    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_STARTED;

    uint32_t local_data_addr = target_address;
    uint32_t local_notification_buffer_addr = notification_mailbox_address;  // Local buffer for preparing notification
    uint32_t remote_notification_dest_addr =
        notification_mailbox_address;  // Remote destination (same offset on sender)
    uint64_t bytes_sent = 0;

    // Fill all packets with data first
    for (uint32_t i = 0; i < num_packets; i++) {
        time_seed = prng_next(time_seed);

        uint32_t curr_local_data_addr = local_data_addr + (i * packet_payload_size_bytes);
        tt_l1_ptr uint32_t* buffer_addr = reinterpret_cast<tt_l1_ptr uint32_t*>(curr_local_data_addr);
        fill_packet_data(buffer_addr, packet_payload_size_bytes / 16, time_seed);

        bytes_sent += packet_payload_size_bytes;
    }

    // Once all data has been filled in L1, notify the sender that it can issue read requests
    notify_receiver(
        dst_dev_id,
        dst_mesh_id,
        noc_x_start,
        noc_y_start,
        local_notification_buffer_addr,
        remote_notification_dest_addr,
        time_seed_init,
        req_notification_size_bytes);

    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_PASS;
    test_results[TT_FABRIC_WORD_CNT_INDEX] = (uint32_t)bytes_sent;
    test_results[TT_FABRIC_WORD_CNT_INDEX + 1] = bytes_sent >> 32;
}
