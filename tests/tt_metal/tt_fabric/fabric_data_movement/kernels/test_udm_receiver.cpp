// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
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

void kernel_main() {
    // TODO: move this into fw once consolidated
    tt::tt_fabric::udm::fabric_local_state_init();

    uint32_t time_seed = time_seed_init;

    int32_t dest_bank_id;
    uint32_t dest_dram_addr;

    tt_l1_ptr uint32_t* start_addr = reinterpret_cast<tt_l1_ptr uint32_t*>(target_address);
    uint32_t mismatch_addr, mismatch_val, expected_val;
    bool match = true;
    uint64_t bytes_received = 0;

    uint32_t notification_addr = notification_mailbox_address;  // Where we receive notifications

    zero_l1_buf(test_results, test_results_size_bytes);
    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_STARTED;

    // Process packets
    for (uint32_t i = 0; i < num_packets; i++) {
        time_seed = prng_next(time_seed);

        // Wait for the notification from sender
        uint32_t curr_notification_addr = notification_addr + i * req_notification_size_bytes;
        volatile tt_l1_ptr PACKET_HEADER_TYPE* received_header =
            wait_for_notification(curr_notification_addr, time_seed, req_notification_size_bytes);

        // Send write ACK back to the sender
        tt::tt_fabric::udm::fabric_fast_write_ack(received_header);

        // Check for data correctness
        match = check_packet_data(
            start_addr, packet_payload_size_bytes / 16, time_seed, mismatch_addr, mismatch_val, expected_val);
        if (!match) {
            break;
        }
        start_addr += packet_payload_size_bytes / 4;
        bytes_received += packet_payload_size_bytes;
    }

    // TODO: move this into fw once consolidated
    tt::tt_fabric::udm::close_fabric_connection();

    if (!match) {
        test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_DATA_MISMATCH;
        test_results[TT_FABRIC_MISC_INDEX + 12] = mismatch_addr;
        test_results[TT_FABRIC_MISC_INDEX + 13] = mismatch_val;
        test_results[TT_FABRIC_MISC_INDEX + 14] = expected_val;
    } else {
        test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_PASS;
    }

    test_results[TT_FABRIC_WORD_CNT_INDEX] = (uint32_t)bytes_received;
    test_results[TT_FABRIC_WORD_CNT_INDEX + 1] = bytes_received >> 32;
}
