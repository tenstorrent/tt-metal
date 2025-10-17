// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/common/kernel_utils.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_traffic_gen.hpp"
#include "fabric/fabric_edm_packet_header.hpp"

constexpr uint32_t test_results_addr_arg = get_compile_time_arg_val(0);
constexpr uint32_t test_results_size_bytes = get_compile_time_arg_val(1);
tt_l1_ptr uint32_t* const test_results = reinterpret_cast<tt_l1_ptr uint32_t*>(test_results_addr_arg);
constexpr uint32_t notification_mailbox_address = get_compile_time_arg_val(2);
uint32_t target_address = get_compile_time_arg_val(3);
constexpr NocSendType noc_send_type = static_cast<NocSendType>(get_compile_time_arg_val(4));
constexpr uint32_t num_send_dir = get_compile_time_arg_val(5);
constexpr bool with_state = get_compile_time_arg_val(6) == 1;
constexpr bool is_chip_multicast = get_compile_time_arg_val(7) == 1;

void kernel_main() {
    uint32_t rt_arg_idx = 0;
    uint16_t packet_payload_size_bytes = static_cast<uint16_t>(get_arg_val<uint32_t>(rt_arg_idx++));
    uint32_t num_packets = get_arg_val<uint32_t>(rt_arg_idx++);
    uint32_t time_seed = get_arg_val<uint32_t>(rt_arg_idx++);

    int32_t dest_bank_id;
    uint32_t dest_dram_addr;
    uint32_t notification_mailbox_address;

    tt_l1_ptr uint32_t* start_addr = reinterpret_cast<tt_l1_ptr uint32_t*>(target_address);
    volatile tt_l1_ptr uint32_t* poll_addr =
        reinterpret_cast<tt_l1_ptr uint32_t*>(target_address + packet_payload_size_bytes - 4);
    uint32_t mismatch_addr, mismatch_val, expected_val;
    bool match = true;
    uint64_t bytes_received = 0;

    zero_l1_buf(test_results, test_results_size_bytes);
    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_STARTED;

    // poll for data
    for (uint32_t i = 0; i < num_packets; i++) {
        time_seed = prng_next(time_seed);
        uint32_t expected_val = time_seed + (packet_payload_size_bytes / 16) - 1;
        if constexpr (noc_send_type == NOC_UNICAST_INLINE_WRITE) {
            expected_val = 0xDEADBEEF;
        }

        WAYPOINT("FPW");
        while (expected_val != *poll_addr) {
            invalidate_l1_cache();
        }
        WAYPOINT("FPD");

        // check for data correctness
        match = check_packet_data(
            start_addr, packet_payload_size_bytes / 16, time_seed, mismatch_addr, mismatch_val, expected_val);
        if (!match) {
            break;
        }
        start_addr += packet_payload_size_bytes / 4;
        poll_addr += packet_payload_size_bytes / 4;
        bytes_received += packet_payload_size_bytes;
    }

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
