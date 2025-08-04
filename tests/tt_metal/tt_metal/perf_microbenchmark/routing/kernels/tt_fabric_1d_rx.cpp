// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/common/kernel_utils.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tt_metal/fabric/hw/inc/tt_fabric.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_traffic_gen.hpp"

// clang-format on

constexpr uint32_t test_results_addr_arg = get_compile_time_arg_val(0);
constexpr uint32_t test_results_size_bytes = get_compile_time_arg_val(1);
tt_l1_ptr uint32_t* const test_results = reinterpret_cast<tt_l1_ptr uint32_t*>(test_results_addr_arg);

constexpr uint32_t target_address = get_compile_time_arg_val(2);
constexpr bool use_dram_dst = get_compile_time_arg_val(3);

void kernel_main() {
    //DPRINT << "=== RX KERNEL START ===" << ENDL();
    
    uint32_t rt_args_idx = 0;
    int32_t dest_bank_id = 0;
    uint32_t dest_dram_addr = 0;
    uint32_t notification_mailbox_address = 0;
    uint32_t packet_payload_size_bytes = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t num_packets = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t time_seed = get_arg_val<uint32_t>(rt_args_idx++);

    //DPRINT << "RX ARGS: packet_size=" << packet_payload_size_bytes << " num_packets=" << num_packets << " seed=" << HEX() << time_seed << ENDL();
    //DPRINT << "RX ARGS: packet_size=" << packet_payload_size_bytes << " num_packets=" << num_packets << " seed=" << HEX() << time_seed << ENDL();
    //DPRINT << "RX ARGS: packet_size=" << packet_payload_size_bytes << " num_packets=" << num_packets << " seed=" << HEX() << time_seed << ENDL();
    if constexpr (use_dram_dst) {
        dest_bank_id = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
        dest_dram_addr = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
        notification_mailbox_address = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
        //DPRINT << "RX DRAM MODE: bank=" << dest_bank_id << " addr=" << HEX() << dest_dram_addr << " mailbox=" << HEX() << notification_mailbox_address << ENDL();
    } else {
        //DPRINT << "RX L1 MODE" << ENDL();
    }
    //DPRINT << "RX ARGS: packet_size=" << packet_payload_size_bytes << " num_packets=" << num_packets << " seed=" << HEX() << time_seed << ENDL();
    //DPRINT << "NUM_PACKETS!=" << num_packets << " PACKET_SIZE=" << packet_payload_size_bytes << ENDL();
    tt_l1_ptr uint32_t* start_addr = reinterpret_cast<tt_l1_ptr uint32_t*>(target_address);
    volatile tt_l1_ptr uint32_t* poll_addr =
        use_dram_dst ? reinterpret_cast<tt_l1_ptr uint32_t*>(notification_mailbox_address)
                     : reinterpret_cast<tt_l1_ptr uint32_t*>(target_address + packet_payload_size_bytes - 4);
    uint32_t mismatch_addr, mismatch_val, expected_val;
    bool match = true;
    uint64_t bytes_received = 0;

    //DPRINT << "RX SETUP: target_addr=" << HEX() << target_address << " start_addr=" << HEX() << (uint32_t)start_addr << " poll_addr=" << HEX() << (uint32_t)poll_addr << ENDL();
    //DPRINT << "NUM_PACKETS@=" << num_packets << " PACKET_SIZE=" << packet_payload_size_bytes << ENDL();
    zero_l1_buf(test_results, test_results_size_bytes);
    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_STARTED;

    //DPRINT << "RX READY - waiting for packets..." << ENDL();
    //DPRINT << "NUM_PACKETS#=" << num_packets << " PACKET_SIZE=" << packet_payload_size_bytes << ENDL();
    if (!use_dram_dst) {
        // poll for data
        //DPRINT << "RX L1 MODE: Starting packet polling loop for " << num_packets << " packets" << ENDL();
        for (uint32_t i = 0; i < num_packets; i++) {
#ifndef BENCHMARK_MODE
            time_seed = prng_next(time_seed);
            uint32_t expected_val = time_seed + (packet_payload_size_bytes / 16) - 1;
            //DPRINT << "RX PACKET " << i << "/" << num_packets << " - waiting for expected_val=" << HEX() << expected_val << " at poll_addr=" << HEX() << (uint32_t)poll_addr << ENDL();

            WAYPOINT("FPW");
            while (expected_val != *poll_addr) {
                invalidate_l1_cache();
            }
            WAYPOINT("FPD");
            //DPRINT << "RX PACKET " << i << " - received! poll_addr_val=" << HEX() << *poll_addr << ENDL();

            // check for data correctness
            match = check_packet_data(
                start_addr, packet_payload_size_bytes / 16, time_seed, mismatch_addr, mismatch_val, expected_val);
            //DPRINT << "RX PACKET " << i << " - data check " << (match ? "PASSED" : "FAILED") << ENDL();
            if (!match) {
                break;
            }
            start_addr += packet_payload_size_bytes / 4;
            poll_addr += packet_payload_size_bytes / 4;
#endif
            bytes_received += packet_payload_size_bytes;
        }
    } else {
#ifndef BENCHMARK_MODE
        //DPRINT << "RX DRAM MODE: Starting DRAM polling for " << num_packets << " packets" << ENDL();

        WAYPOINT("FPW");
        //DPRINT << "RX DRAM MODE: Waiting for notification at poll_addr=" << HEX() << (uint32_t)poll_addr << ENDL();
        while (*poll_addr != 1 /* increment value*/) {
            invalidate_l1_cache();
        }
        WAYPOINT("FPD");
        //DPRINT << "RX DRAM MODE: Notification received! Starting DRAM read..." << ENDL();
        
        // Read all DRAM data in one go
        uint32_t total_data_size_bytes = num_packets * packet_payload_size_bytes;
        uint64_t dram_noc_addr = get_noc_addr_from_bank_id<true>(dest_bank_id, dest_dram_addr);
        //DPRINT << "RX DRAM MODE: Reading " << total_data_size_bytes << " bytes from DRAM addr=" << HEX() << dram_noc_addr << ENDL();
        noc_async_read(dram_noc_addr, (uint32_t)start_addr, total_data_size_bytes);
        noc_async_read_barrier();
        //DPRINT << "RX DRAM MODE: DRAM read completed, verifying packets..." << ENDL();

        // Verify each packet in the data
        tt_l1_ptr uint32_t* packet_addr = start_addr;
        for (uint32_t i = 0; i < num_packets; i++) {
            time_seed = prng_next(time_seed);
            expected_val = time_seed + (packet_payload_size_bytes / 16) - 1;
            //DPRINT << "RX DRAM PACKET " << i << "/" << num_packets << " - verifying packet at addr=" << HEX() << (uint32_t)packet_addr << " expected_val=" << HEX() << expected_val << ENDL();
            match = check_packet_data(
                packet_addr, packet_payload_size_bytes / 16, time_seed, mismatch_addr, mismatch_val, expected_val);
            //DPRINT << "RX DRAM PACKET " << i << " - verification " << (match ? "PASSED" : "FAILED") << ENDL();
            if (!match) {
                break;
            }
            packet_addr += packet_payload_size_bytes / 4;  // Move to next packet
            bytes_received += packet_payload_size_bytes;
        }
    }
#endif

    if (!match) {
        //DPRINT << "RX FAILED: Data mismatch detected!" << ENDL();
        //DPRINT << "RX FAILED: mismatch_addr=" << HEX() << mismatch_addr << " mismatch_val=" << HEX() << mismatch_val << " expected_val=" << HEX() << expected_val << ENDL();
        test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_DATA_MISMATCH;
        test_results[TT_FABRIC_MISC_INDEX + 12] = mismatch_addr;
        test_results[TT_FABRIC_MISC_INDEX + 13] = mismatch_val;
        test_results[TT_FABRIC_MISC_INDEX + 14] = expected_val;
    } else {
        //DPRINT << "RX SUCCESS: All packets verified correctly!" << ENDL();
        //DPRINT << "RX SUCCESS: Total bytes received=" << bytes_received << ENDL();
        test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_PASS;
    }

    test_results[TT_FABRIC_WORD_CNT_INDEX] = (uint32_t)bytes_received;
    test_results[TT_FABRIC_WORD_CNT_INDEX + 1] = bytes_received >> 32;
    
    //DPRINT << "=== RX KERNEL COMPLETE ===" << ENDL();
}
