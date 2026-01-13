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
constexpr uint32_t target_address_base = get_compile_time_arg_val(3);
constexpr NocSendType noc_send_type = static_cast<NocSendType>(get_compile_time_arg_val(4));
constexpr uint16_t packet_payload_size_bytes = static_cast<uint16_t>(get_compile_time_arg_val(5));
constexpr uint32_t num_packets = get_compile_time_arg_val(6);
constexpr uint32_t time_seed_init = get_compile_time_arg_val(7);
constexpr uint32_t req_notification_size_bytes = get_compile_time_arg_val(8);
// Total number of devices in the mesh
constexpr uint32_t num_devices = get_compile_time_arg_val(9);
// Per-sender L1 region size
constexpr uint32_t per_sender_l1_size = get_compile_time_arg_val(10);
// This receiver's device index - skip this slot (no one sends to self)
constexpr uint32_t receiver_device_idx = get_compile_time_arg_val(11);

// Helper function to poll for a value at a specific L1 address
inline void poll_for_value(volatile tt_l1_ptr uint32_t* poll_addr, uint32_t expected_val) {
    WAYPOINT("FPW");
    while (expected_val != *poll_addr) {
        invalidate_l1_cache();
    }
    WAYPOINT("FPD");
}

/*
 * All-to-all receiver kernel.
 * Receives packets from all other devices (N-1 senders).
 * Uses simple indexing: slot i contains data from sender device i.
 * Skips slot = receiver_device_idx (no one sends to self).
 */
void kernel_main() {
    zero_l1_buf(test_results, test_results_size_bytes);
    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_STARTED;

    uint32_t mismatch_addr, mismatch_val, expected_val;
    bool match = true;
    uint64_t bytes_received = 0;

    // Loop through all device slots, skip our own
    for (uint32_t sender_idx = 0; sender_idx < num_devices; sender_idx++) {
        if (sender_idx == receiver_device_idx) {
            continue;  // Skip self - no one sends to their own device
        }

        // Calculate L1 address for this sender's data
        uint32_t target_address = target_address_base + sender_idx * per_sender_l1_size;

        // Use time_seed offset by sender_idx for expected data validation
        uint32_t time_seed = time_seed_init + sender_idx;

        tt_l1_ptr uint32_t* start_addr = reinterpret_cast<tt_l1_ptr uint32_t*>(target_address);
        volatile tt_l1_ptr uint32_t* poll_addr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(target_address + packet_payload_size_bytes - 4);

        // Process packets from this sender
        for (uint32_t i = 0; i < num_packets; i++) {
            time_seed = prng_next(time_seed);

            switch (noc_send_type) {
                case NOC_UNICAST_WRITE: {
                    // Wait for the packet data to arrive by polling on the last word
                    expected_val = time_seed + (packet_payload_size_bytes / 16) - 1;
                    poll_for_value(poll_addr, expected_val);

                    // Check for data correctness
                    match = check_packet_data(
                        start_addr,
                        packet_payload_size_bytes / 16,
                        time_seed,
                        mismatch_addr,
                        mismatch_val,
                        expected_val);
                } break;
                case NOC_UNICAST_INLINE_WRITE: {
                    // Wait for the inline write to arrive by polling on the value
                    expected_val = time_seed;
                    poll_for_value(start_addr, expected_val);

                    // Check data correctness for a single uint32_t
                    uint32_t received_value = *start_addr;
                    if (received_value != expected_val) {
                        match = false;
                        mismatch_addr = reinterpret_cast<uint32_t>(start_addr);
                        mismatch_val = received_value;
                    }
                } break;
                case NOC_UNICAST_ATOMIC_INC: {
                    // Wait for the atomic increment to arrive by polling on the value
                    expected_val = time_seed;
                    poll_for_value(start_addr, expected_val);

                    uint32_t received_value = *start_addr;
                    if (received_value != expected_val) {
                        match = false;
                        mismatch_addr = reinterpret_cast<uint32_t>(start_addr);
                        mismatch_val = received_value;
                    }
                } break;
                default: {
                    ASSERT(false);
                } break;
            }
            if (!match) {
                break;
            }
            start_addr += packet_payload_size_bytes / 4;
            poll_addr += packet_payload_size_bytes / 4;
            bytes_received += packet_payload_size_bytes;
        }
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
