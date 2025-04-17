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
#include "tt_fabric_test_kernels_utils.hpp"
#include <array>

// clang-format on

void kernel_main() {
    size_t rt_args_idx = 0;
    auto worker_config = tt::tt_fabric::ReceiverWorkerConfig::build_from_args(rt_args_idx);

    tt_l1_ptr uint32_t* const test_results = reinterpret_cast<tt_l1_ptr uint32_t*>(worker_config.test_results_address);
    zero_l1_buf(test_results, worker_config::TEST_RESULTS_SIZE_BYTES);
    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_STARTED;

    std::array<uint32_t, worker_config::MAX_NUM_SENDERS_PER_RECEIVER> sender_seeds;
#ifndef BENCHMARK_MODE
    std::array<tt_l1_ptr uint32_t*, worker_config::MAX_NUM_SENDERS_PER_RECEIVER> payload_start_addresses;
    std::array<volatile tt_l1_ptr uint32_t*, worker_config::MAX_NUM_SENDERS_PER_RECEIVER> poll_addresses;
#endif

    for (auto i = 0; i < worker_config.num_senders; i++) {
        sender_seeds[i] = worker_config.time_seed ^ worker_config.sender_ids[i];
#ifndef BENCHMARK_MODE
        payload_start_addresses[i] = reinterpret_cast<tt_l1_ptr uint32_t*>(worker_config.target_addresses[i]);
        poll_addresses[i] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            worker_config.target_addresses[i] + worker_config.packet_payload_size_bytes - 4);
#endif
    }

    uint32_t mismatch_addr, mismatch_val, expected_val;
    bool match = true;
    uint64_t bytes_received = 0;

    for (auto packet_id = 0; packet_id < worker_config.num_packets; packet_id++) {
        for (auto i = 0; i < worker_config.num_senders; i++) {
#ifndef BENCHMARK_MODE
            sender_seeds[i] = prng_next(sender_seeds[i]);
            uint32_t expected_val = sender_seeds[i] + (worker_config.packet_payload_size_bytes / 16) - 1;

            while (expected_val != poll_addresses[i]);

            match = check_packet_data(
                payload_start_addresses[i],
                worker_config.packet_payload_size_bytes / 16,
                sender_seeds[i],
                mismatch_addr,
                mismatch_val,
                expected_val);
            if (!match) {
                break;
            }

            payload_start_addresses[i] += worker_config.packet_payload_size_bytes / 4;
            poll_addresses[i] += worker_config.packet_payload_size_bytes / 4;
#endif
            bytes_received += worker_config.packet_payload_size_bytes;
        }

        if (!match) {
            break;
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
