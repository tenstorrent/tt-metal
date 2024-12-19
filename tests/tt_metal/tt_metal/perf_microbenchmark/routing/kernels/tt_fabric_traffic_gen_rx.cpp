// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
#include "debug/dprint.h"
#include "dataflow_api.h"
#include "tt_fabric/hw/inc/tt_fabric.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_traffic_gen.hpp"
#include "tt_fabric/hw/inc/tt_fabric_interface.h"
// clang-format on

// seed to re-generate the data and validate against incoming data
constexpr uint32_t prng_seed = get_compile_time_arg_val(0);

// total data/payload expected
constexpr uint32_t total_data_kb = get_compile_time_arg_val(1);
constexpr uint64_t total_data_words = ((uint64_t)total_data_kb) * 1024 / PACKET_WORD_SIZE_BYTES;

// max packet size to generate mask
constexpr uint32_t max_packet_size_words = get_compile_time_arg_val(2);
static_assert(max_packet_size_words > 3, "max_packet_size_words must be greater than 3");

// fabric command
constexpr uint32_t test_command = get_compile_time_arg_val(3);

// address to start reading from/poll on
constexpr uint32_t target_address = get_compile_time_arg_val(4);

// atomic increment for the ATOMIC_INC command
constexpr uint32_t atomic_increment = get_compile_time_arg_val(5);

constexpr uint32_t test_results_addr_arg = get_compile_time_arg_val(6);
constexpr uint32_t test_results_size_bytes = get_compile_time_arg_val(7);

tt_l1_ptr uint32_t* const test_results = reinterpret_cast<tt_l1_ptr uint32_t*>(test_results_addr_arg);

#define PAYLOAD_MASK (0xFFFF0000)

void kernel_main() {
    uint64_t processed_packet_words = 0, num_packets = 0;
    tt_l1_ptr volatile uint32_t* poll_addr;
    uint32_t poll_val = 0;
    bool async_wr_check_failed = false;
    uint32_t num_producers = 0;

    // parse runtime args
    num_producers = get_arg_val<uint32_t>(0);

    zero_l1_buf(test_results, test_results_size_bytes);
    test_results[PQ_TEST_STATUS_INDEX] = PACKET_QUEUE_TEST_STARTED;
    test_results[PQ_TEST_MISC_INDEX] = 0xff000000;

    if constexpr (ASYNC_WR == test_command) {
        uint32_t packet_rnd_seed = 0;
        uint64_t curr_packet_words = 0, curr_payload_words = 0, processed_packet_words_src = 0;
        uint32_t max_packet_size_mask, temp;
        uint32_t mismatch_addr, mismatch_val, expected_val;
        tt_l1_ptr uint32_t* read_addr;
        uint32_t start_val = 0;
        bool match;
        tt_l1_ptr uint32_t* src_endpoint_ids;
        tt_l1_ptr uint32_t* target_addresses;

        // parse runtime args relevant to the command
        src_endpoint_ids = reinterpret_cast<tt_l1_ptr uint32_t*>(get_arg_addr(1));
        target_addresses = reinterpret_cast<tt_l1_ptr uint32_t*>(get_arg_addr(1 + num_producers));

        // compute max_packet_size_mask (borrowed from tx kernel)
        temp = max_packet_size_words;
        max_packet_size_mask = 0;
        temp >>= 1;
        while (temp) {
            max_packet_size_mask = (max_packet_size_mask << 1) + 1;
            temp >>= 1;
        }
        if ((max_packet_size_mask + 1) != max_packet_size_words) {
            // max_packet_size_words is not a power of 2
            // snap to next power of 2 mask
            max_packet_size_mask = (max_packet_size_mask << 1) + 1;
        }

        for (uint32_t i = 0; i < num_producers; i++) {
            packet_rnd_seed = prng_seed ^ src_endpoint_ids[i];
            read_addr = reinterpret_cast<tt_l1_ptr uint32_t*>(target_addresses[i]);
            processed_packet_words_src = 0;

            // read out the data
            while (processed_packet_words_src < total_data_words) {
                packet_rnd_seed = prng_next(packet_rnd_seed);

                // get number of words to be read minus the header
                curr_packet_words =
                    packet_rnd_seed_to_size(packet_rnd_seed, max_packet_size_words, max_packet_size_mask);
                curr_payload_words = curr_packet_words - PACKET_HEADER_SIZE_WORDS;

                start_val = packet_rnd_seed & PAYLOAD_MASK;

                // get the value and addr to poll on
                poll_val = start_val + curr_payload_words - 1;
                poll_addr = read_addr + (curr_payload_words * PACKET_WORD_SIZE_BYTES / 4) - 1;

                // poll on the last word in the payload
                while (poll_val != *poll_addr);

                // check correctness
                match = check_packet_data(
                    read_addr, curr_payload_words, start_val, mismatch_addr, mismatch_val, expected_val);
                if (!match) {
                    async_wr_check_failed = true;
                    test_results[PQ_TEST_MISC_INDEX + 12] = mismatch_addr;
                    test_results[PQ_TEST_MISC_INDEX + 13] = mismatch_val;
                    test_results[PQ_TEST_MISC_INDEX + 14] = expected_val;
                    break;
                }

                read_addr += (curr_payload_words * PACKET_WORD_SIZE_BYTES / 4);
                processed_packet_words += curr_packet_words;
                processed_packet_words_src += curr_packet_words;
                num_packets++;
            }
        }
    } else if constexpr (ATOMIC_INC == test_command) {
        poll_addr = reinterpret_cast<tt_l1_ptr uint32_t*>(target_address);
        // TODO: read in wrap boundary as well from compile args
        num_packets = num_producers * ((total_data_words + PACKET_HEADER_SIZE_WORDS - 1) / PACKET_HEADER_SIZE_WORDS);
        poll_val = atomic_increment * num_packets;

        // poll for the final value
        while (poll_val != *poll_addr);

        processed_packet_words = num_packets * PACKET_HEADER_SIZE_WORDS;
    }

    // write out results
    set_64b_result(test_results, processed_packet_words, PQ_TEST_WORD_CNT_INDEX);
    set_64b_result(test_results, num_packets, TX_TEST_IDX_NPKT);

    if (async_wr_check_failed) {
        test_results[PQ_TEST_STATUS_INDEX] = PACKET_QUEUE_TEST_DATA_MISMATCH;
    } else {
        test_results[PQ_TEST_STATUS_INDEX] = PACKET_QUEUE_TEST_PASS;
        test_results[PQ_TEST_MISC_INDEX] = 0xff000005;
    }
}
