// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
#include "debug/dprint.h"
#include "dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_traffic_gen.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_interface.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/common/kernel_utils.hpp"
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

constexpr pkt_dest_size_choices_t pkt_dest_size_choice =
    static_cast<pkt_dest_size_choices_t>(get_compile_time_arg_val(8));

constexpr bool skip_pkt_content_gen = get_compile_time_arg_val(9);

constexpr bool fixed_async_wr_notif_addr = get_compile_time_arg_val(10);

constexpr uint32_t timeout_cycles = get_compile_time_arg_val(11);

#define PAYLOAD_MASK (0xFFFF0000)

// return true if timed-out
inline bool poll_for_value(volatile tt_l1_ptr uint32_t* poll_addr, uint32_t expected_val) {
    uint32_t idle_itr_cnt = 0;
    while (expected_val != *poll_addr) {
#ifdef CHECK_TIMEOUT
        if (++idle_itr_cnt >= timeout_cycles) {
            return true;
        }
#endif
    }
    return false;
}

void kernel_main() {
    uint64_t processed_packet_words = 0, num_packets = 0;
    tt_l1_ptr volatile uint32_t* poll_addr;
    uint32_t poll_val = 0;
    bool async_wr_check_failed = false;
    uint32_t num_producers = 0;
    uint32_t rx_buf_size;
    uint32_t time_seed;
    bool timed_out = false;

    // parse runtime args
    uint32_t rt_args_idx = 0;
    time_seed = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    num_producers = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));
    rx_buf_size = get_arg_val<uint32_t>(increment_arg_idx(rt_args_idx));

    zero_l1_buf(test_results, test_results_size_bytes);
    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_STARTED;
    test_results[TT_FABRIC_MISC_INDEX] = 0xff000000;

    if constexpr (ASYNC_WR & test_command) {
        uint32_t packet_rnd_seed;
        uint64_t curr_packet_words, curr_payload_words, processed_packet_words_src;
        uint32_t max_packet_size_mask, temp;
        uint32_t mismatch_addr, mismatch_val, expected_val;
        tt_l1_ptr uint32_t* base_target_addr;
        tt_l1_ptr uint32_t* read_addr;
        uint32_t rx_addr_hi;
        uint32_t start_val = 0;
        bool match;
        tt_l1_ptr uint32_t* src_endpoint_ids;
        tt_l1_ptr uint32_t* target_addresses;

        // parse runtime args relevant to the command
        src_endpoint_ids =
            reinterpret_cast<tt_l1_ptr uint32_t*>(get_arg_addr(increment_arg_idx(rt_args_idx, num_producers)));
        target_addresses =
            reinterpret_cast<tt_l1_ptr uint32_t*>(get_arg_addr(increment_arg_idx(rt_args_idx, num_producers)));

        // compute max_packet_size_mask (borrowed from tx kernel)
        if constexpr (pkt_dest_size_choice == pkt_dest_size_choices_t::RANDOM) {
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
        }

        for (uint32_t i = 0; i < num_producers; i++) {
            packet_rnd_seed = prng_seed ^ src_endpoint_ids[i];
            base_target_addr = reinterpret_cast<tt_l1_ptr uint32_t*>(target_addresses[i]);
            rx_addr_hi = target_addresses[i] + rx_buf_size;
            read_addr = base_target_addr;
            processed_packet_words_src = 0;
            uint32_t packet_index = 1;

            // if fixed notification address, wait for all the packets
            if constexpr ((test_command & ATOMIC_INC) && fixed_async_wr_notif_addr) {
                uint64_t temp_words = 0, temp_packets = 0;
                uint32_t temp_seed = packet_rnd_seed, temp_poll_val;
                while (temp_words < total_data_words) {
                    if constexpr (pkt_dest_size_choice == pkt_dest_size_choices_t::RANDOM) {
                        temp_seed = prng_next(temp_seed);
                        temp_words += packet_rnd_seed_to_size(temp_seed, max_packet_size_words, max_packet_size_mask);
                    } else if constexpr (
                        pkt_dest_size_choice == pkt_dest_size_choices_t::SAME_START_RNDROBIN_FIX_SIZE) {
                        temp_words += max_packet_size_words;
                    }
                    temp_packets++;
                }

                temp_poll_val = time_seed + packet_index + (temp_packets * atomic_increment);
                poll_addr = base_target_addr;
                timed_out = poll_for_value(poll_addr, temp_poll_val);
                if (timed_out) {
                    break;
                }
            }

            // read out the data
            while (processed_packet_words_src < total_data_words) {
                if constexpr (pkt_dest_size_choice == pkt_dest_size_choices_t::RANDOM) {
                    packet_rnd_seed = prng_next(packet_rnd_seed);

                    // get number of words to be read minus the header
                    curr_packet_words =
                        packet_rnd_seed_to_size(packet_rnd_seed, max_packet_size_words, max_packet_size_mask);
                } else if constexpr (pkt_dest_size_choice == pkt_dest_size_choices_t::SAME_START_RNDROBIN_FIX_SIZE) {
                    curr_packet_words = max_packet_size_words;
                }

                curr_payload_words = curr_packet_words - PACKET_HEADER_SIZE_WORDS;

                // check for wrap
                // if rx is slow, the data validation could fail, need to add sync b/w tx and rx
                if ((uint32_t)read_addr + (curr_payload_words * PACKET_WORD_SIZE_BYTES) > rx_addr_hi) {
                    read_addr = base_target_addr;
                }

                if constexpr (!skip_pkt_content_gen) {
                    start_val = packet_rnd_seed & PAYLOAD_MASK;

                    // get the value and addr to poll on
                    if constexpr (test_command & ATOMIC_INC) {
                        // poll on the first word in the payload
                        poll_addr = read_addr;
                        if constexpr (fixed_async_wr_notif_addr) {
                            // no need to poll further in this case
                            poll_val = *poll_addr;
                        } else {
                            poll_val = time_seed + packet_index + atomic_increment;
                            packet_index++;
                        }
                    } else {
                        // poll on the last word in the payload
                        poll_val = start_val + curr_payload_words - 1;
                        poll_addr = read_addr + (curr_payload_words * PACKET_WORD_SIZE_BYTES / 4) - 1;
                    }

                    timed_out = poll_for_value(poll_addr, poll_val);
                    if (timed_out) {
                        break;
                    }

                    // check correctness
                    match = check_packet_data(
                        read_addr, curr_payload_words, start_val, mismatch_addr, mismatch_val, expected_val);
                    if (!match) {
                        async_wr_check_failed = true;
                        test_results[TT_FABRIC_MISC_INDEX + 12] = mismatch_addr;
                        test_results[TT_FABRIC_MISC_INDEX + 13] = mismatch_val;
                        test_results[TT_FABRIC_MISC_INDEX + 14] = expected_val;
                        break;
                    }
                }

                read_addr += (curr_payload_words * PACKET_WORD_SIZE_BYTES / 4);
                processed_packet_words += curr_packet_words;
                processed_packet_words_src += curr_packet_words;
                num_packets++;
            }

            if (timed_out) {
                break;
            }
        }
    } else if constexpr (ATOMIC_INC == test_command) {
        poll_addr = reinterpret_cast<tt_l1_ptr uint32_t*>(target_address);
        // TODO: read in wrap boundary as well from compile args
        num_packets = num_producers * ((total_data_words + PACKET_HEADER_SIZE_WORDS - 1) / PACKET_HEADER_SIZE_WORDS);
        poll_val = atomic_increment * num_packets;

        // poll for the final value
        processed_packet_words = num_packets * PACKET_HEADER_SIZE_WORDS;
        timed_out = poll_for_value(poll_addr, poll_val);
        if (timed_out) {
            processed_packet_words = 0;
        }
    }

    // write out results
    set_64b_result(test_results, processed_packet_words, TT_FABRIC_WORD_CNT_INDEX);
    set_64b_result(test_results, num_packets, TX_TEST_IDX_NPKT);

    if (async_wr_check_failed) {
        test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_DATA_MISMATCH;
    } else if (timed_out) {
        test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_TIMEOUT;
    } else {
        test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_PASS;
        test_results[TT_FABRIC_MISC_INDEX] = 0xff000005;
    }
}
