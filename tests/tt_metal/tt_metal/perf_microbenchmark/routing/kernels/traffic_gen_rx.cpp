// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tt_metal/impl/dispatch/kernels/packet_queue.hpp"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/traffic_gen.hpp"

packet_input_queue_state_t input_queues[MAX_SWITCH_FAN_IN];

constexpr uint32_t endpoint_id = get_compile_time_arg_val(0);

constexpr uint32_t num_src_endpoints = get_compile_time_arg_val(1);
constexpr uint32_t num_dest_endpoints = get_compile_time_arg_val(2);

static_assert(is_power_of_2(num_src_endpoints), "num_src_endpoints must be a power of 2");
static_assert(is_power_of_2(num_dest_endpoints), "num_dest_endpoints must be a power of 2");

constexpr uint32_t input_queue_id = 0;

constexpr uint32_t queue_start_addr_words = get_compile_time_arg_val(3);
constexpr uint32_t queue_size_words = get_compile_time_arg_val(4);

static_assert(is_power_of_2(queue_size_words), "queue_size_words must be a power of 2");

constexpr uint32_t remote_tx_x = get_compile_time_arg_val(5);
constexpr uint32_t remote_tx_y = get_compile_time_arg_val(6);
constexpr uint32_t remote_tx_queue_id = get_compile_time_arg_val(7);

constexpr DispatchRemoteNetworkType rx_rptr_update_network_type = static_cast<DispatchRemoteNetworkType>(get_compile_time_arg_val(8));

constexpr uint32_t test_results_addr_arg = get_compile_time_arg_val(9);
constexpr uint32_t test_results_size_bytes = get_compile_time_arg_val(10);

tt_l1_ptr uint32_t* const test_results =
    reinterpret_cast<tt_l1_ptr uint32_t*>(test_results_addr_arg);

constexpr uint32_t prng_seed = get_compile_time_arg_val(11);

constexpr uint32_t reserved = get_compile_time_arg_val(12);

constexpr uint32_t max_packet_size_words = get_compile_time_arg_val(13);

constexpr uint32_t disable_data_check = get_compile_time_arg_val(14);

constexpr uint32_t src_endpoint_start_id = get_compile_time_arg_val(15);
constexpr uint32_t dest_endpoint_start_id = get_compile_time_arg_val(16);

constexpr uint32_t timeout_cycles = get_compile_time_arg_val(17);


// predicts size and payload of packets from each destination, should have
// the same random seed as the corresponding traffic_gen_tx
input_queue_rnd_state_t src_rnd_state[num_src_endpoints];


void kernel_main() {

    zero_l1_buf(test_results, test_results_size_bytes);
    test_results[PQ_TEST_STATUS_INDEX] = PACKET_QUEUE_TEST_STARTED;
    test_results[PQ_TEST_MISC_INDEX] = 0xff000000;
    test_results[PQ_TEST_MISC_INDEX+1] = 0xdd000000 | endpoint_id;

    zero_l1_buf(reinterpret_cast<tt_l1_ptr uint32_t*>(queue_start_addr_words*PACKET_WORD_SIZE_BYTES),
                queue_size_words);

    for (uint32_t i = 0; i < num_src_endpoints; i++) {
        src_rnd_state[i].init(prng_seed, src_endpoint_start_id+i);
    }

    packet_input_queue_state_t* input_queue = &(input_queues[input_queue_id]);

    input_queue->init(input_queue_id, queue_start_addr_words, queue_size_words,
                      remote_tx_x, remote_tx_y, remote_tx_queue_id,
                      rx_rptr_update_network_type);

    if (!wait_all_src_dest_ready(input_queue, 1, NULL, 0, timeout_cycles)) {
        test_results[PQ_TEST_STATUS_INDEX] = PACKET_QUEUE_TEST_TIMEOUT;
        return;
    }

    test_results[PQ_TEST_MISC_INDEX] = 0xff000001;

    uint64_t num_words_checked = 0;
    uint32_t curr_packet_payload_words_remaining_to_check = 0;

    uint64_t iter = 0;
    bool timeout = false;
    bool check_failed = false;
    bool all_src_endpoints_last_packet = false;
    bool src_endpoint_last_packet[num_src_endpoints] = {false};
    uint32_t mismatch_addr, mismatch_val, expected_val;
    tt_l1_ptr dispatch_packet_header_t* curr_packet_header_ptr;
    input_queue_rnd_state_t* src_endpoint_rnd_state;
    uint64_t words_sent = 0;
    uint64_t words_cleared = 0;
    uint64_t start_timestamp = get_timestamp();
    uint32_t progress_timestamp = start_timestamp & 0xFFFFFFFF;

    while (!all_src_endpoints_last_packet) {

        iter++;

        bool packet_available = false;
        while (!packet_available) {
            if (timeout_cycles > 0) {
                uint32_t cycles_since_progress = get_timestamp_32b() - progress_timestamp;
                if (cycles_since_progress > timeout_cycles) {
                    test_results[PQ_TEST_MISC_INDEX] = 0xff000006;
                    timeout = true;
                    break;
                }
            }
            uint32_t num_words_available;
            packet_available = input_queue->input_queue_full_packet_available_to_send(num_words_available);
            if (!packet_available) {
                // Mark works as "sent" immediately to keep pipeline from stalling.
                // This is OK since num_words_available comes from the call above, so
                // it's guaranteed to be smaller than the full next packet.
                input_queue->input_queue_advance_words_sent(num_words_available);
                words_sent += num_words_available;
            }
        }

        if (timeout) {
            break;
        }

        curr_packet_header_ptr = input_queue->get_curr_packet_header_ptr();
        uint32_t src_endpoint_id = input_queue->get_curr_packet_src();
        uint32_t src_endpoint_index = src_endpoint_id - src_endpoint_start_id;
        uint32_t curr_packet_size_words = input_queue->get_curr_packet_size_words();
        uint32_t curr_packet_dest = input_queue->get_curr_packet_dest();
        uint32_t curr_packet_tag = input_queue->get_curr_packet_tag();
        uint32_t curr_packet_flags = input_queue->get_curr_packet_flags();

        if (src_endpoint_index >= num_src_endpoints ||
            curr_packet_size_words > max_packet_size_words ||
            endpoint_id != curr_packet_dest) {
                check_failed = true;
                mismatch_addr = reinterpret_cast<uint32_t>(curr_packet_header_ptr);
                mismatch_val = 0;
                expected_val = 0;
                test_results[PQ_TEST_MISC_INDEX+3] = 0xee000001;
                break;
        }

        if (curr_packet_flags & PACKET_TEST_LAST) {
            if (src_endpoint_last_packet[src_endpoint_index] ||
                curr_packet_size_words != 2 ||
                curr_packet_tag != 0xffffffff) {
                    check_failed = true;
                    mismatch_addr = reinterpret_cast<uint32_t>(curr_packet_header_ptr);
                    mismatch_val = 0;
                    expected_val = 0;
                    test_results[PQ_TEST_MISC_INDEX+3] = 0xee000002;
                    break;
            }
            src_endpoint_last_packet[src_endpoint_index] = true;
        } else {
            src_endpoint_rnd_state = &(src_rnd_state[src_endpoint_index]);
            src_endpoint_rnd_state->next_packet_rnd_to_dest(num_dest_endpoints, endpoint_id, dest_endpoint_start_id,
                                                            max_packet_size_words, UINT64_MAX);
            if (src_endpoint_rnd_state->curr_packet_size_words != curr_packet_size_words ||
                src_endpoint_rnd_state->packet_rnd_seed != curr_packet_tag) {
                    check_failed = true;
                    mismatch_addr = reinterpret_cast<uint32_t>(curr_packet_header_ptr);
                    mismatch_val = curr_packet_tag;
                    expected_val = src_endpoint_rnd_state->packet_rnd_seed;
                    test_results[PQ_TEST_MISC_INDEX+3] = 0xee000003;
                    break;
            }
        }

        uint32_t num_words_available = input_queue->input_queue_curr_packet_num_words_available_to_send();
        // we have the packet header info for checking, input queue can now switch to the next packet
        input_queue->input_queue_advance_words_sent(num_words_available);
        words_sent += num_words_available;

        // move rptr_cleared to the packet payload
        input_queue->input_queue_advance_words_cleared(1);
        words_cleared++;

        uint32_t curr_packet_payload_words = curr_packet_size_words-1;
        if (!disable_data_check) {
            uint32_t words_before_wrap = input_queue->get_queue_words_before_rptr_cleared_wrap();
            uint32_t words_after_wrap = 0;
            if (words_before_wrap < curr_packet_payload_words) {
                words_after_wrap = curr_packet_payload_words - words_before_wrap;
            } else {
                words_before_wrap = curr_packet_payload_words;
            }
            if (!check_packet_data(reinterpret_cast<tt_l1_ptr uint32_t*>(input_queue->get_queue_rptr_cleared_addr_bytes()),
                                   words_before_wrap,
                                   (curr_packet_tag & 0xFFFF0000)+1,
                                   mismatch_addr, mismatch_val, expected_val)) {
                check_failed = true;
                test_results[PQ_TEST_MISC_INDEX+3] = 0xee000005;
                test_results[PQ_TEST_MISC_INDEX+4] = words_before_wrap;
                test_results[PQ_TEST_MISC_INDEX+5] = words_after_wrap;
                break;
            }
            input_queue->input_queue_advance_words_cleared(words_before_wrap);
            words_cleared += words_before_wrap;
            if (words_after_wrap > 0) {
                if (!check_packet_data(reinterpret_cast<tt_l1_ptr uint32_t*>(input_queue->get_queue_rptr_cleared_addr_bytes()),
                                       words_after_wrap,
                                       (curr_packet_tag & 0xFFFF0000) + 1 + words_before_wrap,
                                       mismatch_addr, mismatch_val, expected_val)) {
                    check_failed = true;
                    test_results[PQ_TEST_MISC_INDEX+3] = 0xee000006;
                    test_results[PQ_TEST_MISC_INDEX+4] = words_before_wrap;
                    test_results[PQ_TEST_MISC_INDEX+5] = words_after_wrap;
                    break;
                }
                input_queue->input_queue_advance_words_cleared(words_after_wrap);
                words_cleared += words_after_wrap;
            }
        } else {
            input_queue->input_queue_advance_words_cleared(curr_packet_payload_words);
            words_cleared += curr_packet_payload_words;
        }
        progress_timestamp = get_timestamp_32b();
        num_words_checked += curr_packet_size_words;
        all_src_endpoints_last_packet = true;
        uint32_t src_endpoint_last_index_dbg = 0xe0000000;
        for (uint32_t i = 0; i < num_src_endpoints; i++) {
            all_src_endpoints_last_packet &= src_endpoint_last_packet[i];
            if (src_endpoint_last_packet[i]) {
                src_endpoint_last_index_dbg |= (0x1 << i);
            }
        }
        test_results[PQ_TEST_MISC_INDEX+6] = src_endpoint_last_index_dbg;
    }

    uint64_t cycles_elapsed = get_timestamp() - start_timestamp;

    if (!timeout && !check_failed) {
        test_results[PQ_TEST_MISC_INDEX] = 0xff000002;
        input_queue->send_remote_finished_notification();
    }

    set_64b_result(test_results, num_words_checked, PQ_TEST_WORD_CNT_INDEX);
    set_64b_result(test_results, cycles_elapsed, PQ_TEST_CYCLES_INDEX);
    set_64b_result(test_results, iter, PQ_TEST_ITER_INDEX);

    if (timeout) {
        test_results[PQ_TEST_STATUS_INDEX] = PACKET_QUEUE_TEST_TIMEOUT;
        set_64b_result(test_results, words_sent, PQ_TEST_MISC_INDEX+12);
        set_64b_result(test_results, words_cleared, PQ_TEST_MISC_INDEX+14);
        input_queue->dprint_object();
    } else if (check_failed) {
        test_results[PQ_TEST_STATUS_INDEX] = PACKET_QUEUE_TEST_DATA_MISMATCH;
        test_results[PQ_TEST_MISC_INDEX+12] = mismatch_addr;
        test_results[PQ_TEST_MISC_INDEX+12] = mismatch_val;
        test_results[PQ_TEST_MISC_INDEX+12] = expected_val;
        input_queue->dprint_object();
    } else {
        test_results[PQ_TEST_STATUS_INDEX] = PACKET_QUEUE_TEST_PASS;
        test_results[PQ_TEST_MISC_INDEX] = 0xff000005;
    }
}
