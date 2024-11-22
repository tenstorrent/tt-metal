// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tt_fabric/hw/inc/tt_fabric.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_traffic_gen.hpp"
// clang-format on


/*
constexpr uint32_t PACKET_QUEUE_STAUS_MASK = 0xabc00000;
constexpr uint32_t PACKET_QUEUE_TEST_STARTED = PACKET_QUEUE_STAUS_MASK | 0x0;
constexpr uint32_t PACKET_QUEUE_TEST_PASS = PACKET_QUEUE_STAUS_MASK | 0x1;
constexpr uint32_t PACKET_QUEUE_TEST_TIMEOUT = PACKET_QUEUE_STAUS_MASK | 0xdead;
constexpr uint32_t PACKET_QUEUE_TEST_DATA_MISMATCH = PACKET_QUEUE_STAUS_MASK | 0x3;

// indexes of return values in test results buffer
constexpr uint32_t PQ_TEST_STATUS_INDEX = 0;
constexpr uint32_t PQ_TEST_WORD_CNT_INDEX = 2;
constexpr uint32_t PQ_TEST_CYCLES_INDEX = 4;
constexpr uint32_t PQ_TEST_ITER_INDEX = 6;
constexpr uint32_t PQ_TEST_MISC_INDEX = 16;
*/
constexpr uint32_t src_endpoint_id = get_compile_time_arg_val(0);
constexpr uint32_t num_dest_endpoints = get_compile_time_arg_val(1);
static_assert(is_power_of_2(num_dest_endpoints), "num_dest_endpoints must be a power of 2");
constexpr uint32_t dest_endpoint_start_id = get_compile_time_arg_val(2);

constexpr uint32_t req_buffer_start_addr = get_compile_time_arg_val(3);
constexpr uint32_t data_buffer_start_addr = get_compile_time_arg_val(4);
constexpr uint32_t data_buffer_size_words = get_compile_time_arg_val(5);

constexpr uint32_t router_x = get_compile_time_arg_val(6);
constexpr uint32_t router_y = get_compile_time_arg_val(7);

constexpr uint32_t test_results_addr_arg = get_compile_time_arg_val(8);
constexpr uint32_t test_results_size_bytes = get_compile_time_arg_val(9);

tt_l1_ptr uint32_t* const test_results = reinterpret_cast<tt_l1_ptr uint32_t*>(test_results_addr_arg);

constexpr uint32_t prng_seed = get_compile_time_arg_val(10);

constexpr uint32_t total_data_kb = get_compile_time_arg_val(11);
constexpr uint64_t total_data_words = ((uint64_t)total_data_kb) * 1024 / PACKET_WORD_SIZE_BYTES;

constexpr uint32_t max_packet_size_words = get_compile_time_arg_val(12);

static_assert(max_packet_size_words > 3, "max_packet_size_words must be greater than 3");

constexpr uint32_t timeout_cycles = get_compile_time_arg_val(13);

constexpr bool skip_pkt_content_gen = get_compile_time_arg_val(14);
constexpr pkt_dest_size_choices_t pkt_dest_size_choice = static_cast<pkt_dest_size_choices_t>(get_compile_time_arg_val(15));

constexpr uint32_t data_sent_per_iter_low = get_compile_time_arg_val(16);
constexpr uint32_t data_sent_per_iter_high = get_compile_time_arg_val(17);

uint32_t max_packet_size_mask;

// input_queue_rnd_state_t input_queue_state;
auto input_queue_state = select_input_queue<pkt_dest_size_choice>();
volatile local_pull_request_t *local_pull_request = (volatile local_pull_request_t *)(data_buffer_start_addr - 1024);

fvc_producer_state_t test_producer __attribute__((aligned(16)));



uint64_t xy_local_addr;

// generates packets with random size and payload on the input side
inline bool test_buffer_handler() {
    if (input_queue_state.all_packets_done()) {
        return true;
    }

    uint32_t free_words = test_producer.get_num_words_free();
    if (free_words < PACKET_HEADER_SIZE_WORDS) {
        return false;
    }

    // Each call to test_buffer_handler initializes only up to the end
    // of the queue buffer, so we don't need to handle wrapping.
    // TODO: we need to handle wrapping since header is 3 words now.
    // if we have to wrap before next 3 words, header has to wrap.
    uint32_t byte_wr_addr = test_producer.get_local_buffer_write_addr();
    uint32_t words_to_init = std::min(free_words, test_producer.words_before_local_buffer_wrap());
    uint32_t words_initialized = 0;
    uint32_t target_address = 0x100000;
    while (words_initialized < words_to_init) {
        if (input_queue_state.all_packets_done()) {
            break;
        }

        if (!input_queue_state.packet_active()) { // start of a new packet
            input_queue_state.next_packet(num_dest_endpoints, dest_endpoint_start_id, max_packet_size_words, max_packet_size_mask, total_data_words);

            tt_l1_ptr packet_header_t* header_ptr =
                reinterpret_cast<tt_l1_ptr packet_header_t*>(byte_wr_addr);

            header_ptr->routing.flags = FORWARD;
            header_ptr->routing.packet_size_bytes = input_queue_state.curr_packet_size_words * PACKET_WORD_SIZE_BYTES;
            header_ptr->session.command = ASYNC_WR;
            header_ptr->session.target_offset_l = target_address;
            header_ptr->session.target_offset_h = 0x410;
            target_address += header_ptr->routing.packet_size_bytes - PACKET_HEADER_SIZE_BYTES;
            header_ptr->packet_parameters.misc_parameters.words[0] = input_queue_state.packet_rnd_seed;

            words_initialized += PACKET_HEADER_SIZE_WORDS;
            input_queue_state.curr_packet_words_remaining -= PACKET_HEADER_SIZE_WORDS;
            byte_wr_addr += PACKET_HEADER_SIZE_BYTES;
        } else {
            uint32_t words_remaining = words_to_init - words_initialized;
            uint32_t num_words = std::min(words_remaining, input_queue_state.curr_packet_words_remaining);
            if constexpr (!skip_pkt_content_gen) {
                uint32_t start_val =
                (input_queue_state.packet_rnd_seed & 0xFFFF0000) +
                (input_queue_state.curr_packet_size_words - input_queue_state.curr_packet_words_remaining - PACKET_HEADER_SIZE_WORDS);
                fill_packet_data(reinterpret_cast<tt_l1_ptr uint32_t*>(byte_wr_addr), num_words, start_val);
            }
            words_initialized += num_words;
            input_queue_state.curr_packet_words_remaining -= num_words;
            byte_wr_addr += num_words * PACKET_WORD_SIZE_BYTES;
        }
    }
    test_producer.advance_local_wrptr(words_initialized);
    return false;
}

void kernel_main() {

    tt_fabric_init();

    zero_l1_buf(test_results, test_results_size_bytes);
    test_results[PQ_TEST_STATUS_INDEX] = PACKET_QUEUE_TEST_STARTED;
    test_results[PQ_TEST_STATUS_INDEX+1] = (uint32_t) local_pull_request;

    test_results[PQ_TEST_MISC_INDEX] = 0xff000000;
    test_results[PQ_TEST_MISC_INDEX + 1] = 0xcc000000 | src_endpoint_id;

    zero_l1_buf(reinterpret_cast<tt_l1_ptr uint32_t*>(data_buffer_start_addr), data_buffer_size_words * PACKET_WORD_SIZE_BYTES);
    zero_l1_buf((uint32_t*)local_pull_request, sizeof(local_pull_request_t));

    if constexpr (pkt_dest_size_choice == pkt_dest_size_choices_t::RANDOM) {
        input_queue_state.init(src_endpoint_id, prng_seed);
    } else if constexpr (pkt_dest_size_choice == pkt_dest_size_choices_t::SAME_START_RNDROBIN_FIX_SIZE) {
        input_queue_state.init(max_packet_size_words, 0);
    } else {
        input_queue_state.init(src_endpoint_id, prng_seed);
    }

    test_producer.init(data_buffer_start_addr, data_buffer_size_words, 0x0);

    uint32_t temp = max_packet_size_words;
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

/*
    if (!wait_all_src_dest_ready(NULL, 0, output_queue_ptr, 1, timeout_cycles)) {
        test_results[PQ_TEST_STATUS_INDEX] = PACKET_QUEUE_TEST_TIMEOUT;
        return;
    }
*/
    test_results[PQ_TEST_MISC_INDEX] = 0xff000001;

    uint64_t data_words_sent = 0;
    uint64_t iter = 0;
    uint64_t zero_data_sent_iter = 0;
    uint64_t few_data_sent_iter = 0;
    uint64_t many_data_sent_iter = 0;
    uint64_t words_flushed = 0;
    bool timeout = false;
    uint64_t start_timestamp = get_timestamp();
    uint32_t progress_timestamp = start_timestamp & 0xFFFFFFFF;

    uint32_t curr_packet_size = 0;
    uint32_t curr_packet_words_sent = 0;
    uint32_t packet_count = 0;

    while (true) {
        iter++;
#ifdef CHECK_TIMEOUT
        if (timeout_cycles > 0) {
            uint32_t cycles_since_progress = get_timestamp_32b() - progress_timestamp;
            if (cycles_since_progress > timeout_cycles) {
                timeout = true;
                break;
            }
        }
#endif
        bool all_packets_initialized = test_buffer_handler();

        if (test_producer.get_curr_packet_valid()) {
            curr_packet_size = (test_producer.current_packet_header.routing.packet_size_bytes + PACKET_WORD_SIZE_BYTES - 1) >> 4;
            uint32_t curr_data_words_sent = test_producer.pull_data_from_fvc_buffer<FVC_MODE_ENDPOINT>();
            curr_packet_words_sent += curr_data_words_sent;
            data_words_sent += curr_data_words_sent;
            if constexpr (!(data_sent_per_iter_low == 0 && data_sent_per_iter_high == 0)) {
                zero_data_sent_iter += static_cast<uint64_t>(curr_data_words_sent <= 0);
                few_data_sent_iter += static_cast<uint64_t>(curr_data_words_sent <= data_sent_per_iter_low);
                many_data_sent_iter += static_cast<uint64_t>(curr_data_words_sent >= data_sent_per_iter_high);
            }
#ifdef CHECK_TIMEOUT
            progress_timestamp = (curr_data_words_sent > 0) ? get_timestamp_32b() : progress_timestamp;
#endif
            if (curr_packet_words_sent == curr_packet_size) {
                curr_packet_words_sent = 0;
                packet_count++;
                if (packet_count >= 4) {
                    break;
                }
            }
        } else if (all_packets_initialized) {
            break;
        }
        //words_flushed += output_queue_ptr->prev_words_in_flight_check_flush();
    }
/*
    if (!timeout) {
        test_results[PQ_TEST_MISC_INDEX] = 0xff00002;
        if (!output_queue_ptr->output_barrier(timeout_cycles)) {
            timeout = true;
        }
    }
*/
    uint64_t cycles_elapsed = get_timestamp() - start_timestamp;

/*
    if (!timeout) {
        test_results[PQ_TEST_MISC_INDEX] = 0xff00003;
        progress_timestamp = get_timestamp_32b();
        while (!output_queue_ptr->is_remote_finished()) {
            if (timeout_cycles > 0) {
                uint32_t cycles_since_progress = get_timestamp_32b() - progress_timestamp;
                if (cycles_since_progress > timeout_cycles) {
                    timeout = true;
                    break;
                }
            }
        }
    }
*/
    uint64_t num_packets = input_queue_state.get_num_packets();
    set_64b_result(test_results, data_words_sent, PQ_TEST_WORD_CNT_INDEX);
    set_64b_result(test_results, cycles_elapsed, PQ_TEST_CYCLES_INDEX);
    set_64b_result(test_results, iter, PQ_TEST_ITER_INDEX);
    set_64b_result(test_results, total_data_words, TX_TEST_IDX_TOT_DATA_WORDS);
    set_64b_result(test_results, num_packets, TX_TEST_IDX_NPKT);
    set_64b_result(test_results, zero_data_sent_iter, TX_TEST_IDX_ZERO_DATA_WORDS_SENT_ITER);
    set_64b_result(test_results, few_data_sent_iter, TX_TEST_IDX_FEW_DATA_WORDS_SENT_ITER);
    set_64b_result(test_results, many_data_sent_iter, TX_TEST_IDX_MANY_DATA_WORDS_SENT_ITER);

    if (!timeout) {
        test_results[PQ_TEST_STATUS_INDEX] = PACKET_QUEUE_TEST_PASS;
        test_results[PQ_TEST_MISC_INDEX] = 0xff00004;
    } else {
        test_results[PQ_TEST_STATUS_INDEX] = PACKET_QUEUE_TEST_TIMEOUT;
        set_64b_result(test_results, words_flushed, TX_TEST_IDX_WORDS_FLUSHED);
        // these calls lead to code size issues?
        // input_queue_ptr->dprint_object();
        // output_queue_ptr->dprint_object();
    }
}
