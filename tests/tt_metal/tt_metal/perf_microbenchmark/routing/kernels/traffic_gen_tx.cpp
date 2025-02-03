// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tt_metal/impl/dispatch/kernels/packet_queue.hpp"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/traffic_gen.hpp"

constexpr uint32_t src_endpoint_id = get_compile_time_arg_val(0);
constexpr uint32_t num_dest_endpoints = get_compile_time_arg_val(1);

static_assert(is_power_of_2(num_dest_endpoints), "num_dest_endpoints must be a power of 2");

constexpr uint32_t queue_start_addr_words = get_compile_time_arg_val(2);
constexpr uint32_t queue_size_words = get_compile_time_arg_val(3);
constexpr uint32_t queue_size_bytes = queue_size_words * PACKET_WORD_SIZE_BYTES;

static_assert(is_power_of_2(queue_size_words), "queue_size_words must be a power of 2");

constexpr uint32_t remote_rx_queue_start_addr_words = get_compile_time_arg_val(4);
constexpr uint32_t remote_rx_queue_size_words = get_compile_time_arg_val(5);

static_assert(is_power_of_2(remote_rx_queue_size_words), "remote_rx_queue_size_words must be a power of 2");

constexpr uint32_t remote_rx_x = get_compile_time_arg_val(6);
constexpr uint32_t remote_rx_y = get_compile_time_arg_val(7);
constexpr uint32_t remote_rx_queue_id = get_compile_time_arg_val(8);

constexpr DispatchRemoteNetworkType tx_network_type =
    static_cast<DispatchRemoteNetworkType>(get_compile_time_arg_val(9));

constexpr uint32_t test_results_addr_arg = get_compile_time_arg_val(10);
constexpr uint32_t test_results_size_bytes = get_compile_time_arg_val(11);

tt_l1_ptr uint32_t* const test_results = reinterpret_cast<tt_l1_ptr uint32_t*>(test_results_addr_arg);

constexpr uint32_t prng_seed = get_compile_time_arg_val(12);

constexpr uint32_t total_data_kb = get_compile_time_arg_val(13);
constexpr uint64_t total_data_words = ((uint64_t)total_data_kb) * 1024 / PACKET_WORD_SIZE_BYTES;

constexpr uint32_t max_packet_size_words = get_compile_time_arg_val(14);

static_assert(is_power_of_2(max_packet_size_words), "max_packet_size_words must be a power of 2");
static_assert(max_packet_size_words < queue_size_words, "max_packet_size_words must be less than queue_size_words");
static_assert(max_packet_size_words > 2, "max_packet_size_words must be greater than 2");

constexpr uint32_t src_endpoint_start_id = get_compile_time_arg_val(15);
constexpr uint32_t dest_endpoint_start_id = get_compile_time_arg_val(16);

constexpr uint32_t timeout_cycles = get_compile_time_arg_val(17);

constexpr bool skip_pkt_content_gen = get_compile_time_arg_val(18);
constexpr pkt_dest_size_choices_t pkt_dest_size_choice = static_cast<pkt_dest_size_choices_t>(get_compile_time_arg_val(19));

constexpr uint32_t data_sent_per_iter_low = get_compile_time_arg_val(20);
constexpr uint32_t data_sent_per_iter_high = get_compile_time_arg_val(21);

constexpr uint32_t input_queue_id = 0;
constexpr uint32_t output_queue_id = 1;

packet_input_queue_state_t input_queue;
using input_queue_network_sequence = NetworkTypeSequence<DispatchRemoteNetworkType::NONE>;
using input_queue_cb_mode_sequence = CBModeTypeSequence<false>;

packet_output_queue_state_t output_queue;
using output_queue_network_sequence = NetworkTypeSequence<tx_network_type>;
using output_queue_cb_mode_sequence = CBModeTypeSequence<false>;

constexpr packet_input_queue_state_t* input_queue_ptr = &input_queue;
constexpr packet_output_queue_state_t* output_queue_ptr = &output_queue;

// input_queue_rnd_state_t input_queue_state;
auto input_queue_state = select_input_queue<pkt_dest_size_choice>();

// generates packets with random size and payload on the input side
inline bool input_queue_handler() {
    if (input_queue_state.all_packets_done()) {
        return true;
    }

    uint32_t free_words = input_queue_ptr->get_queue_data_num_words_free();
    if (free_words == 0) {
        return false;
    }

    // Each call to input_queue_handler initializes only up to the end
    // of the queue buffer, so we don't need to handle wrapping.
    uint32_t byte_wr_addr = input_queue_ptr->get_queue_wptr_addr_bytes();
    uint32_t words_to_init = std::min(free_words, input_queue_ptr->get_queue_words_before_wptr_wrap());
    uint32_t words_initialized = 0;

    while (words_initialized < words_to_init) {
        if (input_queue_state.all_packets_done()) {
            break;
        }

        if (!input_queue_state.packet_active()) { // start of a new packet
            input_queue_state.next_packet(
                num_dest_endpoints, dest_endpoint_start_id, max_packet_size_words, total_data_words);

            tt_l1_ptr dispatch_packet_header_t* header_ptr =
                reinterpret_cast<tt_l1_ptr dispatch_packet_header_t*>(byte_wr_addr);
            header_ptr->packet_size_bytes = input_queue_state.curr_packet_size_words * PACKET_WORD_SIZE_BYTES;
            header_ptr->packet_src = src_endpoint_id;
            header_ptr->packet_dest = input_queue_state.curr_packet_dest;
            header_ptr->packet_flags = input_queue_state.curr_packet_flags;
            header_ptr->num_cmds = 0;
            header_ptr->tag = input_queue_state.packet_rnd_seed;
            words_initialized++;
            input_queue_state.curr_packet_words_remaining--;
            byte_wr_addr += PACKET_WORD_SIZE_BYTES;
        } else {
            uint32_t words_remaining = words_to_init - words_initialized;
            uint32_t num_words = std::min(words_remaining, input_queue_state.curr_packet_words_remaining);
            if constexpr (!skip_pkt_content_gen) {
                uint32_t start_val =
                (input_queue_state.packet_rnd_seed & 0xFFFF0000) +
                (input_queue_state.curr_packet_size_words - input_queue_state.curr_packet_words_remaining);
                fill_packet_data(reinterpret_cast<tt_l1_ptr uint32_t*>(byte_wr_addr), num_words, start_val);
            }
            words_initialized += num_words;
            input_queue_state.curr_packet_words_remaining -= num_words;
            byte_wr_addr += num_words * PACKET_WORD_SIZE_BYTES;
        }
    }
    input_queue_ptr->advance_queue_local_wptr<true>(words_initialized);
    return false;
}

void kernel_main() {
    zero_l1_buf(test_results, test_results_size_bytes);
    test_results[PQ_TEST_STATUS_INDEX] = PACKET_QUEUE_TEST_STARTED;
    test_results[PQ_TEST_MISC_INDEX] = 0xff000000;
    test_results[PQ_TEST_MISC_INDEX + 1] = 0xcc000000 | src_endpoint_id;

    zero_l1_buf(
        reinterpret_cast<tt_l1_ptr uint32_t*>(queue_start_addr_words * PACKET_WORD_SIZE_BYTES), queue_size_words);

    if constexpr (pkt_dest_size_choice == pkt_dest_size_choices_t::RANDOM) {
        input_queue_state.init(src_endpoint_id, prng_seed);
    } else if constexpr (pkt_dest_size_choice == pkt_dest_size_choices_t::SAME_START_RNDROBIN_FIX_SIZE) {
        input_queue_state.init(max_packet_size_words, 0);
    } else {
        input_queue_state.init(src_endpoint_id, prng_seed);
    }

    input_queue_ptr->init(
        input_queue_id,
        queue_start_addr_words,
        queue_size_words,
        // remote_x, remote_y, remote_queue_id, remote_update_network_type:
        0,
        0,
        0,
        DispatchRemoteNetworkType::NONE);

    output_queue_ptr->init(
        output_queue_id,
        remote_rx_queue_start_addr_words,
        remote_rx_queue_size_words,
        remote_rx_x,
        remote_rx_y,
        remote_rx_queue_id,
        tx_network_type,
        input_queue_ptr,
        1);

    if (!wait_all_input_output_ready<NoNetworkTypeSequence, NoCBModeTypeSequence, output_queue_network_sequence, output_queue_cb_mode_sequence>(NULL, &output_queue, timeout_cycles)) {
        test_results[PQ_TEST_STATUS_INDEX] = PACKET_QUEUE_TEST_TIMEOUT;
        return;
    }

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
        bool all_packets_initialized = input_queue_handler();
        if (input_queue_ptr->get_curr_packet_valid<false>()) {
            bool full_packet_sent;
            uint32_t curr_data_words_sent = output_queue_ptr->forward_data_from_input<tx_network_type, false, DispatchRemoteNetworkType::NONE, false>(
                input_queue_id, full_packet_sent, input_queue.get_end_of_cmd());
            data_words_sent += curr_data_words_sent;
            if constexpr (!(data_sent_per_iter_low == 0 && data_sent_per_iter_high == 0)) {
                zero_data_sent_iter += static_cast<uint64_t>(curr_data_words_sent <= 0);
                few_data_sent_iter += static_cast<uint64_t>(curr_data_words_sent <= data_sent_per_iter_low);
                many_data_sent_iter += static_cast<uint64_t>(curr_data_words_sent >= data_sent_per_iter_high);
            }
#ifdef CHECK_TIMEOUT
            progress_timestamp = (curr_data_words_sent > 0) ? get_timestamp_32b() : progress_timestamp;
#endif
        } else if (all_packets_initialized) {
            break;
        }
        words_flushed += output_queue_ptr->prev_words_in_flight_check_flush<false, input_queue_network_sequence, input_queue_cb_mode_sequence>();
    }

    if (!timeout) {
        test_results[PQ_TEST_MISC_INDEX] = 0xff00002;
        if (!output_queue_ptr->output_barrier<false, input_queue_network_sequence, input_queue_cb_mode_sequence>(timeout_cycles)) {
            timeout = true;
        }
    }

    uint64_t cycles_elapsed = get_timestamp() - start_timestamp;

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
