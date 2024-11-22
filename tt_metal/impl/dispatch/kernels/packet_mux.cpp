// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tt_metal/impl/dispatch/kernels/packet_queue.hpp"
#include "tt_metal/impl/dispatch/kernels/cq_helpers.hpp"

packet_input_queue_state_t input_queues[MAX_SWITCH_FAN_IN];
packet_output_queue_state_t output_queue;

constexpr uint32_t reserved = get_compile_time_arg_val(0);

// assume up to MAX_SWITCH_FAN_IN queues with contiguous storage,
// starting at rx_queue_start_addr
constexpr uint32_t rx_queue_start_addr_words = get_compile_time_arg_val(1);
constexpr uint32_t rx_queue_size_words = get_compile_time_arg_val(2);
constexpr uint32_t rx_queue_size_bytes = rx_queue_size_words*PACKET_WORD_SIZE_BYTES;

static_assert(is_power_of_2(rx_queue_size_words), "rx_queue_size_words must be a power of 2");

constexpr uint32_t mux_fan_in = get_compile_time_arg_val(3);

// FIXME imatosevic - is there a way to do this without explicit indexes?
static_assert(mux_fan_in > 0 && mux_fan_in <= MAX_SWITCH_FAN_IN,
    "mux fan-in 0 or higher than MAX_SWITCH_FAN_IN");
static_assert(MAX_SWITCH_FAN_IN == 4,
    "MAX_SWITCH_FAN_IN must be 4 for the initialization below to work");

constexpr uint32_t remote_rx_x[MAX_SWITCH_FAN_IN] =
    {
        (get_compile_time_arg_val(4) & 0xFF),
        (get_compile_time_arg_val(5) & 0xFF),
        (get_compile_time_arg_val(6) & 0xFF),
        (get_compile_time_arg_val(7) & 0xFF)
    };

constexpr uint32_t remote_rx_y[MAX_SWITCH_FAN_IN] =
    {
        (get_compile_time_arg_val(4) >> 8) & 0xFF,
        (get_compile_time_arg_val(5) >> 8) & 0xFF,
        (get_compile_time_arg_val(6) >> 8) & 0xFF,
        (get_compile_time_arg_val(7) >> 8) & 0xFF
    };

constexpr uint32_t remote_rx_queue_id[MAX_SWITCH_FAN_IN] =
    {
        (get_compile_time_arg_val(4) >> 16) & 0xFF,
        (get_compile_time_arg_val(5) >> 16) & 0xFF,
        (get_compile_time_arg_val(6) >> 16) & 0xFF,
        (get_compile_time_arg_val(7) >> 16) & 0xFF
    };

constexpr DispatchRemoteNetworkType remote_rx_network_type[MAX_SWITCH_FAN_IN] =
    {
        static_cast<DispatchRemoteNetworkType>((get_compile_time_arg_val(4) >> 24) & 0xFF),
        static_cast<DispatchRemoteNetworkType>((get_compile_time_arg_val(5) >> 24) & 0xFF),
        static_cast<DispatchRemoteNetworkType>((get_compile_time_arg_val(6) >> 24) & 0xFF),
        static_cast<DispatchRemoteNetworkType>((get_compile_time_arg_val(7) >> 24) & 0xFF)
    };

constexpr uint32_t remote_tx_queue_start_addr_words = get_compile_time_arg_val(8);
constexpr uint32_t remote_tx_queue_size_words = get_compile_time_arg_val(9);

static_assert(is_power_of_2(remote_tx_queue_size_words), "remote_tx_queue_size_words must be a power of 2");

constexpr uint32_t remote_tx_x = get_compile_time_arg_val(10);
constexpr uint32_t remote_tx_y = get_compile_time_arg_val(11);
constexpr uint32_t remote_tx_queue_id = get_compile_time_arg_val(12);
constexpr DispatchRemoteNetworkType
    tx_network_type =
        static_cast<DispatchRemoteNetworkType>(get_compile_time_arg_val(13));

constexpr uint32_t test_results_buf_addr_arg = get_compile_time_arg_val(14);
constexpr uint32_t test_results_buf_size_bytes = get_compile_time_arg_val(15);

// careful, may be null
tt_l1_ptr uint32_t* const test_results =
    reinterpret_cast<tt_l1_ptr uint32_t*>(test_results_buf_addr_arg);

constexpr uint32_t timeout_cycles = get_compile_time_arg_val(16);

constexpr bool output_depacketize = get_compile_time_arg_val(17);
constexpr uint32_t output_depacketize_info = get_compile_time_arg_val(18);

constexpr uint32_t output_depacketize_log_page_size = output_depacketize_info & 0xFF;
constexpr uint32_t output_depacketize_downstream_sem = (output_depacketize_info >> 8) & 0xFF;
constexpr uint32_t output_depacketize_local_sem = (output_depacketize_info >> 16) & 0xFF;
constexpr bool output_depacketize_remove_header = (output_depacketize_info >> 24) & 0x1;

constexpr uint32_t input_packetize[MAX_SWITCH_FAN_IN] =
    {
        (get_compile_time_arg_val(19) >> 0) & 0x1,
        (get_compile_time_arg_val(20) >> 0) & 0x1,
        (get_compile_time_arg_val(21) >> 0) & 0x1,
        (get_compile_time_arg_val(22) >> 0) & 0x1
    };

constexpr uint32_t input_packetize_log_page_size[MAX_SWITCH_FAN_IN] =
    {
        (get_compile_time_arg_val(19) >> 8) & 0xFF,
        (get_compile_time_arg_val(20) >> 8) & 0xFF,
        (get_compile_time_arg_val(21) >> 8) & 0xFF,
        (get_compile_time_arg_val(22) >> 8) & 0xFF
    };

constexpr uint32_t input_packetize_upstream_sem[MAX_SWITCH_FAN_IN] =
    {
        (get_compile_time_arg_val(19) >> 16) & 0xFF,
        (get_compile_time_arg_val(20) >> 16) & 0xFF,
        (get_compile_time_arg_val(21) >> 16) & 0xFF,
        (get_compile_time_arg_val(22) >> 16) & 0xFF
    };

constexpr uint32_t input_packetize_local_sem[MAX_SWITCH_FAN_IN] =
    {
        (get_compile_time_arg_val(19) >> 24) & 0xFF,
        (get_compile_time_arg_val(20) >> 24) & 0xFF,
        (get_compile_time_arg_val(21) >> 24) & 0xFF,
        (get_compile_time_arg_val(22) >> 24) & 0xFF
    };

constexpr uint32_t input_packetize_src_endpoint[MAX_SWITCH_FAN_IN] =
    {
        (get_compile_time_arg_val(23) >> 0) & 0xFF,
        (get_compile_time_arg_val(23) >> 8) & 0xFF,
        (get_compile_time_arg_val(23) >> 16) & 0xFF,
        (get_compile_time_arg_val(23) >> 24) & 0xFF
    };

constexpr uint32_t input_packetize_dest_endpoint[MAX_SWITCH_FAN_IN] =
    {
        (get_compile_time_arg_val(24) >> 0) & 0xFF,
        (get_compile_time_arg_val(24) >> 8) & 0xFF,
        (get_compile_time_arg_val(24) >> 16) & 0xFF,
        (get_compile_time_arg_val(24) >> 24) & 0xFF
    };


void kernel_main() {

    write_test_results(test_results, PQ_TEST_STATUS_INDEX, PACKET_QUEUE_TEST_STARTED);
    write_test_results(test_results, PQ_TEST_MISC_INDEX, 0xff000000);
    write_test_results(test_results, PQ_TEST_MISC_INDEX+1, 0xaa000000 | mux_fan_in);

    for (uint32_t i = 0; i < mux_fan_in; i++) {
        input_queues[i].init(i, rx_queue_start_addr_words + i*rx_queue_size_words, rx_queue_size_words,
                             remote_rx_x[i], remote_rx_y[i], remote_rx_queue_id[i], remote_rx_network_type[i],
                             input_packetize[i], input_packetize_log_page_size[i],
                             input_packetize_local_sem[i], input_packetize_upstream_sem[i],
                             input_packetize_src_endpoint[i], input_packetize_dest_endpoint[i]);
    }

    output_queue.init(mux_fan_in, remote_tx_queue_start_addr_words, remote_tx_queue_size_words,
                      remote_tx_x, remote_tx_y, remote_tx_queue_id, tx_network_type,
                      input_queues, mux_fan_in,
                      output_depacketize, output_depacketize_log_page_size,
                      output_depacketize_downstream_sem, output_depacketize_local_sem,
                      output_depacketize_remove_header);

    if (!wait_all_src_dest_ready(input_queues, mux_fan_in, &output_queue, 1, timeout_cycles)) {
        write_test_results(test_results, PQ_TEST_STATUS_INDEX, PACKET_QUEUE_TEST_TIMEOUT);
        return;
    }

    write_test_results(test_results, PQ_TEST_MISC_INDEX, 0xff000001);

    uint32_t curr_input = 0;
    bool timeout = false;
    bool dest_finished = false;
    bool curr_input_partial_packet_sent = false;
    uint64_t data_words_sent = 0;
    uint64_t iter = 0;
    uint64_t start_timestamp = get_timestamp();
    uint32_t progress_timestamp = start_timestamp & 0xFFFFFFFF;
    uint32_t heartbeat = 0;
    while (!dest_finished && !timeout) {
        IDLE_ERISC_HEARTBEAT_AND_RETURN(heartbeat);
        iter++;
        if (timeout_cycles > 0) {
            uint32_t cycles_since_progress = get_timestamp_32b() - progress_timestamp;
            if (cycles_since_progress > timeout_cycles) {
                timeout = true;
                break;
            }
        }
        if (input_queues[curr_input].get_curr_packet_valid()) {
            bool full_packet_sent;
            uint32_t words_sent = output_queue.forward_data_from_input(curr_input, full_packet_sent, input_queues[curr_input].get_end_of_cmd());
            data_words_sent += words_sent;
            if ((words_sent > 0) && (timeout_cycles > 0)) {
                progress_timestamp = get_timestamp_32b();
            }
            curr_input_partial_packet_sent = !full_packet_sent;
        }
        if (!curr_input_partial_packet_sent) {
            curr_input++;
            if (curr_input == mux_fan_in) {
                curr_input = 0;
            }
        }
        output_queue.prev_words_in_flight_check_flush();
        dest_finished = output_queue.is_remote_finished();
    }

    if (!timeout) {
        write_test_results(test_results, PQ_TEST_MISC_INDEX, 0xff000002);
        if (!output_queue.output_barrier(timeout_cycles)) {
            timeout = true;
        }
    }

    uint64_t cycles_elapsed = get_timestamp() - start_timestamp;
    if (!timeout) {
        write_test_results(test_results, PQ_TEST_MISC_INDEX, 0xff000003);
        for (uint32_t i = 0; i < mux_fan_in; i++) {
            input_queues[i].send_remote_finished_notification();
        }
    }

    set_64b_result(test_results, data_words_sent, PQ_TEST_WORD_CNT_INDEX);
    set_64b_result(test_results, cycles_elapsed, PQ_TEST_CYCLES_INDEX);
    set_64b_result(test_results, iter, PQ_TEST_ITER_INDEX);

    if (timeout) {
        write_test_results(test_results, PQ_TEST_STATUS_INDEX, PACKET_QUEUE_TEST_TIMEOUT);
        // DPRINT << "mux timeout" << ENDL();
        // input_queues[0].dprint_object();
        // output_queue.dprint_object();
    } else {
        write_test_results(test_results, PQ_TEST_STATUS_INDEX, PACKET_QUEUE_TEST_PASS);
        write_test_results(test_results, PQ_TEST_MISC_INDEX, 0xff00005);
    }
}
