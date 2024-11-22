// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tt_metal/impl/dispatch/kernels/packet_queue.hpp"
#include "tt_metal/impl/dispatch/kernels/cq_helpers.hpp"

packet_input_queue_state_t input_queue;
packet_output_queue_state_t output_queues[MAX_SWITCH_FAN_OUT];

constexpr uint32_t endpoint_id_start_index = get_compile_time_arg_val(0);

constexpr uint32_t rx_queue_start_addr_words = get_compile_time_arg_val(1);
constexpr uint32_t rx_queue_size_words = get_compile_time_arg_val(2);
constexpr uint32_t rx_queue_size_bytes = rx_queue_size_words*PACKET_WORD_SIZE_BYTES;

static_assert(is_power_of_2(rx_queue_size_words), "rx_queue_size_words must be a power of 2");

constexpr uint32_t demux_fan_out = get_compile_time_arg_val(3);

// FIXME imatosevic - is there a way to do this without explicit indexes?
static_assert(demux_fan_out > 0 && demux_fan_out <= MAX_SWITCH_FAN_OUT,
    "demux fan-out 0 or higher than MAX_SWITCH_FAN_OUT");
static_assert(MAX_SWITCH_FAN_OUT == 4,
    "MAX_SWITCH_FAN_OUT must be 4 for the initialization below to work");

constexpr uint32_t remote_tx_x[MAX_SWITCH_FAN_OUT] =
    {
        (get_compile_time_arg_val(4) & 0xFF),
        (get_compile_time_arg_val(5) & 0xFF),
        (get_compile_time_arg_val(6) & 0xFF),
        (get_compile_time_arg_val(7) & 0xFF)
    };

constexpr uint32_t remote_tx_y[MAX_SWITCH_FAN_OUT] =
    {
        (get_compile_time_arg_val(4) >> 8) & 0xFF,
        (get_compile_time_arg_val(5) >> 8) & 0xFF,
        (get_compile_time_arg_val(6) >> 8) & 0xFF,
        (get_compile_time_arg_val(7) >> 8) & 0xFF
    };

constexpr uint32_t remote_tx_queue_id[MAX_SWITCH_FAN_OUT] =
    {
        (get_compile_time_arg_val(4) >> 16) & 0xFF,
        (get_compile_time_arg_val(5) >> 16) & 0xFF,
        (get_compile_time_arg_val(6) >> 16) & 0xFF,
        (get_compile_time_arg_val(7) >> 16) & 0xFF
    };

constexpr DispatchRemoteNetworkType remote_tx_network_type[MAX_SWITCH_FAN_OUT] =
    {
        static_cast<DispatchRemoteNetworkType>((get_compile_time_arg_val(4) >> 24) & 0xFF),
        static_cast<DispatchRemoteNetworkType>((get_compile_time_arg_val(5) >> 24) & 0xFF),
        static_cast<DispatchRemoteNetworkType>((get_compile_time_arg_val(6) >> 24) & 0xFF),
        static_cast<DispatchRemoteNetworkType>((get_compile_time_arg_val(7) >> 24) & 0xFF)
    };

constexpr uint32_t remote_tx_queue_start_addr_words[MAX_SWITCH_FAN_OUT] =
    {
        get_compile_time_arg_val(8),
        get_compile_time_arg_val(10),
        get_compile_time_arg_val(12),
        get_compile_time_arg_val(14)
    };

constexpr uint32_t remote_tx_queue_size_words[MAX_SWITCH_FAN_OUT] =
    {
        get_compile_time_arg_val(9),
        get_compile_time_arg_val(11),
        get_compile_time_arg_val(13),
        get_compile_time_arg_val(15)
    };

static_assert(is_power_of_2(remote_tx_queue_size_words[0]), "remote_tx_queue_size_words must be a power of 2");
static_assert((demux_fan_out < 2) || is_power_of_2(remote_tx_queue_size_words[1]), "remote_tx_queue_size_words must be a power of 2");
static_assert((demux_fan_out < 3) || is_power_of_2(remote_tx_queue_size_words[2]), "remote_tx_queue_size_words must be a power of 2");
static_assert((demux_fan_out < 4) || is_power_of_2(remote_tx_queue_size_words[3]), "remote_tx_queue_size_words must be a power of 2");

constexpr uint32_t remote_rx_x = get_compile_time_arg_val(16);
constexpr uint32_t remote_rx_y = get_compile_time_arg_val(17);
constexpr uint32_t remote_rx_queue_id = get_compile_time_arg_val(18);
constexpr DispatchRemoteNetworkType
    remote_rx_network_type =
        static_cast<DispatchRemoteNetworkType>(get_compile_time_arg_val(19));

static_assert(MAX_DEST_ENDPOINTS <= 32 && MAX_SWITCH_FAN_OUT <= 4,
    "We assume MAX_DEST_ENDPOINTS <= 32 and MAX_SWITCH_FAN_OUT <= 4 for the initialization below to work");

constexpr uint32_t dest_endpoint_output_map_hi = get_compile_time_arg_val(20);
constexpr uint32_t dest_endpoint_output_map_lo = get_compile_time_arg_val(21);

constexpr uint8_t dest_output_queue_id_map[MAX_DEST_ENDPOINTS] =
    {
        (dest_endpoint_output_map_lo >> 0) & 0x3,
        (dest_endpoint_output_map_lo >> 2) & 0x3,
        (dest_endpoint_output_map_lo >> 4) & 0x3,
        (dest_endpoint_output_map_lo >> 6) & 0x3,
        (dest_endpoint_output_map_lo >> 8) & 0x3,
        (dest_endpoint_output_map_lo >> 10) & 0x3,
        (dest_endpoint_output_map_lo >> 12) & 0x3,
        (dest_endpoint_output_map_lo >> 14) & 0x3,
        (dest_endpoint_output_map_lo >> 16) & 0x3,
        (dest_endpoint_output_map_lo >> 18) & 0x3,
        (dest_endpoint_output_map_lo >> 20) & 0x3,
        (dest_endpoint_output_map_lo >> 22) & 0x3,
        (dest_endpoint_output_map_lo >> 24) & 0x3,
        (dest_endpoint_output_map_lo >> 26) & 0x3,
        (dest_endpoint_output_map_lo >> 28) & 0x3,
        (dest_endpoint_output_map_lo >> 30) & 0x3,
        (dest_endpoint_output_map_hi >> 0) & 0x3,
        (dest_endpoint_output_map_hi >> 2) & 0x3,
        (dest_endpoint_output_map_hi >> 4) & 0x3,
        (dest_endpoint_output_map_hi >> 6) & 0x3,
        (dest_endpoint_output_map_hi >> 8) & 0x3,
        (dest_endpoint_output_map_hi >> 10) & 0x3,
        (dest_endpoint_output_map_hi >> 12) & 0x3,
        (dest_endpoint_output_map_hi >> 14) & 0x3,
        (dest_endpoint_output_map_hi >> 16) & 0x3,
        (dest_endpoint_output_map_hi >> 18) & 0x3,
        (dest_endpoint_output_map_hi >> 20) & 0x3,
        (dest_endpoint_output_map_hi >> 22) & 0x3,
        (dest_endpoint_output_map_hi >> 24) & 0x3,
        (dest_endpoint_output_map_hi >> 26) & 0x3,
        (dest_endpoint_output_map_hi >> 28) & 0x3,
        (dest_endpoint_output_map_hi >> 30) & 0x3
    };

constexpr uint32_t output_queue_index_bits = 2;
constexpr uint32_t output_queue_index_mask = (1 << output_queue_index_bits) - 1;

constexpr uint32_t test_results_buf_addr_arg = get_compile_time_arg_val(22);
constexpr uint32_t test_results_buf_size_bytes = get_compile_time_arg_val(23);

// careful, may be null
tt_l1_ptr uint32_t* const test_results =
    reinterpret_cast<tt_l1_ptr uint32_t*>(test_results_buf_addr_arg);

constexpr uint32_t timeout_cycles = get_compile_time_arg_val(24);

constexpr bool output_depacketize[MAX_SWITCH_FAN_OUT] =
    {
        (get_compile_time_arg_val(25) >> 0) & 0x1,
        (get_compile_time_arg_val(25) >> 1) & 0x1,
        (get_compile_time_arg_val(25) >> 2) & 0x1,
        (get_compile_time_arg_val(25) >> 3) & 0x1
    };

constexpr uint32_t output_depacketize_log_page_size[MAX_SWITCH_FAN_OUT] =
    {
        (get_compile_time_arg_val(26) >> 0) & 0xFF,
        (get_compile_time_arg_val(27) >> 0) & 0xFF,
        (get_compile_time_arg_val(28) >> 0) & 0xFF,
        (get_compile_time_arg_val(29) >> 0) & 0xFF
    };

constexpr uint32_t output_depacketize_downstream_sem[MAX_SWITCH_FAN_OUT] =
    {
        (get_compile_time_arg_val(26) >> 8) & 0xFF,
        (get_compile_time_arg_val(27) >> 8) & 0xFF,
        (get_compile_time_arg_val(28) >> 8) & 0xFF,
        (get_compile_time_arg_val(29) >> 8) & 0xFF
    };

constexpr uint32_t output_depacketize_local_sem[MAX_SWITCH_FAN_OUT] =
    {
        (get_compile_time_arg_val(26) >> 16) & 0xFF,
        (get_compile_time_arg_val(27) >> 16) & 0xFF,
        (get_compile_time_arg_val(28) >> 16) & 0xFF,
        (get_compile_time_arg_val(29) >> 16) & 0xFF
    };

constexpr uint32_t output_depacketize_remove_header[MAX_SWITCH_FAN_OUT] =
    {
        (get_compile_time_arg_val(26) >> 24) & 0x1,
        (get_compile_time_arg_val(27) >> 24) & 0x1,
        (get_compile_time_arg_val(28) >> 24) & 0x1,
        (get_compile_time_arg_val(29) >> 24) & 0x1
    };



inline uint8_t dest_output_queue_id(uint32_t dest_endpoint_id) {
    uint32_t dest_endpoint_index = dest_endpoint_id - endpoint_id_start_index;
    return dest_output_queue_id_map[dest_endpoint_index];
}

void kernel_main() {

    write_test_results(test_results, PQ_TEST_STATUS_INDEX, PACKET_QUEUE_TEST_STARTED);
    write_test_results(test_results, PQ_TEST_MISC_INDEX, 0xff000000);
    write_test_results(test_results, PQ_TEST_MISC_INDEX+1, 0xbb000000 | demux_fan_out);
    write_test_results(test_results, PQ_TEST_MISC_INDEX+2, dest_endpoint_output_map_hi);
    write_test_results(test_results, PQ_TEST_MISC_INDEX+3, dest_endpoint_output_map_lo);
    write_test_results(test_results, PQ_TEST_MISC_INDEX+4, endpoint_id_start_index);

    for (uint32_t i = 0; i < demux_fan_out; i++) {
        output_queues[i].init(i + 1, remote_tx_queue_start_addr_words[i], remote_tx_queue_size_words[i],
                              remote_tx_x[i], remote_tx_y[i], remote_tx_queue_id[i], remote_tx_network_type[i],
                              &input_queue, 1,
                              output_depacketize[i], output_depacketize_log_page_size[i],
                              output_depacketize_local_sem[i], output_depacketize_downstream_sem[i],
                              output_depacketize_remove_header[i]);
    }
    input_queue.init(0, rx_queue_start_addr_words, rx_queue_size_words,
                     remote_rx_x, remote_rx_y, remote_rx_queue_id, remote_rx_network_type);

    if (!wait_all_src_dest_ready(&input_queue, 1, output_queues, demux_fan_out, timeout_cycles)) {
        write_test_results(test_results, PQ_TEST_STATUS_INDEX, PACKET_QUEUE_TEST_TIMEOUT);
        return;
    }

    write_test_results(test_results, PQ_TEST_MISC_INDEX, 0xff000001);

    bool timeout = false;
    bool all_outputs_finished = false;
    uint64_t data_words_sent = 0;
    uint64_t iter = 0;
    uint64_t start_timestamp = get_timestamp();
    uint32_t progress_timestamp = start_timestamp & 0xFFFFFFFF;
    uint32_t heartbeat = 0;
    while (!all_outputs_finished && !timeout) {
        IDLE_ERISC_HEARTBEAT_AND_RETURN(heartbeat);
        iter++;
        if (timeout_cycles > 0) {
            uint32_t cycles_since_progress = get_timestamp_32b() - progress_timestamp;
            if (cycles_since_progress > timeout_cycles) {
                timeout = true;
                break;
            }
        }
        if (input_queue.get_curr_packet_valid()) {
            uint32_t dest = input_queue.get_curr_packet_dest();
            uint8_t output_queue_id = dest_output_queue_id(dest);
            bool full_packet_sent;
            uint32_t words_sent = output_queues[output_queue_id].forward_data_from_input(0, full_packet_sent, input_queue.get_end_of_cmd());
            data_words_sent += words_sent;
            if ((words_sent > 0) && (timeout_cycles > 0)) {
                progress_timestamp = get_timestamp_32b();
            }
        }
        all_outputs_finished = true;
        for (uint32_t i = 0; i < demux_fan_out; i++) {
            output_queues[i].prev_words_in_flight_check_flush();
            all_outputs_finished &= output_queues[i].is_remote_finished();
        }
    }

    if (!timeout) {
        write_test_results(test_results, PQ_TEST_MISC_INDEX, 0xff000002);
        for (uint32_t i = 0; i < demux_fan_out; i++) {
            if (!output_queues[i].output_barrier(timeout_cycles)) {
                timeout = true;
                break;
            }
        }
    }

    uint64_t cycles_elapsed = get_timestamp() - start_timestamp;
    if (!timeout) {
        write_test_results(test_results, PQ_TEST_MISC_INDEX, 0xff000003);
        input_queue.send_remote_finished_notification();
    }

    set_64b_result(test_results, data_words_sent, PQ_TEST_WORD_CNT_INDEX);
    set_64b_result(test_results, cycles_elapsed, PQ_TEST_CYCLES_INDEX);
    set_64b_result(test_results, iter, PQ_TEST_ITER_INDEX);

    if (timeout) {
        write_test_results(test_results, PQ_TEST_STATUS_INDEX, PACKET_QUEUE_TEST_TIMEOUT);
        // DPRINT << "demux timeout" << ENDL();
        // // input_queue.dprint_object();
        // output_queues[0].dprint_object();
    } else {
        write_test_results(test_results, PQ_TEST_STATUS_INDEX, PACKET_QUEUE_TEST_PASS);
        write_test_results(test_results, PQ_TEST_MISC_INDEX, 0xff00005);
    }
}
