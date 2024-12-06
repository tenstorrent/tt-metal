// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "packet_queue_ctrl.hpp"
#include "tt_metal/impl/dispatch/kernels/packet_queue_v2.hpp"
#include "tt_metal/impl/dispatch/kernels/cq_helpers.hpp"

using namespace packet_queue;

constexpr uint32_t endpoint_id_start_index = get_compile_time_arg_val(0);

constexpr uint32_t rx_queue_start_addr_words = get_compile_time_arg_val(1);
constexpr uint32_t rx_queue_size_words = get_compile_time_arg_val(2);
constexpr uint32_t rx_queue_size_bytes = rx_queue_size_words*PACKET_WORD_SIZE_BYTES;

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

constexpr uint32_t demux_input_scratch_buffer = get_compile_time_arg_val(30);
constexpr uint32_t demux_input_remote_scratch_buffer = get_compile_time_arg_val(31);

constexpr uint32_t demux_output_scratch_buffers[MAX_SWITCH_FAN_IN] =
    {
        get_compile_time_arg_val(32),
        get_compile_time_arg_val(33),
        get_compile_time_arg_val(34),
        get_compile_time_arg_val(35)
    };
constexpr uint32_t demux_output_remote_scratch_buffers[MAX_SWITCH_FAN_IN] =
    {
        get_compile_time_arg_val(36),
        get_compile_time_arg_val(37),
        get_compile_time_arg_val(38),
        get_compile_time_arg_val(39)
    };

UnsafePacketOutputQueueVariant raw_output_queues[MAX_SWITCH_FAN_OUT];
using output_queue_network_sequence = NetworkTypeSequence<remote_tx_network_type[0], remote_tx_network_type[1], remote_tx_network_type[2], remote_tx_network_type[3]>;
using output_queue_cb_mode_sequence = CBModeTypeSequence<output_depacketize[0], output_depacketize[1], output_depacketize[2], output_depacketize[3]>;

UnsafePacketInputQueueVariant raw_input_queue;
using input_queue_network_sequence = NetworkTypeSequence<remote_rx_network_type>;
using input_queue_cb_mode_sequence = CBModeTypeSequence<false>;

inline uint8_t dest_output_queue_id(uint32_t dest_endpoint_id) {
    uint32_t dest_endpoint_index = dest_endpoint_id - endpoint_id_start_index;
    return dest_output_queue_id_map[dest_endpoint_index];
}

inline void initialize_output_queues() {
    init_params_t init_params{
        .input_queues = &raw_input_queue,
        .num_input_queues = 1,
    };
    process_queues<output_queue_network_sequence, output_queue_cb_mode_sequence>([&]<auto network_type, auto cbmode, auto sequence_i>(auto i) -> bool {
        raw_output_queues[i].template engage<network_type, cbmode, input_queue_network_sequence, input_queue_cb_mode_sequence>();

        auto* active_output_queue = raw_output_queues[i].template get<network_type, cbmode, input_queue_network_sequence, input_queue_cb_mode_sequence>();

        init_params.queue_id = (uint8_t)sequence_i + 1;
        init_params.queue_start_addr_words = remote_tx_queue_start_addr_words[sequence_i];
        init_params.queue_size_words = remote_tx_queue_size_words[sequence_i];
        init_params.remote_queue_id = (uint8_t)remote_tx_queue_id[sequence_i];
        init_params.remote_x = (uint8_t)remote_tx_x[sequence_i];
        init_params.remote_y = (uint8_t)remote_tx_y[sequence_i];

        init_params.ptrs_addr = demux_output_scratch_buffers[sequence_i];
        init_params.remote_ptrs_addr = demux_output_remote_scratch_buffers[sequence_i];

        init_params.local_sem_id = (uint8_t)output_depacketize_local_sem[sequence_i];
        init_params.remote_sem_id = (uint8_t)output_depacketize_downstream_sem[sequence_i];
        init_params.log_page_size = (uint8_t)output_depacketize_log_page_size[sequence_i];

        init_params.unpacketizer_output_remove_header = (uint16_t)output_depacketize_remove_header[sequence_i];

        active_output_queue->init(&init_params);

        return true;
    });
}

void kernel_main() {
    write_test_results(test_results, PQ_TEST_STATUS_INDEX, PACKET_QUEUE_TEST_STARTED);
    write_test_results(test_results, PQ_TEST_MISC_INDEX, 0xff000000);
    write_test_results(test_results, PQ_TEST_MISC_INDEX+1, 0xbb000000 | demux_fan_out);
    write_test_results(test_results, PQ_TEST_MISC_INDEX+2, dest_endpoint_output_map_hi);
    write_test_results(test_results, PQ_TEST_MISC_INDEX+3, dest_endpoint_output_map_lo);
    write_test_results(test_results, PQ_TEST_MISC_INDEX+4, endpoint_id_start_index);

    raw_input_queue.engage<remote_rx_network_type, false>();
    auto* input_queue = raw_input_queue.get<remote_rx_network_type, false>();
    {
        constexpr init_params_t input_queue_init_params{
            .queue_id = 0,
            .queue_start_addr_words = rx_queue_start_addr_words,
            .queue_size_words = rx_queue_size_words,
            .remote_queue_id = remote_rx_queue_id,
            .remote_x = remote_rx_x,
            .remote_y = remote_rx_y,
            .ptrs_addr = demux_input_scratch_buffer,
            .remote_ptrs_addr = demux_input_remote_scratch_buffer,
        };
        input_queue->init(&input_queue_init_params);
    }
    initialize_output_queues();

    if (!wait_all_input_output_ready<input_queue_network_sequence,
                                     input_queue_cb_mode_sequence,
                                     output_queue_network_sequence,
                                     output_queue_cb_mode_sequence>(&raw_input_queue, raw_output_queues, timeout_cycles)) {
        write_test_results(test_results, PQ_TEST_STATUS_INDEX, PACKET_QUEUE_TEST_TIMEOUT);
        return;
    }

    write_test_results(test_results, PQ_TEST_MISC_INDEX, 0xff000001);

    uint64_t start_timestamp = get_timestamp();
    bool all_outputs_finished = false;
    uint64_t data_words_sent = 0;
    uint64_t iter = 0;
    uint32_t heartbeat = 0;
    while (!all_outputs_finished) {
        IDLE_ERISC_HEARTBEAT_AND_RETURN(heartbeat);
        iter++;
        input_queue->advance_if_not_valid();
        if (input_queue->get_curr_packet_valid()) {
            uint32_t output_queue_id = dest_output_queue_id(input_queue->get_curr_packet_dest());
            bool full_packet_sent;
            switch(output_queue_id) {
                case 0:
                    data_words_sent += (raw_output_queues[output_queue_id].get<remote_tx_network_type[0], output_depacketize[0], input_queue_network_sequence, input_queue_cb_mode_sequence>())->forward_data_from_input<0>(full_packet_sent, input_queue->get_end_of_cmd());
                    break;
                case 1:
                    data_words_sent += (raw_output_queues[output_queue_id].get<remote_tx_network_type[1], output_depacketize[1], input_queue_network_sequence, input_queue_cb_mode_sequence>())->forward_data_from_input<0>(full_packet_sent, input_queue->get_end_of_cmd());
                    break;
                case 2:
                    data_words_sent += (raw_output_queues[output_queue_id].get<remote_tx_network_type[2], output_depacketize[2], input_queue_network_sequence, input_queue_cb_mode_sequence>())->forward_data_from_input<0>(full_packet_sent, input_queue->get_end_of_cmd());
                    break;
                case 3:
                    data_words_sent += (raw_output_queues[output_queue_id].get<remote_tx_network_type[3], output_depacketize[3], input_queue_network_sequence, input_queue_cb_mode_sequence>())->forward_data_from_input<0>(full_packet_sent, input_queue->get_end_of_cmd());
                    break;
                default:
                    break;
            }
        }

        all_outputs_finished = true;
        process_queues<output_queue_network_sequence, output_queue_cb_mode_sequence>([&]<auto network_type, auto cbmode, auto sequence_i>(auto) -> bool {
            auto* active_output_queue = raw_output_queues[sequence_i].template get<network_type, cbmode, input_queue_network_sequence, input_queue_cb_mode_sequence>();
            active_output_queue->prev_words_in_flight_check_flush();
            all_outputs_finished &= active_output_queue->is_remote_finished();
            return true;
        });
    }
    input_queue->advance_if_not_valid();

    write_test_results(test_results, PQ_TEST_MISC_INDEX, 0xff000002);

    bool timed_out = false;
    process_queues<output_queue_network_sequence, output_queue_cb_mode_sequence>([&]<auto network_type, auto cbmode, auto sequence_i>(auto) -> bool {
        auto* active_output_queue = raw_output_queues[sequence_i].template get<network_type, cbmode, input_queue_network_sequence, input_queue_cb_mode_sequence>();
        if (!active_output_queue->output_barrier(timeout_cycles)) {
            timed_out = true;
            return false;
        }
        return true;
    });

    uint64_t cycles_elapsed = get_timestamp() - start_timestamp;
    write_test_results(test_results, PQ_TEST_MISC_INDEX, 0xff000003);
    input_queue->send_remote_finished_notification();

    set_64b_result(test_results, data_words_sent, PQ_TEST_WORD_CNT_INDEX);
    set_64b_result(test_results, cycles_elapsed, PQ_TEST_CYCLES_INDEX);
    set_64b_result(test_results, iter, PQ_TEST_ITER_INDEX);

    if (timed_out) {
        write_test_results(test_results, PQ_TEST_STATUS_INDEX, PACKET_QUEUE_TEST_TIMEOUT);
    } else {
        write_test_results(test_results, PQ_TEST_STATUS_INDEX, PACKET_QUEUE_TEST_PASS);
    }
    write_test_results(test_results, PQ_TEST_MISC_INDEX, 0xff00005);
}
