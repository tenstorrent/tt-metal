// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tt_metal/impl/dispatch/kernels/packet_queue_v2.hpp"
#include "tt_metal/impl/dispatch/kernels/cq_helpers.hpp"

using namespace packet_queue;

constexpr uint32_t reserved = get_compile_time_arg_val(0);

// assume up to MAX_SWITCH_FAN_IN queues with contiguous storage,
// starting at rx_queue_start_addr
constexpr uint32_t rx_queue_start_addr_words = get_compile_time_arg_val(1);
constexpr uint32_t rx_queue_size_words = get_compile_time_arg_val(2);
constexpr uint32_t rx_queue_size_bytes = rx_queue_size_words*PACKET_WORD_SIZE_BYTES;

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

constexpr uint32_t mux_input_scratch_buffers[MAX_SWITCH_FAN_IN] =
    {
        get_compile_time_arg_val(25),
        get_compile_time_arg_val(26),
        get_compile_time_arg_val(27),
        get_compile_time_arg_val(28)
    };
constexpr uint32_t mux_input_remote_scratch_buffers[MAX_SWITCH_FAN_IN] =
    {
        get_compile_time_arg_val(29),
        get_compile_time_arg_val(30),
        get_compile_time_arg_val(31),
        get_compile_time_arg_val(32)
    };

constexpr uint32_t mux_output_scratch_buffer = get_compile_time_arg_val(33);
constexpr uint32_t mux_output_remote_scratch_buffer = get_compile_time_arg_val(34);

UnsafePacketInputQueueVariant raw_input_queues[MAX_SWITCH_FAN_IN];
using input_queue_network_sequence = NetworkTypeSequence<remote_rx_network_type[0], remote_rx_network_type[1], remote_rx_network_type[2], remote_rx_network_type[3]>;
using input_queue_cb_mode_sequence = CBModeTypeSequence<input_packetize[0], input_packetize[1], input_packetize[2], input_packetize[3]>;

UnsafePacketOutputQueueVariant raw_output_queue;
constexpr init_params_t output_queue_init_params{
    .queue_id = mux_fan_in,
    .queue_start_addr_words = remote_tx_queue_start_addr_words,
    .queue_size_words = remote_tx_queue_size_words,
    .remote_queue_id = remote_tx_queue_id,
    .remote_x = remote_tx_x,
    .remote_y = remote_tx_y,
    .ptrs_addr = mux_output_scratch_buffer,
    .remote_ptrs_addr = mux_output_remote_scratch_buffer,

    .local_sem_id = output_depacketize_local_sem,
    .remote_sem_id = output_depacketize_downstream_sem,
    .log_page_size = output_depacketize_log_page_size,

    .input_queues = raw_input_queues,
    .num_input_queues = mux_fan_in,
    .unpacketizer_output_remove_header = output_depacketize_remove_header,
};
using output_queue_network_sequence = NetworkTypeSequence<tx_network_type>;
using output_queue_cb_mode_sequence = CBModeTypeSequence<output_depacketize>;

inline void initialize_input_queues() {
    init_params_t init_params{};
    process_queues<input_queue_network_sequence, input_queue_cb_mode_sequence>([&]<auto network_type, auto cbmode, auto sequence_i>(auto) -> bool {
        raw_input_queues[sequence_i].template engage<network_type, cbmode>();

        auto* active_input_queue = raw_input_queues[sequence_i].template get<network_type, cbmode>();

        init_params.queue_id = (uint8_t)sequence_i;
        init_params.queue_start_addr_words = rx_queue_start_addr_words + sequence_i * rx_queue_size_words;
        init_params.queue_size_words = rx_queue_size_words;
        init_params.remote_queue_id = (uint8_t)remote_rx_queue_id[sequence_i];
        init_params.remote_x = (uint8_t)remote_rx_x[sequence_i];
        init_params.remote_y = (uint8_t)remote_rx_y[sequence_i];

        init_params.ptrs_addr = mux_input_scratch_buffers[sequence_i];
        init_params.remote_ptrs_addr = mux_input_remote_scratch_buffers[sequence_i];

        init_params.local_sem_id = (uint8_t)input_packetize_local_sem[sequence_i];
        init_params.remote_sem_id = (uint8_t)input_packetize_upstream_sem[sequence_i];
        init_params.log_page_size = (uint8_t)input_packetize_log_page_size[sequence_i];

        init_params.packetizer_input_src = (uint16_t)input_packetize_src_endpoint[sequence_i];
        init_params.packetizer_input_dest = (uint16_t)input_packetize_dest_endpoint[sequence_i];

        active_input_queue->init(&init_params);

        return true;
    });
}

inline void initialize_output_queues() {
    raw_output_queue.engage<tx_network_type, output_depacketize, input_queue_network_sequence, input_queue_cb_mode_sequence>();
    auto* output_queue = raw_output_queue.get<tx_network_type, output_depacketize, input_queue_network_sequence, input_queue_cb_mode_sequence>();
    output_queue->init(&output_queue_init_params);
}

void kernel_main() {
    initialize_input_queues();
    initialize_output_queues();

    auto* output_queue = raw_output_queue.get<tx_network_type, output_depacketize, input_queue_network_sequence, input_queue_cb_mode_sequence>();

    if (!wait_all_input_output_ready<input_queue_network_sequence,
                                     input_queue_cb_mode_sequence,
                                     output_queue_network_sequence,
                                     output_queue_cb_mode_sequence>(raw_input_queues, &raw_output_queue, timeout_cycles)) {
        write_test_results(test_results, PQ_TEST_STATUS_INDEX, PACKET_QUEUE_TEST_TIMEOUT);
        return;
    }

    bool dest_finished = false;
    bool curr_input_partial_packet_sent = false;
    uint32_t partial_packet_sent_index = 0;
    uint64_t data_words_sent = 0;
    uint64_t iter = 0;
    uint64_t start_timestamp = get_timestamp();
    uint32_t heartbeat = 0;
    while (!dest_finished) {
        IDLE_ERISC_HEARTBEAT_AND_RETURN(heartbeat);
        iter++;

        process_queues<input_queue_network_sequence, input_queue_cb_mode_sequence>([&]<auto network_type, auto cbmode, auto sequence_i>(auto i) -> bool {
            if (curr_input_partial_packet_sent && partial_packet_sent_index != i) return true;
            auto* active_input_queue = raw_input_queues[i].template get<network_type, cbmode>();
            curr_input_partial_packet_sent = false;
            active_input_queue->advance_if_not_valid();
            if (active_input_queue->get_curr_packet_valid()) {
                bool full_packet_sent;
                uint32_t words_sent = output_queue->forward_data_from_input<sequence_i>(full_packet_sent, active_input_queue->get_end_of_cmd());
                data_words_sent += words_sent;
                curr_input_partial_packet_sent = !full_packet_sent;
            }

            if (curr_input_partial_packet_sent) {
                partial_packet_sent_index = i;
                // stop looping at this queue. come back to it at the next iteration from the outer while loop
                return false;
            }

            return true; // keep looping
        });

        output_queue->prev_words_in_flight_check_flush();
        dest_finished = output_queue->is_remote_finished();
    }

    if (!output_queue->output_barrier(timeout_cycles)) {
        write_test_results(test_results, PQ_TEST_STATUS_INDEX, PACKET_QUEUE_TEST_TIMEOUT);
        return;
    }

    uint64_t cycles_elapsed = get_timestamp() - start_timestamp;
    process_queues<input_queue_network_sequence, input_queue_cb_mode_sequence>([&]<auto network_type, auto cbmode, auto sequence_i>(auto i) -> bool {
        auto* active_input_queue = raw_input_queues[i].template get<network_type, cbmode>();
        active_input_queue->send_remote_finished_notification();
        return true;
    });

    set_64b_result(test_results, data_words_sent, PQ_TEST_WORD_CNT_INDEX);
    set_64b_result(test_results, cycles_elapsed, PQ_TEST_CYCLES_INDEX);
    set_64b_result(test_results, iter, PQ_TEST_ITER_INDEX);

    write_test_results(test_results, PQ_TEST_STATUS_INDEX, PACKET_QUEUE_TEST_PASS);
    write_test_results(test_results, PQ_TEST_MISC_INDEX, 0xff00005);
}
