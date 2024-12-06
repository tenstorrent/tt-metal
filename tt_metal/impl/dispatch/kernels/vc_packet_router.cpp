// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tt_metal/impl/dispatch/kernels/cq_helpers.hpp"
#include "packet_queue_ctrl.hpp"
#include "tt_metal/impl/dispatch/kernels/packet_queue_v2.hpp"

using namespace packet_queue;

constexpr uint32_t rx_queue_start_addr_words = get_compile_time_arg_val(1);
constexpr uint32_t rx_queue_size_words = get_compile_time_arg_val(2);
constexpr uint32_t rx_queue_size_bytes = rx_queue_size_words*PACKET_WORD_SIZE_BYTES;

constexpr uint32_t router_lanes = get_compile_time_arg_val(3);

// FIXME imatosevic - is there a way to do this without explicit indexes?
static_assert(router_lanes > 0 && router_lanes <= MAX_SWITCH_FAN_OUT,
    "demux fan-out 0 or higher than MAX_SWITCH_FAN_OUT");
static_assert(MAX_SWITCH_FAN_OUT == 4,
    "MAX_SWITCH_FAN_OUT must be 4 for the initialization below to work");

constexpr uint8_t remote_tx_x[MAX_SWITCH_FAN_OUT] =
    {
        (get_compile_time_arg_val(4) & 0xFF),
        (get_compile_time_arg_val(5) & 0xFF),
        (get_compile_time_arg_val(6) & 0xFF),
        (get_compile_time_arg_val(7) & 0xFF)
    };

constexpr uint8_t remote_tx_y[MAX_SWITCH_FAN_OUT] =
    {
        (get_compile_time_arg_val(4) >> 8) & 0xFF,
        (get_compile_time_arg_val(5) >> 8) & 0xFF,
        (get_compile_time_arg_val(6) >> 8) & 0xFF,
        (get_compile_time_arg_val(7) >> 8) & 0xFF
    };

constexpr uint8_t remote_tx_queue_id[MAX_SWITCH_FAN_OUT] =
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

constexpr uint8_t remote_rx_x[MAX_SWITCH_FAN_OUT] =
    {
        (get_compile_time_arg_val(16) & 0xFF),
        (get_compile_time_arg_val(17) & 0xFF),
        (get_compile_time_arg_val(18) & 0xFF),
        (get_compile_time_arg_val(19) & 0xFF)
    };

constexpr uint8_t remote_rx_y[MAX_SWITCH_FAN_OUT] =
    {
        (get_compile_time_arg_val(16) >> 8) & 0xFF,
        (get_compile_time_arg_val(17) >> 8) & 0xFF,
        (get_compile_time_arg_val(18) >> 8) & 0xFF,
        (get_compile_time_arg_val(19) >> 8) & 0xFF
    };

constexpr uint8_t remote_rx_queue_id[MAX_SWITCH_FAN_OUT] =
    {
        (get_compile_time_arg_val(16) >> 16) & 0xFF,
        (get_compile_time_arg_val(17) >> 16) & 0xFF,
        (get_compile_time_arg_val(18) >> 16) & 0xFF,
        (get_compile_time_arg_val(19) >> 16) & 0xFF
    };

constexpr DispatchRemoteNetworkType remote_rx_network_type[MAX_SWITCH_FAN_OUT] =
    {
        static_cast<DispatchRemoteNetworkType>((get_compile_time_arg_val(16) >> 24) & 0xFF),
        static_cast<DispatchRemoteNetworkType>((get_compile_time_arg_val(17) >> 24) & 0xFF),
        static_cast<DispatchRemoteNetworkType>((get_compile_time_arg_val(18) >> 24) & 0xFF),
        static_cast<DispatchRemoteNetworkType>((get_compile_time_arg_val(19) >> 24) & 0xFF)
    };

constexpr uint32_t kernel_status_buf_addr_arg = get_compile_time_arg_val(22);
constexpr uint32_t kernel_status_buf_size_bytes = get_compile_time_arg_val(23);

// careful, may be null
tt_l1_ptr uint32_t* const kernel_status =
    reinterpret_cast<tt_l1_ptr uint32_t*>(kernel_status_buf_addr_arg);

constexpr uint32_t timeout_cycles = get_compile_time_arg_val(24);

constexpr bool output_depacketize[MAX_SWITCH_FAN_OUT] =
    {
        (get_compile_time_arg_val(25) >> 0) & 0x1,
        (get_compile_time_arg_val(25) >> 1) & 0x1,
        (get_compile_time_arg_val(25) >> 2) & 0x1,
        (get_compile_time_arg_val(25) >> 3) & 0x1
    };

constexpr uint8_t output_depacketize_log_page_size[MAX_SWITCH_FAN_OUT] =
    {
        (get_compile_time_arg_val(26) >> 0) & 0xFF,
        (get_compile_time_arg_val(27) >> 0) & 0xFF,
        (get_compile_time_arg_val(28) >> 0) & 0xFF,
        (get_compile_time_arg_val(29) >> 0) & 0xFF
    };

constexpr uint8_t output_depacketize_downstream_sem[MAX_SWITCH_FAN_OUT] =
    {
        (get_compile_time_arg_val(26) >> 8) & 0xFF,
        (get_compile_time_arg_val(27) >> 8) & 0xFF,
        (get_compile_time_arg_val(28) >> 8) & 0xFF,
        (get_compile_time_arg_val(29) >> 8) & 0xFF
    };

constexpr uint8_t output_depacketize_local_sem[MAX_SWITCH_FAN_OUT] =
    {
        (get_compile_time_arg_val(26) >> 16) & 0xFF,
        (get_compile_time_arg_val(27) >> 16) & 0xFF,
        (get_compile_time_arg_val(28) >> 16) & 0xFF,
        (get_compile_time_arg_val(29) >> 16) & 0xFF
    };

constexpr uint8_t output_depacketize_remove_header[MAX_SWITCH_FAN_OUT] =
    {
        (get_compile_time_arg_val(26) >> 24) & 0x1,
        (get_compile_time_arg_val(27) >> 24) & 0x1,
        (get_compile_time_arg_val(28) >> 24) & 0x1,
        (get_compile_time_arg_val(29) >> 24) & 0x1
    };

constexpr uint8_t input_packetize[MAX_SWITCH_FAN_IN] =
    {
        (get_compile_time_arg_val(30) >> 0) & 0x1,
        (get_compile_time_arg_val(31) >> 0) & 0x1,
        (get_compile_time_arg_val(32) >> 0) & 0x1,
        (get_compile_time_arg_val(33) >> 0) & 0x1
    };

constexpr uint8_t input_packetize_log_page_size[MAX_SWITCH_FAN_IN] =
    {
        (get_compile_time_arg_val(30) >> 8) & 0xFF,
        (get_compile_time_arg_val(31) >> 8) & 0xFF,
        (get_compile_time_arg_val(32) >> 8) & 0xFF,
        (get_compile_time_arg_val(33) >> 8) & 0xFF
    };

constexpr uint8_t input_packetize_upstream_sem[MAX_SWITCH_FAN_IN] =
    {
        (get_compile_time_arg_val(30) >> 16) & 0xFF,
        (get_compile_time_arg_val(31) >> 16) & 0xFF,
        (get_compile_time_arg_val(32) >> 16) & 0xFF,
        (get_compile_time_arg_val(33) >> 16) & 0xFF
    };

constexpr uint8_t input_packetize_local_sem[MAX_SWITCH_FAN_IN] =
    {
        (get_compile_time_arg_val(30) >> 24) & 0xFF,
        (get_compile_time_arg_val(31) >> 24) & 0xFF,
        (get_compile_time_arg_val(32) >> 24) & 0xFF,
        (get_compile_time_arg_val(33) >> 24) & 0xFF
    };

constexpr uint8_t input_packetize_src_endpoint[MAX_SWITCH_FAN_IN] =
    {
        (get_compile_time_arg_val(34) >> 0) & 0xFF,
        (get_compile_time_arg_val(34) >> 8) & 0xFF,
        (get_compile_time_arg_val(34) >> 16) & 0xFF,
        (get_compile_time_arg_val(34) >> 24) & 0xFF
    };

constexpr uint8_t input_packetize_dest_endpoint[MAX_SWITCH_FAN_IN] =
    {
        (get_compile_time_arg_val(35) >> 0) & 0xFF,
        (get_compile_time_arg_val(35) >> 8) & 0xFF,
        (get_compile_time_arg_val(35) >> 16) & 0xFF,
        (get_compile_time_arg_val(35) >> 24) & 0xFF
    };

constexpr uint32_t vc_packet_router_input_scratch_buffers[MAX_SWITCH_FAN_IN] =
    {
        get_compile_time_arg_val(36),
        get_compile_time_arg_val(37),
        get_compile_time_arg_val(38),
        get_compile_time_arg_val(39),
    };

constexpr uint32_t vc_packet_router_input_remote_scratch_buffers[MAX_SWITCH_FAN_IN] =
    {
        get_compile_time_arg_val(40),
        get_compile_time_arg_val(41),
        get_compile_time_arg_val(42),
        get_compile_time_arg_val(43),
    };

constexpr uint32_t vc_packet_router_output_scratch_buffers[MAX_SWITCH_FAN_OUT] =
    {
        get_compile_time_arg_val(44),
        get_compile_time_arg_val(45),
        get_compile_time_arg_val(46),
        get_compile_time_arg_val(47),
    };

constexpr uint32_t vc_packet_router_output_remote_scratch_buffers[MAX_SWITCH_FAN_OUT] =
    {
        get_compile_time_arg_val(48),
        get_compile_time_arg_val(49),
        get_compile_time_arg_val(50),
        get_compile_time_arg_val(51),
    };

UnsafePacketInputQueueVariant raw_input_queues[MAX_SWITCH_FAN_IN];
using input_queue_network_sequence = NetworkTypeSequence<remote_rx_network_type[0], remote_rx_network_type[1], remote_rx_network_type[2], remote_rx_network_type[3]>;
using input_queue_cb_mode_sequence = CBModeTypeSequence<input_packetize[0], input_packetize[1], input_packetize[2], input_packetize[3]>;

UnsafePacketOutputQueueVariant raw_output_queues[MAX_SWITCH_FAN_OUT];
using output_queue_network_sequence = NetworkTypeSequence<remote_tx_network_type[0], remote_tx_network_type[1], remote_tx_network_type[2], remote_tx_network_type[3]>;
using output_queue_cb_mode_sequence = CBModeTypeSequence<output_depacketize[0], output_depacketize[1], output_depacketize[2], output_depacketize[3]>;

inline void initialize_input_queues() {
    init_params_t init_params{
        .queue_size_words = rx_queue_size_words,
    };

    process_queues<input_queue_network_sequence, input_queue_cb_mode_sequence>([&]<auto network_type, auto cbmode, auto sequence_i>(auto) -> bool {
        raw_input_queues[sequence_i].template engage<network_type, cbmode>();

        auto* active_input_queue = raw_input_queues[sequence_i].template get<network_type, cbmode>();
        init_params.queue_id = (uint8_t)sequence_i;
        init_params.queue_start_addr_words = rx_queue_start_addr_words + sequence_i * rx_queue_size_words;
        init_params.queue_size_words = rx_queue_size_words;
        init_params.remote_queue_id = (uint8_t)remote_rx_queue_id[sequence_i];
        init_params.remote_x = remote_rx_x[sequence_i];
        init_params.remote_y = remote_rx_y[sequence_i];
        init_params.ptrs_addr = vc_packet_router_input_scratch_buffers[sequence_i];
        init_params.remote_ptrs_addr = vc_packet_router_input_remote_scratch_buffers[sequence_i];

        init_params.local_sem_id = (uint8_t)input_packetize_local_sem[sequence_i];
        init_params.remote_sem_id = (uint8_t)input_packetize_upstream_sem[sequence_i];
        init_params.log_page_size = (uint8_t)input_packetize_log_page_size[sequence_i];

        init_params.packetizer_input_src = input_packetize_src_endpoint[sequence_i];
        init_params.packetizer_input_dest = input_packetize_dest_endpoint[sequence_i];

        active_input_queue->init(&init_params);

        return true;
    });
}

inline void initialize_output_queues() {
    init_params_t init_params{};

    process_queues<output_queue_network_sequence, output_queue_cb_mode_sequence>([&]<auto network_type, auto cbmode, auto sequence_i>(auto) -> bool {
        // Sequence number for input queues should line up with the output queues
        // input network/cb mode sequence in here is not the same as the global one
        // each output queue only has 1 input queue connected to it
        using this_input_networks = NetworkTypeSequence<remote_rx_network_type[sequence_i]>;
        using this_input_cb_mode = CBModeTypeSequence<input_packetize[sequence_i]>;

        raw_output_queues[sequence_i].template engage<network_type, cbmode, this_input_networks, this_input_cb_mode>();

        auto* active_output_queue = raw_output_queues[sequence_i].template get<network_type, cbmode, this_input_networks, this_input_cb_mode>();

        init_params.queue_id = (uint8_t)sequence_i + router_lanes,
        init_params.queue_start_addr_words = remote_tx_queue_start_addr_words[sequence_i],
        init_params.queue_size_words = remote_tx_queue_size_words[sequence_i],
        init_params.remote_queue_id = (uint8_t)remote_tx_queue_id[sequence_i],
        init_params.remote_x = remote_tx_x[sequence_i],
        init_params.remote_y = remote_tx_y[sequence_i],
        init_params.ptrs_addr = vc_packet_router_output_scratch_buffers[sequence_i],
        init_params.remote_ptrs_addr = vc_packet_router_output_remote_scratch_buffers[sequence_i],

        init_params.local_sem_id = (uint8_t)output_depacketize_local_sem[sequence_i],
        init_params.remote_sem_id = (uint8_t)output_depacketize_downstream_sem[sequence_i],
        init_params.log_page_size = (uint8_t)output_depacketize_log_page_size[sequence_i],

        init_params.input_queues = &raw_input_queues[sequence_i],
        init_params.num_input_queues = 1,

        init_params.unpacketizer_output_remove_header = output_depacketize_remove_header[sequence_i],

        active_output_queue->init(&init_params);

        return true;
    });
}

void kernel_main() {
    write_kernel_status(kernel_status, PQ_TEST_STATUS_INDEX, PACKET_QUEUE_TEST_STARTED);
    write_kernel_status(kernel_status, PQ_TEST_MISC_INDEX, 0xff000000);
    write_kernel_status(kernel_status, PQ_TEST_MISC_INDEX+1, 0xbb000000 | router_lanes);

    initialize_input_queues();
    initialize_output_queues();

    if (!wait_all_input_output_ready<input_queue_network_sequence,
                                     input_queue_cb_mode_sequence,
                                     output_queue_network_sequence,
                                     output_queue_cb_mode_sequence>(raw_input_queues, raw_output_queues, timeout_cycles)) {
        write_kernel_status(kernel_status, PQ_TEST_STATUS_INDEX, PACKET_QUEUE_TEST_TIMEOUT);
        return;
    }

    write_kernel_status(kernel_status, PQ_TEST_MISC_INDEX, 0xff000001);

    bool timeout = false;
    bool all_outputs_finished = false;
    uint64_t data_words_sent = 0;
    uint64_t iter = 0;
    uint64_t start_timestamp = get_timestamp();
    uint32_t progress_timestamp = start_timestamp & 0xFFFFFFFF;
    uint32_t heartbeat = 0;
    while (!all_outputs_finished) {
        IDLE_ERISC_HEARTBEAT_AND_RETURN(heartbeat);
        iter++;

        process_queues<input_queue_network_sequence, input_queue_cb_mode_sequence>([&]<auto network_type, auto cbmode, auto sequence_i>(auto) -> bool {
            using this_input_networks = NetworkTypeSequence<remote_rx_network_type[sequence_i]>;
            using this_input_cb_mode = CBModeTypeSequence<input_packetize[sequence_i]>;

            auto* active_input_queue = raw_input_queues[sequence_i].template get<network_type, cbmode>();
            auto* active_output_queue = raw_output_queues[sequence_i].template get<remote_tx_network_type[sequence_i], output_depacketize[sequence_i], this_input_networks, this_input_cb_mode>();

            active_input_queue->advance_if_not_valid();
            if (active_input_queue->get_curr_packet_valid()) {
                bool full_packet_sent;
                data_words_sent += active_output_queue->template forward_data_from_input<0>(full_packet_sent, active_input_queue->get_end_of_cmd());
            }

            active_output_queue->prev_words_in_flight_check_flush();

            return true;
        });

        if ((iter & 0xFF) == 0) {
            all_outputs_finished = true;
            process_queues<output_queue_network_sequence, output_queue_cb_mode_sequence>([&]<auto network_type, auto cbmode, auto sequence_i>(auto) -> bool {
                auto* active_output_queue = raw_output_queues[sequence_i].template get<network_type, cbmode, input_queue_network_sequence, input_queue_cb_mode_sequence>();
                all_outputs_finished &= active_output_queue->is_remote_finished();
                return true;
            });
        }
    }

    write_kernel_status(kernel_status, PQ_TEST_MISC_INDEX, 0xff000002);
    process_queues<output_queue_network_sequence, output_queue_cb_mode_sequence>([&]<auto network_type, auto cbmode, auto sequence_i>(auto) -> bool {
        auto* active_output_queue = raw_output_queues[sequence_i].template get<network_type, cbmode, input_queue_network_sequence, input_queue_cb_mode_sequence>();
        if (!active_output_queue->output_barrier(timeout_cycles)) {
            timeout = true;
            return false;
        }
        return true;
    });

    uint64_t cycles_elapsed = get_timestamp() - start_timestamp;
    if (!timeout) {
        write_kernel_status(kernel_status, PQ_TEST_MISC_INDEX, 0xff000003);
        process_queues<input_queue_network_sequence, input_queue_cb_mode_sequence>([&]<auto network_type, auto cbmode, auto sequence_i>(auto i) -> bool {
            auto* active_input_queue = raw_input_queues[i].template get<network_type, cbmode>();
            active_input_queue->send_remote_finished_notification();
            return true;
        });
    }

    set_64b_result(kernel_status, data_words_sent, PQ_TEST_WORD_CNT_INDEX);
    set_64b_result(kernel_status, cycles_elapsed, PQ_TEST_CYCLES_INDEX);
    set_64b_result(kernel_status, iter, PQ_TEST_ITER_INDEX);

    if (timeout) {
        write_kernel_status(kernel_status, PQ_TEST_STATUS_INDEX, PACKET_QUEUE_TEST_TIMEOUT);
    } else {
        write_kernel_status(kernel_status, PQ_TEST_STATUS_INDEX, PACKET_QUEUE_TEST_PASS);
        write_kernel_status(kernel_status, PQ_TEST_MISC_INDEX, 0xff00005);
    }
}
