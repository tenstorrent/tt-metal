// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "packet_queue_ctrl.hpp"
#include "tt_metal/impl/dispatch/kernels/packet_queue_v2.hpp"
#include "tt_metal/impl/dispatch/kernels/packet_queue_ctrl.hpp"
#include "tt_metal/impl/dispatch/kernels/cq_helpers.hpp"

using namespace packet_queue;

constexpr uint32_t endpoint_id_start_index = get_compile_time_arg_val(0);
constexpr uint32_t tunnel_lanes = get_compile_time_arg_val(1);
constexpr uint32_t in_queue_start_addr_words = get_compile_time_arg_val(2);
constexpr uint32_t in_queue_size_words = get_compile_time_arg_val(3);
constexpr uint32_t in_queue_size_bytes = in_queue_size_words * PACKET_WORD_SIZE_BYTES;
static_assert(tunnel_lanes <= MAX_TUNNEL_LANES, "cannot have more than 2 tunnel directions.");
static_assert(tunnel_lanes, "tunnel directions cannot be 0. 1 => Unidirectional. 2 => Bidirectional");

constexpr uint32_t remote_receiver_x[MAX_TUNNEL_LANES] =
    {
        (get_compile_time_arg_val(4)  & 0xFF),
        (get_compile_time_arg_val(5)  & 0xFF),
        (get_compile_time_arg_val(6)  & 0xFF),
        (get_compile_time_arg_val(7)  & 0xFF),
        (get_compile_time_arg_val(8)  & 0xFF),
        (get_compile_time_arg_val(9)  & 0xFF),
        (get_compile_time_arg_val(10) & 0xFF),
        (get_compile_time_arg_val(11) & 0xFF),
        (get_compile_time_arg_val(12) & 0xFF),
        (get_compile_time_arg_val(13) & 0xFF)
    };

constexpr uint32_t remote_receiver_y[MAX_TUNNEL_LANES] =
    {
        (get_compile_time_arg_val(4)  >> 8) & 0xFF,
        (get_compile_time_arg_val(5)  >> 8) & 0xFF,
        (get_compile_time_arg_val(6)  >> 8) & 0xFF,
        (get_compile_time_arg_val(7)  >> 8) & 0xFF,
        (get_compile_time_arg_val(8)  >> 8) & 0xFF,
        (get_compile_time_arg_val(9)  >> 8) & 0xFF,
        (get_compile_time_arg_val(10) >> 8) & 0xFF,
        (get_compile_time_arg_val(11) >> 8) & 0xFF,
        (get_compile_time_arg_val(12) >> 8) & 0xFF,
        (get_compile_time_arg_val(13) >> 8) & 0xFF
    };

constexpr uint32_t remote_receiver_queue_id[MAX_TUNNEL_LANES] =
    {
        (get_compile_time_arg_val(4)  >> 16) & 0xFF,
        (get_compile_time_arg_val(5)  >> 16) & 0xFF,
        (get_compile_time_arg_val(6)  >> 16) & 0xFF,
        (get_compile_time_arg_val(7)  >> 16) & 0xFF,
        (get_compile_time_arg_val(8)  >> 16) & 0xFF,
        (get_compile_time_arg_val(9)  >> 16) & 0xFF,
        (get_compile_time_arg_val(10) >> 16) & 0xFF,
        (get_compile_time_arg_val(11) >> 16) & 0xFF,
        (get_compile_time_arg_val(12) >> 16) & 0xFF,
        (get_compile_time_arg_val(13) >> 16) & 0xFF
    };

constexpr DispatchRemoteNetworkType remote_receiver_network_type[MAX_TUNNEL_LANES] =
    {
        static_cast<DispatchRemoteNetworkType>((get_compile_time_arg_val(4)  >> 24) & 0xFF),
        static_cast<DispatchRemoteNetworkType>((get_compile_time_arg_val(5)  >> 24) & 0xFF),
        static_cast<DispatchRemoteNetworkType>((get_compile_time_arg_val(6)  >> 24) & 0xFF),
        static_cast<DispatchRemoteNetworkType>((get_compile_time_arg_val(7)  >> 24) & 0xFF),
        static_cast<DispatchRemoteNetworkType>((get_compile_time_arg_val(8)  >> 24) & 0xFF),
        static_cast<DispatchRemoteNetworkType>((get_compile_time_arg_val(9)  >> 24) & 0xFF),
        static_cast<DispatchRemoteNetworkType>((get_compile_time_arg_val(10) >> 24) & 0xFF),
        static_cast<DispatchRemoteNetworkType>((get_compile_time_arg_val(11) >> 24) & 0xFF),
        static_cast<DispatchRemoteNetworkType>((get_compile_time_arg_val(12) >> 24) & 0xFF),
        static_cast<DispatchRemoteNetworkType>((get_compile_time_arg_val(13) >> 24) & 0xFF)
    };

constexpr uint32_t remote_receiver_queue_start_addr_words[MAX_TUNNEL_LANES] =
    {
        get_compile_time_arg_val(14),
        get_compile_time_arg_val(16),
        get_compile_time_arg_val(18),
        get_compile_time_arg_val(20),
        get_compile_time_arg_val(22),
        get_compile_time_arg_val(24),
        get_compile_time_arg_val(26),
        get_compile_time_arg_val(28),
        get_compile_time_arg_val(30),
        get_compile_time_arg_val(32)
    };

constexpr uint32_t remote_receiver_queue_size_words[MAX_TUNNEL_LANES] =
    {
        get_compile_time_arg_val(15),
        get_compile_time_arg_val(17),
        get_compile_time_arg_val(19),
        get_compile_time_arg_val(21),
        get_compile_time_arg_val(23),
        get_compile_time_arg_val(25),
        get_compile_time_arg_val(27),
        get_compile_time_arg_val(29),
        get_compile_time_arg_val(31),
        get_compile_time_arg_val(33)
    };

constexpr uint32_t remote_sender_x[MAX_TUNNEL_LANES] =
    {
        (get_compile_time_arg_val(34) & 0xFF),
        (get_compile_time_arg_val(35) & 0xFF),
        (get_compile_time_arg_val(36) & 0xFF),
        (get_compile_time_arg_val(37) & 0xFF),
        (get_compile_time_arg_val(38) & 0xFF),
        (get_compile_time_arg_val(39) & 0xFF),
        (get_compile_time_arg_val(40) & 0xFF),
        (get_compile_time_arg_val(41) & 0xFF),
        (get_compile_time_arg_val(42) & 0xFF),
        (get_compile_time_arg_val(43) & 0xFF)
    };

constexpr uint32_t remote_sender_y[MAX_TUNNEL_LANES] =
    {
        (get_compile_time_arg_val(34) >> 8) & 0xFF,
        (get_compile_time_arg_val(35) >> 8) & 0xFF,
        (get_compile_time_arg_val(36) >> 8) & 0xFF,
        (get_compile_time_arg_val(37) >> 8) & 0xFF,
        (get_compile_time_arg_val(38) >> 8) & 0xFF,
        (get_compile_time_arg_val(39) >> 8) & 0xFF,
        (get_compile_time_arg_val(40) >> 8) & 0xFF,
        (get_compile_time_arg_val(41) >> 8) & 0xFF,
        (get_compile_time_arg_val(42) >> 8) & 0xFF,
        (get_compile_time_arg_val(43) >> 8) & 0xFF
    };

constexpr uint32_t remote_sender_queue_id[MAX_TUNNEL_LANES] =
    {
        (get_compile_time_arg_val(34) >> 16) & 0xFF,
        (get_compile_time_arg_val(35) >> 16) & 0xFF,
        (get_compile_time_arg_val(36) >> 16) & 0xFF,
        (get_compile_time_arg_val(37) >> 16) & 0xFF,
        (get_compile_time_arg_val(38) >> 16) & 0xFF,
        (get_compile_time_arg_val(39) >> 16) & 0xFF,
        (get_compile_time_arg_val(40) >> 16) & 0xFF,
        (get_compile_time_arg_val(41) >> 16) & 0xFF,
        (get_compile_time_arg_val(42) >> 16) & 0xFF,
        (get_compile_time_arg_val(43) >> 16) & 0xFF
    };

constexpr DispatchRemoteNetworkType remote_sender_network_type[MAX_TUNNEL_LANES] =
    {
        static_cast<DispatchRemoteNetworkType>((get_compile_time_arg_val(34) >> 24) & 0xFF),
        static_cast<DispatchRemoteNetworkType>((get_compile_time_arg_val(35) >> 24) & 0xFF),
        static_cast<DispatchRemoteNetworkType>((get_compile_time_arg_val(36) >> 24) & 0xFF),
        static_cast<DispatchRemoteNetworkType>((get_compile_time_arg_val(37) >> 24) & 0xFF),
        static_cast<DispatchRemoteNetworkType>((get_compile_time_arg_val(38) >> 24) & 0xFF),
        static_cast<DispatchRemoteNetworkType>((get_compile_time_arg_val(39) >> 24) & 0xFF),
        static_cast<DispatchRemoteNetworkType>((get_compile_time_arg_val(40) >> 24) & 0xFF),
        static_cast<DispatchRemoteNetworkType>((get_compile_time_arg_val(41) >> 24) & 0xFF),
        static_cast<DispatchRemoteNetworkType>((get_compile_time_arg_val(42) >> 24) & 0xFF),
        static_cast<DispatchRemoteNetworkType>((get_compile_time_arg_val(43) >> 24) & 0xFF)
    };

constexpr uint32_t kernel_status_buf_addr_arg = get_compile_time_arg_val(44);
constexpr uint32_t kernel_status_buf_size_bytes = get_compile_time_arg_val(45);

// careful, may be null
tt_l1_ptr uint32_t* const kernel_status = reinterpret_cast<tt_l1_ptr uint32_t*>(kernel_status_buf_addr_arg);

constexpr uint32_t timeout_cycles = get_compile_time_arg_val(46);
constexpr uint32_t inner_stop_mux_d_bypass = get_compile_time_arg_val(47);

constexpr uint32_t vc_eth_tunneler_input_scratch_buffers[MAX_TUNNEL_LANES] = {
    get_compile_time_arg_val(48),
    get_compile_time_arg_val(49),
    get_compile_time_arg_val(50),
    get_compile_time_arg_val(51),
    get_compile_time_arg_val(52),
    get_compile_time_arg_val(53),
    get_compile_time_arg_val(54),
    get_compile_time_arg_val(55),
    get_compile_time_arg_val(56),
    get_compile_time_arg_val(57),
};

constexpr uint32_t vc_eth_tunneler_input_remote_scratch_buffers[MAX_TUNNEL_LANES] = {
    get_compile_time_arg_val(58),
    get_compile_time_arg_val(59),
    get_compile_time_arg_val(60),
    get_compile_time_arg_val(61),
    get_compile_time_arg_val(62),
    get_compile_time_arg_val(63),
    get_compile_time_arg_val(64),
    get_compile_time_arg_val(65),
    get_compile_time_arg_val(66),
    get_compile_time_arg_val(67),
};

constexpr uint32_t vc_eth_tunneler_output_scratch_buffers[MAX_TUNNEL_LANES] = {
    get_compile_time_arg_val(68),
    get_compile_time_arg_val(69),
    get_compile_time_arg_val(70),
    get_compile_time_arg_val(71),
    get_compile_time_arg_val(72),
    get_compile_time_arg_val(73),
    get_compile_time_arg_val(74),
    get_compile_time_arg_val(75),
    get_compile_time_arg_val(76),
    get_compile_time_arg_val(77),
};

constexpr uint32_t vc_eth_tunneler_output_remote_scratch_buffers[MAX_TUNNEL_LANES] = {
    get_compile_time_arg_val(78),
    get_compile_time_arg_val(79),
    get_compile_time_arg_val(80),
    get_compile_time_arg_val(81),
    get_compile_time_arg_val(82),
    get_compile_time_arg_val(83),
    get_compile_time_arg_val(84),
    get_compile_time_arg_val(85),
    get_compile_time_arg_val(86),
    get_compile_time_arg_val(87),
};

UnsafePacketInputQueueVariant raw_input_queues[MAX_TUNNEL_LANES];
using input_queue_network_sequence = NetworkTypeSequence<remote_sender_network_type[0],
                                                         remote_sender_network_type[1],
                                                         remote_sender_network_type[2],
                                                         remote_sender_network_type[3],
                                                         remote_sender_network_type[4],
                                                         remote_sender_network_type[5],
                                                         remote_sender_network_type[6],
                                                         remote_sender_network_type[7],
                                                         remote_sender_network_type[8],
                                                         remote_sender_network_type[9]>;
using input_queue_cb_mode_sequence = CBModeTypeSequence<false,
                                                        false,
                                                        false,
                                                        false,
                                                        false,
                                                        false,
                                                        false,
                                                        false,
                                                        false,
                                                        false>;

UnsafePacketOutputQueueVariant raw_output_queues[MAX_TUNNEL_LANES];
using output_queue_network_sequence = NetworkTypeSequence<remote_receiver_network_type[0],
                                                          remote_receiver_network_type[1],
                                                          remote_receiver_network_type[2],
                                                          remote_receiver_network_type[3],
                                                          remote_receiver_network_type[4],
                                                          remote_receiver_network_type[5],
                                                          remote_receiver_network_type[6],
                                                          remote_receiver_network_type[7],
                                                          remote_receiver_network_type[8],
                                                          remote_receiver_network_type[9]>;
using output_queue_cb_mode_sequence = CBModeTypeSequence<false,
                                                         false,
                                                         false,
                                                         false,
                                                         false,
                                                         false,
                                                         false,
                                                         false,
                                                         false,
                                                         false>;


void initialize_input_queues() {
    init_params_t init_params{};
    process_queues<input_queue_network_sequence, input_queue_cb_mode_sequence>([&]<auto network_type, auto cbmode, auto sequence_i>(auto) -> bool {
        raw_input_queues[sequence_i].template engage<network_type, cbmode>();

        auto* active_input_queue = raw_input_queues[sequence_i].template get<network_type, cbmode>();
        init_params.queue_id = (uint8_t)sequence_i;
        init_params.queue_start_addr_words = in_queue_start_addr_words + sequence_i * in_queue_size_words;
        init_params.queue_size_words = in_queue_size_words;
        init_params.remote_queue_id = (uint8_t)remote_sender_queue_id[sequence_i];
        init_params.remote_x = remote_sender_x[sequence_i];
        init_params.remote_y = remote_sender_y[sequence_i];
        init_params.ptrs_addr = vc_eth_tunneler_input_scratch_buffers[sequence_i];
        init_params.remote_ptrs_addr = vc_eth_tunneler_input_remote_scratch_buffers[sequence_i];

        active_input_queue->init(&init_params);

        return true;
    });
}

void initialize_output_queue() {
    init_params_t init_params{};
    process_queues<output_queue_network_sequence, output_queue_cb_mode_sequence>([&]<auto network_type, auto cbmode, auto sequence_i>(auto) -> bool {
        // Sequence number for input queues should line up with the output queues
        // input network/cb mode sequence in here is not the same as the global one
        // each output queue only has 1 input queue connected to it
        using this_input_networks = NetworkTypeSequence<remote_sender_network_type[sequence_i]>;
        using this_input_cb_mode = CBModeTypeSequence<false>;

        raw_output_queues[sequence_i].template engage<network_type, cbmode, this_input_networks, this_input_cb_mode>();

        auto* active_output_queue = raw_output_queues[sequence_i].template get<network_type, cbmode, this_input_networks, this_input_cb_mode>();

        init_params.queue_id = (uint8_t)sequence_i + tunnel_lanes,
        init_params.queue_start_addr_words = remote_receiver_queue_start_addr_words[sequence_i],
        init_params.queue_size_words = remote_receiver_queue_size_words[sequence_i],
        init_params.remote_queue_id = (uint8_t)remote_receiver_queue_id[sequence_i],
        init_params.remote_x = remote_receiver_x[sequence_i],
        init_params.remote_y = remote_receiver_y[sequence_i],
        init_params.ptrs_addr = vc_eth_tunneler_output_scratch_buffers[sequence_i],
        init_params.remote_ptrs_addr = vc_eth_tunneler_output_remote_scratch_buffers[sequence_i],

        init_params.input_queues = &raw_input_queues[sequence_i],
        init_params.num_input_queues = 1,

        active_output_queue->init(&init_params);

        return true;
    });
}

void kernel_main() {
    rtos_context_switch_ptr = (void (*)())RtosTable[0];

    write_kernel_status(kernel_status, PQ_TEST_STATUS_INDEX, PACKET_QUEUE_TEST_STARTED);
    write_kernel_status(kernel_status, PQ_TEST_MISC_INDEX, 0xff000000);
    write_kernel_status(kernel_status, PQ_TEST_MISC_INDEX + 1, 0xbb000000);
    write_kernel_status(kernel_status, PQ_TEST_MISC_INDEX + 2, 0xAABBCCDD);
    write_kernel_status(kernel_status, PQ_TEST_MISC_INDEX + 3, 0xDDCCBBAA);
    write_kernel_status(kernel_status, PQ_TEST_MISC_INDEX + 4, endpoint_id_start_index);

    // initialize_input_queues();
    // initialize_output_queue();
    DPRINT << "wait for init\n";

    // if (!wait_all_input_output_ready<input_queue_network_sequence,
    //                                  input_queue_cb_mode_sequence,
    //                                  output_queue_network_sequence,
    //                                  output_queue_cb_mode_sequence>(raw_input_queues, raw_output_queues, timeout_cycles)) {
    //     write_kernel_status(kernel_status, PQ_TEST_STATUS_INDEX, PACKET_QUEUE_TEST_TIMEOUT);
    //     return;
    // }

    // write_kernel_status(kernel_status, PQ_TEST_MISC_INDEX, 0xff000001);

    // DPRINT << "ethernet kernels init\n";

    // bool all_outputs_finished = false;
    // uint64_t data_words_sent = 0;
    // uint64_t iter = 0;
    // uint64_t start_timestamp = get_timestamp();
    // uint32_t switch_counter = 0;
    // uint32_t heartbeat = 0;
    // while (!all_outputs_finished) {
    //     IDLE_ERISC_HEARTBEAT_AND_RETURN(heartbeat);
    //     iter++;
    //     switch_counter++;
    //     all_outputs_finished = switch_counter >= PACKET_QUEUE_ETH_SWITCH_LOOPS;
    //     process_queues<input_queue_network_sequence, input_queue_cb_mode_sequence>([&]<auto network_type, auto cbmode, auto sequence_i>(auto) -> bool {
    //         using this_input_networks = NetworkTypeSequence<remote_sender_network_type[sequence_i]>;
    //         using this_input_cb_mode = CBModeTypeSequence<false>;

    //         auto* active_input_queue = raw_input_queues[sequence_i].template get<network_type, cbmode>();
    //         auto* active_output_queue = raw_output_queues[sequence_i].template get<remote_receiver_network_type[sequence_i], false, this_input_networks, this_input_cb_mode>();

    //         active_input_queue->handle_recv();
    //         active_output_queue->handle_recv();

    //         // No progress will be made when either of the queues are waiting
    //         // If we are waiting too long, no progress will be made, and
    //         // the while loop will exit
    //         if (active_input_queue->busy() || active_output_queue->busy()) {
    //             all_outputs_finished = false;
    //             return true;
    //         }

    //         active_input_queue->advance_if_not_valid();
    //         if (active_input_queue->get_curr_packet_valid()) {
    //             bool full_packet_sent;
    //             uint32_t words_sent = active_output_queue->template forward_data_from_input<0>(full_packet_sent, active_input_queue->get_end_of_cmd());
    //             data_words_sent += words_sent;
    //             if (words_sent > 0) {
    //                 switch_counter = 0;
    //                 all_outputs_finished = false;
    //             }

    //             active_output_queue->prev_words_in_flight_check_flush();
    //         }

    //         if (switch_counter >= PACKET_QUEUE_ETH_SWITCH_LOOPS) {
    //             bool output_finished = active_output_queue->is_remote_finished();
    //             if (output_finished) {
    //                 uint32_t return_vc = (inner_stop_mux_d_bypass >> 24) & 0xFF;
    //                 if ((sequence_i == return_vc) && (inner_stop_mux_d_bypass != 0)) {
    //                     active_input_queue->set_final_remote_xy(inner_stop_mux_d_bypass & 0xFF, (inner_stop_mux_d_bypass >> 8) & 0xFF);
    //                     active_input_queue->set_remote_ready_status_addr((inner_stop_mux_d_bypass >> 16) & 0xFF);
    //                 }
    //                 active_input_queue->send_remote_finished_notification();
    //             }
    //             all_outputs_finished &= output_finished;
    //         }

    //         return true;
    //     });

    //     uint32_t launch_msg_rd_ptr = *GET_MAILBOX_ADDRESS_DEV(launch_msg_rd_ptr);
    //     tt_l1_ptr launch_msg_t * const launch_msg = GET_MAILBOX_ADDRESS_DEV(launch[launch_msg_rd_ptr]);
    //     if (launch_msg->kernel_config.exit_erisc_kernel) {
    //         return;
    //     }
    //     // need to optimize this.
    //     // context switch to base fw is very costly.
    //     if (switch_counter >= PACKET_QUEUE_ETH_SWITCH_LOOPS) {
    //         internal_::risc_context_switch();
    //         switch_counter = PACKET_QUEUE_ETH_SWITCH_LOOPS;
    //     }

    // }

    // bool timeout = false;
    // write_kernel_status(kernel_status, PQ_TEST_MISC_INDEX, 0xff000002);
    // process_queues<output_queue_network_sequence, output_queue_cb_mode_sequence>([&]<auto network_type, auto cbmode, auto sequence_i>(auto) -> bool {
    //     auto* active_output_queue = raw_output_queues[sequence_i].template get<network_type, cbmode, input_queue_network_sequence, input_queue_cb_mode_sequence>();
    //     if (!active_output_queue->output_barrier(timeout_cycles)) {
    //         timeout = true;
    //         return false;
    //     }
    //     return true;
    // });

    // uint64_t cycles_elapsed = get_timestamp() - start_timestamp;
    // write_kernel_status(kernel_status, PQ_TEST_MISC_INDEX, 0xff000003);

    // set_64b_result(kernel_status, data_words_sent, PQ_TEST_WORD_CNT_INDEX);
    // set_64b_result(kernel_status, cycles_elapsed, PQ_TEST_CYCLES_INDEX);
    // set_64b_result(kernel_status, iter, PQ_TEST_ITER_INDEX);

    // if (timeout) {
    //     write_kernel_status(kernel_status, PQ_TEST_STATUS_INDEX, PACKET_QUEUE_TEST_TIMEOUT);
    // } else {
    //     write_kernel_status(kernel_status, PQ_TEST_STATUS_INDEX, PACKET_QUEUE_TEST_PASS);
    //     write_kernel_status(kernel_status, PQ_TEST_MISC_INDEX, 0xff00005);
    // }
}
