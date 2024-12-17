// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "tt_metal/impl/dispatch/kernels/packet_queue.hpp"

constexpr uint32_t endpoint_id_start_index = get_compile_time_arg_val(0);
constexpr uint32_t tunnel_lanes = get_compile_time_arg_val(1);
constexpr uint32_t in_queue_start_addr_words = get_compile_time_arg_val(2);
constexpr uint32_t in_queue_size_words = get_compile_time_arg_val(3);
constexpr uint32_t in_queue_size_bytes = in_queue_size_words * PACKET_WORD_SIZE_BYTES;
static_assert(is_power_of_2(in_queue_size_words), "in_queue_size_words must be a power of 2");
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

static_assert(is_power_of_2(remote_receiver_queue_size_words[0]), "remote_receiver_queue_size_words must be a power of 2");
static_assert(is_power_of_2(remote_receiver_queue_size_words[1]), "remote_receiver_queue_size_words must be a power of 2");
static_assert(is_power_of_2(remote_receiver_queue_size_words[2]), "remote_receiver_queue_size_words must be a power of 2");
static_assert(is_power_of_2(remote_receiver_queue_size_words[3]), "remote_receiver_queue_size_words must be a power of 2");
static_assert(is_power_of_2(remote_receiver_queue_size_words[4]), "remote_receiver_queue_size_words must be a power of 2");
static_assert(is_power_of_2(remote_receiver_queue_size_words[5]), "remote_receiver_queue_size_words must be a power of 2");
static_assert(is_power_of_2(remote_receiver_queue_size_words[6]), "remote_receiver_queue_size_words must be a power of 2");
static_assert(is_power_of_2(remote_receiver_queue_size_words[7]), "remote_receiver_queue_size_words must be a power of 2");
static_assert(is_power_of_2(remote_receiver_queue_size_words[8]), "remote_receiver_queue_size_words must be a power of 2");
static_assert(is_power_of_2(remote_receiver_queue_size_words[9]), "remote_receiver_queue_size_words must be a power of 2");

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

packet_input_queue_state_t input_queues[MAX_TUNNEL_LANES];
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

packet_output_queue_state_t output_queues[MAX_TUNNEL_LANES];
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

#define SWITCH_THRESHOLD 32
void kernel_main() {
    rtos_context_switch_ptr = (void (*)())RtosTable[0];

    write_kernel_status(kernel_status, PQ_TEST_STATUS_INDEX, PACKET_QUEUE_TEST_STARTED);
    write_kernel_status(kernel_status, PQ_TEST_MISC_INDEX, 0xff000000);
    write_kernel_status(kernel_status, PQ_TEST_MISC_INDEX + 1, 0xbb000000);
    write_kernel_status(kernel_status, PQ_TEST_MISC_INDEX + 2, 0xAABBCCDD);
    write_kernel_status(kernel_status, PQ_TEST_MISC_INDEX + 3, 0xDDCCBBAA);
    write_kernel_status(kernel_status, PQ_TEST_MISC_INDEX + 4, endpoint_id_start_index);

    for (uint32_t i = 0; i < tunnel_lanes; i++) {
        input_queues[i].init(
            i,
            in_queue_start_addr_words + i * in_queue_size_words,
            in_queue_size_words,
            remote_sender_x[i],
            remote_sender_y[i],
            remote_sender_queue_id[i],
            remote_sender_network_type[i]);
    }

    for (uint32_t i = 0; i < tunnel_lanes; i++) {
        output_queues[i].init(
            i + tunnel_lanes, //MAX_TUNNEL_LANES,
            remote_receiver_queue_start_addr_words[i],
            remote_receiver_queue_size_words[i],
            remote_receiver_x[i],
            remote_receiver_y[i],
            remote_receiver_queue_id[i],
            remote_receiver_network_type[i],
            &input_queues[i],
            1);
    }

    if (!wait_all_input_output_ready<input_queue_network_sequence,
                                     input_queue_cb_mode_sequence,
                                     output_queue_network_sequence,
                                     output_queue_cb_mode_sequence>(input_queues, output_queues, timeout_cycles)) {
        write_kernel_status(kernel_status, PQ_TEST_STATUS_INDEX, PACKET_QUEUE_TEST_TIMEOUT);
        return;
    }

    write_kernel_status(kernel_status, PQ_TEST_MISC_INDEX, 0xff000001);

    bool all_outputs_finished = false;
    uint64_t data_words_sent = 0;
    uint64_t iter = 0;
    uint64_t start_timestamp = get_timestamp();
    uint32_t switch_counter = 0;
    uint32_t progress_timestamp = start_timestamp & 0xFFFFFFFF;
    bool timeout = false;
    while (!all_outputs_finished) {
        if constexpr (timeout_cycles > 0) {
            uint32_t cycles_since_progress = get_timestamp_32b() - progress_timestamp;
            if (cycles_since_progress > timeout_cycles) {
                timeout = true;
                break;
            }
        }
        iter++;
        switch_counter++;
        all_outputs_finished = switch_counter >= SWITCH_THRESHOLD;
        process_queues<input_queue_network_sequence, input_queue_cb_mode_sequence>([&]<auto input_network_type, auto input_cb_mode, auto sequence_i>(auto) -> bool {
            using remote_input_networks = NetworkTypeSequence<remote_sender_network_type[sequence_i]>;
            using remote_input_cb_modes = CBModeTypeSequence<false>;

            if (input_queues[sequence_i].template get_curr_packet_valid<input_cb_mode>()) {
                bool full_packet_sent;
                uint32_t words_sent = output_queues[sequence_i].template forward_data_from_input<remote_receiver_network_type[sequence_i], false, remote_sender_network_type[sequence_i], false>(0, full_packet_sent, input_queues[sequence_i].get_end_of_cmd());
                data_words_sent += words_sent;
                if (words_sent > 0) {
                    switch_counter = 0;
                    all_outputs_finished = false;
                }
            }
            output_queues[sequence_i].template prev_words_in_flight_check_flush<false, remote_input_networks, remote_input_cb_modes>();
            if (switch_counter >= SWITCH_THRESHOLD) {
                bool output_finished = output_queues[sequence_i].is_remote_finished();
                if (output_finished) {
                    uint32_t return_vc = (inner_stop_mux_d_bypass >> 24) & 0xFF;
                    if ((sequence_i == return_vc) && (inner_stop_mux_d_bypass != 0)) {
                        input_queues[sequence_i].set_end_remote_queue(
                            (inner_stop_mux_d_bypass >> 16) & 0xFF, // remote_queue_id
                            inner_stop_mux_d_bypass & 0xFF, // remote_x
                            (inner_stop_mux_d_bypass >> 8) & 0xFF // remote_y
                        );
                    }
                    input_queues[sequence_i].template send_remote_finished_notification<input_network_type, input_cb_mode>();
                }
                all_outputs_finished &= output_finished;
            }

            return true;
        });

        uint32_t launch_msg_rd_ptr = *GET_MAILBOX_ADDRESS_DEV(launch_msg_rd_ptr);
        tt_l1_ptr launch_msg_t * const launch_msg = GET_MAILBOX_ADDRESS_DEV(launch[launch_msg_rd_ptr]);
        if (launch_msg->kernel_config.exit_erisc_kernel) {
            return;
        }
        // need to optimize this.
        // context switch to base fw is very costly.
        if (switch_counter >= SWITCH_THRESHOLD) {
            internal_::risc_context_switch();
            switch_counter = SWITCH_THRESHOLD;
        }

    }

    timeout = false;
    write_kernel_status(kernel_status, PQ_TEST_MISC_INDEX, 0xff000002);
    process_queues<output_queue_network_sequence, output_queue_cb_mode_sequence>([&]<auto network_type, auto cb_mode, auto sequence_i>(auto) -> bool {
        // inputs for this output queue
        using remote_input_networks = NetworkTypeSequence<remote_sender_network_type[sequence_i]>;
        using remote_input_cb_modes = CBModeTypeSequence<false>;

        if (!output_queues[sequence_i].template output_barrier<cb_mode, remote_input_networks, remote_input_cb_modes>(timeout_cycles)) {
            timeout = true;
            return false;
        }
        return true;
    });

    uint64_t cycles_elapsed = get_timestamp() - start_timestamp;
    write_kernel_status(kernel_status, PQ_TEST_MISC_INDEX, 0xff000003);

    set_64b_result(kernel_status, data_words_sent, PQ_TEST_WORD_CNT_INDEX);
    set_64b_result(kernel_status, cycles_elapsed, PQ_TEST_CYCLES_INDEX);
    set_64b_result(kernel_status, iter, PQ_TEST_ITER_INDEX);

    write_kernel_status(kernel_status, PQ_TEST_STATUS_INDEX, PACKET_QUEUE_TEST_PASS);
    write_kernel_status(kernel_status, PQ_TEST_MISC_INDEX, 0xff00005);
}
