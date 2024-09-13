// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
#include "dataflow_api.h"
#include "tt_metal/impl/dispatch/kernels/packet_queue.hpp"
// clang-format on

packet_input_queue_state_t input_queues[MAX_TUNNEL_LANES];
packet_output_queue_state_t output_queues[MAX_TUNNEL_LANES];

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

#define SWITCH_THRESHOLD 16
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

    if (!wait_all_src_dest_ready(input_queues, tunnel_lanes, output_queues, tunnel_lanes, timeout_cycles)) {
        write_kernel_status(kernel_status, PQ_TEST_STATUS_INDEX, PACKET_QUEUE_TEST_TIMEOUT);
        return;
    }

    write_kernel_status(kernel_status, PQ_TEST_MISC_INDEX, 0xff000001);

    bool all_outputs_finished = false;
    uint64_t data_words_sent = 0;
    uint64_t iter = 0;
    uint64_t start_timestamp = get_timestamp();
    uint32_t switch_counter = 0;
    while (!all_outputs_finished) {
        iter++;
        switch_counter++;
        all_outputs_finished = switch_counter >= SWITCH_THRESHOLD;
        for (uint32_t i = 0; i < tunnel_lanes; i++) {
            if (input_queues[i].get_curr_packet_valid()) {
                bool full_packet_sent;
                uint32_t words_sent =
                    output_queues[i].forward_data_from_input(0, full_packet_sent, input_queues[i].get_end_of_cmd());
                data_words_sent += words_sent;
                if (words_sent > 0) {
                    switch_counter = 0;
                    all_outputs_finished = false;
                }
            }
            output_queues[i].prev_words_in_flight_check_flush();
            if (switch_counter >= SWITCH_THRESHOLD) {
                bool output_finished = output_queues[i].is_remote_finished();
                if (output_finished) {
                    uint32_t return_vc = (inner_stop_mux_d_bypass >> 24) & 0xFF;
                    if ((i == return_vc) && (inner_stop_mux_d_bypass != 0)) {
                        input_queues[i].remote_x = inner_stop_mux_d_bypass & 0xFF;
                        input_queues[i].remote_y = (inner_stop_mux_d_bypass >> 8) & 0xFF;
                        input_queues[i].set_remote_ready_status_addr((inner_stop_mux_d_bypass >> 16) & 0xFF);
                    }
                    input_queues[i].send_remote_finished_notification();
                }
                all_outputs_finished &= output_finished;
            }
        }

        tt_l1_ptr launch_msg_t * const launch_msg = GET_MAILBOX_ADDRESS_DEV(launch);
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

    write_kernel_status(kernel_status, PQ_TEST_MISC_INDEX, 0xff000002);
    for (uint32_t i = 0; i < tunnel_lanes; i++) {
        output_queues[i].output_barrier();
    }

    uint64_t cycles_elapsed = get_timestamp() - start_timestamp;
    write_kernel_status(kernel_status, PQ_TEST_MISC_INDEX, 0xff000003);

    set_64b_result(kernel_status, data_words_sent, PQ_TEST_WORD_CNT_INDEX);
    set_64b_result(kernel_status, cycles_elapsed, PQ_TEST_CYCLES_INDEX);
    set_64b_result(kernel_status, iter, PQ_TEST_ITER_INDEX);

    write_kernel_status(kernel_status, PQ_TEST_STATUS_INDEX, PACKET_QUEUE_TEST_PASS);
    write_kernel_status(kernel_status, PQ_TEST_MISC_INDEX, 0xff00005);
}
