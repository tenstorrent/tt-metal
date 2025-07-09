// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compile_time_args.h"
#include "dataflow_api.h"

#include "tt_metal/fabric/hw/inc/edm_fabric/compile_time_arg_tmp.hpp"

#include <array>
#include <utility>

// CHANNEL CONSTANTS
// ETH TXQ SELECTION

constexpr uint32_t DEFAULT_ETH_TXQ = 0;

// STREAM REGISTER ASSIGNMENT
// senders update this stream
constexpr uint32_t to_receiver_0_pkts_sent_id = 0;
// senders update this stream
constexpr uint32_t to_receiver_1_pkts_sent_id = 1;
// receivers updates the reg on this stream
constexpr uint32_t to_sender_0_pkts_acked_id = 2;
// receivers updates the reg on this stream
constexpr uint32_t to_sender_1_pkts_acked_id = 3;
// receivers updates the reg on this stream
constexpr uint32_t to_sender_2_pkts_acked_id = 4;
// receivers updates the reg on this stream
constexpr uint32_t to_sender_3_pkts_acked_id = 5;
// receivers updates the reg on this stream
constexpr uint32_t to_sender_4_pkts_acked_id = 6;
// receivers updates the reg on this stream
constexpr uint32_t to_sender_0_pkts_completed_id = 7;
// receivers updates the reg on this stream
constexpr uint32_t to_sender_1_pkts_completed_id = 8;
// receivers updates the reg on this stream
constexpr uint32_t to_sender_2_pkts_completed_id = 9;
// receivers updates the reg on this stream
constexpr uint32_t to_sender_3_pkts_completed_id = 10;
// receivers updates the reg on this stream
constexpr uint32_t to_sender_4_pkts_completed_id = 11;

/*
Receiver channel side registers are defined here to receive free-slot credits from downstream sender channels.

                                North Router
                        ┌───────────────────────────────────┐
                        │                                   │
                        │  ┌────┐ ┌────┐ ┌────┐ ┌────┐      │
                        │  │    │ │    │ │    │ │    │      │
                        │  │    │ │    │ │    │ │    │      │
                        │  └────┘ └────┘ └────┘ └────┘      │
                        │  ┌────┐ ┌────┐ ┌────┐ ┌────┐      │
                        │  │    │ │    │ │    │ │    │      │
                        │  │    │ │    │ │    │ │    │      │
                        │  │    │ │    │ │    │ │    │      │
                        │  │    │ │    │ │    │ │    │      │
                        │  └────┘ └─┬──┘ └────┘ └────┘      │
    West Router         └───────────┼───────────────────────┘        East Router
 ┌─────────────────────┐            │                             ┌────────────────────────────┐
 │                     │            │                             │                            │
 │                     │            │                             │                            │
 │               ┌────┐│ (increment)│    Acks From East           │┌──────────────┐ ┌────┐     │
 │   Free Slots  │    ◄┼────────────┼───────────────────┐         ││              │ │    │ E   │
 │     East      │    ││            │                   │         ││              │ │    │     │
 │               └────┘│            │                   │         │└──────────────┘ └────┘     │
 │                 12  │            │                   │         │                            │
 │               ┌────┐│            │                   │         │┌──────────────┐ ┌────┐     │
 │   Free Slots  │    ││            │                   │         ││              │ │    │ W   │
 │     West      │    ││            │                   └─────────┼┼              │ │    │     │
 │               └────┘│            │                             │└──────────────┘ └────┘     │
 │                 13  │            │                             │                            │
 │               ┌────┐│ (increment)│                             │┌──────────────┐ ┌────┐     │
 │   Free Slots  │    │◄────────────┘                             ││              │ │    │ N   │
 │     North     │    ││  Acks From North                         ││              │ │    │     │
 │               └────┘│                                          │└──────────────┘ └────┘     │
 │                 14  │                                          │                            │
 │               ┌────┐│  Acks From South                         │┌──────────────┐ ┌────┐     │
 │   Free Slots  │    │◄────────────────┐                         ││              │ │    │ S   │
 │     South     │    ││ (increment)    │                         ││              │ │    │     │
 │               └────┘│                │                         │└──────────────┘ └────┘     │
 │                 15  │                │                         │                            │
 │                     │                │                         │                            │
 │                     │                │                         │                            │
 └─────────────────────┘  ┌─────────────┼───────────────────┐     └────────────────────────────┘
                          │   ┌────┐ ┌──┼─┐ ┌────┐ ┌────┐   │
                          │   │    │ │    │ │    │ │    │   │
                          │   │    │ │    │ │    │ │    │   │
                          │   │    │ │    │ │    │ │    │   │
                          │   │    │ │    │ │    │ │    │   │
                          │   │    │ │    │ │    │ │    │   │
                          │   │    │ │    │ │    │ │    │   │
                          │   └────┘ └────┘ └────┘ └────┘   │
                          │   ┌────┐ ┌────┐ ┌────┐ ┌────┐   │
                          │   │    │ │    │ │    │ │    │   │
                          │   │    │ │    │ │    │ │    │   │
                          │   └────┘ └────┘ └────┘ └────┘   │
                          │                                 │
                          └─────────────────────────────────┘
                                   South Router
*/
constexpr size_t NUM_ROUTER_CARDINAL_DIRECTIONS = 4;
constexpr uint32_t receiver_channel_0_free_slots_from_east_stream_id = 12;
constexpr uint32_t receiver_channel_0_free_slots_from_west_stream_id = 13;
constexpr uint32_t receiver_channel_0_free_slots_from_north_stream_id = 14;
constexpr uint32_t receiver_channel_0_free_slots_from_south_stream_id = 15;

// For post-dateline connection. We only have one counter here because if we are
// post-dateline, there is only one other possible post-dateline consumer that we
// can forward to (the consumer in the same direction). Switching directions/turning
// requires directing back to pre-dateline consumer channels (in those cases, we'd
// use the receiver_channel_0_free_slots_* location). For the time being, this is
// placeholder until 2D torus is implemented
constexpr uint32_t receiver_channel_1_free_slots_from_downstream_stream_id = 16;

// These are the
// Slot 17 is defined in the edm_fabric_worker_adapter
constexpr uint32_t sender_channel_1_free_slots_stream_id = 18;
constexpr uint32_t sender_channel_2_free_slots_stream_id = 19;
constexpr uint32_t sender_channel_3_free_slots_stream_id = 20;
constexpr uint32_t sender_channel_4_free_slots_stream_id = 21;
constexpr uint32_t vc1_sender_channel_free_slots_stream_id = 22;

constexpr size_t MAX_NUM_RECEIVER_CHANNELS = 2;
constexpr size_t MAX_NUM_SENDER_CHANNELS = 5;

// Compile Time args

constexpr bool SPECIAL_MARKER_CHECK_ENABLED = true;
constexpr size_t SENDER_CHANNEL_NOC_CONFIG_START_IDX = 0;
constexpr size_t NUM_SENDER_CHANNELS = get_compile_time_arg_val(SENDER_CHANNEL_NOC_CONFIG_START_IDX);
constexpr size_t NUM_RECEIVER_CHANNELS_CT_ARG_IDX = SENDER_CHANNEL_NOC_CONFIG_START_IDX + 1;
constexpr size_t NUM_RECEIVER_CHANNELS = get_compile_time_arg_val(NUM_RECEIVER_CHANNELS_CT_ARG_IDX);
constexpr size_t NUM_FORWARDING_PATHS_CT_ARG_IDX = NUM_RECEIVER_CHANNELS_CT_ARG_IDX + 1;
constexpr size_t NUM_FORWARDING_PATHS = get_compile_time_arg_val(NUM_FORWARDING_PATHS_CT_ARG_IDX);
constexpr size_t wait_for_host_signal_IDX = NUM_FORWARDING_PATHS_CT_ARG_IDX + 1;
constexpr bool wait_for_host_signal = get_compile_time_arg_val(wait_for_host_signal_IDX);
constexpr size_t MAIN_CT_ARGS_START_IDX = wait_for_host_signal_IDX + 1;

static_assert(
    NUM_RECEIVER_CHANNELS <= NUM_SENDER_CHANNELS,
    "NUM_RECEIVER_CHANNELS must be less than or equal to NUM_SENDER_CHANNELS");
static_assert(
    NUM_RECEIVER_CHANNELS <= MAX_NUM_RECEIVER_CHANNELS,
    "NUM_RECEIVER_CHANNELS must be less than or equal to MAX_NUM_RECEIVER_CHANNELS");
static_assert(
    NUM_SENDER_CHANNELS <= MAX_NUM_SENDER_CHANNELS,
    "NUM_SENDER_CHANNELS must be less than or equal to MAX_NUM_SENDER_CHANNELS");
static_assert(wait_for_host_signal_IDX == 3, "wait_for_host_signal_IDX must be 3");
static_assert(
    get_compile_time_arg_val(wait_for_host_signal_IDX) == 0 || get_compile_time_arg_val(wait_for_host_signal_IDX) == 1,
    "wait_for_host_signal must be 0 or 1");
static_assert(MAIN_CT_ARGS_START_IDX == 4, "MAIN_CT_ARGS_START_IDX must be 4");

constexpr uint32_t SWITCH_INTERVAL =
#ifndef DEBUG_PRINT_ENABLED
    get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 0);
#else
    0;
#endif
constexpr bool enable_first_level_ack = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 1);
constexpr bool fuse_receiver_flush_and_completion_ptr = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 2);
constexpr bool enable_ring_support = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 3);
constexpr bool dateline_connection = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 4);
constexpr bool is_handshake_sender = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 5) != 0;
constexpr size_t handshake_addr = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 6);

static_assert(enable_first_level_ack == 0, "enable_first_level_ack must be 0");
static_assert(fuse_receiver_flush_and_completion_ptr == 1, "fuse_receiver_flush_and_completion_ptr must be 0");
static_assert(!enable_ring_support || NUM_RECEIVER_CHANNELS > 1, "Ring support requires at least 2 receiver channels");
// TODO: Pipe from host
constexpr size_t NUM_USED_RECEIVER_CHANNELS = NUM_FORWARDING_PATHS;

constexpr size_t VC0_RECEIVER_CHANNEL = dateline_connection ? 1 : 0;
// On a dateline connection, we would never forward through the dateline on VC1

// VC1/dateline vc is the last of available sender channels.
// For 1D, its 2, For 2D its 4.
constexpr size_t VC1_SENDER_CHANNEL = NUM_SENDER_CHANNELS - 1;

// Doesn't REALLY matter but for consistency I picked the next available ID
constexpr size_t worker_info_offset_past_connection_semaphore = 32;

// the size of one of the buffers within a sender channel
// For example if `channel_buffer_size` = 4k, with `SENDER_NUM_BUFFERS` = 2
// then the total amount of buffering for that
constexpr size_t channel_buffer_size = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 7);

constexpr size_t SENDER_NUM_BUFFERS_IDX = MAIN_CT_ARGS_START_IDX + 8;
constexpr std::array<size_t, NUM_SENDER_CHANNELS> SENDER_NUM_BUFFERS_ARRAY =
    fill_array_with_next_n_args<size_t, SENDER_NUM_BUFFERS_IDX, NUM_SENDER_CHANNELS>();

// dateline edm recv channel 0 has 0 buffer slots, dateline upstream channel 1 has 0 buffer.
constexpr size_t RECEIVER_NUM_BUFFERS_IDX = SENDER_NUM_BUFFERS_IDX + NUM_SENDER_CHANNELS;
constexpr std::array<size_t, NUM_RECEIVER_CHANNELS> RECEIVER_NUM_BUFFERS_ARRAY =
    fill_array_with_next_n_args<size_t, RECEIVER_NUM_BUFFERS_IDX, NUM_RECEIVER_CHANNELS>();

constexpr size_t REMOTE_RECEIVER_NUM_BUFFERS_IDX = RECEIVER_NUM_BUFFERS_IDX + NUM_RECEIVER_CHANNELS;
constexpr std::array<size_t, NUM_RECEIVER_CHANNELS> REMOTE_RECEIVER_NUM_BUFFERS_ARRAY =
    fill_array_with_next_n_args<size_t, REMOTE_RECEIVER_NUM_BUFFERS_IDX, NUM_RECEIVER_CHANNELS>();

constexpr size_t NUM_DOWNSTREAM_CHANNELS = NUM_FORWARDING_PATHS;
constexpr size_t DOWNSTREAM_SENDER_NUM_BUFFERS_IDX = REMOTE_RECEIVER_NUM_BUFFERS_IDX + NUM_RECEIVER_CHANNELS;
constexpr std::array<size_t, NUM_DOWNSTREAM_CHANNELS> DOWNSTREAM_SENDER_NUM_BUFFERS_ARRAY =
    fill_array_with_next_n_args<size_t, DOWNSTREAM_SENDER_NUM_BUFFERS_IDX, NUM_DOWNSTREAM_CHANNELS>();
// TODO: remove DOWNSTREAM_SENDER_NUM_BUFFERS and use TMP on downstream sender channels.
constexpr size_t DOWNSTREAM_SENDER_NUM_BUFFERS = DOWNSTREAM_SENDER_NUM_BUFFERS_ARRAY[0];

constexpr size_t SKIP_CHANNEL_IDX = DOWNSTREAM_SENDER_NUM_BUFFERS_IDX + NUM_DOWNSTREAM_CHANNELS;
constexpr bool skip_receiver_channel_1_connection = get_compile_time_arg_val(SKIP_CHANNEL_IDX);
constexpr bool skip_sender_channel_1_connection = get_compile_time_arg_val(SKIP_CHANNEL_IDX + 1);
constexpr bool skip_sender_vc1_channel_connection = get_compile_time_arg_val(SKIP_CHANNEL_IDX + 2);

constexpr size_t MAIN_CT_ARGS_IDX_1 = SKIP_CHANNEL_IDX + 3;
constexpr size_t local_sender_0_channel_address = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_1);
constexpr size_t local_sender_channel_0_connection_info_addr = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_1 + 1);
constexpr size_t local_sender_1_channel_address = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_1 + 2);
constexpr size_t local_sender_channel_1_connection_info_addr = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_1 + 3);
constexpr size_t local_sender_2_channel_address = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_1 + 4);
constexpr size_t local_sender_channel_2_connection_info_addr = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_1 + 5);
constexpr size_t local_sender_3_channel_address = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_1 + 6);
constexpr size_t local_sender_channel_3_connection_info_addr = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_1 + 7);
constexpr size_t local_sender_4_channel_address = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_1 + 8);
constexpr size_t local_sender_channel_4_connection_info_addr = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_1 + 9);
constexpr size_t local_receiver_0_channel_buffer_address = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_1 + 10);
constexpr size_t remote_receiver_0_channel_buffer_address = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_1 + 11);
constexpr size_t local_receiver_1_channel_buffer_address = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_1 + 12);
constexpr size_t remote_receiver_1_channel_buffer_address = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_1 + 13);
constexpr size_t remote_sender_0_channel_address = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_1 + 14);
constexpr size_t remote_sender_1_channel_address = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_1 + 15);
constexpr size_t remote_sender_2_channel_address = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_1 + 16);
constexpr size_t remote_sender_3_channel_address = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_1 + 17);
constexpr size_t remote_sender_4_channel_address = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_1 + 18);

// TODO: CONVERT TO SEMAPHORE
constexpr size_t MAIN_CT_ARGS_IDX_2 = MAIN_CT_ARGS_IDX_1 + 19;
constexpr uint32_t termination_signal_addr = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_2);
constexpr uint32_t edm_local_sync_ptr_addr =
    wait_for_host_signal ? get_compile_time_arg_val(MAIN_CT_ARGS_IDX_2 + 1) : 0;
constexpr uint32_t edm_status_ptr_addr = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_2 + 2);

// Per-channel counters
constexpr size_t MAIN_CT_ARGS_IDX_3 = MAIN_CT_ARGS_IDX_2 + 3;
constexpr bool enable_fabric_counters = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_3 + 0) != 0;
constexpr size_t receiver_channel_0_counters_address = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_3 + 1);
constexpr size_t receiver_channel_1_counters_address = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_3 + 2);
constexpr size_t sender_channel_0_counters_address = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_3 + 3);
constexpr size_t sender_channel_1_counters_address = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_3 + 4);
constexpr size_t sender_channel_2_counters_address = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_3 + 5);
constexpr size_t sender_channel_3_counters_address = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_3 + 6);
constexpr size_t sender_channel_4_counters_address = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_3 + 7);

constexpr size_t MAIN_CT_ARGS_IDX_4 = MAIN_CT_ARGS_IDX_3 + 8;
constexpr bool enable_packet_header_recording = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_4 + 0) != 0;
constexpr size_t receiver_0_completed_packet_header_cb_address = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_4 + 1);
constexpr size_t receiver_0_completed_packet_header_cb_size_headers = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_4 + 2);
constexpr size_t receiver_1_completed_packet_header_cb_address = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_4 + 3);
constexpr size_t receiver_1_completed_packet_header_cb_size_headers = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_4 + 4);
constexpr size_t sender_0_completed_packet_header_cb_address = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_4 + 5);
constexpr size_t sender_0_completed_packet_header_cb_size_headers = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_4 + 6);
constexpr size_t sender_1_completed_packet_header_cb_address = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_4 + 7);
constexpr size_t sender_1_completed_packet_header_cb_size_headers = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_4 + 8);
constexpr size_t sender_2_completed_packet_header_cb_address = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_4 + 9);
constexpr size_t sender_2_completed_packet_header_cb_size_headers = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_4 + 10);
constexpr size_t sender_3_completed_packet_header_cb_address = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_4 + 11);
constexpr size_t sender_3_completed_packet_header_cb_size_headers = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_4 + 12);
constexpr size_t sender_4_completed_packet_header_cb_address = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_4 + 13);
constexpr size_t sender_4_completed_packet_header_cb_size_headers = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_4 + 14);

constexpr size_t sender_channel_serviced_args_idx = MAIN_CT_ARGS_IDX_4 + 15;
constexpr std::array<bool, MAX_NUM_SENDER_CHANNELS> is_sender_channel_serviced =
    fill_array_with_next_n_args<bool, sender_channel_serviced_args_idx, MAX_NUM_SENDER_CHANNELS>();
constexpr size_t receiver_channel_serviced_args_idx =
    sender_channel_serviced_args_idx + is_sender_channel_serviced.size();
constexpr std::array<bool, MAX_NUM_RECEIVER_CHANNELS> is_receiver_channel_serviced =
    fill_array_with_next_n_args<bool, receiver_channel_serviced_args_idx, MAX_NUM_RECEIVER_CHANNELS>();
constexpr size_t MAIN_CT_ARGS_IDX_5 = receiver_channel_serviced_args_idx + is_receiver_channel_serviced.size();

constexpr bool enable_ethernet_handshake = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_5) != 0;
constexpr bool enable_context_switch = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_5 + 1) != 0;
constexpr bool enable_interrupts = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_5 + 2) != 0;
constexpr size_t sender_txq_id = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_5 + 3);
constexpr size_t receiver_txq_id = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_5 + 4);
constexpr size_t iterations_between_ctx_switch_and_teardown_checks = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_5 + 5);
constexpr size_t is_2d_fabric = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_5 + 6);
constexpr size_t my_direction = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_5 + 7);
constexpr size_t num_eth_ports = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_5 + 8);

// If true, the sender channel will spin inside send_next_data until the eth_txq is not busy, rather than checking
// eth_txq_is_busy() being false as a prerequisite for sending the next packet
constexpr bool ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_5 + 9) != 0;
constexpr bool ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_5 + 10) != 0;

constexpr size_t DEFAULT_NUM_ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_5 + 11);

// Context switch timeouts
constexpr size_t DEFAULT_HANDSHAKE_CONTEXT_SWITCH_TIMEOUT = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_5 + 12);
constexpr bool IDLE_CONTEXT_SWITCHING = get_compile_time_arg_val(MAIN_CT_ARGS_IDX_5 + 13) != 0;

constexpr size_t SPECIAL_MARKER_0_IDX = MAIN_CT_ARGS_IDX_5 + 14;
constexpr size_t SPECIAL_MARKER_0 = 0x00c0ffee;
static_assert(
    !SPECIAL_MARKER_CHECK_ENABLED || get_compile_time_arg_val(SPECIAL_MARKER_0_IDX) == SPECIAL_MARKER_0,
    "Special marker 0 not found. This implies some arguments were misaligned between host and device. Double check the "
    "CT args.");

constexpr size_t SKIP_LIVENESS_CHECK_ARG_IDX = SPECIAL_MARKER_0_IDX + SPECIAL_MARKER_CHECK_ENABLED;
constexpr std::array<bool, NUM_SENDER_CHANNELS> sender_ch_live_check_skip =
    fill_array_with_next_n_args<bool, SKIP_LIVENESS_CHECK_ARG_IDX, NUM_SENDER_CHANNELS>();

constexpr size_t SENDER_CHANNEL_ACK_NOC_IDS_START_IDX = SKIP_LIVENESS_CHECK_ARG_IDX + NUM_SENDER_CHANNELS;
constexpr size_t SENDER_CHANNEL_ACK_CMD_BUF_IDS_START_IDX = SENDER_CHANNEL_ACK_NOC_IDS_START_IDX + NUM_SENDER_CHANNELS;
constexpr std::array<size_t, NUM_SENDER_CHANNELS> sender_channel_ack_noc_ids =
    fill_array_with_next_n_args<size_t, SENDER_CHANNEL_ACK_NOC_IDS_START_IDX, NUM_SENDER_CHANNELS>();
constexpr std::array<uint8_t, NUM_SENDER_CHANNELS> sender_channel_ack_cmd_buf_ids =
    fill_array_with_next_n_args<uint8_t, SENDER_CHANNEL_ACK_CMD_BUF_IDS_START_IDX, NUM_SENDER_CHANNELS>();

constexpr size_t RX_CH_FWD_NOC_IDS_START_IDX = SENDER_CHANNEL_ACK_CMD_BUF_IDS_START_IDX + NUM_SENDER_CHANNELS;
constexpr size_t RX_CH_FWD_DATA_CMD_BUF_IDS_START_IDX = RX_CH_FWD_NOC_IDS_START_IDX + NUM_RECEIVER_CHANNELS;
constexpr size_t RX_CH_FWD_SYNC_CMD_BUF_IDS_START_IDX = RX_CH_FWD_DATA_CMD_BUF_IDS_START_IDX + NUM_RECEIVER_CHANNELS;
constexpr size_t RX_CH_LOCAL_WRITE_NOC_ID_IDX = RX_CH_FWD_SYNC_CMD_BUF_IDS_START_IDX + NUM_RECEIVER_CHANNELS;
constexpr size_t RX_CH_LOCAL_WRITE_CMD_BUF_ID_IDX = RX_CH_LOCAL_WRITE_NOC_ID_IDX + NUM_RECEIVER_CHANNELS;

constexpr std::array<size_t, NUM_RECEIVER_CHANNELS> receiver_channel_forwarding_noc_ids =
    fill_array_with_next_n_args<size_t, RX_CH_FWD_NOC_IDS_START_IDX, NUM_RECEIVER_CHANNELS>();
constexpr std::array<uint8_t, NUM_RECEIVER_CHANNELS> receiver_channel_forwarding_data_cmd_buf_ids =
    fill_array_with_next_n_args<uint8_t, RX_CH_FWD_DATA_CMD_BUF_IDS_START_IDX, NUM_RECEIVER_CHANNELS>();
constexpr std::array<uint8_t, NUM_RECEIVER_CHANNELS> receiver_channel_forwarding_sync_cmd_buf_ids =
    fill_array_with_next_n_args<uint8_t, RX_CH_FWD_SYNC_CMD_BUF_IDS_START_IDX, NUM_RECEIVER_CHANNELS>();
constexpr std::array<size_t, NUM_RECEIVER_CHANNELS> receiver_channel_local_write_noc_ids =
    fill_array_with_next_n_args<size_t, RX_CH_LOCAL_WRITE_NOC_ID_IDX, NUM_RECEIVER_CHANNELS>();
constexpr std::array<uint8_t, NUM_RECEIVER_CHANNELS> receiver_channel_local_write_cmd_buf_ids =
    fill_array_with_next_n_args<uint8_t, RX_CH_LOCAL_WRITE_CMD_BUF_ID_IDX, NUM_RECEIVER_CHANNELS>();

// TODO: Add a special marker in CT args so we don't misalign unintentionally
constexpr size_t EDM_NOC_VC_IDX = RX_CH_LOCAL_WRITE_CMD_BUF_ID_IDX + NUM_RECEIVER_CHANNELS;
constexpr size_t SPECIAL_MARKER_1_IDX = EDM_NOC_VC_IDX + 1;
constexpr size_t SPECIAL_MARKER_1 = 0x10c0ffee;
static_assert(
    !SPECIAL_MARKER_CHECK_ENABLED || get_compile_time_arg_val(SPECIAL_MARKER_1_IDX) == SPECIAL_MARKER_1,
    "Special marker 1 not found. This implies some arguments were misaligned between host and device. Double check the "
    "CT args.");

constexpr size_t HOST_SIGNAL_ARGS_START_IDX = SPECIAL_MARKER_1_IDX + SPECIAL_MARKER_CHECK_ENABLED;
// static_assert(HOST_SIGNAL_ARGS_START_IDX == 56, "HOST_SIGNAL_ARGS_START_IDX must be 56");
// TODO: Add type safe getter
constexpr bool is_local_handshake_master =
    bool(conditional_get_compile_time_arg<wait_for_host_signal, HOST_SIGNAL_ARGS_START_IDX + 0>());
constexpr uint32_t local_handshake_master_eth_chan =
    conditional_get_compile_time_arg<wait_for_host_signal, HOST_SIGNAL_ARGS_START_IDX + 1>();
constexpr uint32_t num_local_edms =
    conditional_get_compile_time_arg<wait_for_host_signal, HOST_SIGNAL_ARGS_START_IDX + 2>();
constexpr uint32_t edm_channels_mask =
    conditional_get_compile_time_arg<wait_for_host_signal, HOST_SIGNAL_ARGS_START_IDX + 3>();

constexpr size_t VC1_RECEIVER_CHANNEL = 1;

constexpr size_t sender_channel_base_id = 0;
constexpr size_t receiver_channel_base_id = NUM_SENDER_CHANNELS;

// TRANSACTION IDS
// TODO: Pass this value from host
constexpr uint8_t NUM_TRANSACTION_IDS = enable_ring_support ? 8 : 4;

constexpr std::array<uint8_t, MAX_NUM_RECEIVER_CHANNELS> RX_CH_TRID_STARTS =
    initialize_receiver_channel_trid_starts<MAX_NUM_RECEIVER_CHANNELS, NUM_TRANSACTION_IDS>();

constexpr std::array<uint32_t, MAX_NUM_RECEIVER_CHANNELS> to_receiver_packets_sent_streams =
    take_first_n_elements<MAX_NUM_RECEIVER_CHANNELS, MAX_NUM_RECEIVER_CHANNELS, uint32_t>(
        std::array<uint32_t, MAX_NUM_RECEIVER_CHANNELS>{to_receiver_0_pkts_sent_id, to_receiver_1_pkts_sent_id});

// not in symbol table - because not used
constexpr std::array<uint32_t, MAX_NUM_SENDER_CHANNELS> to_sender_packets_acked_streams =
    take_first_n_elements<MAX_NUM_SENDER_CHANNELS, MAX_NUM_SENDER_CHANNELS, uint32_t>(
        std::array<uint32_t, MAX_NUM_SENDER_CHANNELS>{
            to_sender_0_pkts_acked_id, to_sender_1_pkts_acked_id, to_sender_2_pkts_acked_id,
            to_sender_3_pkts_acked_id, to_sender_4_pkts_acked_id});

// data section
constexpr std::array<uint32_t, MAX_NUM_SENDER_CHANNELS> to_sender_packets_completed_streams =
    take_first_n_elements<MAX_NUM_SENDER_CHANNELS, MAX_NUM_SENDER_CHANNELS, uint32_t>(
        std::array<uint32_t, MAX_NUM_SENDER_CHANNELS>{
            to_sender_0_pkts_completed_id, to_sender_1_pkts_completed_id, to_sender_2_pkts_completed_id,
            to_sender_3_pkts_completed_id, to_sender_4_pkts_completed_id});

// Miscellaneous configuration

// TODO: move this to compile time args if we need to enable it
constexpr bool enable_trid_flush_check_on_noc_txn = false;

namespace tt::tt_fabric {
static_assert(
    receiver_channel_local_write_noc_ids[0] == edm_to_local_chip_noc,
    "edm_to_local_chip_noc must equal to receiver_channel_local_write_noc_ids");
static constexpr uint8_t edm_to_downstream_noc = receiver_channel_forwarding_noc_ids[0];
static constexpr uint8_t worker_handshake_noc = sender_channel_ack_noc_ids[0];
constexpr bool local_chip_noc_equals_downstream_noc =
    receiver_channel_forwarding_noc_ids[0] == receiver_channel_local_write_noc_ids[0];
static constexpr uint8_t local_chip_data_cmd_buf = receiver_channel_local_write_cmd_buf_ids[0];
static constexpr uint8_t forward_and_local_write_noc_vc = get_compile_time_arg_val(EDM_NOC_VC_IDX);

}  // namespace tt::tt_fabric
