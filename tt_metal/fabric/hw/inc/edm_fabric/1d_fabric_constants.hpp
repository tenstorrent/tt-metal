// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dataflow_api.h"

#include <array>

// CHANNEL CONSTANTS

constexpr size_t NUM_LINE_SENDER_CHANNELS = 2;
constexpr size_t NUM_RING_SENDER_CHANNELS = 3;
// This is a placeholder for sizing arrays and other types properly
// - we padd up for constexpr arrays
constexpr size_t NUM_SENDER_CHANNELS = std::max(NUM_LINE_SENDER_CHANNELS, NUM_RING_SENDER_CHANNELS);

constexpr size_t NUM_LINE_RECEIVER_CHANNELS = 1;
constexpr size_t NUM_RING_RECEIVER_CHANNELS = 2;
constexpr size_t NUM_RECEIVER_CHANNELS = std::max(NUM_LINE_RECEIVER_CHANNELS, NUM_RING_RECEIVER_CHANNELS);

constexpr size_t VC1_RECEIVER_CHANNEL = 1;

constexpr size_t receiver_channel_base_id = NUM_SENDER_CHANNELS;

// TRANSACTION IDS
constexpr uint8_t NUM_TRANSACTION_IDS = 4;

constexpr uint8_t RX_CH0_TRID_START = 0;
constexpr uint8_t RX_CH1_TRID_START = NUM_TRANSACTION_IDS;
constexpr std::array<uint8_t, NUM_RECEIVER_CHANNELS> RX_CH_TRID_STARTS = {
    RX_CH0_TRID_START,
    RX_CH1_TRID_START,
};

// ETH TXQ SELECTION
constexpr uint32_t DEFAULT_ETH_TXQ = 0;
constexpr uint32_t DEFAULT_NUM_ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD = 32;

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
constexpr uint32_t to_sender_0_pkts_completed_id = 5;
// receivers updates the reg on this stream
constexpr uint32_t to_sender_1_pkts_completed_id = 6;
// receivers updates the reg on this stream
constexpr uint32_t to_sender_2_pkts_completed_id = 7;

constexpr std::array<uint32_t, NUM_RECEIVER_CHANNELS> to_receiver_packets_sent_streams = {
    to_receiver_0_pkts_sent_id, to_receiver_1_pkts_sent_id};

// not in symbol table - because not used
constexpr std::array<uint32_t, NUM_SENDER_CHANNELS> to_sender_packets_acked_streams = {
    {to_sender_0_pkts_acked_id, to_sender_1_pkts_acked_id, to_sender_2_pkts_acked_id}};

// data section
constexpr std::array<uint32_t, NUM_SENDER_CHANNELS> to_sender_packets_completed_streams = {
    {to_sender_0_pkts_completed_id, to_sender_1_pkts_completed_id, to_sender_2_pkts_completed_id}};

// Miscellaneous configuration
constexpr uint32_t DEFAULT_ITERATIONS_BETWEEN_CTX_SWITCH_AND_TEARDOWN_CHECKS = 32;
constexpr size_t DEFAULT_HANDSHAKE_CONTEXT_SWITCH_TIMEOUT = 0;

// Compile Time args
constexpr uint32_t SWITCH_INTERVAL =
#ifndef DEBUG_PRINT_ENABLED
    get_compile_time_arg_val(0);
#else
    0;
#endif

constexpr bool enable_first_level_ack = get_compile_time_arg_val(1);
constexpr bool fuse_receiver_flush_and_completion_ptr = get_compile_time_arg_val(2);
constexpr bool enable_ring_support = get_compile_time_arg_val(3);
constexpr bool dateline_connection = get_compile_time_arg_val(4);
constexpr bool is_handshake_sender = get_compile_time_arg_val(5) != 0;
constexpr size_t handshake_addr = get_compile_time_arg_val(6);

// TODO: Pipe from host
constexpr size_t NUM_USED_SENDER_CHANNELS = enable_ring_support ? NUM_RING_SENDER_CHANNELS : NUM_LINE_SENDER_CHANNELS;
constexpr size_t NUM_USED_RECEIVER_CHANNELS =
    enable_ring_support ? NUM_RING_RECEIVER_CHANNELS : NUM_LINE_RECEIVER_CHANNELS;
constexpr size_t VC0_RECEIVER_CHANNEL = dateline_connection ? 1 : 0;
// On a dateline connection, we would never forward through the dateline on VC1

// Doesn't REALLY matter but for consistency I picked the next available ID
constexpr size_t worker_info_offset_past_connection_semaphore = 32;

// the size of one of the buffers within a sender channel
// For example if `channel_buffer_size` = 4k, with `SENDER_NUM_BUFFERS` = 2
// then the total amount of buffering for that
constexpr size_t channel_buffer_size = get_compile_time_arg_val(7);

constexpr size_t SENDER_NUM_BUFFERS = get_compile_time_arg_val(8);
constexpr size_t RECEIVER_NUM_BUFFERS = get_compile_time_arg_val(9);
constexpr size_t local_sender_0_channel_address = get_compile_time_arg_val(10);
constexpr size_t local_sender_channel_0_connection_info_addr = get_compile_time_arg_val(11);
constexpr size_t local_sender_1_channel_address = get_compile_time_arg_val(12);
constexpr size_t local_sender_channel_1_connection_info_addr = get_compile_time_arg_val(13);
constexpr size_t local_sender_2_channel_address = get_compile_time_arg_val(14);
constexpr size_t local_sender_channel_2_connection_info_addr = get_compile_time_arg_val(15);
constexpr size_t local_receiver_0_channel_buffer_address = get_compile_time_arg_val(16);
constexpr size_t remote_receiver_0_channel_buffer_address = get_compile_time_arg_val(17);
constexpr size_t local_receiver_1_channel_buffer_address = get_compile_time_arg_val(18);
constexpr size_t remote_receiver_1_channel_buffer_address = get_compile_time_arg_val(19);
constexpr size_t remote_sender_0_channel_address = get_compile_time_arg_val(20);
constexpr size_t remote_sender_1_channel_address = get_compile_time_arg_val(21);
constexpr size_t remote_sender_2_channel_address = get_compile_time_arg_val(22);

// TODO: CONVERT TO SEMAPHORE
constexpr uint32_t termination_signal_addr = get_compile_time_arg_val(23);
#ifdef WAIT_FOR_HOST_SIGNAL
constexpr uint32_t edm_local_sync_ptr_addr = get_compile_time_arg_val(24);
#endif
constexpr uint32_t edm_status_ptr_addr = get_compile_time_arg_val(25);

constexpr bool persistent_mode = get_compile_time_arg_val(26) != 0;

// Per-channel counters
constexpr bool enable_fabric_counters = get_compile_time_arg_val(27) != 0;
constexpr size_t receiver_channel_0_counters_address = get_compile_time_arg_val(28);
constexpr size_t receiver_channel_1_counters_address = get_compile_time_arg_val(29);
constexpr size_t sender_channel_0_counters_address = get_compile_time_arg_val(30);
constexpr size_t sender_channel_1_counters_address = get_compile_time_arg_val(31);
constexpr size_t sender_channel_2_counters_address = get_compile_time_arg_val(32);

constexpr bool enable_packet_header_recording = get_compile_time_arg_val(33) != 0;
constexpr size_t receiver_0_completed_packet_header_cb_address = get_compile_time_arg_val(34);
constexpr size_t receiver_0_completed_packet_header_cb_size_headers = get_compile_time_arg_val(35);
constexpr size_t receiver_1_completed_packet_header_cb_address = get_compile_time_arg_val(36);
constexpr size_t receiver_1_completed_packet_header_cb_size_headers = get_compile_time_arg_val(37);
constexpr size_t sender_0_completed_packet_header_cb_address = get_compile_time_arg_val(38);
constexpr size_t sender_0_completed_packet_header_cb_size_headers = get_compile_time_arg_val(39);
constexpr size_t sender_1_completed_packet_header_cb_address = get_compile_time_arg_val(40);
constexpr size_t sender_1_completed_packet_header_cb_size_headers = get_compile_time_arg_val(41);
constexpr size_t sender_2_completed_packet_header_cb_address = get_compile_time_arg_val(42);
constexpr size_t sender_2_completed_packet_header_cb_size_headers = get_compile_time_arg_val(43);

#ifdef WAIT_FOR_HOST_SIGNAL
constexpr bool is_local_handshake_master = get_compile_time_arg_val(44);
constexpr uint32_t local_handshake_master_eth_chan = get_compile_time_arg_val(45);
constexpr uint32_t num_local_edms = get_compile_time_arg_val(46);
constexpr uint32_t edm_channels_mask = get_compile_time_arg_val(47);
#endif
