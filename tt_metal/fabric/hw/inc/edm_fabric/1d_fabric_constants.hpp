// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dataflow_api.h"

#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_utils.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/compile_time_arg_tmp.hpp"

#include <array>
#include <utility>

// CHANNEL CONSTANTS
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

constexpr size_t MAX_NUM_RECEIVER_CHANNELS = 2;
constexpr size_t MAX_NUM_SENDER_CHANNELS = 3;

// Compile Time args

constexpr bool SPECIAL_MARKER_CHECK_ENABLED = true;
constexpr size_t SENDER_CHANNEL_NOC_CONFIG_START_IDX = 0;
constexpr size_t NUM_RECEIVER_CHANNELS_CT_ARG_IDX = SENDER_CHANNEL_NOC_CONFIG_START_IDX + 1;
constexpr size_t NUM_SENDER_CHANNELS = get_compile_time_arg_val(SENDER_CHANNEL_NOC_CONFIG_START_IDX);
constexpr size_t NUM_RECEIVER_CHANNELS = get_compile_time_arg_val(NUM_RECEIVER_CHANNELS_CT_ARG_IDX);
constexpr size_t wait_for_host_signal_IDX = NUM_RECEIVER_CHANNELS_CT_ARG_IDX + 1;
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
static_assert(wait_for_host_signal_IDX == 2, "wait_for_host_signal_IDX must be 3");
static_assert(
    get_compile_time_arg_val(wait_for_host_signal_IDX) == 0 || get_compile_time_arg_val(wait_for_host_signal_IDX) == 1,
    "wait_for_host_signal must be 0 or 1");
static_assert(MAIN_CT_ARGS_START_IDX == 3, "MAIN_CT_ARGS_START_IDX must be 3");

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
constexpr size_t NUM_USED_RECEIVER_CHANNELS = NUM_RECEIVER_CHANNELS;
constexpr size_t VC0_RECEIVER_CHANNEL = dateline_connection ? 1 : 0;
// On a dateline connection, we would never forward through the dateline on VC1

// Doesn't REALLY matter but for consistency I picked the next available ID
constexpr size_t worker_info_offset_past_connection_semaphore = 32;

// the size of one of the buffers within a sender channel
// For example if `channel_buffer_size` = 4k, with `SENDER_NUM_BUFFERS` = 2
// then the total amount of buffering for that
constexpr size_t channel_buffer_size = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 7);

constexpr size_t SENDER_NUM_BUFFERS = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 8);
constexpr size_t RECEIVER_NUM_BUFFERS = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 9);
constexpr size_t local_sender_0_channel_address = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 10);
constexpr size_t local_sender_channel_0_connection_info_addr = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 11);
constexpr size_t local_sender_1_channel_address = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 12);
constexpr size_t local_sender_channel_1_connection_info_addr = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 13);
constexpr size_t local_sender_2_channel_address = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 14);
constexpr size_t local_sender_channel_2_connection_info_addr = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 15);
constexpr size_t local_receiver_0_channel_buffer_address = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 16);
constexpr size_t remote_receiver_0_channel_buffer_address = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 17);
constexpr size_t local_receiver_1_channel_buffer_address = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 18);
constexpr size_t remote_receiver_1_channel_buffer_address = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 19);
constexpr size_t remote_sender_0_channel_address = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 20);
constexpr size_t remote_sender_1_channel_address = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 21);
constexpr size_t remote_sender_2_channel_address = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 22);

// TODO: CONVERT TO SEMAPHORE
constexpr uint32_t termination_signal_addr = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 23);
constexpr uint32_t edm_local_sync_ptr_addr =
    wait_for_host_signal ? get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 24) : 0;
constexpr uint32_t edm_status_ptr_addr = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 25);

constexpr bool persistent_mode = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 26) != 0;

// Per-channel counters
constexpr bool enable_fabric_counters = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 27) != 0;
constexpr size_t receiver_channel_0_counters_address = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 28);
constexpr size_t receiver_channel_1_counters_address = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 29);
constexpr size_t sender_channel_0_counters_address = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 30);
constexpr size_t sender_channel_1_counters_address = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 31);
constexpr size_t sender_channel_2_counters_address = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 32);

constexpr bool enable_packet_header_recording = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 33) != 0;
constexpr size_t receiver_0_completed_packet_header_cb_address = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 34);
constexpr size_t receiver_0_completed_packet_header_cb_size_headers =
    get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 35);
constexpr size_t receiver_1_completed_packet_header_cb_address = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 36);
constexpr size_t receiver_1_completed_packet_header_cb_size_headers =
    get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 37);
constexpr size_t sender_0_completed_packet_header_cb_address = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 38);
constexpr size_t sender_0_completed_packet_header_cb_size_headers =
    get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 39);
constexpr size_t sender_1_completed_packet_header_cb_address = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 40);
constexpr size_t sender_1_completed_packet_header_cb_size_headers =
    get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 41);
constexpr size_t sender_2_completed_packet_header_cb_address = get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 42);
constexpr size_t sender_2_completed_packet_header_cb_size_headers =
    get_compile_time_arg_val(MAIN_CT_ARGS_START_IDX + 43);

constexpr size_t SPECIAL_MARKER_0_IDX = MAIN_CT_ARGS_START_IDX + 44;
constexpr size_t SPECIAL_MARKER_0 = 0x00c0ffee;
static_assert(
    !SPECIAL_MARKER_CHECK_ENABLED || get_compile_time_arg_val(SPECIAL_MARKER_0_IDX) == SPECIAL_MARKER_0,
    "Special marker 0 not found. This implies some arguments were misaligned between host and device. Double check the "
    "CT args.");

constexpr size_t SENDER_CHANNEL_ACK_NOC_IDS_START_IDX = SPECIAL_MARKER_0_IDX + SPECIAL_MARKER_CHECK_ENABLED;
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
constexpr size_t SPECIAL_MARKER_1_IDX = RX_CH_LOCAL_WRITE_CMD_BUF_ID_IDX + NUM_RECEIVER_CHANNELS;
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

constexpr size_t receiver_channel_base_id = NUM_SENDER_CHANNELS;

// TRANSACTION IDS
constexpr uint8_t NUM_TRANSACTION_IDS = 4;

constexpr std::array<uint8_t, MAX_NUM_RECEIVER_CHANNELS> RX_CH_TRID_STARTS =
    initialize_receiver_channel_trid_starts<MAX_NUM_RECEIVER_CHANNELS, NUM_TRANSACTION_IDS>();

constexpr std::array<uint32_t, MAX_NUM_RECEIVER_CHANNELS> to_receiver_packets_sent_streams =
    take_first_n_elements<MAX_NUM_RECEIVER_CHANNELS, MAX_NUM_RECEIVER_CHANNELS, uint32_t>(
        std::array<uint32_t, MAX_NUM_RECEIVER_CHANNELS>{to_receiver_0_pkts_sent_id, to_receiver_1_pkts_sent_id});

// not in symbol table - because not used
constexpr std::array<uint32_t, MAX_NUM_SENDER_CHANNELS> to_sender_packets_acked_streams =
    take_first_n_elements<MAX_NUM_SENDER_CHANNELS, MAX_NUM_SENDER_CHANNELS, uint32_t>(
        std::array<uint32_t, MAX_NUM_SENDER_CHANNELS>{
            to_sender_0_pkts_acked_id, to_sender_1_pkts_acked_id, to_sender_2_pkts_acked_id});

// data section
constexpr std::array<uint32_t, MAX_NUM_SENDER_CHANNELS> to_sender_packets_completed_streams =
    take_first_n_elements<MAX_NUM_SENDER_CHANNELS, MAX_NUM_SENDER_CHANNELS, uint32_t>(
        std::array<uint32_t, MAX_NUM_SENDER_CHANNELS>{
            to_sender_0_pkts_completed_id, to_sender_1_pkts_completed_id, to_sender_2_pkts_completed_id});

// Miscellaneous configuration
constexpr uint32_t DEFAULT_ITERATIONS_BETWEEN_CTX_SWITCH_AND_TEARDOWN_CHECKS = 32;
constexpr size_t DEFAULT_HANDSHAKE_CONTEXT_SWITCH_TIMEOUT = 0;

namespace tt::tt_fabric {
static_assert(
    receiver_channel_forwarding_noc_ids[0] == edm_to_local_chip_noc,
    "edm_to_local_chip_noc must be 1 otherwise packet header setup must be modified to account for the final fabric "
    "router not necessarily using noc1 for writing");

static constexpr uint8_t local_chip_data_cmd_buf = receiver_channel_local_write_cmd_buf_ids[0];

}  // namespace tt::tt_fabric
