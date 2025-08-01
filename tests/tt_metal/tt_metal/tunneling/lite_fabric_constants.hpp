// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Substitute for 1d_fabric_constants.hpp

#pragma once

#include <cstddef>
#include <cstdint>
#include <array>
#include "fabric_edm_packet_header.hpp"

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
constexpr uint32_t receiver_channel_0_free_slots_from_east_stream_id = 12;
constexpr uint32_t receiver_channel_0_free_slots_from_west_stream_id = 13;
constexpr uint32_t receiver_channel_0_free_slots_from_north_stream_id = 14;
constexpr uint32_t receiver_channel_0_free_slots_from_south_stream_id = 15;
constexpr uint32_t sender_channel_0_free_slots_stream_id = 17;
constexpr uint32_t sender_channel_1_free_slots_stream_id = 18;
constexpr uint32_t sender_channel_2_free_slots_stream_id = 19;
constexpr uint32_t sender_channel_3_free_slots_stream_id = 20;
constexpr uint32_t sender_channel_4_free_slots_stream_id = 21;
constexpr uint32_t vc1_sender_channel_free_slots_stream_id = 22;

constexpr size_t MAX_NUM_RECEIVER_CHANNELS = 2;

constexpr size_t MAX_NUM_SENDER_CHANNELS = 5;

constexpr uint32_t NUM_RECEIVER_CHANNELS = 1;

constexpr uint32_t NUM_SENDER_CHANNELS = 1;

constexpr size_t VC1_SENDER_CHANNEL = NUM_SENDER_CHANNELS - 1;

constexpr uint8_t NUM_TRANSACTION_IDS = 4;

constexpr std::array<size_t, NUM_SENDER_CHANNELS> SENDER_NUM_BUFFERS_ARRAY = {8};

constexpr std::array<size_t, NUM_RECEIVER_CHANNELS> RECEIVER_NUM_BUFFERS_ARRAY = {16};

static_assert(NUM_SENDER_CHANNELS == 1);

constexpr uint32_t CHANNEL_BUFFER_SIZE = 4096 + sizeof(tt::tt_fabric::LowLatencyPacketHeader);

constexpr uint32_t CHANNEL_BUFFER_SLOTS = 8;

constexpr size_t RECEIVER_CHANNEL_BASE_ID = NUM_SENDER_CHANNELS;
constexpr size_t SENDER_CHANNEL_BASE_ID = 0;

#if defined(KERNEL_BUILD) || defined(FW_BUILD)
#include "tt_metal/fabric/hw/inc/edm_fabric/compile_time_arg_tmp.hpp"
#include "noc_nonblocking_api.h"

constexpr std::array<uint32_t, MAX_NUM_RECEIVER_CHANNELS> to_receiver_packets_sent_streams =
    take_first_n_elements<MAX_NUM_RECEIVER_CHANNELS, MAX_NUM_RECEIVER_CHANNELS, uint32_t>(
        std::array<uint32_t, MAX_NUM_RECEIVER_CHANNELS>{to_receiver_0_pkts_sent_id, to_receiver_1_pkts_sent_id});

constexpr std::array<uint32_t, MAX_NUM_SENDER_CHANNELS> to_sender_packets_acked_streams =
    take_first_n_elements<MAX_NUM_SENDER_CHANNELS, MAX_NUM_SENDER_CHANNELS, uint32_t>(
        std::array<uint32_t, MAX_NUM_SENDER_CHANNELS>{
            to_sender_0_pkts_acked_id,
            to_sender_1_pkts_acked_id,
            to_sender_2_pkts_acked_id,
            to_sender_3_pkts_acked_id,
            to_sender_4_pkts_acked_id});

constexpr std::array<uint32_t, MAX_NUM_SENDER_CHANNELS> to_sender_packets_completed_streams =
    take_first_n_elements<MAX_NUM_SENDER_CHANNELS, MAX_NUM_SENDER_CHANNELS, uint32_t>(
        std::array<uint32_t, MAX_NUM_SENDER_CHANNELS>{
            to_sender_0_pkts_completed_id,
            to_sender_1_pkts_completed_id,
            to_sender_2_pkts_completed_id,
            to_sender_3_pkts_completed_id,
            to_sender_4_pkts_completed_id});

static constexpr std::array<uint32_t, MAX_NUM_SENDER_CHANNELS> sender_channel_free_slots_stream_ids = {
    sender_channel_0_free_slots_stream_id,
    sender_channel_1_free_slots_stream_id,
    sender_channel_2_free_slots_stream_id,
    sender_channel_3_free_slots_stream_id,
    sender_channel_4_free_slots_stream_id};

static_assert(std::is_same_v<PACKET_HEADER_TYPE, tt::tt_fabric::LowLatencyPacketHeader>);

constexpr bool use_posted_writes_for_connection_open = false;

constexpr bool is_2d_fabric = false;

constexpr uint32_t my_direction = 0;  // No direction for 1D fabric

// Always using NOC0 and default cmd bufs
// Used for acks
constexpr std::array<uint8_t, NUM_SENDER_CHANNELS> sender_channel_ack_cmd_buf_ids = {BRISC_AT_CMD_BUF};

namespace tt::tt_fabric {

constexpr uint8_t worker_handshake_noc = noc_index;
constexpr uint8_t edm_to_downstream_noc = 0;                 // Used?
constexpr bool local_chip_noc_equals_downstream_noc = true;  // Used?
}
#endif
