// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

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
