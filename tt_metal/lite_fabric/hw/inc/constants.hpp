// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <array>
#include "tt_metal/lite_fabric/hw/inc/header.hpp"

#if defined(KERNEL_BUILD) || defined(FW_BUILD)
#include "tt_metal/fabric/hw/inc/edm_fabric/compile_time_arg_tmp.hpp"
#include "noc_nonblocking_api.h"
#endif

namespace lite_fabric {

// STREAM REGISTER ASSIGNMENT
// senders update this stream
constexpr uint32_t to_receiver_0_pkts_sent_id = 23;
// receivers updates the reg on this stream
constexpr uint32_t to_sender_0_pkts_acked_id = 24;
// receivers updates the reg on this stream
constexpr uint32_t to_sender_0_pkts_completed_id = 25;

constexpr uint32_t to_receiver_1_pkts_sent_id = 24;
constexpr uint32_t to_sender_1_pkts_acked_id = 25;
constexpr uint32_t to_sender_1_pkts_completed_id = 26;

constexpr size_t MAX_NUM_RECEIVER_CHANNELS = 2;

constexpr size_t MAX_NUM_SENDER_CHANNELS = 5;

constexpr std::array<uint32_t, MAX_NUM_RECEIVER_CHANNELS> to_receiver_pkts_sent_ids = {
    to_receiver_0_pkts_sent_id, to_receiver_1_pkts_sent_id};
constexpr std::array<uint32_t, MAX_NUM_SENDER_CHANNELS> to_sender_pkts_acked_ids = {
    to_sender_0_pkts_acked_id, to_sender_1_pkts_acked_id, 0, 0, 0};
constexpr std::array<uint32_t, MAX_NUM_SENDER_CHANNELS> to_sender_pkts_completed_ids = {
    to_sender_0_pkts_completed_id, to_sender_1_pkts_completed_id, 0, 0, 0};

// Only 1 receiver because 1 erisc
constexpr uint32_t NUM_RECEIVER_CHANNELS = 1;

constexpr uint32_t NUM_USED_RECEIVER_CHANNELS = 1;

// Only 1 sender because no upstream edm
constexpr uint32_t NUM_SENDER_CHANNELS = 1;

constexpr size_t VC1_SENDER_CHANNEL = NUM_SENDER_CHANNELS - 1;

constexpr uint8_t NUM_TRANSACTION_IDS = 4;

constexpr std::array<size_t, NUM_SENDER_CHANNELS> SENDER_NUM_BUFFERS_ARRAY = {2};

constexpr std::array<size_t, NUM_RECEIVER_CHANNELS> RECEIVER_NUM_BUFFERS_ARRAY = {2};

constexpr std::array<size_t, NUM_RECEIVER_CHANNELS> REMOTE_RECEIVER_NUM_BUFFERS_ARRAY = RECEIVER_NUM_BUFFERS_ARRAY;

static_assert(NUM_SENDER_CHANNELS == 1);

// Additional 16B to be used only for unaligned reads/writes
constexpr uint32_t CHANNEL_BUFFER_SIZE = 2048 + 16 + sizeof(lite_fabric::FabricLiteHeader);

constexpr size_t RECEIVER_CHANNEL_BASE_ID = NUM_SENDER_CHANNELS;
constexpr size_t SENDER_CHANNEL_BASE_ID = 0;

#if defined(KERNEL_BUILD) || defined(FW_BUILD)

// Always using NOC0 and default cmd bufs
// Used for acks
constexpr std::array<uint8_t, NUM_SENDER_CHANNELS> sender_channel_ack_cmd_buf_ids = {BRISC_AT_CMD_BUF};

#endif

constexpr bool use_posted_writes_for_connection_open = false;

constexpr bool is_2d_fabric = false;

constexpr uint32_t my_direction = 0;  // No direction for 1D fabric

// Not using multi txq / 2 erisc
constexpr uint32_t NUM_ACTIVE_ERISCS = 1;
constexpr uint32_t DEFAULT_ETH_TXQ = 0;
constexpr bool multi_txq_enabled = false;
constexpr uint32_t sender_txq_id = DEFAULT_ETH_TXQ;
constexpr uint32_t receiver_txq_id = DEFAULT_ETH_TXQ;
constexpr bool enable_first_level_ack = false;
constexpr std::array<size_t, NUM_RECEIVER_CHANNELS> local_receiver_completion_counter_ptrs = {0};
constexpr std::array<size_t, NUM_RECEIVER_CHANNELS> local_receiver_ack_counter_ptrs = {0};
constexpr std::array<size_t, NUM_RECEIVER_CHANNELS> to_sender_remote_completion_counter_addrs = {0};
constexpr std::array<size_t, NUM_RECEIVER_CHANNELS> to_sender_remote_ack_counter_addrs = {0};

// Misc
constexpr std::array<size_t, MAX_NUM_RECEIVER_CHANNELS> RX_CH_TRID_STARTS = {0, 4};
constexpr bool fuse_receiver_flush_and_completion_ptr = true;
constexpr uint8_t num_eth_ports = 32;  // Not used in 1D fabric
constexpr bool ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK = false;
constexpr bool ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA = false;
constexpr bool SKIP_CONNECTION_LIVENESS_CHECK = false;
constexpr bool enable_ring_support = false;
constexpr bool enable_trid_flush_check_on_noc_txn = false;

#if defined(KERNEL_BUILD) || defined(FW_BUILD)
constexpr uint8_t local_chip_data_cmd_buf = BRISC_WR_CMD_BUF;  // Used
constexpr uint8_t worker_handshake_noc = 0;
#endif

// Default NoC to use for Reads/Writes
constexpr uint8_t edm_to_local_chip_noc = 0;
constexpr uint8_t forward_and_local_write_noc_vc = 2;  // FabricEriscDatamoverConfig::DEFAULT_NOC_VC

constexpr uint8_t edm_to_downstream_noc = 0;                 // Used?
constexpr bool local_chip_noc_equals_downstream_noc = true;  // Used?

}  // namespace lite_fabric
