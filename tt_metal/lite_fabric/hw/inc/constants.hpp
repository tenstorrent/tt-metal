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
// Consult tt_metal/fabric/erisc_datamover_builder.hpp StreamRegAssignments to ensure no conflicts
constexpr uint32_t to_receiver_0_pkts_sent_id = 23;
constexpr uint32_t to_sender_0_pkts_acked_id = 24;
constexpr uint32_t to_sender_0_pkts_completed_id = 25;
constexpr uint32_t to_receiver_1_pkts_sent_id = 26;
constexpr uint32_t to_sender_1_pkts_acked_id = 27;
constexpr uint32_t to_sender_1_pkts_completed_id = 28;

// Max two but only using 1 for now
constexpr size_t MAX_NUM_RECEIVER_CHANNELS = 2;
constexpr size_t MAX_NUM_SENDER_CHANNELS = 2;

constexpr std::array<uint32_t, MAX_NUM_RECEIVER_CHANNELS> to_receiver_pkts_sent_ids = {
    to_receiver_0_pkts_sent_id, to_receiver_1_pkts_sent_id};
constexpr std::array<uint32_t, MAX_NUM_SENDER_CHANNELS> to_sender_pkts_acked_ids = {
    to_sender_0_pkts_acked_id, to_sender_1_pkts_acked_id};
constexpr std::array<uint32_t, MAX_NUM_SENDER_CHANNELS> to_sender_pkts_completed_ids = {
    to_sender_0_pkts_completed_id, to_sender_1_pkts_completed_id};

constexpr uint32_t NUM_RECEIVER_CHANNELS = 1;
constexpr uint32_t NUM_USED_RECEIVER_CHANNELS = 1;
constexpr uint32_t NUM_SENDER_CHANNELS = 1;

constexpr uint8_t NUM_TRANSACTION_IDS = 4;

constexpr std::array<size_t, NUM_SENDER_CHANNELS> SENDER_NUM_BUFFERS_ARRAY = {2};

constexpr std::array<size_t, NUM_RECEIVER_CHANNELS> RECEIVER_NUM_BUFFERS_ARRAY = {2};

constexpr std::array<size_t, NUM_RECEIVER_CHANNELS> REMOTE_RECEIVER_NUM_BUFFERS_ARRAY = RECEIVER_NUM_BUFFERS_ARRAY;

static_assert(NUM_SENDER_CHANNELS == 1);

// Alignment for read and write to work on all core types
constexpr uint32_t GLOBAL_ALIGNMENT = 64;
// Additional space reserved for data alignment
constexpr uint32_t ALIGNMENT_BUFFER_SIZE = GLOBAL_ALIGNMENT;
constexpr uint32_t CHANNEL_BUFFER_SIZE = 2048 + ALIGNMENT_BUFFER_SIZE + sizeof(lite_fabric::FabricLiteHeader);

constexpr size_t RECEIVER_CHANNEL_BASE_ID = NUM_SENDER_CHANNELS;
constexpr size_t SENDER_CHANNEL_BASE_ID = 0;

// Not using multi txq / 2 erisc
constexpr uint32_t DEFAULT_ETH_TXQ = 0;
constexpr bool multi_txq_enabled = false;
constexpr uint32_t sender_txq_id = DEFAULT_ETH_TXQ;
constexpr uint32_t receiver_txq_id = DEFAULT_ETH_TXQ;
constexpr bool enable_first_level_ack = false;
constexpr std::array<size_t, NUM_RECEIVER_CHANNELS> local_receiver_completion_counter_ptrs = {0};
constexpr std::array<size_t, NUM_RECEIVER_CHANNELS> local_receiver_ack_counter_ptrs = {0};
constexpr std::array<size_t, NUM_RECEIVER_CHANNELS> to_sender_remote_completion_counter_addrs = {0};
constexpr std::array<size_t, NUM_RECEIVER_CHANNELS> to_sender_remote_ack_counter_addrs = {0};

#if defined(KERNEL_BUILD) || defined(FW_BUILD)
constexpr uint8_t local_chip_data_cmd_buf = BRISC_WR_CMD_BUF;
#endif

// Default NoC to use for Reads/Writes
constexpr uint8_t edm_to_local_chip_noc = 0;
constexpr uint8_t forward_and_local_write_noc_vc = 2;  // FabricEriscDatamoverConfig::DEFAULT_NOC_VC
constexpr uint8_t edm_to_downstream_noc = 0;

}  // namespace lite_fabric
