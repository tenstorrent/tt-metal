// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>

#include "perf_counters/eth_counters_types.h"

// Blackhole ethernet core counters. Include through perf_counters/eth_counters_inventory.h.
// Three TX and three RX queues, 0x1000 apart: TX queues from 0xFFB90000, RX queues from 0xFFB94000. One TXQ_WORD_CNT
// tick is 96 bytes on the wire (a keepalive is one tick) and one RXQ_WORD_CNT tick is 96 bytes by the register
// description. RXQ_BYTE_CNT counts payload bytes written to L1.
// The L1 values are the live status the base firmware refreshes about once a second from the MAC/PCS block
// (boot_results_t::eth_live_status at 0x7CE00), low word first; the queue counts there are 64-bit copies of the queue
// registers. Ethernet firmware 1.5.0 and newer writes them.

namespace llk::perf::eth::blackhole
{

inline constexpr Layout layout = {0xFFB90000u, 0xFFB94000u, 0x1000u, 3u, 3u, 96u, 96u};

inline constexpr std::array<QueueEntry, 5> tx_queue_counters = {{
    {EthCounter::TXQ_TRANSFER_CNT, 0x30},
    {EthCounter::TXQ_PKT_START_CNT, 0x34},
    {EthCounter::TXQ_PKT_END_CNT, 0x3C},
    {EthCounter::TXQ_WORD_CNT, 0x40},
    {EthCounter::TXQ_RESEND_CNT, 0x68},
}};

inline constexpr std::array<QueueEntry, 5> rx_queue_counters = {{
    {EthCounter::RXQ_WORD_CNT, 0x14},
    {EthCounter::RXQ_PKT_START_CNT, 0x24},
    {EthCounter::RXQ_PKT_END_CNT, 0x28},
    {EthCounter::RXQ_PKT_DROP_CNT, 0x4C},
    {EthCounter::RXQ_BYTE_CNT, 0x04},
}};

inline constexpr std::array<L1Entry, 25> l1_counters = {{
    {EthCounter::FW_RETRAIN_COUNT, NO_QUEUE, 1, WordOrder::LOW_FIRST, 0x7CE00},
    {EthCounter::FW_LINK_FLAP_COUNT, NO_QUEUE, 1, WordOrder::LOW_FIRST, 0x7CE08},
    {EthCounter::FW_LINK_POLL_ALIVE_COUNT, NO_QUEUE, 1, WordOrder::LOW_FIRST, 0x7CE0C},
    {EthCounter::FW_FRAMES_TXD, NO_QUEUE, 2, WordOrder::LOW_FIRST, 0x7CE20},
    {EthCounter::FW_FRAMES_TXD_OK, NO_QUEUE, 2, WordOrder::LOW_FIRST, 0x7CE28},
    {EthCounter::FW_FRAMES_TXD_BADFCS, NO_QUEUE, 2, WordOrder::LOW_FIRST, 0x7CE30},
    {EthCounter::FW_BYTES_TXD, NO_QUEUE, 2, WordOrder::LOW_FIRST, 0x7CE38},
    {EthCounter::FW_BYTES_TXD_OK, NO_QUEUE, 2, WordOrder::LOW_FIRST, 0x7CE40},
    {EthCounter::FW_BYTES_TXD_BADFCS, NO_QUEUE, 2, WordOrder::LOW_FIRST, 0x7CE48},
    {EthCounter::FW_FRAMES_RXD, NO_QUEUE, 2, WordOrder::LOW_FIRST, 0x7CE50},
    {EthCounter::FW_FRAMES_RXD_OK, NO_QUEUE, 2, WordOrder::LOW_FIRST, 0x7CE58},
    {EthCounter::FW_FRAMES_RXD_BADFCS, NO_QUEUE, 2, WordOrder::LOW_FIRST, 0x7CE60},
    {EthCounter::FW_FRAMES_RXD_DROPPED, NO_QUEUE, 2, WordOrder::LOW_FIRST, 0x7CE68},
    {EthCounter::FW_BYTES_RXD, NO_QUEUE, 2, WordOrder::LOW_FIRST, 0x7CE70},
    {EthCounter::FW_BYTES_RXD_OK, NO_QUEUE, 2, WordOrder::LOW_FIRST, 0x7CE78},
    {EthCounter::FW_BYTES_RXD_BADFCS, NO_QUEUE, 2, WordOrder::LOW_FIRST, 0x7CE80},
    {EthCounter::FW_BYTES_RXD_DROPPED, NO_QUEUE, 2, WordOrder::LOW_FIRST, 0x7CE88},
    {EthCounter::FW_CORR_CW, NO_QUEUE, 2, WordOrder::LOW_FIRST, 0x7CE90},
    {EthCounter::FW_UNCORR_CW, NO_QUEUE, 2, WordOrder::LOW_FIRST, 0x7CE98},
    {EthCounter::FW_TXQ_RESEND_CNT, 0, 2, WordOrder::LOW_FIRST, 0x7CEA0},
    {EthCounter::FW_TXQ_RESEND_CNT, 1, 2, WordOrder::LOW_FIRST, 0x7CEA8},
    {EthCounter::FW_TXQ_RESEND_CNT, 2, 2, WordOrder::LOW_FIRST, 0x7CEB0},
    {EthCounter::FW_RXQ_PKT_DROP, 0, 2, WordOrder::LOW_FIRST, 0x7CEB8},
    {EthCounter::FW_RXQ_PKT_DROP, 1, 2, WordOrder::LOW_FIRST, 0x7CEC0},
    {EthCounter::FW_RXQ_PKT_DROP, 2, 2, WordOrder::LOW_FIRST, 0x7CEC8},
}};

} // namespace llk::perf::eth::blackhole
