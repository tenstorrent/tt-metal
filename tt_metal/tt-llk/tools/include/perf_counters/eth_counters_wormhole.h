// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>

#include "perf_counters/eth_counters_types.h"

// Wormhole ethernet core counters. Include through perf_counters/eth_counters_inventory.h.
// Two TX and two RX queues, 0x1000 apart: TX queues from 0xFFB90000, RX queues from 0xFFB92000. One queue word is
// 16 bytes by the controller description (the MAC data path is 128 bits wide); a host read on silicon has not been tried.
// The base firmware keeps four values in L1, the 64-bit ones as high word, then low word; they exist on ethernet
// firmware 6.14.0 and newer.

namespace llk::perf::eth::wormhole
{

inline constexpr Layout layout = {0xFFB90000u, 0xFFB92000u, 0x1000u, 2u, 2u, 16u, 16u};

inline constexpr std::array<QueueEntry, 5> tx_queue_counters = {{
    {EthCounter::TXQ_TRANSFER_CNT, 0x30},
    {EthCounter::TXQ_PKT_START_CNT, 0x34},
    {EthCounter::TXQ_PKT_END_CNT, 0x3C},
    {EthCounter::TXQ_WORD_CNT, 0x40},
    {EthCounter::TXQ_RESEND_CNT, 0x68},
}};

inline constexpr std::array<QueueEntry, 4> rx_queue_counters = {{
    {EthCounter::RXQ_WORD_CNT, 0x14},
    {EthCounter::RXQ_PKT_START_CNT, 0x24},
    {EthCounter::RXQ_PKT_END_CNT, 0x28},
    {EthCounter::RXQ_PKT_DROP_CNT, 0x4C},
}};

inline constexpr std::array<L1Entry, 4> l1_counters = {{
    {EthCounter::FW_RETRAIN_COUNT, NO_QUEUE, 1, WordOrder::LOW_FIRST, 0x1EDC},
    {EthCounter::FW_CRC_ERR, NO_QUEUE, 1, WordOrder::LOW_FIRST, 0x1F7C},
    {EthCounter::FW_CORR_CW, NO_QUEUE, 2, WordOrder::HIGH_FIRST, 0x1F90},
    {EthCounter::FW_UNCORR_CW, NO_QUEUE, 2, WordOrder::HIGH_FIRST, 0x1F98},
}};

} // namespace llk::perf::eth::wormhole
