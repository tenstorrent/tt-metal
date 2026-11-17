// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

namespace llk::perf::eth
{

// Counters of an ethernet tile. The host reads them over PCIe or the NoC: queue registers, and words the base
// firmware keeps in L1. Names only: append new counters at the end, never reorder or remove.
enum class EthCounter : std::uint16_t
{
    UNDEF = 0,
    // TX queue registers, free running 32-bit, one set per queue.
    TXQ_TRANSFER_CNT,
    TXQ_PKT_START_CNT,
    TXQ_PKT_END_CNT,
    TXQ_WORD_CNT,
    TXQ_RESEND_CNT,
    // RX queue registers, free running 32-bit, one set per queue.
    RXQ_WORD_CNT,
    RXQ_PKT_START_CNT,
    RXQ_PKT_END_CNT,
    RXQ_PKT_DROP_CNT,
    RXQ_BYTE_CNT,
    // Words the base firmware refreshes in L1 from the MAC/PCS block, 64-bit unless noted.
    FW_RETRAIN_COUNT,
    FW_LINK_FLAP_COUNT,
    FW_LINK_POLL_ALIVE_COUNT,
    FW_FRAMES_TXD,
    FW_FRAMES_TXD_OK,
    FW_FRAMES_TXD_BADFCS,
    FW_BYTES_TXD,
    FW_BYTES_TXD_OK,
    FW_BYTES_TXD_BADFCS,
    FW_FRAMES_RXD,
    FW_FRAMES_RXD_OK,
    FW_FRAMES_RXD_BADFCS,
    FW_FRAMES_RXD_DROPPED,
    FW_BYTES_RXD,
    FW_BYTES_RXD_OK,
    FW_BYTES_RXD_BADFCS,
    FW_BYTES_RXD_DROPPED,
    FW_CORR_CW,
    FW_UNCORR_CW,
    FW_TXQ_RESEND_CNT,
    FW_RXQ_PKT_DROP,
    FW_CRC_ERR,
};

enum class Arch : std::uint8_t
{
    BLACKHOLE = 0,
    WORMHOLE,
    QUASAR,
};

inline constexpr std::size_t NUM_ARCHES = 3;

// {counter name, register offset inside the register space of one queue}; every counter is a free running 32-bit register.
using QueueEntry = std::pair<EthCounter, std::uint32_t>;

// Which word of a 64-bit L1 value comes first in memory.
enum class WordOrder : std::uint8_t
{
    LOW_FIRST,
    HIGH_FIRST,
};

inline constexpr std::uint8_t NO_QUEUE = 0xFF;

// A counter the base firmware keeps in the L1 of the ethernet core: byte address, width in 32-bit words (1 or 2).
struct L1Entry
{
    EthCounter counter;
    std::uint8_t queue; // NO_QUEUE when the value does not belong to a queue
    std::uint8_t words;
    WordOrder order;
    std::uint32_t address;
};

struct QueueTable
{
    const QueueEntry* data;
    std::size_t size;
};

struct L1Table
{
    const L1Entry* data;
    std::size_t size;
};

// What the queue tables are relative to. A zero base or word size means the value is not known.
struct Layout
{
    std::uint32_t txq_base;     // address of the register space of TX queue 0
    std::uint32_t rxq_base;     // address of the register space of RX queue 0
    std::uint32_t queue_stride; // address distance between two queues of one kind
    std::uint32_t num_txq;
    std::uint32_t num_rxq;
    std::uint32_t txq_word_bytes; // bytes behind one TXQ_WORD_CNT tick
    std::uint32_t rxq_word_bytes; // bytes behind one RXQ_WORD_CNT tick
};

} // namespace llk::perf::eth
