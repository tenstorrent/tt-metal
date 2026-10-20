// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Layout and result slots shared by the Ethernet TX queue probe's host test and kernel. Offsets are from the start
// of the active Ethernet core's unreserved L1; both chips use the same layout.

#pragma once

#include <cstdint>

namespace eth_txq_probe {

constexpr uint32_t kHandshakeOffset = 0x0;
constexpr uint32_t kResultsOffset = 0x100;
constexpr uint32_t kReverseLandOffset = 0x400;    // sender: latest count from the receiver's reverse messages
constexpr uint32_t kQ1CounterLandOffset = 0x410;  // receiver: count of packets sent on TXQ1 (its pointer update)
constexpr uint32_t kDoneLandOffset = 0x420;       // sender: receiver's done flag
constexpr uint32_t kDoneSrcOffset = 0x430;        // receiver: source word of the done flag
constexpr uint32_t kSmallRingOffset = 0x800;      // 16 B source words for counter sends, reused round-robin
constexpr uint32_t kSmallRingEntries = 64;
constexpr uint32_t kSlotsOffset = 0x1000;  // sender: source slots; receiver: landing slots (same addresses)
constexpr uint32_t kSlotBytes = 4352;      // fits a 4096 B payload plus the largest 2D header
constexpr uint32_t kNoQueue = 0xFF;

// Compile-time args.
enum CtArg : uint32_t {
    CT_NUM_DATA_QUEUES,  // 1: every packet on TXQ0; 2: even packets on TXQ0, odd on TXQ1
    CT_REVERSE_QUEUE,    // receiver's queue for reverse messages, or kNoQueue
    CT_REVERSE_EVERY,    // one reverse message per this many packets (0: none)
    CT_GATED,            // sender treats the reverse counts as credits for its slots
    CT_COUNT,
};

// Runtime args.
enum RtArg : uint32_t {
    RT_BASE,
    RT_IS_SENDER,
    RT_PAYLOAD,
    RT_NUM_PACKETS,
    RT_WARMUP_PACKETS,
    RT_NUM_SLOTS,           // power of two
    RT_SEQ_UPDATE_TIMEOUT,  // receiver: keepalive period for its TXQ0 during the run (0: leave the firmware value)
    RT_COUNT,
};

constexpr uint32_t kNumQueues = 2;    // the probe touches TXQ0 and TXQ1 only
constexpr uint32_t kTxqCounters = 5;  // TRANSFER_CNT, PKT_START_CNT, PKT_END_CNT, WORD_CNT, RESEND_CNT
constexpr uint32_t kTxqConfig = 5;    // CTRL, MAX_PKT_SIZE, REMOTE_SEQ_TIMEOUT, LOCAL_SEQ_UPDATE_TIMEOUT, ACCEPT_AHEAD

// RX queue registers (tt-isa-documentation BlackholeA0/EthernetTile/EthernetTxRx.md); Metal has no defines for them.
constexpr uint32_t kRxq0RegsStart = 0xFFB94000;
constexpr uint32_t kRxqRegsSize = 0x1000;
constexpr uint32_t kRxqPacketDropCnt = 0x4C;  // packets discarded by the RX queue (e.g. sequence number mismatch)

// Result words (uint32_t) at kResultsOffset. Times are the core's mcycle (1.35 GHz, the Ethernet clock).
enum Result : uint32_t {
    R_STATUS,        // kRunning, kDone or an error code
    R_ERR_INDEX,     // packet index at the error
    R_T_FIRST,       // sender: before packet 0; receiver: packet 0 seen
    R_T_WARM,        // packet RT_WARMUP_PACKETS issued / seen
    R_T_LAST,        // sender: after the last issue; receiver: last packet seen
    R_MERGE_ERRORS,  // receiver: landing slot did not hold this packet (or a later one for the same slot)
    R_TXQ_BEFORE,    // [kNumQueues][kTxqCounters] after the handshake
    R_TXQ_AFTER = R_TXQ_BEFORE + kNumQueues * kTxqCounters,
    R_TXQ_CONFIG = R_TXQ_AFTER + kNumQueues * kTxqCounters,           // [kNumQueues][kTxqConfig] at the end
    R_SEQ_TIMEOUT_FIRMWARE = R_TXQ_CONFIG + kNumQueues * kTxqConfig,  // TXQ0 keepalive period found at start
    R_SEQ_TIMEOUT_USED,  // TXQ0 keepalive period read back at the end of the run, before restoring
    R_RXQ_DROPS_BEFORE,  // [kNumQueues] RX queue drop counters after the handshake
    R_RXQ_DROPS_AFTER = R_RXQ_DROPS_BEFORE + kNumQueues,
    R_COUNT = R_RXQ_DROPS_AFTER + kNumQueues,
};

enum Status : uint32_t {
    kRunning = 0,
    kDone = 1,
    kErrSenderTxq = 0x10,
    kErrSenderCredits,
    kErrSenderDone,
    kErrReceiverQ0,
    kErrReceiverQ1,
    kErrReceiverTxq,
};

}  // namespace eth_txq_probe
