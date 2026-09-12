// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Device-side access to the Blackhole Ethernet tile's IEEE-1588 hardware: the eth_ctrl PTP timer, the per-queue TX
// timestamp command, the Rianta RSm410 MAC's two-step TX timestamp FIFO and the RX classifier's timestamp FIFO; the
// tile's clocks are eth_ptp_clock.hpp. A StampSession owns the two timestamp FIFOs, one TX header row, one TCAM row
// and one flow label for as long as it is open, and end() restores everything begin() changed. Register offsets
// follow tt-isa-documentation (EthernetRxClassifier.md, the eth_ctrl and TXQ maps) and the Rianta RSm410 register
// guide; the TXQ and RISC bases are tt_eth_ss_regs.h's.
//
// The PTP timer counts refclk (50 MHz on Blackhole) and is independent of AICLK, so it is immune to DVFS. Nothing in
// the shipped firmware enables its main counter; ptp_timer_start() does, and leaves it running after the session.
// Verified on silicon (2026-09-10), where the documents differ or are silent:
//   - TXQ TIMESTAMP.ts_cmd is sticky: every frame the queue sends is stamped until it is written back to 0.
//   - MAC FIFO word order is tag LO, tag HI, txTS LO, txTS HI (the Rianta text has the pairs reversed).
//   - The RX FIFO head is readable before the pop, and the pop bit is a 1-then-0 strobe.
//   - The RX stamp is pushed at start-of-frame, before the RX queue's L1 write of the frame is visible to the ERISC,
//     so a poller that drains the FIFO while waiting for a frame consumes that frame's stamp.
//   - The link is never idle: each TX queue sends a sequence-number keepalive after LOCAL_SEQ_UPDATE_TIMEOUT idle
//     cycles (0x1f40 at boot; measured 8002 AICLK cycles apart), and every keepalive gets an RX stamp under the
//     no-match label. A keepalive waiting at the MAC when the stamp request is armed is stamped under the frames'
//     tag. The timeout cannot be held off to avoid that: with it at its maximum the peer's queue retransmits
//     (duplicate frames, so duplicate ingress stamps, in every round after the first), the updates being its acks.
//   - The queue's counters see every hand-off, keepalives included, and in a fixed order: PKT_START_CNT moves 86
//     cycles after the command (the L1 fetch), PKT_END_CNT 22 cycles later, the MAC FIFO entry 65 cycles after that
//     (35 after a keepalive's count), whatever the frame's size. WORD_CNT counts 96-byte units on the wire: one for a
//     keepalive or a frame of up to 64 bytes of payload, two from 80 bytes of payload to 128. Keepalives carry no
//     sequence number (the RX queues' sequence registers stand still through them) and, per the ISA documentation, are
//     generated only when the queue has no other packet to carry the numbers, so none is sent between or behind queued
//     frames.
//   - Reading the LO half of CFR or PTP64NS captures its HI half (tt_ptp_timer.sv: the HI register loads on the LO
//     read strobe). The wall clock has two HI addresses: WALL_CLOCK_1 is live, WALL_CLOCK_1_AT is the value at the
//     last LO read; only the latter pairs with LO.
//
// Measured on p150 links over 0.5 m passive DAC: one way inside the stamps 33-35 ns, a receiver's turnaround
// ~350 ns; 256 exchanges averaged per side dither the 20 ns tick to ~0.4 ns per round.
//
// One end of a link:
//   using Session = StampSession<kTxq, kHeaderRow, kTcamRow, kLabel>;   // the configuration is compile-time: no loads
//   static Session sess;                                    // on the per-frame path; the ERISC runs no dynamic init
//   sess.begin();                                           // false if the timer never acknowledged its rate
//   stamps_arm(sess, tag);                                  // every frame the queue sends from here is stamped
//   internal_::eth_send_packet(Session::kTxq, ...);         // ... per frame
//   tx_stamps_drain(tag_lo, sink); rx_stamps_drain(sess, sink);  // egress stamps; the peer's frames' ingress stamps
//   stamps_disarm(sess);                                    // once the last frame's egress stamp has been drained
//   sess.end();
// The two FIFOs are the tile's: a second session on the same tile would consume the first one's stamps.

#pragma once

#include <cstdint>

#include "internal/ethernet/dataflow_api.h"
#include "internal/ethernet/eth_ptp_clock.hpp"

namespace tt::tt_metal::eth_ptp {

// Per TX queue
constexpr uint32_t kTxqRegsBase = ETH_TXQ0_REGS_START;
constexpr uint32_t kTxqStride = ETH_TXQ_REGS_SIZE;
constexpr uint32_t kNumTxq = NUM_ETH_QUEUES;
constexpr uint32_t kTxqTimestampOff = 0x90;      // [2:0] ts_cmd, [21:16] ts_offset (2-byte units)
constexpr uint32_t kTxqRxTimestampLoOff = 0x94;  // drives TX_RX_TS[31:0], echoed in the MAC FIFO as the tag
constexpr uint32_t kTxqRxTimestampHiOff = 0x98;  // drives TX_RX_TS[63:32]
constexpr uint32_t kTxqLocalSeqUpdateTimeoutOff = ETH_TXQ_LOCAL_SEQ_UPDATE_TIMEOUT;
constexpr uint32_t kTxqPktCfgSelSwOff =
    0x80;  // header row per software-initiated frame type: [3:0] raw, [7:4] reg write, [11:8] packet
constexpr uint32_t txq_reg(uint32_t q, uint32_t off) { return kTxqRegsBase + q * kTxqStride + off; }

enum TsCmd : uint32_t {
    TS_CMD_NOP = 0,
    TS_CMD_ONE_STEP_CORRECTION = 1,
    TS_CMD_ONE_STEP_ORIGIN = 2,
    TS_CMD_TWO_STEP_FIFO = 3,
};

// TX header-template rows (eth_ctrl TXPKT_CFG). Firmware programs rows 0..2 for queues 0..2 (DA broadcast /
// multicast / 02:00:00:00:00:00 unicast, which MAC_RX_ADDR_ROUTING maps to RXQ0/1/2) and leaves 3..9 free.
constexpr uint32_t kTxPktCfgBase = 0xFFB98200;
constexpr uint32_t kTxPktCfgStride = 0x80;
constexpr uint32_t kTxPktCfgInsertCtlOff = 0x00;
constexpr uint32_t kTxPktCfgCustomHdrOff = 0x04;
constexpr uint32_t kTxPktCfgMacSaLoOff = 0x10;
constexpr uint32_t kTxPktCfgMacSaHiOff = 0x14;
constexpr uint32_t kTxPktCfgMacDaLoOff = 0x18;
constexpr uint32_t kTxPktCfgMacDaHiOff = 0x1C;
constexpr uint32_t kTxPktCfgEthertypeOff = 0x20;
constexpr uint32_t kTxPktCfgVlan1Off = 0x24;
constexpr uint32_t kTxPktCfgVlan2Off = 0x28;
constexpr uint32_t tx_pkt_cfg_reg(uint32_t row, uint32_t off) { return kTxPktCfgBase + row * kTxPktCfgStride + off; }

// Rianta RSm410 MAC (RSM410_A_REG_MAP_BASE_ADDR 0xFFBA0000)
constexpr uint32_t kMacTxCfg = 0xFFBA2200;        // [1] tx_enb [4] tx_ts_fifo_enb [11] tx_origin_timestamp_mode
constexpr uint32_t kMacTxCfgTsFifoEnb = 1u << 4;  // 1 = FIFO entry for every packet, 0 = only TS_CMD == 3
constexpr uint32_t kMacTxDelay = 0xFFBA2218;      // [15:0] added to txTimestamp
constexpr uint32_t kMacTxInt = 0xFFBA2288;        // W1C
constexpr uint32_t kMacTxIntRaw = 0xFFBA2290;     // [2] ts_fifo_full [3] ts_fifo_not_empty [4] err_ts_offset
constexpr uint32_t kMacTxIntTsFifoNotEmpty = 1u << 3;
constexpr uint32_t kMacTsFifoFullThresh = 0xFFBA2300;
constexpr uint32_t kMacTsFifo0 = 0xFFBA2E00;  // reading pops; empty reads 0xFFFFFFFF
constexpr uint32_t kMacTsFifo1 = 0xFFBA2E04;
constexpr uint32_t kMacTsFifo2 = 0xFFBA2E08;
constexpr uint32_t kMacTsFifo3 = 0xFFBA2E0C;

// RX classifier timestamp handling ("TH") file, and the flow-table bits that feed it
constexpr uint32_t kRxThTsLow = 0xFFB9D800;
constexpr uint32_t kRxThTsHigh = 0xFFB9D804;
constexpr uint32_t kRxThTsLabel = 0xFFB9D808;  // [5:0] flow label, [31] valid_ts
constexpr uint32_t kRxThStatus =
    0xFFB9D810;  // [3:0] entries [16] full [17] becoming_full [18] empty [30] flush [31] pop
constexpr uint32_t kRxThStatusEntriesMask = 0xF;
constexpr uint32_t kRxThStatusEmpty = 1u << 18;
constexpr uint32_t kRxThStatusFlush = 1u << 30;
constexpr uint32_t kRxThStatusPop = 1u << 31;
constexpr uint32_t kRxThLabelMask = 0x1F;
constexpr uint32_t kRxThLabelValid = 1u << 31;
constexpr uint32_t kRxFdOverrideDecision = 0xFFB9D000;  // [1:0]: 0 accept all (ignore flow table), 2 use flow table
constexpr uint32_t kRxFlNoMatchActions = 0xFFB9CD04;    // [1:0] queue [2] drop [3] rm_hdr [4] keep_timestamp
constexpr uint32_t kRxFlKeepTimestamp = 1u << 4;

// RX classifier flow lookup (64-row TCAM) and flow table, per tt-isa-documentation EthernetRxClassifier.md.
constexpr uint32_t kRxFlTcamRowMappingBase = 0xFFB9CC00;  // + 4 * row: [2:0] priority, [21:16] flow-table row
constexpr uint32_t kRxFlTcamRowUpdate = 0xFFB9CD40;       // [5:0] row, [8] enable, [16] write, [31] go
constexpr uint32_t kRxFlTcamTupleTypeWrite = 0xFFB9CD80;  // 0 = not IP
constexpr uint32_t kRxFlTcamSaWrite0 = 0xFFB9CD90;        // 4 words
constexpr uint32_t kRxFlTcamDaWrite0 = 0xFFB9CDA0;        // 4 words; not-IP rows: first six bytes = MAC DA
constexpr uint32_t kRxFlTcamNonIpAddrFlagsWrite = 0xFFB9CDB0;
constexpr uint32_t kRxFlTcamEthertypeWrite = 0xFFB9CDC0;
constexpr uint32_t kRxFlTcamPriorityWrite = 0xFFB9CDC4;
constexpr uint32_t kRxFlTcamUpdate = 0xFFB9CDF0;  // [5:0] row [8] mask [9] write [10] not IP [19] DA [20] SA [21] kind
                                                  // [22] ethertype [23] l2 pri [31] go
constexpr uint32_t kRxFlFtableLabels = 0xFFB9CE80;   // [4:0] label, copied into every TH FIFO entry of the flow
constexpr uint32_t kRxFlFtableActions = 0xFFB9CE84;  // NO_MATCH_ACTIONS layout
constexpr uint32_t kRxFlFtableVlan = 0xFFB9CE88;
constexpr uint32_t kRxFlFtableSwMetadata = 0xFFB9CE8C;
constexpr uint32_t kRxFlFtableUpdate = 0xFFB9CEA0;  // [5:0] row, [8] write, [31] go

// Identity of a stamped frame: a locally administered unicast DA whose middle four bytes are 0xA5. The RX rule
// compares only those four bytes, so it holds whichever byte order the TCAM stores addresses in, and no firmware
// frame (broadcast, 01:00:.. multicast, 02:00:.. unicast) can match it.
constexpr uint64_t kStampFrameDa = 0x02A5'A5A5'A5A5ull;

constexpr uint32_t kTimerLeadTicks = 5000;    // the scheduled rate update lands this far ahead of the CFR: 100 us
constexpr uint32_t kTimerAckSpins = 200'000;  // polls of the update status before ptp_timer_start gives up
constexpr uint32_t kTxStampSpins = 4096;      // drains of the MAC FIFO for a burst's last egress stamp before it is given up

// Register-level operations. Each one changes tile state that outlives the kernel; StampSession is the pairing of
// changes and their restores, and the API below it is what a kernel is meant to call.
namespace raw {

using eth_ptp::rd;
using eth_ptp::wr;

// The tag the MAC files queue q's egress stamps under, sampled when a frame is handed to the MAC.
inline __attribute__((always_inline)) void txq_set_tag(uint32_t q, uint64_t tag) {
    wr(txq_reg(q, kTxqRxTimestampLoOff), static_cast<uint32_t>(tag));
    wr(txq_reg(q, kTxqRxTimestampHiOff), static_cast<uint32_t>(tag >> 32));
}
// Arms the MAC to push {tag, egress timestamp} into its FIFO for every frame queue q sends until the command is
// cleared.
inline __attribute__((always_inline)) void txq_request_two_step(uint32_t q, uint64_t tag) {
    txq_set_tag(q, tag);
    wr(txq_reg(q, kTxqTimestampOff), TS_CMD_TWO_STEP_FIFO);
}
inline __attribute__((always_inline)) void txq_clear_timestamp_cmd(uint32_t q) {
    wr(txq_reg(q, kTxqTimestampOff), TS_CMD_NOP);
}
inline __attribute__((always_inline)) uint32_t txq_pkt_start_cnt(uint32_t q) {
    return rd(txq_reg(q, ETH_TXQ_PKT_START_CNT));
}
inline __attribute__((always_inline)) uint32_t txq_word_cnt(uint32_t q) { return rd(txq_reg(q, ETH_TXQ_WORD_CNT)); }

struct MacTxStamp {
    uint64_t tag;  // the TX_RX_TS value the queue drove when the frame entered the MAC
    uint64_t tx_ts;
};
inline __attribute__((always_inline)) bool mac_tx_fifo_not_empty() {
    return (rd(kMacTxIntRaw) & kMacTxIntTsFifoNotEmpty) != 0;
}
// Pops one entry; word 0 must be read first, it is what advances the FIFO.
inline __attribute__((always_inline)) bool mac_tx_fifo_pop(MacTxStamp& out) {
    const uint32_t w0 = rd(kMacTsFifo0);
    if (w0 == 0xFFFFFFFFu) {
        return false;
    }
    const uint32_t w1 = rd(kMacTsFifo1);
    const uint32_t w2 = rd(kMacTsFifo2);
    const uint32_t w3 = rd(kMacTsFifo3);
    out.tag = (static_cast<uint64_t>(w1) << 32) | w0;
    out.tx_ts = (static_cast<uint64_t>(w3) << 32) | w2;
    return true;
}
// Pops one entry reading its tag's low word and the stamp only; the tag's high word is the caller's own.
inline __attribute__((always_inline)) bool mac_tx_fifo_pop_ts(uint32_t& tag_lo, uint64_t& tx_ts) {
    const uint32_t w0 = rd(kMacTsFifo0);
    if (w0 == 0xFFFFFFFFu) {
        return false;
    }
    const uint32_t w2 = rd(kMacTsFifo2);
    const uint32_t w3 = rd(kMacTsFifo3);
    tag_lo = w0;
    tx_ts = (static_cast<uint64_t>(w3) << 32) | w2;
    return true;
}
inline void mac_tx_fifo_drain() {
    MacTxStamp junk;
    while (mac_tx_fifo_pop(junk)) {
    }
}

struct RxStamp {
    uint64_t rx_ts;
    uint32_t label;
    bool valid;
};
inline __attribute__((always_inline)) uint32_t rx_th_entries() { return rd(kRxThStatus) & kRxThStatusEntriesMask; }
// Reads the head entry, then pops it.
inline __attribute__((always_inline)) bool rx_th_pop(RxStamp& out) {
    if (rd(kRxThStatus) & kRxThStatusEmpty) {
        return false;
    }
    const uint32_t lo = rd(kRxThTsLow);
    const uint32_t hi = rd(kRxThTsHigh);
    const uint32_t label = rd(kRxThTsLabel);
    wr(kRxThStatus, kRxThStatusPop);
    wr(kRxThStatus, 0);
    out.rx_ts = (static_cast<uint64_t>(hi) << 32) | lo;
    out.label = label & kRxThLabelMask;
    out.valid = (label & kRxThLabelValid) != 0;
    return true;
}
inline void rx_th_flush() { wr(kRxThStatus, kRxThStatusFlush); }

// Holds the TX queues' idle sequence-number keepalives off. Returns the previous timeouts for restore.
inline void txq_keepalives_off(uint32_t prev[kNumTxq]) {
    for (uint32_t q = 0; q < kNumTxq; q++) {
        prev[q] = rd(txq_reg(q, kTxqLocalSeqUpdateTimeoutOff));
        wr(txq_reg(q, kTxqLocalSeqUpdateTimeoutOff), 0xFFFFFFFFu);
    }
}
inline void txq_keepalives_restore(const uint32_t prev[kNumTxq]) {
    for (uint32_t q = 0; q < kNumTxq; q++) {
        wr(txq_reg(q, kTxqLocalSeqUpdateTimeoutOff), prev[q]);
    }
}

// Every received frame records its RX timestamp via the no-match flow row's keep-timestamp action. The flow
// decision override is left alone unless asked: at its default of 0 the classifier still applies the table's
// keep-timestamp action and only ignores its drop decisions, so no other traffic is affected.
struct RxThPrev {
    uint32_t no_match_actions;
    uint32_t override_decision;
};
inline RxThPrev rx_timestamps_enable_all(bool set_override = false) {
    RxThPrev prev{rd(kRxFlNoMatchActions), rd(kRxFdOverrideDecision)};
    rx_th_flush();
    wr(kRxFlNoMatchActions, prev.no_match_actions | kRxFlKeepTimestamp);
    if (set_override) {
        wr(kRxFdOverrideDecision, 2);
    }
    return prev;
}
inline void rx_timestamps_restore(const RxThPrev& prev) {
    wr(kRxFdOverrideDecision, prev.override_decision);
    wr(kRxFlNoMatchActions, prev.no_match_actions);
    rx_th_flush();
}

struct TxHeaderPrev {
    uint32_t sel_sw;
    uint32_t da_lo;
    uint32_t da_hi;
};
// Points queue q's software-initiated frames at header row `row`, a copy of the queue's boot row with DA `da`.
// Hardware-generated frames (sequence-number keepalives) keep the boot row, so only frames the kernel sends carry
// the identity.
inline TxHeaderPrev tx_header_row_install(uint32_t q, uint32_t row, uint64_t da) {
    const uint32_t sel = txq_reg(q, kTxqPktCfgSelSwOff);
    TxHeaderPrev prev{
        rd(sel), rd(tx_pkt_cfg_reg(row, kTxPktCfgMacDaLoOff)), rd(tx_pkt_cfg_reg(row, kTxPktCfgMacDaHiOff))};
    const uint32_t copied[] = {
        kTxPktCfgInsertCtlOff,
        kTxPktCfgCustomHdrOff,
        kTxPktCfgMacSaLoOff,
        kTxPktCfgMacSaHiOff,
        kTxPktCfgEthertypeOff,
        kTxPktCfgVlan1Off,
        kTxPktCfgVlan2Off};
    for (uint32_t off : copied) {
        wr(tx_pkt_cfg_reg(row, off), rd(tx_pkt_cfg_reg(q, off)));
    }
    wr(tx_pkt_cfg_reg(row, kTxPktCfgMacDaLoOff), static_cast<uint32_t>(da));
    wr(tx_pkt_cfg_reg(row, kTxPktCfgMacDaHiOff), static_cast<uint32_t>(da >> 32));
    wr(sel, row | (row << 4) | (row << 8));
    return prev;
}
inline void tx_header_row_restore(uint32_t q, uint32_t row, const TxHeaderPrev& prev) {
    wr(txq_reg(q, kTxqPktCfgSelSwOff), prev.sel_sw);
    wr(tx_pkt_cfg_reg(row, kTxPktCfgMacDaLoOff), prev.da_lo);
    wr(tx_pkt_cfg_reg(row, kTxPktCfgMacDaHiOff), prev.da_hi);
}

inline void rx_tcam_write_pattern(
    uint32_t row,
    bool mask,
    uint32_t da_w0,
    uint32_t da_w1,
    uint32_t rest,
    uint32_t flags,
    uint32_t etype,
    uint32_t pri) {
    wr(kRxFlTcamTupleTypeWrite, 0);
    wr(kRxFlTcamEthertypeWrite, etype);
    wr(kRxFlTcamPriorityWrite, pri);
    for (uint32_t i = 0; i < 4; i++) {
        wr(kRxFlTcamSaWrite0 + 4 * i, rest);
    }
    wr(kRxFlTcamDaWrite0, da_w0);
    wr(kRxFlTcamDaWrite0 + 4, da_w1);
    wr(kRxFlTcamDaWrite0 + 8, rest);
    wr(kRxFlTcamDaWrite0 + 12, rest);
    wr(kRxFlTcamNonIpAddrFlagsWrite, flags);
    wr(kRxFlTcamUpdate,
       row | (mask ? 1u << 8 : 0u) | (1u << 9) | (1u << 10) | (1u << 19) | (1u << 20) | (1u << 21) | (1u << 22) |
           (1u << 23) | (1u << 31));
}
// One TCAM row matching kStampFrameDa's middle bytes, mapped to a flow-table row whose only action is to keep the
// RX timestamp under `label`. Nothing else about how frames are handled changes.
inline void rx_stamp_rule_install(uint32_t row, uint32_t label) {
    rx_tcam_write_pattern(row, false, 0xA5A5A500u, 0x000000A5u, 0u, 0u, 0u, 0u);
    rx_tcam_write_pattern(row, true, 0x000000FFu, 0xFFFFFF00u, 0xFFFFFFFFu, 0x000F000Fu, 0x000FFFFFu, 0x7u);
    wr(kRxFlTcamRowMappingBase + 4 * row, (row << 16) | 7u);
    wr(kRxFlFtableActions, kRxFlKeepTimestamp);
    wr(kRxFlFtableVlan, 0);
    wr(kRxFlFtableLabels, label & kRxThLabelMask);
    wr(kRxFlFtableSwMetadata, 0);
    wr(kRxFlFtableUpdate, row | (1u << 8) | (1u << 31));
    wr(kRxFlTcamRowUpdate, row | (1u << 8) | (1u << 16) | (1u << 31));
}
inline void rx_stamp_rule_remove(uint32_t row) {
    wr(kRxFlTcamRowUpdate, row | (1u << 16) | (1u << 31));
    wr(kRxFlFtableActions, 0);
    wr(kRxFlFtableVlan, 0);
    wr(kRxFlFtableLabels, 0);
    wr(kRxFlFtableSwMetadata, 0);
    wr(kRxFlFtableUpdate, row | (1u << 8) | (1u << 31));
}

}  // namespace raw

// Programs the per-tick increment through the timer's scheduled-update mechanism and enables the main counter.
// A timer already running at the requested increment is left alone. If the hardware never acknowledges the update
// within spin_limit polls the timer is left as it was and false is returned.
__attribute__((noinline, cold)) inline bool ptp_timer_start(uint32_t pti, uint32_t lead_ticks, uint32_t spin_limit) {
    if ((raw::rd(kPtpTimerCtrl) & 1u) && (raw::rd(kPtpPtiStat) & 0xFFFFFFu) == pti) {
        return true;
    }
    const uint64_t target = read_cfr() + lead_ticks;
    raw::wr(kPtpFutureCfrLo, static_cast<uint32_t>(target));
    raw::wr(kPtpFutureCfrHi, static_cast<uint32_t>(target >> 32));
    raw::wr(kPtpFuturePti, pti);
    raw::wr(kPtpUpdatePti, 1);
    bool acked = false;
    for (uint32_t i = 0; i < spin_limit && !acked; i++) {
        acked = (raw::rd(kPtpUpdateStat) & kUpdateStatPtiAck) != 0;
    }
    raw::wr(kPtpUpdatePti, 0);
    if (acked) {
        raw::wr(kPtpTimerCtrl, 1);
    }
    return acked;
}

// The once-per-session routines are cold: in a kernel built -O3 (the fabric router) they would otherwise unroll into
// a couple of KB of a 26 KB kernel budget shared with the router, for code that runs once.
// PTP64NS minus the CFR count in ns. The two counters advance on the same 50 MHz edge, so the difference is one
// constant, a multiple of the tick, for as long as the timer runs; but a single pair of reads puts the register latency
// between them (a tick or more) into it, and a stall between the two reads puts in several -- measured once at kernel
// start, that sat in every stamp of one side for the whole run as an 80 ns link bias that came and went between
// launches. Each pair is read in both orders so the skew cancels in the sum, the median over pairs discards a stalled
// one, and the result is rounded to the tick.
__attribute__((noinline, cold)) inline int64_t ptp_offset_ns() {
    constexpr int kPairs = 16;
    constexpr uint32_t kTick = kNsPerRefclkTick;
    static_assert(kTick == 20);
    int64_t sum2[kPairs];
    for (int i = 0; i < kPairs; i++) {
        const uint64_t c1 = read_cfr();
        const uint64_t n1 = read_ptp64ns();
        const uint64_t n2 = read_ptp64ns();
        const uint64_t c2 = read_cfr();
        const int64_t k1 = static_cast<int64_t>(n1 - ((c1 << 4) + (c1 << 2)));
        const int64_t k2 = static_cast<int64_t>(n2 - ((c2 << 4) + (c2 << 2)));
        int64_t v = k1 + k2;
        int j = i;
        for (; j > 0 && sum2[j - 1] > v; j--) {
            sum2[j] = sum2[j - 1];
        }
        sum2[j] = v;
    }
    const int64_t k = (sum2[kPairs / 2 - 1] + sum2[kPairs / 2]) >> 2;
    // Rounded to the tick in 32-bit arithmetic (2^32 = 16 mod 20): a 64-bit division here is the largest routine
    // in an otherwise small kernel.
    const bool neg = k < 0;
    const uint64_t a = neg ? static_cast<uint64_t>(-k) : static_cast<uint64_t>(k);
    const uint32_t r = ((static_cast<uint32_t>(a >> 32) % kTick) * 16u + static_cast<uint32_t>(a) % kTick) % kTick;
    const uint64_t rounded = a - r + (r >= kTick / 2 ? kTick : 0u);
    return neg ? -static_cast<int64_t>(rounded) : static_cast<int64_t>(rounded);
}

// One end's use of the tile's 1588 hardware: frames sent on queue Txq carry kStampFrameDa through header row
// HeaderRow, TCAM row TcamRow recognises them on the way in and files their ingress stamps under Label. The
// configuration is the type, so every per-frame register address and the label compare are immediates.
template <uint32_t Txq, uint32_t HeaderRow, uint32_t TcamRow, uint32_t Label>
struct StampSession {
    static constexpr uint32_t kTxq = Txq;
    static constexpr uint32_t kHeaderRow = HeaderRow;
    static constexpr uint32_t kTcamRow = TcamRow;
    static constexpr uint32_t kLabel = Label;
    static_assert(Txq < kNumTxq);
    static_assert(Label <= kRxThLabelMask);

    bool timer_ok = false;      // the PTP timer runs at kPtiRefclk; hardware stamps are meaningless otherwise
    int64_t ptp_offset_ns = 0;  // PTP64NS minus the CFR count in ns while this session's timer runs
    raw::TxHeaderPrev hdr_prev{};
    uint32_t no_match_prev = 0;

    __attribute__((noinline, cold)) bool begin() {
        timer_ok = ptp_timer_start(kPtiRefclk, kTimerLeadTicks, kTimerAckSpins);
        ptp_offset_ns = eth_ptp::ptp_offset_ns();
        no_match_prev = raw::rd(kRxFlNoMatchActions);
        raw::rx_th_flush();
        raw::wr(kRxFlNoMatchActions, no_match_prev & ~kRxFlKeepTimestamp);
        raw::rx_stamp_rule_install(kTcamRow, kLabel);
        hdr_prev = raw::tx_header_row_install(kTxq, kHeaderRow, kStampFrameDa);
        raw::txq_clear_timestamp_cmd(kTxq);
        raw::mac_tx_fifo_drain();
        return timer_ok;
    }
    __attribute__((noinline, cold)) void end() {
        raw::txq_clear_timestamp_cmd(kTxq);
        raw::tx_header_row_restore(kTxq, kHeaderRow, hdr_prev);
        raw::rx_stamp_rule_remove(kTcamRow);
        raw::wr(kRxFlNoMatchActions, no_match_prev);
        raw::rx_th_flush();
    }
};

// A run of stamped frames on the session's queue: stamps_arm(tag) once, then every frame the queue sends until
// stamps_disarm() is stamped under `tag` into the MAC's 128-deep egress FIFO, and every frame the peer's session
// sends is stamped on ingress under this session's label into the 16-deep RX FIFO. The drains take whatever a FIFO
// holds, each stamp to sink(uint64_t ts), and return how many carried the tag's low word or the label; the rest are
// discarded. Ingress stamps are pushed at start-of-frame, before the frame's bytes are visible in L1, so a frame
// that has been seen has its stamp waiting.
//
// ts_cmd is sticky and the queue reports idle before the MAC has taken a frame's command (measured: a clear at queue
// idle lost the stamp of one frame in fifteen), so disarm only once tx_stamps_drain has delivered the last frame's
// stamp. The tag, like the command, is sampled when the queue latches a frame's command, 24 to 32 cycles after the
// write (measured: a request armed 24 cycles after the command still stamps the frame, one armed 32 after does not)
// and some 60 before PKT_START_CNT counts the hand-off; a keepalive samples them 15 to 73 cycles before its count.
// A keepalive of the queue's own that samples an armed request is stamped under the tag like a frame, and the
// counters are what tell them apart (eth_ptp_link.hpp, collect_burst).
template <typename Session>
inline __attribute__((always_inline)) void stamps_arm(const Session&, uint64_t tag) {
    raw::txq_request_two_step(Session::kTxq, tag);
}
template <typename Session>
inline __attribute__((always_inline)) void stamps_disarm(const Session&) {
    raw::txq_clear_timestamp_cmd(Session::kTxq);
}
// Points the queue's software frames at its boot header row (`boot`), under which the peer's classifier leaves them
// unstamped like the queue's keepalives, or back at the session's row. The row is latched with a frame's command,
// so switch back once the queue reports the command taken.
template <typename Session>
inline __attribute__((always_inline)) void tx_header_row_select(const Session& s, bool boot) {
    raw::wr(txq_reg(Session::kTxq, kTxqPktCfgSelSwOff), boot ? s.hdr_prev.sel_sw : Session::kHeaderRow * 0x111u);
}
template <typename Sink>
inline __attribute__((always_inline)) uint32_t tx_stamps_drain(uint32_t tag_lo, Sink&& sink) {
    uint32_t got_tag = 0;
    uint64_t ts = 0;
    uint32_t n = 0;
    while (raw::mac_tx_fifo_pop_ts(got_tag, ts)) {
        if (got_tag == tag_lo) {
            sink(ts);
            n++;
        }
    }
    return n;
}
template <typename Session, typename Sink>
inline __attribute__((always_inline)) uint32_t rx_stamps_drain(const Session&, Sink&& sink) {
    raw::RxStamp s;
    uint32_t n = 0;
    while (raw::rx_th_pop(s)) {
        if (s.valid && s.label == Session::kLabel) {
            sink(s.rx_ts);
            n++;
        }
    }
    return n;
}

}  // namespace tt::tt_metal::eth_ptp
