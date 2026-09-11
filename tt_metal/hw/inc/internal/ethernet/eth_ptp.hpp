// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Device-side access to the Blackhole Ethernet tile's IEEE-1588 hardware: the eth_ctrl PTP timer, the per-queue TX
// timestamp command, the Rianta RSm410 MAC's two-step TX timestamp FIFO and the RX classifier's timestamp FIFO, ; the
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
//     cycles (0x1f40 at boot, ~6 us), and every keepalive gets an RX stamp under the no-match label.
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
//   const uint64_t t0 = send_and_stamp(sess, tag, src, dst, words);  // the MAC's egress stamp of that frame, 0 if none
//   RxStamps rx; ... const uint64_t t1 = rx.take<Session>();       // the newest ingress stamp under the label, 0 if
//   none sess.end();
// The two FIFOs are the tile's: a second session on the same tile would consume the first one's stamps.

#pragma once

#include <cstdint>

#include "internal/ethernet/dataflow_api.h"
#include "internal/ethernet/eth_ptp_clock.hpp"

namespace tt::tt_metal::eth_ptp {

// Per TX queue
constexpr uint32_t kTxqRegsBase = ETH_TXQ0_REGS_START;
constexpr uint32_t kTxqStride = ETH_TXQ_REGS_SIZE;
constexpr uint32_t kNumTxq = 3;
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
constexpr uint32_t kTxStampSpins = 4096;      // polls of the MAC FIFO for a frame's egress stamp before it is given up

// Register-level operations. Each one changes tile state that outlives the kernel; StampSession is the pairing of
// changes and their restores, and the API below it is what a kernel is meant to call.
namespace raw {

using eth_ptp::rd;
using eth_ptp::wr;

// Arms the MAC to push {tag, egress timestamp} into its FIFO for every frame queue q sends until the command is
// cleared.
inline __attribute__((always_inline)) void txq_request_two_step(uint32_t q, uint64_t tag) {
    wr(txq_reg(q, kTxqRxTimestampLoOff), static_cast<uint32_t>(tag));
    wr(txq_reg(q, kTxqRxTimestampHiOff), static_cast<uint32_t>(tag >> 32));
    wr(txq_reg(q, kTxqTimestampOff), TS_CMD_TWO_STEP_FIFO);
}
inline __attribute__((always_inline)) void txq_clear_timestamp_cmd(uint32_t q) {
    wr(txq_reg(q, kTxqTimestampOff), TS_CMD_NOP);
}

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
__attribute__((noinline)) inline bool ptp_timer_start(uint32_t pti, uint32_t lead_ticks, uint32_t spin_limit) {
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

// PTP64NS - 20 * CFR. The two counters advance on the same 50 MHz edge, so the difference is one constant, a multiple
// of 20 ns, for as long as the timer runs; but a single pair of reads puts the register latency between them (a tick
// or more) into it, and a stall between the two reads puts in several -- measured once at kernel start, that sat in
// every stamp of one side for the whole run as an 80 ns link bias that came and went between launches. Each pair is
// read in both orders so the skew cancels in the sum, the median over pairs discards a stalled one, and the result
// is rounded to the tick.
__attribute__((noinline)) inline int64_t ptp_ns_minus_20cfr() {
    constexpr int kPairs = 16;
    int64_t sum2[kPairs];
    for (int i = 0; i < kPairs; i++) {
        const uint64_t c1 = read_cfr();
        const uint64_t n1 = read_ptp64ns();
        const uint64_t n2 = read_ptp64ns();
        const uint64_t c2 = read_cfr();
        const int64_t k1 = static_cast<int64_t>(n1) - static_cast<int64_t>(c1) * 20;
        const int64_t k2 = static_cast<int64_t>(n2) - static_cast<int64_t>(c2) * 20;
        int64_t v = k1 + k2;
        int j = i;
        for (; j > 0 && sum2[j - 1] > v; j--) {
            sum2[j] = sum2[j - 1];
        }
        sum2[j] = v;
    }
    const int64_t k = (sum2[kPairs / 2 - 1] + sum2[kPairs / 2]) / 4;
    return ((k + (k >= 0 ? 10 : -10)) / 20) * 20;
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

    bool timer_ok = false;       // the PTP timer runs at kPtiRefclk; hardware stamps are meaningless otherwise
    int64_t ns_minus_20cfr = 0;  // PTP64NS - 20 * CFR while this session's timer runs
    raw::TxHeaderPrev hdr_prev{};
    uint32_t no_match_prev = 0;

    __attribute__((noinline)) bool begin() {
        timer_ok = ptp_timer_start(kPtiRefclk, kTimerLeadTicks, kTimerAckSpins);
        ns_minus_20cfr = ptp_ns_minus_20cfr();
        no_match_prev = raw::rd(kRxFlNoMatchActions);
        raw::rx_th_flush();
        raw::wr(kRxFlNoMatchActions, no_match_prev & ~kRxFlKeepTimestamp);
        raw::rx_stamp_rule_install(kTcamRow, kLabel);
        hdr_prev = raw::tx_header_row_install(kTxq, kHeaderRow, kStampFrameDa);
        raw::txq_clear_timestamp_cmd(kTxq);
        raw::mac_tx_fifo_drain();
        return timer_ok;
    }
    __attribute__((noinline)) void end() {
        raw::txq_clear_timestamp_cmd(kTxq);
        raw::tx_header_row_restore(kTxq, kHeaderRow, hdr_prev);
        raw::rx_stamp_rule_remove(kTcamRow);
        raw::wr(kRxFlNoMatchActions, no_match_prev);
        raw::rx_th_flush();
    }
};

// Issues one frame of `words` 16-byte words on the session's queue under a two-step stamp request and returns once
// the queue is idle again. The request stays armed until collect_tx_stamp clears it: measured, a clear at queue
// idle loses the stamp of one frame in fifteen, so the queue reports idle before the MAC has taken the frame's
// command. `before_issue` runs with the queue armed, right before the frame is issued: where a software stamp of the
// same instant belongs.
template <typename Session, typename BeforeIssue>
inline __attribute__((always_inline)) void send_stamped(
    const Session&, uint64_t tag, uint32_t src_addr, uint32_t dst_addr, uint32_t words, BeforeIssue&& before_issue) {
    while (internal_::eth_txq_is_busy(Session::kTxq)) {
    }
    raw::txq_request_two_step(Session::kTxq, tag);
    before_issue();
    internal_::eth_send_packet(Session::kTxq, src_addr >> 4, dst_addr >> 4, words);
    while (internal_::eth_txq_is_busy(Session::kTxq)) {
    }
}
template <typename Session>
inline __attribute__((always_inline)) void send_stamped(
    const Session& s, uint64_t tag, uint32_t src_addr, uint32_t dst_addr, uint32_t words) {
    send_stamped(s, tag, src_addr, dst_addr, words, [] {});
}

// The MAC's egress stamp of the frame send_stamped issued under `tag`: waits up to `spins` polls for the FIFO to
// fill, clears the queue's stamp request (ts_cmd is sticky: a queue still armed at its idle timeout would stamp its
// own keepalive under the same tag, so collect within that timeout, ~6 us), then pops entries up to the tag's.
// 0 if no stamp appeared or the tag was not among them.
template <typename Session>
inline __attribute__((always_inline)) uint64_t collect_tx_stamp(const Session&, uint64_t tag, uint32_t spins) {
    for (uint32_t spin = 0; spin < spins && !raw::mac_tx_fifo_not_empty(); spin++) {
    }
    raw::txq_clear_timestamp_cmd(Session::kTxq);
    raw::MacTxStamp s;
    while (raw::mac_tx_fifo_pop(s)) {
        if (s.tag == tag) {
            return s.tx_ts;
        }
    }
    return 0;
}

// send_stamped followed by collect_tx_stamp, waiting up to kTxStampSpins for the stamp.
template <typename Session, typename BeforeIssue>
inline __attribute__((always_inline)) uint64_t send_and_stamp(
    const Session& s, uint64_t tag, uint32_t src_addr, uint32_t dst_addr, uint32_t words, BeforeIssue&& before_issue) {
    send_stamped(s, tag, src_addr, dst_addr, words, before_issue);
    return collect_tx_stamp(s, tag, kTxStampSpins);
}
template <typename Session>
inline __attribute__((always_inline)) uint64_t
send_and_stamp(const Session& s, uint64_t tag, uint32_t src_addr, uint32_t dst_addr, uint32_t words) {
    return send_and_stamp(s, tag, src_addr, dst_addr, words, [] {});
}

// Ingress stamps land in the RX FIFO at start-of-frame, before the frame's bytes are visible in L1, so a wait loop
// polls them as they come and take() hands over the newest one under the session's label once the frame has been
// seen.
struct RxStamps {
    uint64_t newest = 0;
    bool have = false;
    template <typename Session>
    __attribute__((always_inline)) void poll() {
        raw::RxStamp s;
        while (raw::rx_th_pop(s)) {
            if (s.valid && s.label == Session::kLabel) {
                newest = s.rx_ts;
                have = true;
            }
        }
    }
    template <typename Session>
    __attribute__((always_inline)) uint64_t take() {
        poll<Session>();
        const uint64_t ts = have ? newest : 0;
        have = false;
        return ts;
    }
};

}  // namespace tt::tt_metal::eth_ptp
