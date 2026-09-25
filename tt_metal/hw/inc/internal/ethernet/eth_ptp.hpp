// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Device-side access to the Blackhole Ethernet tile's IEEE-1588 hardware, one type per block: the eth_ctrl PTP timer
// (PtpTimer), a TX queue's timestamp command and hand-off counters (TxQueue), a TX header row (TxHeaderRow), an RX
// classifier stamp rule (RxStampRule), the Rianta RSm410 MAC's two-step TX timestamp FIFO (TxStampFifo) and the RX
// classifier's timestamp FIFO (RxStampFifo); the tile's clocks are eth_ptp_clock.hpp. The blocks that change tile
// state keep what they changed and put it back. Register offsets follow tt-isa-documentation (EthernetRxClassifier.md,
// the eth_ctrl and TXQ maps) and the Rianta RSm410 register guide; the TXQ and RISC bases are tt_eth_ss_regs.h's.
//
// The PTP timer counts refclk (50 MHz on Blackhole) and is independent of AICLK, so it is immune to DVFS. Nothing in
// the shipped firmware enables its main counter; PtpTimer::start() does, and leaves it running.
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
// ~350 ns; 96 exchanges averaged per side dither the 20 ns tick to ~0.6 ns per round.
//
// One end of a link, its configuration compile-time so the per-frame path has no loads (the ERISC runs no dynamic
// init, so the blocks are statics or members):
//   PtpTimer timer; TxHeaderRow<kTxq, kRow> header; RxStampRule<kTcamRow, kLabel> rule;
//   timer.start(); rule.install(); header.install();      // start() is false if the timer never took its rate
//   TxQueue<kTxq>{}.arm_two_step(tag);                     // every frame the queue sends from here is stamped
//   internal_::eth_send_packet(kTxq, ...);                 // ... per frame
//   TxStampFifo{}.drain(tag_lo, sink);                     // this end's egress stamps
//   RxStampFifo{}.drain<kLabel>(sink);                     // the peer's frames' ingress stamps
//   TxQueue<kTxq>{}.disarm();                              // once the last frame's egress stamp has been drained
//   header.restore(); rule.remove();
// The two FIFOs are the tile's: a second user on the same tile would consume the first one's stamps.

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
constexpr uint32_t kRxThStatusFull = 1u << 16;
constexpr uint32_t kRxThStatusEmpty = 1u << 18;
constexpr uint32_t kRxThStatusFlush = 1u << 30;
constexpr uint32_t kRxThStatusPop = 1u << 31;
constexpr uint32_t kRxThLabelMask = 0x1F;
constexpr uint32_t kRxThLabelValid = 1u << 31;
// [1:0]: 0 accept all, 2 use the flow table. At 0 the classifier still applies the table's keep-timestamp action and
// only ignores its drop decisions, so the stamp rule needs no change here.
constexpr uint32_t kRxFdOverrideDecision = 0xFFB9D000;
constexpr uint32_t kRxFlNoMatchActions = 0xFFB9CD04;  // [1:0] queue [2] drop [3] rm_hdr [4] keep_timestamp
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
constexpr uint32_t kTimerAckSpins = 200'000;  // polls of the update status before PtpTimer::start gives up

// The eth_ctrl PTP timer. start() enables its main counter, then schedules, at a CFR tick kTimerLeadTicks ahead, both
// its per-tick increment and a restart of its timestamp from 0, which lands on the tick after the one scheduled (the
// timer compares the CFR count against it, then updates); PTP64NS is 20 ns per tick from there, so its offset from the
// CFR count is known exactly. The counter runs before the restart lands, so it counts from that very tick.
// Cold: in a kernel built -O3 (the fabric router) the once-per-link routines would otherwise unroll into a couple of KB
// of a 26 KB kernel budget shared with the router.
struct PtpTimer {
    bool ok = false;        // the restart landed; the timer's stamps are meaningless otherwise
    int64_t offset_64 = 0;  // PTP64NS minus the CFR count in ns, in 64ths of a ns: -20 ns times the restart tick

    // False, and the timestamp as it was, if the hardware never acknowledges both updates.
    __attribute__((noinline, cold)) bool start() {
        wr(kPtpTimerCtrl, 1);
        const uint64_t at = read_cfr() + kTimerLeadTicks;
        wr(kPtpFutureCfrLo, static_cast<uint32_t>(at));
        wr(kPtpFutureCfrHi, static_cast<uint32_t>(at >> 32));
        wr(kPtpFuturePti, kPtiRefclk);
        wr(kPtpFutureTimestampLo, 0);
        wr(kPtpFutureTimestampHi, 0);
        wr(kPtpUpdatePti, 1);
        wr(kPtpUpdateTimestamp, 1);
        constexpr uint32_t kAcks = kUpdateStatPtiAck | kUpdateStatTsAck;
        uint32_t stat = 0;
        for (uint32_t i = 0; i < kTimerAckSpins && (stat & kAcks) != kAcks; i++) {
            stat = rd(kPtpUpdateStat);
        }
        wr(kPtpUpdatePti, 0);
        wr(kPtpUpdateTimestamp, 0);
        ok = (stat & kAcks) == kAcks;
        offset_64 = -static_cast<int64_t>(at + 1) * kNsPerRefclkTick * 64;
        return ok;
    }
};

// A frame sent under arm_in_frame() carries the MAC's egress time kFrameStampField bytes into its payload: sixteen
// bits of ORIGIN_TIMESTAMP_MSBS then PTP64NS, big-endian (ts_offset kFrameStampOffset, in 2-byte units; the field
// lands two bytes past it, measured). The peer reads the stamp out of the frame in its own L1, so no core reads the
// MAC: under fabric traffic a router's reads of any MAC register, the egress FIFO or a status word, wedged the link.
constexpr uint32_t kFrameStampOffset = 40;
constexpr uint32_t kFrameStampField = 2 * kFrameStampOffset + 2;
constexpr uint32_t kFrameStampHiWord = (kFrameStampField + 2) / 4;
static_assert((kFrameStampField + 2) % 4 == 0);
FORCE_INLINE uint64_t frame_stamp(const volatile uint32_t* payload) {
    return (static_cast<uint64_t>(__builtin_bswap32(payload[kFrameStampHiWord])) << 32) |
           __builtin_bswap32(payload[kFrameStampHiWord + 1]);
}

// TX queue Q's timestamp command and hand-off counters. arm_two_step(tag) has the MAC push {tag, egress stamp} into
// TxStampFifo for every frame the queue sends; arm_in_frame() has it write each frame's egress stamp into the frame.
// The command is sticky until disarm(), and the queue reports idle before the MAC has taken a frame's command
// (measured: a clear at queue idle lost the stamp of one frame in fifteen), so disarm only once the last frame's
// stamp is in. The command and tag are sampled when the queue latches a frame's command, 24 to 32 cycles after the
// write (measured: a request armed 24 cycles after the command still stamps the frame, one armed 32 after does not)
// and some 60 before PKT_START_CNT counts the hand-off; a keepalive samples them 15 to 73 cycles before its count, and
// one that samples an armed request is stamped like a frame (the counters tell them apart).
template <uint32_t Q>
struct TxQueue {
    static_assert(Q < kNumTxq);
    FORCE_INLINE void arm_two_step(uint64_t tag) const {
        wr(txq_reg(Q, kTxqRxTimestampLoOff), static_cast<uint32_t>(tag));
        wr(txq_reg(Q, kTxqRxTimestampHiOff), static_cast<uint32_t>(tag >> 32));
        wr(txq_reg(Q, kTxqTimestampOff), TS_CMD_TWO_STEP_FIFO);
    }
    FORCE_INLINE void arm_in_frame() const {
        wr(txq_reg(Q, kTxqTimestampOff), TS_CMD_ONE_STEP_ORIGIN | (kFrameStampOffset << 16));
    }
    FORCE_INLINE void disarm() const { wr(txq_reg(Q, kTxqTimestampOff), TS_CMD_NOP); }
    FORCE_INLINE uint32_t packets_started() const { return rd(txq_reg(Q, ETH_TXQ_PKT_START_CNT)); }
    // In 96-byte units on the wire: one for a keepalive or a frame of up to 64 bytes of payload, two to 128.
    FORCE_INLINE uint32_t words_sent() const { return rd(txq_reg(Q, ETH_TXQ_WORD_CNT)); }
};

// Header row Row as a copy of queue Q's boot row with DA kStampFrameDa. install() points the queue's software frames
// at it; hardware-generated frames (sequence-number keepalives) keep the boot row, so only frames the kernel sends
// carry the identity. select() points the software frames at this row or back at the boot row, under which the peer's
// classifier leaves them unstamped; the row is latched with a frame's command, so switch back once the queue reports
// the command taken. restore() puts the row's DA and the selection back.
template <uint32_t Q, uint32_t Row>
struct TxHeaderRow {
    uint32_t sel_boot = 0, da_prev[2] = {};

    __attribute__((noinline, cold)) void install() {
        sel_boot = rd(txq_reg(Q, kTxqPktCfgSelSwOff));
        da_prev[0] = rd(tx_pkt_cfg_reg(Row, kTxPktCfgMacDaLoOff));
        da_prev[1] = rd(tx_pkt_cfg_reg(Row, kTxPktCfgMacDaHiOff));
        for (uint32_t off :
             {kTxPktCfgInsertCtlOff,
              kTxPktCfgCustomHdrOff,
              kTxPktCfgMacSaLoOff,
              kTxPktCfgMacSaHiOff,
              kTxPktCfgEthertypeOff,
              kTxPktCfgVlan1Off,
              kTxPktCfgVlan2Off}) {
            wr(tx_pkt_cfg_reg(Row, off), rd(tx_pkt_cfg_reg(Q, off)));
        }
        wr(tx_pkt_cfg_reg(Row, kTxPktCfgMacDaLoOff), static_cast<uint32_t>(kStampFrameDa));
        wr(tx_pkt_cfg_reg(Row, kTxPktCfgMacDaHiOff), static_cast<uint32_t>(kStampFrameDa >> 32));
        select(true);
    }
    __attribute__((noinline, cold)) void restore() const {
        wr(txq_reg(Q, kTxqPktCfgSelSwOff), sel_boot);
        wr(tx_pkt_cfg_reg(Row, kTxPktCfgMacDaLoOff), da_prev[0]);
        wr(tx_pkt_cfg_reg(Row, kTxPktCfgMacDaHiOff), da_prev[1]);
    }
    FORCE_INLINE void select(bool stamped) const {
        wr(txq_reg(Q, kTxqPktCfgSelSwOff), stamped ? Row * 0x111u : sel_boot);
    }
};

// The RX classifier's ingress timestamp FIFO, 16 deep: a stamp for every frame a rule keeps it for, pushed at
// start-of-frame, before the frame's bytes are visible in L1, so a frame that has been seen has its stamp waiting.
struct RxStampFifo {
    struct Entry {
        uint64_t ts;
        uint32_t label;
        bool valid;
    };
    // Reads the head entry, then pops it.
    FORCE_INLINE bool pop(Entry& e) const {
        if (rd(kRxThStatus) & kRxThStatusEmpty) {
            return false;
        }
        const uint32_t lo = rd(kRxThTsLow);
        const uint32_t hi = rd(kRxThTsHigh);
        const uint32_t label = rd(kRxThTsLabel);
        wr(kRxThStatus, kRxThStatusPop);
        wr(kRxThStatus, 0);
        e.ts = (static_cast<uint64_t>(hi) << 32) | lo;
        e.label = label & kRxThLabelMask;
        e.valid = (label & kRxThLabelValid) != 0;
        return true;
    }
    // Exactly n entries, and never full since the last pop.
    FORCE_INLINE bool holds_exactly(uint32_t n) const {
        return (rd(kRxThStatus) & (kRxThStatusFull | kRxThStatusEntriesMask)) == n;
    }
    FORCE_INLINE void flush() const { wr(kRxThStatus, kRxThStatusFlush); }
    // Each valid stamp under Label to sink(uint64_t), the rest discarded; returns how many went to sink. At most one
    // FIFO's worth per call, so a stream of stamped frames cannot hold a router's core here.
    template <uint32_t Label, typename Sink>
    FORCE_INLINE uint32_t drain(Sink&& sink) const {
        Entry e;
        uint32_t n = 0;
        for (uint32_t i = 0; i <= kRxThStatusEntriesMask && pop(e); i++) {
            if (e.valid && e.label == Label) {
                sink(e.ts);
                n++;
            }
        }
        return n;
    }
};

// A TCAM row matching kStampFrameDa's middle bytes, mapped to a flow-table row whose only action is to keep the RX
// timestamp under Label. install() also clears the keep-timestamp action of frames no rule matches, so the FIFO holds
// only the rule's stamps, and flushes what the FIFO held; remove() puts the no-match actions back.
template <uint32_t Row, uint32_t Label>
struct RxStampRule {
    static_assert(Label <= kRxThLabelMask);
    static constexpr uint32_t kLabel = Label;
    uint32_t no_match_prev = 0;

    __attribute__((noinline, cold)) void install() {
        no_match_prev = rd(kRxFlNoMatchActions);
        RxStampFifo{}.flush();
        wr(kRxFlNoMatchActions, no_match_prev & ~kRxFlKeepTimestamp);
        write_pattern(false, 0xA5A5A500u, 0x000000A5u, 0u, 0u, 0u, 0u);
        write_pattern(true, 0x000000FFu, 0xFFFFFF00u, 0xFFFFFFFFu, 0x000F000Fu, 0x000FFFFFu, 0x7u);
        wr(kRxFlTcamRowMappingBase + 4 * Row, (Row << 16) | 7u);
        write_flow(kRxFlKeepTimestamp, Label);
        wr(kRxFlTcamRowUpdate, Row | (1u << 8) | (1u << 16) | (1u << 31));
    }
    __attribute__((noinline, cold)) void remove() const {
        wr(kRxFlTcamRowUpdate, Row | (1u << 16) | (1u << 31));
        write_flow(0, 0);
        wr(kRxFlNoMatchActions, no_match_prev);
        RxStampFifo{}.flush();
    }

private:
    static void write_pattern(
        bool mask, uint32_t da_w0, uint32_t da_w1, uint32_t rest, uint32_t flags, uint32_t etype, uint32_t pri) {
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
           Row | (mask ? 1u << 8 : 0u) | (1u << 9) | (1u << 10) | (1u << 19) | (1u << 20) | (1u << 21) | (1u << 22) |
               (1u << 23) | (1u << 31));
    }
    static void write_flow(uint32_t actions, uint32_t label) {
        wr(kRxFlFtableActions, actions);
        wr(kRxFlFtableVlan, 0);
        wr(kRxFlFtableLabels, label & kRxThLabelMask);
        wr(kRxFlFtableSwMetadata, 0);
        wr(kRxFlFtableUpdate, Row | (1u << 8) | (1u << 31));
    }
};

// The MAC's two-step egress timestamp FIFO, 128 deep: {tag, stamp} for every frame a queue sends under arm_two_step.
struct TxStampFifo {
    // One entry: its tag's low word and the stamp (the tag's high word is the caller's own). Word 0 must be read
    // first, it is what advances the FIFO; an empty FIFO reads 0xFFFFFFFF there.
    FORCE_INLINE bool pop(uint32_t& tag_lo, uint64_t& ts) const {
        const uint32_t w0 = rd(kMacTsFifo0);
        if (w0 == 0xFFFFFFFFu) {
            return false;
        }
        const uint32_t w2 = rd(kMacTsFifo2);
        const uint32_t w3 = rd(kMacTsFifo3);
        tag_lo = w0;
        ts = (static_cast<uint64_t>(w3) << 32) | w2;
        return true;
    }
    FORCE_INLINE bool empty() const { return (rd(kMacTxIntRaw) & kMacTxIntTsFifoNotEmpty) == 0; }
    void clear() const {
        uint32_t tag_lo = 0;
        uint64_t ts = 0;
        while (pop(tag_lo, ts)) {
        }
    }
    // Each stamp whose tag's low word is tag_lo to sink(uint64_t), the rest discarded; returns how many went to sink.
    template <typename Sink>
    FORCE_INLINE uint32_t drain(uint32_t tag_lo, Sink&& sink) const {
        uint32_t got = 0, n = 0;
        uint64_t ts = 0;
        while (pop(got, ts)) {
            if (got == tag_lo) {
                sink(ts);
                n++;
            }
        }
        return n;
    }
};

}  // namespace tt::tt_metal::eth_ptp
