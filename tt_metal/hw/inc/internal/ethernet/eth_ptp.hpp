// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Device-side access to the Blackhole Ethernet tile's IEEE 1588 timestamping hardware.
//
// IEEE 1588, the Precision Time Protocol, compares two machines' clocks using the send and receive times of packets,
// captured by the network hardware as the packets pass. The tile does this with several independent blocks, each
// wrapped in a type here:
//   - PtpTimer is the timestamp counter. It runs off the 50 MHz reference clock, so AI clock changes don't affect it.
//     The firmware never starts it; start() does, and leaves it running.
//   - TxQueue turns on timestamping for one transmit queue: each packet's send time is either written into the packet
//     (arm_in_frame) or pushed, with a tag of the caller's choosing, into TxStampFifo (arm_two_step).
//   - TxHeaderRow gives the packets sent here a distinctive destination address, and RxStampRule has the receiver
//     record the arrival times of packets with that address in RxStampFifo.
// Every block except PtpTimer puts back what it changes. The clocks are in eth_ptp_clock.hpp. Register layouts follow
// the Tenstorrent ISA documentation and the register guide of the tile's Rianta RSm410 MAC.
//
// Found on silicon, where the documentation is silent or wrong:
//   - A queue's timestamp setting applies to every packet it sends until it is cleared.
//   - The MAC's send-time queue returns tag low, tag high, time low, time high; the vendor's guide swaps the pairs.
//   - The receive-time queue's head can be read before it is removed, and removing it takes a write of 1, then of 0.
//   - A receive time is recorded as a packet starts to arrive, before its data is in L1.
//   - Each transmit queue sends a keepalive after about 8,000 idle cycles, but never between queued packets, and
//     keepalives are timestamped like any other packet. A keepalive is only headers and padding, so an in-frame stamp
//     near the start of the payload lands harmlessly in the padding; at offset 40, past its end, every keepalive was
//     lost. The interval can't just be raised: at its maximum, the peer starts retransmitting.
//   - A queue's packet-start count moves 86 cycles after a send command, its packet-end count 22 cycles later, and the
//     MAC's send-time entry appears 65 cycles after that, whatever the packet's size.
//
// On p150 links over 0.5 m of copper, the one-way delay between a packet's two stamps is 33-35 ns and a reply takes
// about 350 ns. A stamp resolves 20 ns; averaging 96 exchanges each way brings a round to about 0.6 ns.
//
// Typical use (each end's configuration is fixed at compile time, so sending needs no memory loads):
//   PtpTimer timer; TxHeaderRow<kTxq, kRow> header; RxStampRule<kTcamRow, kLabel> rule;
//   timer.start(); rule.install(); header.install();  // start() is false if the timer never took its rate
//   TxQueue<kTxq>{}.arm_in_frame();                    // every packet the queue sends now carries its send time
//   internal_::eth_send_packet(kTxq, ...);             // the peer reads the send time out of the packet
//   RxStampFifo{}.drain<kLabel>(sink);                 // the arrival times of the peer's packets
//   TxQueue<kTxq>{}.disarm(); header.restore(); rule.remove();
// The two timestamp queues belong to the tile, so a second user on the same tile would take the first one's entries.

#pragma once

#include <cstdint>

#include "internal/ethernet/dataflow_api.h"
#include "internal/ethernet/eth_ptp_clock.hpp"

namespace tt::tt_metal::eth_ptp {

enum TsCmd : uint32_t {
    TS_CMD_NOP = 0,
    TS_CMD_ONE_STEP_CORRECTION = 1,
    TS_CMD_ONE_STEP_ORIGIN = 2,
    TS_CMD_TWO_STEP_FIFO = 3,
};

struct TxqTimestamp {
    uint32_t cmd : 3;  // TsCmd
    uint32_t rsvd0 : 13;
    uint32_t offset : 6;  // where an in-frame stamp goes, in 2-byte units
    uint32_t rsvd1 : 10;
};

// The header table row each kind of software-sent packet uses.
struct TxqPktCfgSelSw {
    uint32_t raw : 4;
    uint32_t reg_write : 4;
    uint32_t packet : 4;
    uint32_t rsvd : 20;
};

constexpr uint32_t txq_base(uint32_t q) { return ETH_TXQ0_REGS_START + q * ETH_TXQ_REGS_SIZE; }
constexpr Reg<TxqPktCfgSelSw> txq_pkt_cfg_sel_sw(uint32_t q) { return {txq_base(q) + ETH_TXQ_TXPKT_CFG_SEL_SW}; }
constexpr Reg<TxqTimestamp> txq_timestamp(uint32_t q) { return {txq_base(q) + ETH_TXQ_TIMESTAMP}; }
constexpr Reg<> txq_rx_timestamp_lo(uint32_t q) { return {txq_base(q) + ETH_TXQ_RX_TIMESTAMP_LO}; }
constexpr Reg<> txq_rx_timestamp_hi(uint32_t q) { return {txq_base(q) + ETH_TXQ_RX_TIMESTAMP_HI}; }
constexpr Reg<> txq_word_cnt(uint32_t q) { return {txq_base(q) + ETH_TXQ_WORD_CNT}; }

constexpr Reg<> txpkt_cfg(uint32_t row, uint32_t reg) {
    return {ETH_TXPKT_CFG_REGS_START + row * ETH_TXPKT_CFG_REGS_SIZE + reg};
}

struct MacTxCfg {
    uint32_t rsvd0 : 1;
    uint32_t tx_enable : 1;
    uint32_t rsvd1 : 2;
    uint32_t ts_fifo_enable : 1;  // 1: a send time for every packet; 0: only under a two-step request
    uint32_t rsvd2 : 6;
    uint32_t origin_timestamp_mode : 1;
    uint32_t rsvd3 : 20;
};

struct MacTxInt {
    uint32_t rsvd0 : 2;
    uint32_t ts_fifo_full : 1;
    uint32_t ts_fifo_not_empty : 1;
    uint32_t ts_offset_error : 1;
    uint32_t rsvd1 : 27;
};

constexpr Reg<MacTxCfg> kMacTxCfg{ETH_MAC_REGS_START + ETH_MAC_TX_CFG};
constexpr Reg<> kMacTxDelay{ETH_MAC_REGS_START + ETH_MAC_TX_DELAY};
constexpr Reg<MacTxInt> kMacTxInt{ETH_MAC_REGS_START + ETH_MAC_TX_INT};
constexpr Reg<MacTxInt> kMacTxIntRaw{ETH_MAC_REGS_START + ETH_MAC_TX_INT_RAW};
constexpr Reg<> kMacTsFifoFullThresh{ETH_MAC_REGS_START + ETH_MAC_TS_FIFO_FULL_THRESH};
constexpr Reg<> kMacTsFifo0{ETH_MAC_REGS_START + ETH_MAC_TS_FIFO_0};
constexpr Reg<> kMacTsFifo1{ETH_MAC_REGS_START + ETH_MAC_TS_FIFO_1};
constexpr Reg<> kMacTsFifo2{ETH_MAC_REGS_START + ETH_MAC_TS_FIFO_2};
constexpr Reg<> kMacTsFifo3{ETH_MAC_REGS_START + ETH_MAC_TS_FIFO_3};

struct RxThTsLabel {
    uint32_t label : 5;
    uint32_t rsvd0 : 1;  // set in the entries a rule records (measured), so not part of the label
    uint32_t rsvd1 : 25;
    uint32_t valid : 1;
};

struct RxThStatus {
    uint32_t entries : 4;
    uint32_t rsvd0 : 12;
    uint32_t full : 1;
    uint32_t nearly_full : 1;
    uint32_t empty : 1;
    uint32_t rsvd1 : 11;
    uint32_t flush : 1;
    uint32_t pop : 1;
};

constexpr Reg<> kRxThTsLow{ETH_RX_TH_REGS_START + ETH_RX_TH_TS_LOW};
constexpr Reg<> kRxThTsHigh{ETH_RX_TH_REGS_START + ETH_RX_TH_TS_HIGH};
constexpr Reg<RxThTsLabel> kRxThTsLabel{ETH_RX_TH_REGS_START + ETH_RX_TH_TS_LABEL};
constexpr Reg<RxThStatus> kRxThStatus{ETH_RX_TH_REGS_START + ETH_RX_TH_STATUS};

// The RX classifier's match table and flow table, per EthernetRxClassifier.md in the ISA documentation.
struct RxFlowActions {
    uint32_t rx_queue : 2;
    uint32_t drop : 1;
    uint32_t strip_headers : 1;
    uint32_t record_rx_time : 1;
    uint32_t prepend_sw_metadata : 1;
    uint32_t prepend_hw_metadata : 1;
    uint32_t rsvd : 25;
};

struct RxTcamRowMapping {
    uint32_t priority : 3;  // the larger wins
    uint32_t rsvd0 : 13;
    uint32_t ftable_row : 6;
    uint32_t rsvd1 : 10;
};

struct RxTcamRowUpdate {
    uint32_t row : 6;
    uint32_t rsvd0 : 2;
    uint32_t enable : 1;
    uint32_t rsvd1 : 7;
    uint32_t write : 1;
    uint32_t rsvd2 : 14;
    uint32_t go : 1;
};

// Each update_* bit has the row take that part of its pattern from the write registers; the rest is left unchanged.
struct RxTcamUpdate {
    uint32_t row : 6;
    uint32_t rsvd0 : 2;
    uint32_t mask : 1;  // the row's mask bits rather than its values
    uint32_t write : 1;
    uint32_t not_ip : 1;
    uint32_t rsvd1 : 5;
    uint32_t update_protocol : 1;
    uint32_t update_dst_port : 1;
    uint32_t update_src_port : 1;
    uint32_t update_da : 1;
    uint32_t update_sa : 1;
    uint32_t update_row_kind : 1;
    uint32_t update_ethertype : 1;
    uint32_t update_l2_priority : 1;
    uint32_t rsvd2 : 7;
    uint32_t go : 1;
};

struct RxTcamNonIpAddrFlags {
    uint32_t augmented_da : 4;
    uint32_t rsvd0 : 12;
    uint32_t augmented_sa : 4;
    uint32_t rsvd1 : 12;
};

struct RxTcamEthertype {
    uint32_t value : 16;
    uint32_t augmented : 4;
    uint32_t rsvd : 12;
};

struct RxTcamPriority {
    uint32_t pcp : 3;
    uint32_t rsvd : 29;
};

struct RxFtableUpdate {
    uint32_t row : 6;
    uint32_t rsvd0 : 2;
    uint32_t write : 1;
    uint32_t rsvd1 : 22;
    uint32_t go : 1;
};

constexpr uint32_t rx_classifier(uint32_t reg) { return ETH_RX_CLASSIFIER_REGS_START + reg; }
constexpr Reg<RxFlowActions> kRxNoMatchActions{rx_classifier(ETH_RX_CLASSIFIER_NO_MATCH_ACTIONS)};
constexpr Reg<RxTcamRowMapping> rx_tcam_row_mapping(uint32_t row) {
    return {rx_classifier(ETH_RX_CLASSIFIER_TCAM_ROW_MAPPING) + 4 * row};
}
constexpr Reg<RxTcamRowUpdate> kRxTcamRowUpdate{rx_classifier(ETH_RX_CLASSIFIER_TCAM_ROW_UPDATE)};
constexpr Reg<> kRxTcamTupleTypeWrite{rx_classifier(ETH_RX_CLASSIFIER_TCAM_TUPLE_TYPE_WRITE)};  // 0: not IP
constexpr Reg<> rx_tcam_sa_write(uint32_t i) { return {rx_classifier(ETH_RX_CLASSIFIER_TCAM_SA_WRITE) + 4 * i}; }
constexpr Reg<> rx_tcam_da_write(uint32_t i) { return {rx_classifier(ETH_RX_CLASSIFIER_TCAM_DA_WRITE) + 4 * i}; }
constexpr Reg<RxTcamNonIpAddrFlags> kRxTcamNonIpAddrFlagsWrite{
    rx_classifier(ETH_RX_CLASSIFIER_TCAM_NON_IP_ADDR_FLAGS_WRITE)};
constexpr Reg<RxTcamEthertype> kRxTcamEthertypeWrite{rx_classifier(ETH_RX_CLASSIFIER_TCAM_ETHERTYPE_WRITE)};
constexpr Reg<RxTcamPriority> kRxTcamPriorityWrite{rx_classifier(ETH_RX_CLASSIFIER_TCAM_PRIORITY_WRITE)};
constexpr Reg<RxTcamUpdate> kRxTcamUpdate{rx_classifier(ETH_RX_CLASSIFIER_TCAM_UPDATE)};
constexpr Reg<> kRxFtableLabels{rx_classifier(ETH_RX_CLASSIFIER_FTABLE_LABELS)};
constexpr Reg<RxFlowActions> kRxFtableActions{rx_classifier(ETH_RX_CLASSIFIER_FTABLE_ACTIONS)};
constexpr Reg<> kRxFtableVlan{rx_classifier(ETH_RX_CLASSIFIER_FTABLE_VLAN)};
constexpr Reg<> kRxFtableSwMetadata{rx_classifier(ETH_RX_CLASSIFIER_FTABLE_SW_METADATA)};
constexpr Reg<RxFtableUpdate> kRxFtableUpdate{rx_classifier(ETH_RX_CLASSIFIER_FTABLE_UPDATE)};

// A non-IP match-table row's pattern, laid out as the pattern write registers take it.
struct RxTcamNonIpPattern {
    uint32_t sa[4];
    uint32_t da[4];
    RxTcamNonIpAddrFlags addr_flags;
    RxTcamEthertype ethertype;
    RxTcamPriority priority;
};

// The destination address that marks the packets sent here: a locally administered unicast address with 0xA5 in its
// middle four bytes. The receive rule compares only those four bytes, so the match table's byte order doesn't matter,
// and no packet the firmware sends (broadcast, 01:00:.. multicast or 02:00:.. unicast) can match.
constexpr uint64_t kStampFrameDa = 0x02A5'A5A5'A5A5ull;
// The rule's row: its values, and its mask, in which a 1 bit means don't care, so only the four 0xA5 bytes count.
constexpr RxTcamNonIpPattern kStampRowValues{.da = {0xA5A5'A500u, 0x0000'00A5u}};
constexpr RxTcamNonIpPattern kStampRowMask{
    .sa = {~0u, ~0u, ~0u, ~0u},
    .da = {0x0000'00FFu, 0xFFFF'FF00u, ~0u, ~0u},
    .addr_flags = {.augmented_da = 0xF, .augmented_sa = 0xF},
    .ethertype = {.value = 0xFFFF, .augmented = 0xF},
    .priority = {.pcp = 7}};

constexpr uint32_t kTimerLeadTicks = 5000;    // how far ahead the timer's start is scheduled: 100 us of reference ticks
constexpr uint32_t kTimerAckSpins = 200'000;  // polls of the update status before PtpTimer::start gives up

// The PTP timer. start() enables its counter and schedules its rate, and a reset of its time to 0, for kTimerLeadTicks
// reference ticks ahead. The reset lands one tick after the scheduled one, and since the counter is already running,
// the time advances 20 ns per tick from that very tick, so its offset from the reference count is known exactly.
// The routines that run once per link are cold, so that the -O3 fabric router doesn't unroll them into its 26 KB code
// budget.
struct PtpTimer {
    bool ok = false;        // the reset landed; the timer's stamps are meaningless otherwise
    int64_t offset_64 = 0;  // PTP time minus the reference count in ns, in 64ths of a ns: -20 ns times the reset tick

    // False, with the time left as it was, if the hardware never acknowledges both updates.
    __attribute__((noinline, cold)) bool start() {
        kPtpTimerCtrl.write(1);
        const uint64_t at = read_cfr() + kTimerLeadTicks;
        kPtpFutureCfrLo.write(static_cast<uint32_t>(at));
        kPtpFutureCfrHi.write(static_cast<uint32_t>(at >> 32));
        kPtpFuturePti.write(kPtiRefclk);
        kPtpFutureTimestampLo.write(0);
        kPtpFutureTimestampHi.write(0);
        kPtpUpdatePti.write(1);
        kPtpUpdateTimestamp.write(1);
        PtpUpdateStat stat{};
        for (uint32_t i = 0; i < kTimerAckSpins && !(stat.pti_ack && stat.timestamp_ack); i++) {
            stat = kPtpUpdateStat.read();
        }
        kPtpUpdatePti.write(0);
        kPtpUpdateTimestamp.write(0);
        ok = stat.pti_ack && stat.timestamp_ack;
        offset_64 = -static_cast<int64_t>(at + 1) * kNsPerRefclkTick * 64;
        return ok;
    }
};

// Under arm_in_frame(), each packet carries its send time kFrameStampField bytes into its payload, as 16 high bits and
// then the 64-bit PTP time, big-endian; the field lands 2 bytes past the queue's offset setting, which is in 2-byte
// units (measured). The receiver reads the time from its own L1, so no core has to read the MAC, which stalled the
// link under fabric traffic. The field falls in what is only padding in a keepalive, so an armed queue's keepalives
// arrive intact.
constexpr uint32_t kFrameStampOffset = 2;
constexpr uint32_t kFrameStampField = 2 * kFrameStampOffset + 2;
constexpr uint32_t kFrameStampHiWord = (kFrameStampField + 2) / 4;
static_assert((kFrameStampField + 2) % 4 == 0);
FORCE_INLINE uint64_t frame_stamp(const volatile uint32_t* payload) {
    return (static_cast<uint64_t>(__builtin_bswap32(payload[kFrameStampHiWord])) << 32) |
           __builtin_bswap32(payload[kFrameStampHiWord + 1]);
}

// Timestamping on transmit queue Q, in effect until disarm(): arm_in_frame() writes each packet's send time into the
// packet, and arm_two_step(tag) pushes it, with the tag, into TxStampFifo. The queue samples the setting 24 to 32
// cycles after a send command (measured), so arming just after the command still covers that packet. It reports idle
// before the MAC has taken the setting, though (disarming at idle lost one packet's time in fifteen), so to stamp only
// chosen packets, disarm once the last one's time is in.
template <uint32_t Q>
struct TxQueue {
    static_assert(Q < NUM_ETH_QUEUES);
    FORCE_INLINE void arm_two_step(uint64_t tag) const {
        txq_rx_timestamp_lo(Q).write(static_cast<uint32_t>(tag));
        txq_rx_timestamp_hi(Q).write(static_cast<uint32_t>(tag >> 32));
        txq_timestamp(Q).write({.cmd = TS_CMD_TWO_STEP_FIFO});
    }
    FORCE_INLINE void arm_in_frame() const {
        txq_timestamp(Q).write({.cmd = TS_CMD_ONE_STEP_ORIGIN, .offset = kFrameStampOffset});
    }
    FORCE_INLINE void disarm() const { txq_timestamp(Q).write({.cmd = TS_CMD_NOP}); }
    // In 96-byte units on the wire: one for a keepalive or a packet of up to 64 bytes of payload, two up to 128.
    FORCE_INLINE uint32_t words_sent() const { return txq_word_cnt(Q).read(); }
};

// Header table row Row as a copy of queue Q's boot row with the destination address kStampFrameDa; the firmware uses
// rows 0 to 2 and leaves 3 to 9 free. install() points the queue's software-sent packets at it (keepalives keep the
// boot row), select() switches them between the two rows from the next send command the queue takes, and restore()
// undoes install().
template <uint32_t Q, uint32_t Row>
struct TxHeaderRow {
    TxqPktCfgSelSw sel_boot{};
    uint32_t da_prev[2] = {};

    __attribute__((noinline, cold)) void install() {
        sel_boot = txq_pkt_cfg_sel_sw(Q).read();
        da_prev[0] = txpkt_cfg(Row, ETH_TXPKT_CFG_MAC_DA_LO).read();
        da_prev[1] = txpkt_cfg(Row, ETH_TXPKT_CFG_MAC_DA_HI).read();
        for (uint32_t reg :
             {ETH_TXPKT_CFG_INSERT_CTL,
              ETH_TXPKT_CFG_CUSTOM_HDR,
              ETH_TXPKT_CFG_MAC_SA_LO,
              ETH_TXPKT_CFG_MAC_SA_HI,
              ETH_TXPKT_CFG_ETHERTYPE,
              ETH_TXPKT_CFG_VLAN1,
              ETH_TXPKT_CFG_VLAN2}) {
            txpkt_cfg(Row, reg).write(txpkt_cfg(Q, reg).read());
        }
        txpkt_cfg(Row, ETH_TXPKT_CFG_MAC_DA_LO).write(static_cast<uint32_t>(kStampFrameDa));
        txpkt_cfg(Row, ETH_TXPKT_CFG_MAC_DA_HI).write(static_cast<uint32_t>(kStampFrameDa >> 32));
        select(true);
    }
    __attribute__((noinline, cold)) void restore() const {
        txq_pkt_cfg_sel_sw(Q).write(sel_boot);
        txpkt_cfg(Row, ETH_TXPKT_CFG_MAC_DA_LO).write(da_prev[0]);
        txpkt_cfg(Row, ETH_TXPKT_CFG_MAC_DA_HI).write(da_prev[1]);
    }
    FORCE_INLINE void select(bool stamped) const {
        txq_pkt_cfg_sel_sw(Q).write(stamped ? TxqPktCfgSelSw{.raw = Row, .reg_write = Row, .packet = Row} : sel_boot);
    }
};

// The receive-time queue. A packet's time is recorded as it starts to arrive, so once its data is visible in L1, its
// time is already waiting.
struct RxStampFifo {
    static constexpr uint32_t kDepth = 16;
    struct Entry {
        uint64_t ts;
        uint32_t label;
        bool valid;
    };
    // Reads the head entry, then removes it.
    FORCE_INLINE bool pop(Entry& e) const {
        if (kRxThStatus.read().empty) {
            return false;
        }
        const uint32_t lo = kRxThTsLow.read();
        const uint32_t hi = kRxThTsHigh.read();
        const RxThTsLabel label = kRxThTsLabel.read();
        kRxThStatus.write({.pop = 1});
        kRxThStatus.write({});
        e.ts = (static_cast<uint64_t>(hi) << 32) | lo;
        e.label = label.label;
        e.valid = label.valid;
        return true;
    }
    // Exactly n entries, and not full at any point since the last pop.
    FORCE_INLINE bool holds_exactly(uint32_t n) const {
        constexpr RxThStatus kCounted{.entries = 0xF, .full = 1};
        return (bits(kRxThStatus.read()) & bits(kCounted)) == n;
    }
    FORCE_INLINE void flush() const { kRxThStatus.write({.flush = 1}); }
    // Passes each valid time under Label to sink(uint64_t), discards the rest, and returns how many it passed. It takes
    // at most one queue's worth per call, so a stream of stamped packets can't hold a router's core here.
    template <uint32_t Label, typename Sink>
    FORCE_INLINE uint32_t drain(Sink&& sink) const {
        Entry e;
        uint32_t n = 0;
        for (uint32_t i = 0; i < kDepth && pop(e); i++) {
            if (e.valid && e.label == Label) {
                sink(e.ts);
                n++;
            }
        }
        return n;
    }
};

// A match-table row for kStampFrameDa, pointing at a flow-table row whose only action is to record the receive time
// under Label. install() also stops the recording of times for packets that match no rule, so the queue holds only
// this rule's times, and empties the queue; remove() undoes install().
template <uint32_t Row, uint32_t Label>
struct RxStampRule {
    static_assert(Label < 32);  // flow-table labels are five bits
    static constexpr uint32_t kLabel = Label;
    RxFlowActions no_match_prev{};

    __attribute__((noinline, cold)) void install() {
        no_match_prev = kRxNoMatchActions.read();
        RxStampFifo{}.flush();
        RxFlowActions no_match = no_match_prev;
        no_match.record_rx_time = 0;
        kRxNoMatchActions.write(no_match);
        write_row(kStampRowValues, false);
        write_row(kStampRowMask, true);
        rx_tcam_row_mapping(Row).write({.priority = 7, .ftable_row = Row});
        write_flow({.record_rx_time = 1}, Label);
        kRxTcamRowUpdate.write({.row = Row, .enable = 1, .write = 1, .go = 1});
    }
    __attribute__((noinline, cold)) void remove() const {
        kRxTcamRowUpdate.write({.row = Row, .write = 1, .go = 1});
        write_flow({}, 0);
        kRxNoMatchActions.write(no_match_prev);
        RxStampFifo{}.flush();
    }

private:
    FORCE_INLINE static void write_row(const RxTcamNonIpPattern& p, bool mask) {
        kRxTcamTupleTypeWrite.write(0);
        kRxTcamEthertypeWrite.write(p.ethertype);
        kRxTcamPriorityWrite.write(p.priority);
        for (uint32_t i = 0; i < 4; i++) {
            rx_tcam_sa_write(i).write(p.sa[i]);
        }
        for (uint32_t i = 0; i < 4; i++) {
            rx_tcam_da_write(i).write(p.da[i]);
        }
        kRxTcamNonIpAddrFlagsWrite.write(p.addr_flags);
        kRxTcamUpdate.write(
            {.row = Row,
             .mask = mask,
             .write = 1,
             .not_ip = 1,
             .update_da = 1,
             .update_sa = 1,
             .update_row_kind = 1,
             .update_ethertype = 1,
             .update_l2_priority = 1,
             .go = 1});
    }
    FORCE_INLINE static void write_flow(RxFlowActions actions, uint32_t label) {
        kRxFtableActions.write(actions);
        kRxFtableVlan.write(0);
        kRxFtableLabels.write(label);
        kRxFtableSwMetadata.write(0);
        kRxFtableUpdate.write({.row = Row, .write = 1, .go = 1});
    }
};

// The MAC's send-time queue, 128 entries deep, holding the tag and send time of each packet sent under arm_two_step().
struct TxStampFifo {
    // Removes one entry: its tag's low word and the send time (the caller knows the tag's high word). Word 0 must be
    // read first, since reading it advances the queue; an empty queue reads 0xFFFFFFFF there.
    FORCE_INLINE bool pop(uint32_t& tag_lo, uint64_t& ts) const {
        const uint32_t w0 = kMacTsFifo0.read();
        if (w0 == 0xFFFFFFFFu) {
            return false;
        }
        const uint32_t w2 = kMacTsFifo2.read();
        const uint32_t w3 = kMacTsFifo3.read();
        tag_lo = w0;
        ts = (static_cast<uint64_t>(w3) << 32) | w2;
        return true;
    }
    FORCE_INLINE bool empty() const { return !kMacTxIntRaw.read().ts_fifo_not_empty; }
    void clear() const {
        uint32_t tag_lo = 0;
        uint64_t ts = 0;
        while (pop(tag_lo, ts)) {
        }
    }
    // Passes each time whose tag's low word is tag_lo to sink(uint64_t), discards the rest, and returns how many it
    // passed.
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
