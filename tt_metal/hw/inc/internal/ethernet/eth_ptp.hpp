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

constexpr uint32_t kTxqRegsBase = ETH_TXQ0_REGS_START;
constexpr uint32_t kTxqStride = ETH_TXQ_REGS_SIZE;
constexpr uint32_t kNumTxq = NUM_ETH_QUEUES;
constexpr uint32_t kTxqPktCfgSelSwOff = 0x80;
constexpr uint32_t kTxqTimestampOff = 0x90;
constexpr uint32_t kTxqRxTimestampLoOff = 0x94;  // the two-step tag, low word; the MAC returns it with the send time
constexpr uint32_t kTxqRxTimestampHiOff = 0x98;  // the two-step tag, high word
constexpr uint32_t txq_reg(uint32_t q, uint32_t off) { return kTxqRegsBase + q * kTxqStride + off; }

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

// The header row each kind of software-sent packet uses.
struct TxqPktCfgSelSw {
    uint32_t raw : 4;
    uint32_t reg_write : 4;
    uint32_t packet : 4;
    uint32_t rsvd : 20;
};

// The transmit header table (the Ethernet controller's TXPKT_CFG rows). The firmware programs rows 0 to 2 for queues 0
// to 2, with destination addresses broadcast, multicast and 02:00:00:00:00:00, which the receiving MAC routes to
// receive queues 0, 1 and 2. Rows 3 to 9 are left free.
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

// The Rianta RSm410 MAC's registers (its register map starts at 0xFFBA0000).
constexpr uint32_t kMacTxCfg = 0xFFBA2200;
constexpr uint32_t kMacTxDelay = 0xFFBA2218;   // [15:0] a fixed delay the MAC adds to every send time
constexpr uint32_t kMacTxInt = 0xFFBA2288;     // MacTxInt; writing 1 to a bit clears it
constexpr uint32_t kMacTxIntRaw = 0xFFBA2290;  // MacTxInt
constexpr uint32_t kMacTsFifoFullThresh = 0xFFBA2300;
constexpr uint32_t kMacTsFifo0 = 0xFFBA2E00;  // reading it removes the head entry; an empty queue reads 0xFFFFFFFF
constexpr uint32_t kMacTsFifo1 = 0xFFBA2E04;
constexpr uint32_t kMacTsFifo2 = 0xFFBA2E08;
constexpr uint32_t kMacTsFifo3 = 0xFFBA2E0C;

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

// The receive classifier's receive-time queue (its timestamp handling registers).
constexpr uint32_t kRxThTsLow = 0xFFB9D800;
constexpr uint32_t kRxThTsHigh = 0xFFB9D804;
constexpr uint32_t kRxThTsLabel = 0xFFB9D808;
constexpr uint32_t kRxThStatus = 0xFFB9D810;

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

// The receive classifier's match table (a 64-row content-addressable memory) and the flow table its rows point at,
// per the ISA documentation's EthernetRxClassifier.md. The flow table's row 64 holds the actions for packets no row
// matches.
constexpr uint32_t kRxFlNoMatchActions = 0xFFB9CD04;
constexpr uint32_t kRxFlTcamRowMappingBase = 0xFFB9CC00;  // + 4 * row
constexpr uint32_t kRxFlTcamRowUpdate = 0xFFB9CD40;
constexpr uint32_t kRxFlTcamTupleTypeWrite = 0xFFB9CD80;  // [1:0] the row's kind; 0 is not IP
constexpr uint32_t kRxFlTcamSaWrite0 = 0xFFB9CD90;        // four words
constexpr uint32_t kRxFlTcamDaWrite0 = 0xFFB9CDA0;  // four words; a non-IP row's first six bytes are the MAC address
constexpr uint32_t kRxFlTcamNonIpAddrFlagsWrite = 0xFFB9CDB0;
constexpr uint32_t kRxFlTcamEthertypeWrite = 0xFFB9CDC0;
constexpr uint32_t kRxFlTcamPriorityWrite = 0xFFB9CDC4;
constexpr uint32_t kRxFlTcamUpdate = 0xFFB9CDF0;
constexpr uint32_t kRxFlFtableLabels = 0xFFB9CE80;
constexpr uint32_t kRxFlFtableActions = 0xFFB9CE84;
constexpr uint32_t kRxFlFtableVlan = 0xFFB9CE88;
constexpr uint32_t kRxFlFtableSwMetadata = 0xFFB9CE8C;
constexpr uint32_t kRxFlFtableUpdate = 0xFFB9CEA0;

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

// The destination address that marks the packets sent here: a locally administered unicast address whose middle four
// bytes are 0xA5. The receive rule compares only those four bytes, so it matches whichever byte order the match table
// stores addresses in, and no packet the firmware sends (broadcast, 01:00:.. multicast, 02:00:.. unicast) can match.
constexpr uint64_t kStampFrameDa = 0x02A5'A5A5'A5A5ull;

constexpr uint32_t kTimerLeadTicks = 5000;    // how far ahead the timer's start is scheduled: 100 us of reference ticks
constexpr uint32_t kTimerAckSpins = 200'000;  // polls of the update status before PtpTimer::start gives up

// The Ethernet controller's PTP timer. start() enables the counter, then schedules two updates for kTimerLeadTicks
// reference ticks later: the per-tick increment, and a reset of the time to 0. The reset takes effect one tick after
// the scheduled one (the timer compares the reference count with the scheduled value, then updates), and from there
// the time advances 20 ns per tick, so its offset from the reference count is known exactly. The counter is already
// running when the reset lands, so it counts from that very tick.
// The routines that run once per link are marked cold: in a kernel built at -O3 (the fabric router) they would
// otherwise unroll into a couple of KB of the 26 KB code budget the router shares.
struct PtpTimer {
    bool ok = false;        // the reset landed; the timer's stamps are meaningless otherwise
    int64_t offset_64 = 0;  // PTP time minus the reference count in ns, in 64ths of a ns: -20 ns times the reset tick

    // False, with the time left as it was, if the hardware never acknowledges both updates.
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
        PtpUpdateStat stat{};
        for (uint32_t i = 0; i < kTimerAckSpins && !(stat.pti_ack && stat.timestamp_ack); i++) {
            stat = rd<PtpUpdateStat>(kPtpUpdateStat);
        }
        wr(kPtpUpdatePti, 0);
        wr(kPtpUpdateTimestamp, 0);
        ok = stat.pti_ack && stat.timestamp_ack;
        offset_64 = -static_cast<int64_t>(at + 1) * kNsPerRefclkTick * 64;
        return ok;
    }
};

// A packet sent under arm_in_frame() carries its send time kFrameStampField bytes into its payload, as 16 high bits
// followed by the 64-bit PTP time, big-endian. The queue's offset setting (kFrameStampOffset) is in 2-byte units, and
// the field lands 2 bytes after it (measured). The receiver reads the time straight out of the packet in its own L1,
// so no core has to read the MAC: under fabric traffic, a router reading any MAC register, the send-time queue or a
// status word stalled the link. The field falls where a keepalive has only padding, so an armed queue's keepalives
// arrive intact.
constexpr uint32_t kFrameStampOffset = 2;
constexpr uint32_t kFrameStampField = 2 * kFrameStampOffset + 2;
constexpr uint32_t kFrameStampHiWord = (kFrameStampField + 2) / 4;
static_assert((kFrameStampField + 2) % 4 == 0);
FORCE_INLINE uint64_t frame_stamp(const volatile uint32_t* payload) {
    return (static_cast<uint64_t>(__builtin_bswap32(payload[kFrameStampHiWord])) << 32) |
           __builtin_bswap32(payload[kFrameStampHiWord + 1]);
}

// Timestamping on transmit queue Q. arm_two_step(tag) has the MAC push the tag and send time of every packet the queue
// sends into TxStampFifo; arm_in_frame() has it write each packet's send time into the packet. Either stays in effect
// until disarm(). The queue reports idle before the MAC has picked up a packet's setting (measured: disarming at idle
// lost the time of one packet in fifteen), so to stamp only particular packets, disarm once the last one's time is in.
// The queue samples the setting and tag 24 to 32 cycles after a send command is written (measured: armed 24 cycles
// after the command, the packet is still stamped; armed 32 cycles after, it is not), about 60 cycles before the
// packet-start count moves. A keepalive samples them 15 to 73 cycles before its count moves, and one that finds a
// request set is stamped like any other packet.
template <uint32_t Q>
struct TxQueue {
    static_assert(Q < kNumTxq);
    FORCE_INLINE void arm_two_step(uint64_t tag) const {
        wr(txq_reg(Q, kTxqRxTimestampLoOff), static_cast<uint32_t>(tag));
        wr(txq_reg(Q, kTxqRxTimestampHiOff), static_cast<uint32_t>(tag >> 32));
        wr(txq_reg(Q, kTxqTimestampOff), TxqTimestamp{.cmd = TS_CMD_TWO_STEP_FIFO});
    }
    FORCE_INLINE void arm_in_frame() const {
        wr(txq_reg(Q, kTxqTimestampOff), TxqTimestamp{.cmd = TS_CMD_ONE_STEP_ORIGIN, .offset = kFrameStampOffset});
    }
    FORCE_INLINE void disarm() const { wr(txq_reg(Q, kTxqTimestampOff), TxqTimestamp{.cmd = TS_CMD_NOP}); }
    // In 96-byte units on the wire: one for a keepalive or a packet of up to 64 bytes of payload, two up to 128.
    FORCE_INLINE uint32_t words_sent() const { return rd(txq_reg(Q, ETH_TXQ_WORD_CNT)); }
};

// Header row Row, set up as a copy of queue Q's row at boot with the destination address kStampFrameDa. install()
// points the packets software sends on the queue at it, while the packets the hardware generates (the keepalives)
// keep the boot row, so only the packets sent here carry the address. select() switches software packets between this
// row and the boot row, whose packets the receiver's classifier records no time for; a packet picks up its row when
// the queue takes its send command, so switch back only once the queue reports the command taken. restore() puts back
// the row's address and the queue's selection.
template <uint32_t Q, uint32_t Row>
struct TxHeaderRow {
    TxqPktCfgSelSw sel_boot{};
    uint32_t da_prev[2] = {};

    __attribute__((noinline, cold)) void install() {
        sel_boot = rd<TxqPktCfgSelSw>(txq_reg(Q, kTxqPktCfgSelSwOff));
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
        wr(txq_reg(Q, kTxqPktCfgSelSwOff),
           stamped ? TxqPktCfgSelSw{.raw = Row, .reg_write = Row, .packet = Row} : sel_boot);
    }
};

// The receive classifier's receive-time queue. It holds a time for every packet a rule asks to have timed, recorded as
// the packet starts to arrive and before its data is visible in L1, so a packet that has been seen already has its
// time waiting.
struct RxStampFifo {
    static constexpr uint32_t kDepth = 16;
    struct Entry {
        uint64_t ts;
        uint32_t label;
        bool valid;
    };
    // Reads the head entry, then removes it.
    FORCE_INLINE bool pop(Entry& e) const {
        if (rd<RxThStatus>(kRxThStatus).empty) {
            return false;
        }
        const uint32_t lo = rd(kRxThTsLow);
        const uint32_t hi = rd(kRxThTsHigh);
        const RxThTsLabel label = rd<RxThTsLabel>(kRxThTsLabel);
        wr(kRxThStatus, RxThStatus{.pop = 1});
        wr(kRxThStatus, RxThStatus{});
        e.ts = (static_cast<uint64_t>(hi) << 32) | lo;
        e.label = label.label;
        e.valid = label.valid;
        return true;
    }
    // Exactly n entries, and not full at any point since the last pop.
    FORCE_INLINE bool holds_exactly(uint32_t n) const {
        constexpr RxThStatus kCounted{.entries = 0xF, .full = 1};
        return (rd(kRxThStatus) & __builtin_bit_cast(uint32_t, kCounted)) == n;
    }
    FORCE_INLINE void flush() const { wr(kRxThStatus, RxThStatus{.flush = 1}); }
    // Passes each valid time under Label to sink(uint64_t) and discards the rest; returns how many it passed. It takes
    // at most one queue's worth per call, so a stream of stamped packets cannot hold a router's core here.
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

// A match-table row for kStampFrameDa's middle bytes, pointing at a flow-table row whose only action is to record the
// receive time under Label. install() also turns off time recording for packets that match no rule, so the queue
// holds only this rule's times, and empties the queue; remove() restores the no-match actions.
template <uint32_t Row, uint32_t Label>
struct RxStampRule {
    static_assert(Label < 32);  // flow-table labels are five bits
    static constexpr uint32_t kLabel = Label;
    RxFlowActions no_match_prev{};

    __attribute__((noinline, cold)) void install() {
        no_match_prev = rd<RxFlowActions>(kRxFlNoMatchActions);
        RxStampFifo{}.flush();
        RxFlowActions no_match = no_match_prev;
        no_match.record_rx_time = 0;
        wr(kRxFlNoMatchActions, no_match);
        write_pattern(false, 0xA5A5A500u, 0x000000A5u, 0u, {}, {}, {});
        // A mask bit of 1 means don't care: of the whole pattern, only the destination's four 0xA5 bytes are compared.
        write_pattern(
            true,
            0x000000FFu,
            0xFFFFFF00u,
            0xFFFFFFFFu,
            {.augmented_da = 0xF, .augmented_sa = 0xF},
            {.value = 0xFFFF, .augmented = 0xF},
            {.pcp = 7});
        wr(kRxFlTcamRowMappingBase + 4 * Row, RxTcamRowMapping{.priority = 7, .ftable_row = Row});
        write_flow({.record_rx_time = 1}, Label);
        wr(kRxFlTcamRowUpdate, RxTcamRowUpdate{.row = Row, .enable = 1, .write = 1, .go = 1});
    }
    __attribute__((noinline, cold)) void remove() const {
        wr(kRxFlTcamRowUpdate, RxTcamRowUpdate{.row = Row, .write = 1, .go = 1});
        write_flow({}, 0);
        wr(kRxFlNoMatchActions, no_match_prev);
        RxStampFifo{}.flush();
    }

private:
    FORCE_INLINE static void write_pattern(
        bool mask,
        uint32_t da_w0,
        uint32_t da_w1,
        uint32_t rest,
        RxTcamNonIpAddrFlags flags,
        RxTcamEthertype etype,
        RxTcamPriority pri) {
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
           RxTcamUpdate{
               .row = Row,
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
        wr(kRxFlFtableActions, actions);
        wr(kRxFlFtableVlan, 0);
        wr(kRxFlFtableLabels, label);
        wr(kRxFlFtableSwMetadata, 0);
        wr(kRxFlFtableUpdate, RxFtableUpdate{.row = Row, .write = 1, .go = 1});
    }
};

// The MAC's send-time queue, 128 entries deep: the tag and send time of every packet a queue sends under arm_two_step.
struct TxStampFifo {
    // Removes one entry: its tag's low word and the send time (the tag's high word is the caller's own). Word 0 must be
    // read first, since reading it is what advances the queue; an empty queue reads 0xFFFFFFFF there.
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
    FORCE_INLINE bool empty() const { return !rd<MacTxInt>(kMacTxIntRaw).ts_fifo_not_empty; }
    void clear() const {
        uint32_t tag_lo = 0;
        uint64_t ts = 0;
        while (pop(tag_lo, ts)) {
        }
    }
    // Passes each time whose tag's low word is tag_lo to sink(uint64_t) and discards the rest; returns how many it
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
