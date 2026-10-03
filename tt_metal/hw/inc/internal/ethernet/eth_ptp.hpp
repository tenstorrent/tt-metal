// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Blackhole Ethernet IEEE 1588 hardware: the PTP timer, the TX queues' timestamping, the TX header table, the RX
// classifier's stamp rules, and the RX and TX timestamp FIFOs, plus the refclk and ERISC wall clock reads that go with
// them.
//
// These blocks belong to the tile: a second user on the same tile would take the first one's FIFO entries, and
// restarting the PTP timer restarts every user's PTP time. Setup is cold and out of line, so an -O3 caller keeps it out
// of its hot code; per-frame operations are inline.

#pragma once

#if !defined(ARCH_BLACKHOLE)
#error "eth_ptp.hpp is Blackhole only"
#endif

#include <cstdint>
#include <optional>
#include <type_traits>

#include "internal/ethernet/tt_eth_ss_regs.h"
#include "internal/risc_attribs.h"

namespace tt::tt_metal::eth_ptp {

constexpr uint32_t kRefclkHz = 50'000'000u;
constexpr uint32_t kNsPerRefclkTick = 1'000'000'000u / kRefclkHz;

constexpr uint64_t join_words(uint32_t hi, uint32_t lo) { return (static_cast<uint64_t>(hi) << 32) | lo; }

template <typename T>
constexpr uint32_t word_of(const T& value) {
    static_assert(sizeof(T) == sizeof(uint32_t));
    return __builtin_bit_cast(uint32_t, value);
}

// A register bound to its layout. A layout struct names every bit, reserved ones included, so that a value built with
// designated initializers has all its other bits zero.
template <typename T = uint32_t>
struct Reg {
    static_assert(sizeof(T) == sizeof(uint32_t));
    static_assert(std::has_unique_object_representations_v<T>, "a register layout names every bit");
    uint32_t addr;
    FORCE_INLINE T read() const { return __builtin_bit_cast(T, *reinterpret_cast<volatile uint32_t*>(addr)); }
    FORCE_INLINE void write(T value) const { *reinterpret_cast<volatile uint32_t*>(addr) = word_of(value); }
};

// ---------------------------------------------------------------------------------------------------------------------
// The refclk and the ERISC wall clock

// The refclk count lives in the PTP timer block. Reading a counter's low word latches its high word.
constexpr Reg<> kRefclkLo{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_CFR_LO};
constexpr Reg<> kRefclkHi{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_CFR_HI};
// The wall clock's low word latches WALL_CLOCK_1_AT; WALL_CLOCK_1 is live.
constexpr Reg<> kWallClockLo{ETH_RISC_REGS_START + ETH_RISC_WALL_CLOCK_0};
constexpr Reg<> kWallClockHi{ETH_RISC_REGS_START + ETH_RISC_WALL_CLOCK_1_AT};

namespace detail {
FORCE_INLINE uint64_t read_latched(Reg<> lo_reg, Reg<> hi_reg) {
    const uint32_t lo = lo_reg.read();
    return join_words(hi_reg.read(), lo);
}
}  // namespace detail
FORCE_INLINE uint64_t read_refclk() { return detail::read_latched(kRefclkLo, kRefclkHi); }

// The wall clock and the refclk, read together.
struct Instant {
    uint64_t wall, refclk;
};
// The wall clock's low read latches its high word for only a few cycles, and the refclk read between the two takes
// longer, so a low word that wrapped before the high read would tear the pair by 2^32.
FORCE_INLINE Instant read_instant() {
    while (true) {
        const uint32_t wall_hi_before = kWallClockHi.read();
        const uint32_t wall_lo = kWallClockLo.read();
        const uint32_t refclk_lo = kRefclkLo.read();
        const uint32_t wall_hi = kWallClockHi.read();
        const uint32_t refclk_hi = kRefclkHi.read();
        if (wall_hi == wall_hi_before) {
            return {join_words(wall_hi, wall_lo), join_words(refclk_hi, refclk_lo)};
        }
    }
}

// The count the cores see moves 4 refclk ticks at a time, once per 80 ns: it crosses into the tile's clock domain
// through a handshake.
constexpr uint32_t kRefclkTicksPerUpdate = 4;

// The low words of the wall clock and the refclk, read in that order.
struct ClocksLo {
    uint32_t wall, refclk;
};
FORCE_INLINE ClocksLo read_clocks_lo() { return {kWallClockLo.read(), kRefclkLo.read()}; }
// read_clocks_lo() just after the refclk's low word changed.
FORCE_INLINE ClocksLo await_refclk_update() {
    uint32_t prev = kRefclkLo.read();
    while (true) {
        const ClocksLo now = read_clocks_lo();
        if (now.refclk != prev) {
            return now;
        }
        prev = now.refclk;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// The PTP timer

struct PtpTimerCtrl {
    uint32_t enable : 1;
    uint32_t rsvd : 31;
};
// The PTP ns the timer adds per refclk tick, in 8.16 fixed point.
struct PtpTickIncrement {
    uint32_t ns_q8_16 : 24;
    uint32_t rsvd : 8;
};
struct PtpUpdateRequest {
    uint32_t request : 1;
    uint32_t rsvd : 31;
};
struct PtpUpdateStat {
    uint32_t tick_increment_pending : 1;
    uint32_t time_pending : 1;
    uint32_t rsvd0 : 6;
    uint32_t tick_increment_ack : 1;
    uint32_t time_ack : 1;
    uint32_t rsvd1 : 6;
    uint32_t tick_increment_late : 1;
    uint32_t time_late : 1;
    uint32_t rsvd2 : 14;
};
// How many PTP ns the SYNC timers run ahead of the main one.
struct PtpSyncOffset {
    uint32_t ns : 8;
    uint32_t rsvd : 24;
};

constexpr Reg<PtpTimerCtrl> kPtpTimerCtrl{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_CTRL};
// A scheduled update: at refclk count kPtpFutureRefclk*, the timer takes the future tick increment and time that
// kPtpUpdate* request.
constexpr Reg<> kPtpFutureRefclkLo{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_FUTURE_CFR_LO};
constexpr Reg<> kPtpFutureRefclkHi{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_FUTURE_CFR_HI};
constexpr Reg<PtpTickIncrement> kPtpFutureTickIncrement{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_FUTURE_PTI};
constexpr Reg<> kPtpFutureTimeLo{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_FUTURE_TIMESTAMP_LO};
constexpr Reg<> kPtpFutureTimeHi{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_FUTURE_TIMESTAMP_HI};
constexpr Reg<PtpUpdateRequest> kPtpUpdateTickIncrement{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_UPDATE_PTI};
constexpr Reg<PtpUpdateRequest> kPtpUpdateTime{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_UPDATE_TIMESTAMP};
constexpr Reg<PtpUpdateStat> kPtpUpdateStat{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_UPDATE_STAT};
constexpr Reg<PtpSyncOffset> kPtpSyncOffset{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_SYNC_OFFSET1};
constexpr Reg<PtpTickIncrement> kPtpTickIncrementInUse{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_PTI_STAT};
// The PTP time as 32-bit seconds and 32-bit ns, and as 64-bit ns; the SYNC timers' copies run kPtpSyncOffset ahead.
constexpr Reg<> kPtpSecondsNsLo{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_32S_32NS_LO};
constexpr Reg<> kPtpSecondsNsHi{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_32S_32NS_HI};
constexpr Reg<> kPtpNsLo{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_64NS_LO};
constexpr Reg<> kPtpNsHi{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_64NS_HI};
constexpr Reg<> kPtpSyncSecondsNsLo{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_SYNC_32S_32NS_LO};
constexpr Reg<> kPtpSyncSecondsNsHi{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_SYNC_32S_32NS_HI};
constexpr Reg<> kPtpSyncNsLo{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_SYNC_64NS_LO};
constexpr Reg<> kPtpSyncNsHi{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_SYNC_64NS_HI};

FORCE_INLINE uint64_t read_ptp_ns() { return detail::read_latched(kPtpNsLo, kPtpNsHi); }

// Far enough ahead that the scheduled restart is still in the future once all of its setup writes have landed.
constexpr uint32_t kPtpRestartLeadTicks = kRefclkHz / 10'000;

// Restarts the tile's PTP time at 0, counting kNsPerRefclkTick per refclk tick, and returns PTP ns minus refclk ticks *
// kNsPerRefclkTick once the restart has landed. Every PTP stamp the tile takes restarts with it.
__attribute__((noinline, cold)) inline int64_t restart_ptp_timer() {
    kPtpTimerCtrl.write({.enable = 1});
    const uint64_t restart_at = read_refclk() + kPtpRestartLeadTicks;
    kPtpFutureRefclkLo.write(static_cast<uint32_t>(restart_at));
    kPtpFutureRefclkHi.write(static_cast<uint32_t>(restart_at >> 32));
    kPtpFutureTickIncrement.write({.ns_q8_16 = kNsPerRefclkTick << 16});
    kPtpFutureTimeLo.write(0);
    kPtpFutureTimeHi.write(0);
    kPtpUpdateTickIncrement.write({.request = 1});
    kPtpUpdateTime.write({.request = 1});
    PtpUpdateStat stat{};
    while (!(stat.tick_increment_ack && stat.time_ack)) {
        stat = kPtpUpdateStat.read();
    }
    kPtpUpdateTickIncrement.write({});
    kPtpUpdateTime.write({});
    // The restart lands one tick after the scheduled one; a stamp taken before then is on the old PTP time.
    const uint64_t landed_at = restart_at + 1;
    while (read_refclk() <= landed_at) {
    }
    return -static_cast<int64_t>(landed_at) * kNsPerRefclkTick;
}

// ---------------------------------------------------------------------------------------------------------------------
// TX queues: which packets get a send-time stamp, and how

enum class StampMode : uint32_t { None = 0, OneStepCorrection = 1, OneStepInFrame = 2, TwoStepFifo = 3 };

// Where a one-step stamp goes in the frame, in the 2-byte units the queue's offset setting counts.
constexpr uint32_t kFrameStampOffsetSetting = 2;

struct TxqStampControl {
    StampMode mode : 3;
    uint32_t rsvd0 : 13;
    uint32_t in_frame_offset : 6;  // in 2-byte units
    uint32_t rsvd1 : 10;
};

// The TX header table row each kind of software-issued packet uses.
struct TxqHeaderSelect {
    uint32_t raw : 4;
    uint32_t reg_write : 4;
    uint32_t packet : 4;
    uint32_t rsvd : 20;
};

constexpr Reg<TxqHeaderSelect> txq_header_select(uint32_t queue) {
    return {ETH_TXQ0_REGS_START + queue * ETH_TXQ_REGS_SIZE + ETH_TXQ_TXPKT_CFG_SEL_SW};
}
// The queue samples its stamp control 24 to 32 cycles after a send command (measured), so arming just after the
// command still covers that packet. It reports idle before the MAC has taken the setting, though, so to stamp only
// chosen packets, disarm once the last one's time is in.
constexpr Reg<TxqStampControl> txq_stamp_control(uint32_t queue) {
    return {ETH_TXQ0_REGS_START + queue * ETH_TXQ_REGS_SIZE + ETH_TXQ_TIMESTAMP};
}
// The tag a two-step stamp carries into the TX timestamp FIFO, which returns its low word.
constexpr Reg<> txq_two_step_tag_lo(uint32_t queue) {
    return {ETH_TXQ0_REGS_START + queue * ETH_TXQ_REGS_SIZE + ETH_TXQ_RX_TIMESTAMP_LO};
}
constexpr Reg<> txq_two_step_tag_hi(uint32_t queue) {
    return {ETH_TXQ0_REGS_START + queue * ETH_TXQ_REGS_SIZE + ETH_TXQ_RX_TIMESTAMP_HI};
}
// In 96-byte units on the wire, not the 16-byte words ETH_TXQ_WORD_CNT's description says (measured): one for a
// keepalive or a packet of up to 64 bytes of payload, two up to 128.
constexpr Reg<> txq_word_count(uint32_t queue) {
    return {ETH_TXQ0_REGS_START + queue * ETH_TXQ_REGS_SIZE + ETH_TXQ_WORD_CNT};
}

// `tag` must not be tx_stamp_fifo::kEmptyTag, which an empty FIFO reads as.
FORCE_INLINE void txq_arm_two_step(uint32_t queue, uint32_t tag) {
    txq_two_step_tag_lo(queue).write(tag);
    txq_stamp_control(queue).write(TxqStampControl{.mode = StampMode::TwoStepFifo});
}
FORCE_INLINE void txq_arm_in_frame(uint32_t queue) {
    txq_stamp_control(queue).write(
        TxqStampControl{.mode = StampMode::OneStepInFrame, .in_frame_offset = kFrameStampOffsetSetting});
}
FORCE_INLINE void txq_disarm(uint32_t queue) {
    txq_stamp_control(queue).write(TxqStampControl{.mode = StampMode::None});
}

// A one-step stamp is 16 high bits and then the 64-bit PTP time in ns, big-endian. It starts 2 bytes past the queue's
// offset setting (measured). Keepalives are stamped like any other packet: at offset setting 2, the only one validated,
// the stamp lands in their padding, but at 40, past its end, every keepalive was lost.
constexpr uint32_t kFrameStampOffsetBytes = 2 * kFrameStampOffsetSetting + 2;
constexpr uint32_t kFrameStampBytes = 10;
constexpr uint32_t kFrameStampTimeOffsetBytes = kFrameStampOffsetBytes + 2;  // past the 16 high bits
static_assert(kFrameStampTimeOffsetBytes % sizeof(uint32_t) == 0);
constexpr uint32_t kFrameStampTimeWord = kFrameStampTimeOffsetBytes / sizeof(uint32_t);

// The start of a frame, where the MAC writes its one-step stamp. A type of its own, so that a store to it doesn't make
// the compiler reload a client's own uint32_t members.
struct FrameStampSlot {
    uint32_t words[(kFrameStampOffsetBytes + kFrameStampBytes + 3) / sizeof(uint32_t)];
};

FORCE_INLINE uint64_t frame_stamp_ns(const volatile FrameStampSlot& slot) {
    return join_words(
        __builtin_bswap32(slot.words[kFrameStampTimeWord]), __builtin_bswap32(slot.words[kFrameStampTimeWord + 1]));
}
// A frame the MAC didn't stamp carries whatever its slot held.
FORCE_INLINE void clear_frame_stamp(volatile FrameStampSlot& slot) {
    slot.words[kFrameStampTimeWord] = 0;
    slot.words[kFrameStampTimeWord + 1] = 0;
}

// ---------------------------------------------------------------------------------------------------------------------
// The TX header table

constexpr uint32_t kTxHeaderRows = 10;

// The firmware points each TX queue at the header row of the same number, rows 0 to 2, sends only on queue 0, and
// leaves rows 3 to 9 free. install() points Queue at Row, a copy of row Queue except for the destination `da`;
// restore() points it back. restore() is only valid after install().
template <uint32_t Queue, uint32_t Row>
class TxHeaderRow {
public:
    static_assert(Queue < NUM_ETH_QUEUES && Row >= NUM_ETH_QUEUES && Row < kTxHeaderRows);

    __attribute__((noinline, cold)) void install(uint64_t da) {
        const auto field = [](uint32_t row, uint32_t offset) {
            return Reg<>{ETH_TXPKT_CFG_REGS_START + row * ETH_TXPKT_CFG_REGS_SIZE + offset};
        };
        select_before_ = txq_header_select(Queue).read();
        for (uint32_t reg :
             {ETH_TXPKT_CFG_INSERT_CTL,
              ETH_TXPKT_CFG_CUSTOM_HDR,
              ETH_TXPKT_CFG_MAC_SA_LO,
              ETH_TXPKT_CFG_MAC_SA_HI,
              ETH_TXPKT_CFG_ETHERTYPE,
              ETH_TXPKT_CFG_VLAN1,
              ETH_TXPKT_CFG_VLAN2}) {
            field(Row, reg).write(field(Queue, reg).read());
        }
        field(Row, ETH_TXPKT_CFG_MAC_DA_LO).write(static_cast<uint32_t>(da));
        field(Row, ETH_TXPKT_CFG_MAC_DA_HI).write(static_cast<uint32_t>(da >> 32));
        txq_header_select(Queue).write({.raw = Row, .reg_write = Row, .packet = Row});
    }
    __attribute__((noinline, cold)) void restore() const { txq_header_select(Queue).write(select_before_); }

private:
    TxqHeaderSelect select_before_{};
};

// ---------------------------------------------------------------------------------------------------------------------
// The RX classifier: TCAM rows match frames, and each row's flow-table row says what to do with them

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
constexpr uint32_t kRxTcamTopPriority = 7;
// Flow-table row kRxTcamRows is the no-match row.
constexpr uint32_t kRxTcamRows = 64;

struct RxTcamRowUpdate {
    uint32_t row : 6;
    uint32_t rsvd0 : 2;
    uint32_t enable : 1;
    uint32_t rsvd1 : 7;
    uint32_t write : 1;
    uint32_t rsvd2 : 14;
    uint32_t go : 1;
};

enum class RxTcamTupleType : uint32_t { NonIp = 0 };

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

struct RxFtableLabel {
    uint32_t label : 5;
    uint32_t rsvd : 27;
};

struct RxFtableUpdate {
    uint32_t row : 6;
    uint32_t rsvd0 : 2;
    uint32_t write : 1;
    uint32_t rsvd1 : 22;
    uint32_t go : 1;
};

constexpr Reg<RxFlowActions> kRxNoMatchActions{ETH_RX_CLASSIFIER_REGS_START + ETH_RX_CLASSIFIER_NO_MATCH_ACTIONS};
constexpr Reg<RxTcamRowMapping> rx_tcam_row_mapping(uint32_t row) {
    return {ETH_RX_CLASSIFIER_REGS_START + ETH_RX_CLASSIFIER_TCAM_ROW_MAPPING + row * sizeof(uint32_t)};
}
constexpr Reg<RxTcamRowUpdate> kRxTcamRowUpdate{ETH_RX_CLASSIFIER_REGS_START + ETH_RX_CLASSIFIER_TCAM_ROW_UPDATE};
constexpr Reg<RxTcamTupleType> kRxTcamTupleTypeWrite{
    ETH_RX_CLASSIFIER_REGS_START + ETH_RX_CLASSIFIER_TCAM_TUPLE_TYPE_WRITE};
constexpr Reg<> rx_tcam_sa_write(uint32_t word) {
    return {ETH_RX_CLASSIFIER_REGS_START + ETH_RX_CLASSIFIER_TCAM_SA_WRITE + word * sizeof(uint32_t)};
}
constexpr Reg<> rx_tcam_da_write(uint32_t word) {
    return {ETH_RX_CLASSIFIER_REGS_START + ETH_RX_CLASSIFIER_TCAM_DA_WRITE + word * sizeof(uint32_t)};
}
constexpr Reg<RxTcamNonIpAddrFlags> kRxTcamNonIpAddrFlagsWrite{
    ETH_RX_CLASSIFIER_REGS_START + ETH_RX_CLASSIFIER_TCAM_NON_IP_ADDR_FLAGS_WRITE};
constexpr Reg<RxTcamEthertype> kRxTcamEthertypeWrite{
    ETH_RX_CLASSIFIER_REGS_START + ETH_RX_CLASSIFIER_TCAM_ETHERTYPE_WRITE};
constexpr Reg<RxTcamPriority> kRxTcamPriorityWrite{
    ETH_RX_CLASSIFIER_REGS_START + ETH_RX_CLASSIFIER_TCAM_PRIORITY_WRITE};
constexpr Reg<RxTcamUpdate> kRxTcamUpdate{ETH_RX_CLASSIFIER_REGS_START + ETH_RX_CLASSIFIER_TCAM_UPDATE};
constexpr Reg<RxFtableLabel> kRxFtableLabel{ETH_RX_CLASSIFIER_REGS_START + ETH_RX_CLASSIFIER_FTABLE_LABELS};
constexpr Reg<RxFlowActions> kRxFtableActions{ETH_RX_CLASSIFIER_REGS_START + ETH_RX_CLASSIFIER_FTABLE_ACTIONS};
constexpr Reg<> kRxFtableVlan{ETH_RX_CLASSIFIER_REGS_START + ETH_RX_CLASSIFIER_FTABLE_VLAN};
constexpr Reg<> kRxFtableSwMetadata{ETH_RX_CLASSIFIER_REGS_START + ETH_RX_CLASSIFIER_FTABLE_SW_METADATA};
constexpr Reg<RxFtableUpdate> kRxFtableUpdate{ETH_RX_CLASSIFIER_REGS_START + ETH_RX_CLASSIFIER_FTABLE_UPDATE};

// A non-IP TCAM pattern. The address fields are four words wide, the classifier's IPv6 width. The firmware sends only
// broadcast, 01:00:.. multicast and 02:00:.. unicast frames.
struct RxTcamNonIpPattern {
    uint32_t sa[4];
    uint32_t da[4];
    RxTcamNonIpAddrFlags addr_flags;
    RxTcamEthertype ethertype;
    RxTcamPriority priority;
};

// ---------------------------------------------------------------------------------------------------------------------
// The RX timestamp FIFO: each entry is a received frame's arrival time and the label of the rule that recorded it

struct RxStampLabel {
    uint32_t label : 5;
    uint32_t recorded_by_rule : 1;  // set in every entry a rule records (measured), so not part of the label
    uint32_t rsvd : 25;
    uint32_t valid : 1;
};
struct RxStampStatus {
    uint32_t entries : 4;
    uint32_t rsvd0 : 12;
    uint32_t full : 1;
    uint32_t nearly_full : 1;
    uint32_t empty : 1;
    uint32_t rsvd1 : 13;
};
// The same register as RxStampStatus, written.
struct RxStampCommand {
    uint32_t rsvd : 30;
    uint32_t flush : 1;
    uint32_t pop : 1;
};

constexpr Reg<> kRxStampTimeLo{ETH_RX_TH_REGS_START + ETH_RX_TH_TS_LOW};
constexpr Reg<> kRxStampTimeHi{ETH_RX_TH_REGS_START + ETH_RX_TH_TS_HIGH};
constexpr Reg<RxStampLabel> kRxStampLabel{ETH_RX_TH_REGS_START + ETH_RX_TH_TS_LABEL};
constexpr Reg<RxStampStatus> kRxStampStatus{ETH_RX_TH_REGS_START + ETH_RX_TH_STATUS};
constexpr Reg<RxStampCommand> kRxStampCommand{ETH_RX_TH_REGS_START + ETH_RX_TH_STATUS};

// A frame's time is recorded as it starts to arrive, so once its data is visible in L1, its time is already waiting. A
// frame the link sends again (a Go-back-N resend) is recorded again, though the receiver then drops it as a duplicate.
namespace rx_stamp_fifo {
constexpr uint32_t kDepth = 16;
// Removes the head entry, which must exist, and returns its arrival time. kRxStampLabel holds the head's label until
// then. Removing it takes a pop of 1, then of 0 (measured).
FORCE_INLINE uint64_t pop() {
    const uint32_t lo = kRxStampTimeLo.read();
    const uint32_t hi = kRxStampTimeHi.read();
    kRxStampCommand.write({.pop = 1});
    kRxStampCommand.write({});
    return join_words(hi, lo);
}
template <uint32_t Count>
FORCE_INLINE bool holds_exactly() {
    static_assert(0 < Count && Count < kDepth, "a full FIFO reads as 0 entries");
    return kRxStampStatus.read().entries == Count;
}
// Takes effect on the write of 1 alone (measured).
FORCE_INLINE void flush() { kRxStampCommand.write({.flush = 1}); }
}  // namespace rx_stamp_fifo

// Records the arrival time of every frame that matches `values` where `mask` is 0, under kLabel, with TCAM row Row
// and its flow-table row; matched frames are otherwise handled as unmatched ones are. remove() is only valid after
// install().
template <uint32_t Row, uint32_t Label>
class RxStampRule {
public:
    static_assert(Row < kRxTcamRows);
    static_assert(Label < 32);  // flow-table labels are five bits
    static constexpr uint32_t kLabel = Label;

    // Also stops unmatched frames recording their times, then flushes the FIFO.
    __attribute__((noinline, cold)) void install(const RxTcamNonIpPattern& values, const RxTcamNonIpPattern& mask) {
        no_match_before_ = kRxNoMatchActions.read();
        RxFlowActions actions = no_match_before_;
        actions.record_rx_time = 0;
        kRxNoMatchActions.write(actions);
        rx_stamp_fifo::flush();
        write_tcam_row(values, false);
        write_tcam_row(mask, true);
        rx_tcam_row_mapping(Row).write({.priority = kRxTcamTopPriority, .ftable_row = Row});
        actions.record_rx_time = 1;
        write_flow(actions, Label);
        kRxTcamRowUpdate.write({.row = Row, .enable = 1, .write = 1, .go = 1});
    }
    // Disables the row, empties its flow-table row, flushes the FIFO and restores the unmatched frames' actions.
    __attribute__((noinline, cold)) void remove() const {
        kRxTcamRowUpdate.write({.row = Row, .write = 1, .go = 1});
        write_flow({}, 0);
        rx_stamp_fifo::flush();
        kRxNoMatchActions.write(no_match_before_);
    }

private:
    FORCE_INLINE static void write_tcam_row(const RxTcamNonIpPattern& pattern, bool mask) {
        kRxTcamTupleTypeWrite.write(RxTcamTupleType::NonIp);
        kRxTcamEthertypeWrite.write(pattern.ethertype);
        kRxTcamPriorityWrite.write(pattern.priority);
        for (uint32_t word = 0; word < 4; word++) {
            rx_tcam_sa_write(word).write(pattern.sa[word]);
        }
        for (uint32_t word = 0; word < 4; word++) {
            rx_tcam_da_write(word).write(pattern.da[word]);
        }
        kRxTcamNonIpAddrFlagsWrite.write(pattern.addr_flags);
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
        kRxFtableLabel.write({.label = label});
        kRxFtableSwMetadata.write(0);
        kRxFtableUpdate.write({.row = Row, .write = 1, .go = 1});
    }

    RxFlowActions no_match_before_{};
};

// ---------------------------------------------------------------------------------------------------------------------
// The MAC: send-time stamping, and the TX timestamp FIFO a two-step stamp lands in

struct MacTxCfg {
    uint32_t rsvd0 : 1;
    uint32_t tx_enable : 1;
    uint32_t rsvd1 : 2;
    uint32_t stamp_fifo_every_packet : 1;  // 0: only packets armed for a two-step stamp
    uint32_t rsvd2 : 6;
    uint32_t origin_timestamp_mode : 1;
    uint32_t rsvd3 : 20;
};
struct MacTxInt {
    uint32_t rsvd0 : 2;
    uint32_t stamp_fifo_full : 1;
    uint32_t stamp_fifo_not_empty : 1;
    uint32_t stamp_offset_error : 1;
    uint32_t rsvd1 : 27;
};

// PTP ns added to every send-time stamp.
struct MacTxDelay {
    uint32_t ns : 16;
    uint32_t rsvd : 16;
};

constexpr Reg<MacTxCfg> kMacTxCfg{ETH_MAC_REGS_START + ETH_MAC_TX_CFG};
constexpr Reg<MacTxDelay> kMacTxDelay{ETH_MAC_REGS_START + ETH_MAC_TX_DELAY};
constexpr Reg<MacTxInt> kMacTxInt{ETH_MAC_REGS_START + ETH_MAC_TX_INT};
constexpr Reg<MacTxInt> kMacTxIntRaw{ETH_MAC_REGS_START + ETH_MAC_TX_INT_RAW};
constexpr Reg<> kMacTxStampFifoFullThreshold{ETH_MAC_REGS_START + ETH_MAC_TS_FIFO_FULL_THRESH};
// An entry's four words are its tag's low and high words and then its time's, not the order the vendor's guide gives
// (measured). Reading the first pops the entry, and an empty FIFO reads all ones there.
constexpr Reg<> kMacTxStampTagLo{ETH_MAC_REGS_START + ETH_MAC_TS_FIFO_0};
constexpr Reg<> kMacTxStampTagHi{ETH_MAC_REGS_START + ETH_MAC_TS_FIFO_1};
constexpr Reg<> kMacTxStampTimeLo{ETH_MAC_REGS_START + ETH_MAC_TS_FIFO_2};
constexpr Reg<> kMacTxStampTimeHi{ETH_MAC_REGS_START + ETH_MAC_TS_FIFO_3};

namespace tx_stamp_fifo {
constexpr uint32_t kEmptyTag = 0xFFFFFFFFu;
struct Entry {
    uint32_t tag;
    uint64_t time_ns;
};
FORCE_INLINE std::optional<Entry> pop() {
    const uint32_t tag = kMacTxStampTagLo.read();
    if (tag == kEmptyTag) {
        return std::nullopt;
    }
    const uint32_t time_lo = kMacTxStampTimeLo.read();
    return Entry{.tag = tag, .time_ns = join_words(kMacTxStampTimeHi.read(), time_lo)};
}
FORCE_INLINE bool empty() { return !kMacTxIntRaw.read().stamp_fifo_not_empty; }
__attribute__((noinline, cold)) inline void clear() {
    while (pop()) {
    }
}
}  // namespace tx_stamp_fifo

}  // namespace tt::tt_metal::eth_ptp
