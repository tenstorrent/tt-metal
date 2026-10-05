// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Register access for the Blackhole Ethernet IEEE 1588 (PTP) hardware, and reads of the refclk, the 50 MHz clock that
// drives the PTP timer of every Ethernet tile on the chip. Its frequency is fixed, unlike the AICLK the cores run on.
//
// The MAC stamps frames with the tile's PTP time. When a TX queue is set to stamp, the MAC writes each frame's send
// time either into the frame itself or into the TX timestamp FIFO. On receive, the RX classifier looks each frame up in
// a TCAM, a table of patterns. Each TCAM row has a flow-table row that says what happens to the frames it matches, such
// as which RX queue they go to and whether their arrival times go into the RX timestamp FIFO, along with a label the
// row sets. The TX header table holds the headers that outgoing frames carry. Setting a frame's destination address
// there lets a TCAM row on the receiving tile pick that frame out.
//
// Only one kernel per tile may use this hardware, because once a kernel removes a FIFO entry no other kernel sees it,
// and restarting the PTP timer changes the PTP time for the whole tile.

#pragma once

#if !defined(ARCH_BLACKHOLE)
#error "eth_ptp.hpp is Blackhole only"
#endif

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <type_traits>

#include "internal/ethernet/tt_eth_ss_regs.h"
#include "internal/risc_attribs.h"

//////////////////////////////
// TX queue registers for choosing the header row and for stamping, as offsets within each queue's register block
#define ETH_TXQ_TXPKT_CFG_SEL_SW 0x80
#define ETH_TXQ_TIMESTAMP 0x90
#define ETH_TXQ_RX_TIMESTAMP_LO 0x94
#define ETH_TXQ_RX_TIMESTAMP_HI 0x98

//////////////////////////////
// eth_ctrl TX header table, row i at ETH_TXPKT_CFG_REGS_START + i * ETH_TXPKT_CFG_REGS_SIZE
#define ETH_TXPKT_CFG_REGS_START 0xFFB98200
#define ETH_TXPKT_CFG_REGS_SIZE 0x80
#define ETH_TXPKT_CFG_INSERT_CTL 0x00
#define ETH_TXPKT_CFG_CUSTOM_HDR 0x04
#define ETH_TXPKT_CFG_MAC_SA_LO 0x10
#define ETH_TXPKT_CFG_MAC_SA_HI 0x14
#define ETH_TXPKT_CFG_MAC_DA_LO 0x18
#define ETH_TXPKT_CFG_MAC_DA_HI 0x1C
#define ETH_TXPKT_CFG_ETHERTYPE 0x20
#define ETH_TXPKT_CFG_VLAN1 0x24
#define ETH_TXPKT_CFG_VLAN2 0x28

//////////////////////////////
// eth_ctrl PTP timer
#define ETH_PTP_TIMER_REGS_START 0xFFB98800
#define ETH_PTP_TIMER_CTRL 0x00
#define ETH_PTP_TIMER_FUTURE_CFR_LO 0x04
#define ETH_PTP_TIMER_FUTURE_CFR_HI 0x08
#define ETH_PTP_TIMER_FUTURE_PTI 0x0C
#define ETH_PTP_TIMER_FUTURE_TIMESTAMP_LO 0x10
#define ETH_PTP_TIMER_FUTURE_TIMESTAMP_HI 0x14
#define ETH_PTP_TIMER_UPDATE_PTI 0x20
#define ETH_PTP_TIMER_UPDATE_TIMESTAMP 0x24
#define ETH_PTP_TIMER_UPDATE_STAT 0x40
#define ETH_PTP_TIMER_CFR_LO 0x50
#define ETH_PTP_TIMER_CFR_HI 0x54
#define ETH_PTP_TIMER_64NS_LO 0x60
#define ETH_PTP_TIMER_64NS_HI 0x64

//////////////////////////////
// RX classifier TCAM and flow table
#define ETH_RX_CLASSIFIER_REGS_START 0xFFB9C000
#define ETH_RX_CLASSIFIER_TCAM_ROW_MAPPING 0xC00
#define ETH_RX_CLASSIFIER_NO_MATCH_ACTIONS 0xD04
#define ETH_RX_CLASSIFIER_TCAM_ROW_UPDATE 0xD40
#define ETH_RX_CLASSIFIER_TCAM_TUPLE_TYPE_WRITE 0xD80
#define ETH_RX_CLASSIFIER_TCAM_SA_WRITE 0xD90
#define ETH_RX_CLASSIFIER_TCAM_DA_WRITE 0xDA0
#define ETH_RX_CLASSIFIER_TCAM_NON_IP_ADDR_FLAGS_WRITE 0xDB0
#define ETH_RX_CLASSIFIER_TCAM_ETHERTYPE_WRITE 0xDC0
#define ETH_RX_CLASSIFIER_TCAM_PRIORITY_WRITE 0xDC4
#define ETH_RX_CLASSIFIER_TCAM_UPDATE 0xDF0
#define ETH_RX_CLASSIFIER_FTABLE_LABELS 0xE80
#define ETH_RX_CLASSIFIER_FTABLE_ACTIONS 0xE84
#define ETH_RX_CLASSIFIER_FTABLE_VLAN 0xE88
#define ETH_RX_CLASSIFIER_FTABLE_SW_METADATA 0xE8C
#define ETH_RX_CLASSIFIER_FTABLE_UPDATE 0xEA0

//////////////////////////////
// RX classifier timestamp FIFO
#define ETH_RX_TH_REGS_START 0xFFB9D800
#define ETH_RX_TH_TS_LOW 0x00
#define ETH_RX_TH_TS_HIGH 0x04
#define ETH_RX_TH_TS_LABEL 0x08
#define ETH_RX_TH_STATUS 0x10

//////////////////////////////
// RSm410 MAC TX config, interrupts and TX timestamp FIFO
#define ETH_MAC_REGS_START 0xFFBA0000
#define ETH_MAC_TX_CFG 0x2200
#define ETH_MAC_TX_DELAY 0x2218
#define ETH_MAC_TX_INT 0x2288
#define ETH_MAC_TX_INT_RAW 0x2290
#define ETH_MAC_TS_FIFO_FULL_THRESH 0x2300
#define ETH_MAC_TS_FIFO_0 0x2E00
#define ETH_MAC_TS_FIFO_1 0x2E04
#define ETH_MAC_TS_FIFO_2 0x2E08
#define ETH_MAC_TS_FIFO_3 0x2E0C

namespace eth_ptp {

constexpr uint32_t kRefclkHz = 50'000'000u;
constexpr uint32_t kNsPerRefclkTick = 1'000'000'000u / kRefclkHz;

constexpr uint64_t join_words(uint32_t hi, uint32_t lo) { return (static_cast<uint64_t>(hi) << 32) | lo; }

template <typename T>
constexpr uint32_t word_of(const T& value) {
    static_assert(sizeof(T) == sizeof(uint32_t));
    return __builtin_bit_cast(uint32_t, value);
}

// A memory-mapped register whose bits are laid out by T.
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

// The PTP timer block's count of refclk ticks since power-on. Reading the low word of any PTP timer counter holds its
// high word until that low word is read again, so the low word must be read first.
constexpr Reg<> kRefclkLo{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_CFR_LO};
constexpr Reg<> kRefclkHi{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_CFR_HI};
// The ERISC wall clock, the core's 64-bit count of AICLK cycles. Reading its low word holds the high word in
// WALL_CLOCK_1_AT, which kWallClockHi reads, but only for the core's very next load (measured). After that,
// kWallClockHi reads the live high word.
constexpr Reg<> kWallClockLo{ETH_RISC_REGS_START + ETH_RISC_WALL_CLOCK_0};
constexpr Reg<> kWallClockHi{ETH_RISC_REGS_START + ETH_RISC_WALL_CLOCK_1_AT};

FORCE_INLINE uint64_t read_latched(Reg<> lo_reg, Reg<> hi_reg) {
    const uint32_t lo = lo_reg.read();
    return join_words(hi_reg.read(), lo);
}
FORCE_INLINE uint64_t read_refclk() { return read_latched(kRefclkLo, kRefclkHi); }

// The wall clock and the refclk, read together.
struct Instant {
    uint64_t wall, refclk;
};
// Reads the wall clock's high word before and after its low word, and retries if they differ, since the low word may
// have wrapped in between.
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

// As the cores read it, the refclk count advances by 4 every 80 ns, because it crosses into the tile's clock domain
// through a handshake.
constexpr uint32_t kRefclkTicksPerUpdate = 4;

struct ClocksLo {
    uint32_t wall, refclk;
};
// Waits for the refclk's next update and returns both clocks' low words from the first loop iteration that sees it, so
// the wall clock reading is within one iteration of the update.
FORCE_INLINE ClocksLo await_refclk_update() {
    const uint32_t before = kRefclkLo.read();
    while (true) {
        const ClocksLo now{kWallClockLo.read(), kRefclkLo.read()};
        if (now.refclk != before) {
            return now;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// The PTP timer

struct PtpTimerCtrl {
    uint32_t enable : 1;
    uint32_t rsvd : 31;
};
// PTP nanoseconds added per refclk tick, in 8.16 fixed point.
struct PtpTickIncrement {
    uint32_t ns_fixed_point : 24;
    uint32_t rsvd : 8;
};
constexpr uint32_t kPtpTickIncrementFractionBits = 16;
// Changing `request` from 0 to 1 schedules a load into the timer for when the refclk count reaches
// kPtpFutureRefclkLo/Hi. In kPtpUpdateTickIncrement it loads kPtpFutureTickIncrement, and in kPtpUpdateTime it loads
// kPtpFutureTimeSeconds/Ns. Clearing `request` clears that load's ack and late flags, ready for the next request.
struct PtpUpdateRequest {
    uint32_t request : 1;
    uint32_t rsvd : 31;
};
// The progress of the two scheduled loads, the tick increment's and the time's. Each is pending until the refclk count
// reaches the scheduled count, acked once the timer has loaded its value, and late if the scheduled count had already
// passed when it was requested.
struct PtpUpdateStatus {
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

constexpr Reg<PtpTimerCtrl> kPtpTimerCtrl{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_CTRL};
// The refclk count at which the scheduled loads happen, and the values they load.
constexpr Reg<> kPtpFutureRefclkLo{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_FUTURE_CFR_LO};
constexpr Reg<> kPtpFutureRefclkHi{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_FUTURE_CFR_HI};
constexpr Reg<PtpTickIncrement> kPtpFutureTickIncrement{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_FUTURE_PTI};
constexpr Reg<> kPtpFutureTimeNs{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_FUTURE_TIMESTAMP_LO};
constexpr Reg<> kPtpFutureTimeSeconds{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_FUTURE_TIMESTAMP_HI};
constexpr Reg<PtpUpdateRequest> kPtpUpdateTickIncrement{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_UPDATE_PTI};
constexpr Reg<PtpUpdateRequest> kPtpUpdateTime{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_UPDATE_TIMESTAMP};
constexpr Reg<PtpUpdateStatus> kPtpUpdateStatus{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_UPDATE_STAT};
// The tile's PTP time in ns, which the MAC stamps frames with.
constexpr Reg<> kPtpNsLo{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_64NS_LO};
constexpr Reg<> kPtpNsHi{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_64NS_HI};

FORCE_INLINE uint64_t read_ptp_ns() { return read_latched(kPtpNsLo, kPtpNsHi); }

// The restart is scheduled 20 us ahead. That is far longer than its dozen register writes take, so the scheduled time
// is still in the future when they finish.
constexpr uint32_t kPtpRestartLeadTicks = 20'000 / kNsPerRefclkTick;

// Restarts the tile's PTP time so it reads the refclk count times kNsPerRefclkTick, and returns once the restart has
// taken effect. Every stamp the tile takes afterwards uses that time. Before the restart, the PTP time has no fixed
// relation to the refclk count, because it counts only while the timer is enabled, and its reset increment is meant for
// a 27 MHz clock.
__attribute__((noinline, cold)) inline void restart_ptp_timer() {
    kPtpTimerCtrl.write({.enable = 1});
    const uint64_t restart_at = read_refclk() + kPtpRestartLeadTicks;
    // The restart takes effect one tick after the scheduled time. Stamps taken before then still use the old PTP time.
    const uint64_t landed_at = restart_at + 1;
    // The timer loads a time as whole seconds and nanoseconds. They are split by shift-and-subtract because a 64-bit
    // division, or a 64-bit shift by a variable amount, would cost far more code size.
    uint64_t nanoseconds = landed_at * kNsPerRefclkTick;
    uint32_t seconds = 0;
    uint64_t chunk = uint64_t{1'000'000'000} << 31;
    for (uint32_t bit = 1u << 31; bit != 0; bit >>= 1, chunk >>= 1) {
        if (nanoseconds >= chunk) {
            nanoseconds -= chunk;
            seconds |= bit;
        }
    }
    kPtpFutureRefclkLo.write(static_cast<uint32_t>(restart_at));
    kPtpFutureRefclkHi.write(static_cast<uint32_t>(restart_at >> 32));
    kPtpFutureTickIncrement.write({.ns_fixed_point = kNsPerRefclkTick << kPtpTickIncrementFractionBits});
    kPtpFutureTimeNs.write(static_cast<uint32_t>(nanoseconds));
    kPtpFutureTimeSeconds.write(seconds);
    kPtpUpdateTickIncrement.write({.request = 1});
    kPtpUpdateTime.write({.request = 1});
    PtpUpdateStatus status;
    do {
        status = kPtpUpdateStatus.read();
    } while (!(status.tick_increment_ack && status.time_ack));
    kPtpUpdateTickIncrement.write({});
    kPtpUpdateTime.write({});
    while (read_refclk() <= landed_at) {
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// TX queue timestamping

// What the MAC does with a frame's send time. OneStepInFrame writes it into the frame, and TwoStepFifo puts it in the
// TX timestamp FIFO. OneStepCorrection is for a switch that forwards PTP frames, and adds the time the frame spent in
// the switch to the correction field of its PTP header.
enum class StampMode : uint32_t { None = 0, OneStepCorrection = 1, OneStepInFrame = 2, TwoStepFifo = 3 };

// The queue's offset setting for one-step stamps, in 2-byte units. The MAC writes the stamp starting 2 bytes past that
// offset (measured). The MAC also stamps the keepalive frames the queue sends on its own, and with the setting at 2
// their stamp lands in their padding rather than past their end, where it would corrupt them.
constexpr uint32_t kFrameStampOffsetSetting = 2;

// How a queue stamps the frames it sends.
struct TxqStampControl {
    StampMode mode : 3;
    uint32_t rsvd0 : 13;
    uint32_t in_frame_offset : 6;
    uint32_t rsvd1 : 10;
};

// The TX header row a queue's frames take their header from, for each kind of send the ERISC issues on it. Frames the
// queue generates itself, its sequence-number updates and keepalives, take theirs from the row its TXPKT_CFG_SEL_HW
// register selects instead.
struct TxqHeaderSelect {
    uint32_t raw : 4;
    uint32_t reg_write : 4;
    uint32_t packet : 4;
    uint32_t rsvd : 20;
};

template <typename T = uint32_t>
constexpr Reg<T> txq_reg(uint32_t queue, uint32_t offset) {
    return {ETH_TXQ0_REGS_START + queue * ETH_TXQ_REGS_SIZE + offset};
}

constexpr Reg<TxqHeaderSelect> txq_header_select(uint32_t queue) {
    return txq_reg<TxqHeaderSelect>(queue, ETH_TXQ_TXPKT_CFG_SEL_SW);
}
// The queue reads its stamp control 24 to 32 cycles after a send command (measured), so a change made just after a send
// still applies to that packet. The queue also reports idle before the MAC has read the setting, so change it only
// after the last stamped packet's timestamp has arrived, not when the queue goes idle.
constexpr Reg<TxqStampControl> txq_stamp_control(uint32_t queue) {
    return txq_reg<TxqStampControl>(queue, ETH_TXQ_TIMESTAMP);
}
// The tag, a number the queue stores with each two-step stamp in the TX timestamp FIFO so that software can tell which
// frame the stamp belongs to. txq_arm_two_step() and tx_stamp_fifo::pop() use only its low word.
constexpr Reg<> txq_two_step_tag_lo(uint32_t queue) { return txq_reg(queue, ETH_TXQ_RX_TIMESTAMP_LO); }
constexpr Reg<> txq_two_step_tag_hi(uint32_t queue) { return txq_reg(queue, ETH_TXQ_RX_TIMESTAMP_HI); }
// How many words the queue has sent, as a running count. A word is 96 bytes on the wire, not 16 bytes as the
// ETH_TXQ_WORD_CNT description says (measured). A keepalive, or a packet with up to 64 bytes of payload, is one word. A
// packet with up to 128 bytes is two.
constexpr Reg<> txq_word_count(uint32_t queue) { return txq_reg(queue, ETH_TXQ_WORD_CNT); }

// Arms the queue to put each packet's send time into the TX timestamp FIFO with `tag`. `tag` must not be
// tx_stamp_fifo::kEmptyTag, which is what an empty FIFO reads.
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

// The start of a frame, where the MAC writes a one-step stamp. The stamp is an 80-bit big-endian value whose low 64
// bits are the PTP time in ns.
struct FrameStampSlot {
    uint32_t head[2];  // frame bytes the stamp leaves alone, then the stamp's 16 high bits
    uint32_t time_big_endian[2];
};
static_assert(offsetof(FrameStampSlot, time_big_endian) == 2 * kFrameStampOffsetSetting + 4);

FORCE_INLINE uint64_t frame_stamp_ns(const volatile FrameStampSlot& slot) {
    return join_words(__builtin_bswap32(slot.time_big_endian[0]), __builtin_bswap32(slot.time_big_endian[1]));
}
// Zeroes the slot's stamp, because a frame the MAC doesn't stamp keeps whatever its slot held before.
FORCE_INLINE void clear_frame_stamp(volatile FrameStampSlot& slot) {
    slot.time_big_endian[0] = 0;
    slot.time_big_endian[1] = 0;
}

// ---------------------------------------------------------------------------------------------------------------------
// The TX header table

constexpr uint32_t kTxHeaderRows = 10;

// Lets TX queue `Queue` send its frames with TX header row `Row`, a spare row, so that they carry a destination address
// of their own. install() copies every other header field from the row the queue's packet sends use now, so only the
// destination changes. uninstall() points the queue back at its old rows and is valid only after install(). The
// Ethernet firmware sends only on TX queue 0 and points TX queues 0 to 2 at TX header rows 0 to 2, leaving rows 3 to 9
// free.
template <uint32_t Queue, uint32_t Row>
class TxHeaderRow {
public:
    static_assert(Queue < NUM_ETH_QUEUES && Row >= NUM_ETH_QUEUES && Row < kTxHeaderRows);

    __attribute__((noinline, cold)) void install(uint64_t destination) {
        const auto field = [](uint32_t row, uint32_t offset) {
            return Reg<>{ETH_TXPKT_CFG_REGS_START + row * ETH_TXPKT_CFG_REGS_SIZE + offset};
        };
        select_before_ = txq_header_select(Queue).read();
        for (uint32_t offset :
             {ETH_TXPKT_CFG_INSERT_CTL,
              ETH_TXPKT_CFG_CUSTOM_HDR,
              ETH_TXPKT_CFG_MAC_SA_LO,
              ETH_TXPKT_CFG_MAC_SA_HI,
              ETH_TXPKT_CFG_ETHERTYPE,
              ETH_TXPKT_CFG_VLAN1,
              ETH_TXPKT_CFG_VLAN2}) {
            field(Row, offset).write(field(select_before_.packet, offset).read());
        }
        field(Row, ETH_TXPKT_CFG_MAC_DA_LO).write(static_cast<uint32_t>(destination));
        field(Row, ETH_TXPKT_CFG_MAC_DA_HI).write(static_cast<uint32_t>(destination >> 32));
        txq_header_select(Queue).write({.raw = Row, .reg_write = Row, .packet = Row});
    }
    __attribute__((noinline, cold)) void uninstall() const { txq_header_select(Queue).write(select_before_); }

private:
    TxqHeaderSelect select_before_{};
};

// ---------------------------------------------------------------------------------------------------------------------
// The RX classifier. It compares each received frame against its TCAM rows, each a pattern with a mask, and the
// flow-table row of the highest-priority match says what to do with the frame.

// What a flow-table row does with the frames that reach it.
struct RxFlowActions {
    uint32_t rx_queue : 2;
    uint32_t drop : 1;
    uint32_t strip_headers : 1;
    uint32_t record_rx_time : 1;
    uint32_t prepend_software_metadata : 1;
    uint32_t prepend_hardware_metadata : 1;
    uint32_t rsvd : 25;
};

// The flow-table row that a TCAM row sends its matches to, and the row's priority when several rows match one frame.
struct RxTcamRowMapping {
    uint32_t priority : 3;  // higher wins
    uint32_t rsvd0 : 13;
    uint32_t flow_table_row : 6;
    uint32_t rsvd1 : 10;
};
constexpr uint32_t kRxTcamTopPriority = 7;
constexpr uint32_t kRxTcamRows = 64;

// Enables or disables a TCAM row. A write with `write` and `go` set applies `enable` to row `row`.
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

// Copies the TCAM write registers into row `row` once `go` is set, into its mask if `mask` is set and its values
// otherwise. Each update_* bit makes the row take that field from the write registers. Fields whose bit is clear keep
// their values.
struct RxTcamUpdate {
    uint32_t row : 6;
    uint32_t rsvd0 : 2;
    uint32_t mask : 1;
    uint32_t write : 1;
    uint32_t non_ip : 1;
    uint32_t rsvd1 : 5;
    uint32_t update_protocol : 1;
    uint32_t update_destination_port : 1;
    uint32_t update_source_port : 1;
    uint32_t update_destination : 1;
    uint32_t update_source : 1;
    uint32_t update_tuple_type : 1;
    uint32_t update_ethertype : 1;
    uint32_t update_l2_priority : 1;
    uint32_t rsvd2 : 7;
    uint32_t go : 1;
};

// The augmented fields are 4-bit classes that the classifier gives a frame's destination address and EtherType from its
// tables of well-known and user-set values, and 0 for values in none of them. The source address never gets a class.
struct RxTcamNonIpAddrFlags {
    uint32_t augmented_destination : 4;
    uint32_t rsvd0 : 12;
    uint32_t augmented_source : 4;
    uint32_t rsvd1 : 12;
};

struct RxTcamEthertype {
    uint32_t value : 16;
    uint32_t augmented : 4;
    uint32_t rsvd : 12;
};

struct RxTcamPriority {
    uint32_t priority_code_point : 3;
    uint32_t rsvd : 29;
};

// The label a flow-table row puts on the entries it records in the RX timestamp FIFO.
struct RxFlowTableLabel {
    uint32_t label : 5;
    uint32_t rsvd : 27;
};

// Copies the flow-table write registers into row `row` once `go` is set.
struct RxFlowTableUpdate {
    uint32_t row : 6;
    uint32_t rsvd0 : 2;
    uint32_t write : 1;
    uint32_t rsvd1 : 22;
    uint32_t go : 1;
};

// What the classifier does with frames that no TCAM row matches. These actions are flow-table row kRxTcamRows, just
// past the rows that TCAM rows can point to.
constexpr Reg<RxFlowActions> kRxNoMatchActions{ETH_RX_CLASSIFIER_REGS_START + ETH_RX_CLASSIFIER_NO_MATCH_ACTIONS};
constexpr Reg<RxTcamRowMapping> rx_tcam_row_mapping(uint32_t row) {
    return {ETH_RX_CLASSIFIER_REGS_START + ETH_RX_CLASSIFIER_TCAM_ROW_MAPPING + row * sizeof(uint32_t)};
}
constexpr Reg<RxTcamRowUpdate> kRxTcamRowUpdate{ETH_RX_CLASSIFIER_REGS_START + ETH_RX_CLASSIFIER_TCAM_ROW_UPDATE};
// The TCAM write registers, which kRxTcamUpdate copies into a row.
constexpr Reg<RxTcamTupleType> kRxTcamTupleTypeWrite{
    ETH_RX_CLASSIFIER_REGS_START + ETH_RX_CLASSIFIER_TCAM_TUPLE_TYPE_WRITE};
constexpr Reg<> rx_tcam_source_write(uint32_t word) {
    return {ETH_RX_CLASSIFIER_REGS_START + ETH_RX_CLASSIFIER_TCAM_SA_WRITE + word * sizeof(uint32_t)};
}
constexpr Reg<> rx_tcam_destination_write(uint32_t word) {
    return {ETH_RX_CLASSIFIER_REGS_START + ETH_RX_CLASSIFIER_TCAM_DA_WRITE + word * sizeof(uint32_t)};
}
constexpr Reg<RxTcamNonIpAddrFlags> kRxTcamNonIpAddrFlagsWrite{
    ETH_RX_CLASSIFIER_REGS_START + ETH_RX_CLASSIFIER_TCAM_NON_IP_ADDR_FLAGS_WRITE};
constexpr Reg<RxTcamEthertype> kRxTcamEthertypeWrite{
    ETH_RX_CLASSIFIER_REGS_START + ETH_RX_CLASSIFIER_TCAM_ETHERTYPE_WRITE};
constexpr Reg<RxTcamPriority> kRxTcamPriorityWrite{
    ETH_RX_CLASSIFIER_REGS_START + ETH_RX_CLASSIFIER_TCAM_PRIORITY_WRITE};
constexpr Reg<RxTcamUpdate> kRxTcamUpdate{ETH_RX_CLASSIFIER_REGS_START + ETH_RX_CLASSIFIER_TCAM_UPDATE};
// The flow-table write registers, which kRxFlowTableUpdate copies into a row.
constexpr Reg<RxFlowTableLabel> kRxFlowTableLabel{ETH_RX_CLASSIFIER_REGS_START + ETH_RX_CLASSIFIER_FTABLE_LABELS};
constexpr Reg<RxFlowActions> kRxFlowTableActions{ETH_RX_CLASSIFIER_REGS_START + ETH_RX_CLASSIFIER_FTABLE_ACTIONS};
constexpr Reg<> kRxFlowTableVlan{ETH_RX_CLASSIFIER_REGS_START + ETH_RX_CLASSIFIER_FTABLE_VLAN};
constexpr Reg<> kRxFlowTableSoftwareMetadata{ETH_RX_CLASSIFIER_REGS_START + ETH_RX_CLASSIFIER_FTABLE_SW_METADATA};
constexpr Reg<RxFlowTableUpdate> kRxFlowTableUpdate{ETH_RX_CLASSIFIER_REGS_START + ETH_RX_CLASSIFIER_FTABLE_UPDATE};

// The pattern of a non-IP TCAM row, which matches on a frame's Ethernet header alone. Its address fields are four words
// wide to fit an IPv6 address.
struct RxTcamNonIpPattern {
    uint32_t source[4];
    uint32_t destination[4];
    RxTcamNonIpAddrFlags addr_flags;
    RxTcamEthertype ethertype;
    RxTcamPriority priority;
};
// A TCAM row's values and mask. A frame matches if it equals `values` on every bit where `mask` is 0.
struct RxTcamNonIpMatch {
    RxTcamNonIpPattern values, mask;
};
// Returns a match for frames whose destination address is `destination`, regardless of the rest of the frame. The
// Ethernet firmware sets the destinations of TX header rows 0 to 2 to ff:ff:ff:ff:ff:ff, 01:00:00:00:00:00 and
// 02:00:00:00:00:00, so a destination other than those never matches a firmware frame. `destination` holds the address
// the same way as in TxHeaderRow::install().
constexpr RxTcamNonIpMatch rx_tcam_match_destination(uint64_t destination) {
    return {
        .values = {.destination = {static_cast<uint32_t>(destination), static_cast<uint32_t>(destination >> 32)}},
        .mask = {
            .source = {~0u, ~0u, ~0u, ~0u},
            .destination = {0u, ~0xFFFFu, ~0u, ~0u},
            .addr_flags = {.augmented_destination = 0xF, .augmented_source = 0xF},
            .ethertype = {.value = 0xFFFF, .augmented = 0xF},
            .priority = {.priority_code_point = 7}}};
}

// ---------------------------------------------------------------------------------------------------------------------
// The RX timestamp FIFO. Each entry holds a received frame's arrival time and the label of the flow-table row that
// recorded it.

// The label of the FIFO's head entry.
struct RxStampLabel {
    uint32_t label : 5;
    uint32_t recorded_by_rule : 1;  // set on every entry a TCAM match records (measured), so not part of the label
    uint32_t rsvd : 25;
    uint32_t valid : 1;
};
// The FIFO's fill level. `entries` reads 0 when the FIFO is full, because its four bits can't hold kDepth.
struct RxStampStatus {
    uint32_t entries : 4;
    uint32_t rsvd0 : 12;
    uint32_t full : 1;
    uint32_t nearly_full : 1;
    uint32_t empty : 1;
    uint32_t rsvd1 : 13;
};
// The RxStampStatus register's write layout, which removes the head entry or empties the FIFO.
struct RxStampCommand {
    uint32_t rsvd : 30;
    uint32_t flush : 1;
    uint32_t pop : 1;
};

// The head entry's arrival time in PTP ns.
constexpr Reg<> kRxStampTimeLo{ETH_RX_TH_REGS_START + ETH_RX_TH_TS_LOW};
constexpr Reg<> kRxStampTimeHi{ETH_RX_TH_REGS_START + ETH_RX_TH_TS_HIGH};
constexpr Reg<RxStampLabel> kRxStampLabel{ETH_RX_TH_REGS_START + ETH_RX_TH_TS_LABEL};
constexpr Reg<RxStampStatus> kRxStampStatus{ETH_RX_TH_REGS_START + ETH_RX_TH_STATUS};
constexpr Reg<RxStampCommand> kRxStampCommand{ETH_RX_TH_REGS_START + ETH_RX_TH_STATUS};

// A frame's time is recorded when it starts to arrive, so by the time its data is visible in L1, its timestamp is
// already in the FIFO. A frame the link resends (Go-back-N) is recorded again, even though the receiver drops it as a
// duplicate.
namespace rx_stamp_fifo {
constexpr uint32_t kDepth = 16;
// Removes the head entry, which must exist, and returns its arrival time. Read the entry's label from kRxStampLabel
// before calling this, because the label moves on to the next entry once this one is removed. The removal takes a write
// of 1 followed by a write of 0 (measured).
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
// Empties the FIFO. Unlike a removal, this needs no write of 0 afterwards (measured).
FORCE_INLINE void flush() { kRxStampCommand.write({.flush = 1}); }
}  // namespace rx_stamp_fifo

// A classifier rule that records the arrival time of each frame matching `match` in the RX timestamp FIFO, with label
// `Label`. It uses the TCAM row and the flow-table row numbered `Row`. Matched frames still go where unmatched frames
// go, because the rule copies the actions for unmatched frames and only adds the recording. uninstall() undoes
// install() and is valid only after it.
template <uint32_t Row, uint32_t Label>
class RxStampRule {
public:
    static_assert(Row < kRxTcamRows);
    static_assert(Label < 32, "flow-table labels are five bits");

    // Stops recording the arrival times of unmatched frames and empties the FIFO before installing the rule, so that
    // afterwards the FIFO holds only the entries the rule records.
    __attribute__((noinline, cold)) void install(const RxTcamNonIpMatch& match) {
        no_match_before_ = kRxNoMatchActions.read();
        RxFlowActions actions = no_match_before_;
        actions.record_rx_time = 0;
        kRxNoMatchActions.write(actions);
        rx_stamp_fifo::flush();
        write_tcam_row(match.values, false);
        write_tcam_row(match.mask, true);
        rx_tcam_row_mapping(Row).write({.priority = kRxTcamTopPriority, .flow_table_row = Row});
        actions.record_rx_time = 1;
        write_flow(actions, Label);
        kRxTcamRowUpdate.write({.row = Row, .enable = 1, .write = 1, .go = 1});
    }
    __attribute__((noinline, cold)) void uninstall() const {
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
        for (uint32_t word = 0; word < std::size(pattern.source); word++) {
            rx_tcam_source_write(word).write(pattern.source[word]);
            rx_tcam_destination_write(word).write(pattern.destination[word]);
        }
        kRxTcamNonIpAddrFlagsWrite.write(pattern.addr_flags);
        kRxTcamUpdate.write(
            {.row = Row,
             .mask = mask,
             .write = 1,
             .non_ip = 1,
             .update_destination = 1,
             .update_source = 1,
             .update_tuple_type = 1,
             .update_ethertype = 1,
             .update_l2_priority = 1,
             .go = 1});
    }
    FORCE_INLINE static void write_flow(RxFlowActions actions, uint32_t label) {
        kRxFlowTableActions.write(actions);
        kRxFlowTableVlan.write(0);
        kRxFlowTableLabel.write({.label = label});
        kRxFlowTableSoftwareMetadata.write(0);
        kRxFlowTableUpdate.write({.row = Row, .write = 1, .go = 1});
    }

    RxFlowActions no_match_before_{};
};

// ---------------------------------------------------------------------------------------------------------------------
// The MAC's send-time stamping, and the TX timestamp FIFO that two-step stamps go to

// The MAC's TX settings.
struct MacTxCfg {
    uint32_t sw_reset : 1;
    uint32_t tx_enable : 1;
    uint32_t disable_on_frag : 1;
    uint32_t pad_enable : 1;
    uint32_t stamp_fifo_every_packet : 1;  // every packet's send time goes to the FIFO, not only two-step ones
    uint32_t mib_count_snapshot_clear : 1;
    uint32_t mib_count_snapshot_periodic : 1;
    uint32_t active_mib_clear_on_read : 1;
    uint32_t snapshot_mib_clear_on_read : 1;
    uint32_t start_align : 2;
    uint32_t origin_timestamp_mode : 1;
    uint32_t short_alignment_marker_period : 1;
    uint32_t pma_data_width : 1;
    uint32_t rsvd : 18;
};
// The MAC's TX interrupts. stamp_fifo_full sets when the TX timestamp FIFO holds at least kMacTxStampFifoFullThreshold
// words, and stamp_offset_error when a stamp's offset falls outside its frame, which then goes out with a bad CRC.
struct MacTxInterrupts {
    uint32_t staging_fifo_underflow : 1;
    uint32_t staging_fifo_overflow : 1;
    uint32_t stamp_fifo_full : 1;
    uint32_t stamp_fifo_not_empty : 1;
    uint32_t stamp_offset_error : 1;
    uint32_t rsvd1 : 27;
};

// PTP nanoseconds added to every send-time stamp, for the delay from the MAC through the SerDes.
struct MacTxDelay {
    uint32_t ns : 16;
    uint32_t rsvd : 16;
};

constexpr Reg<MacTxCfg> kMacTxCfg{ETH_MAC_REGS_START + ETH_MAC_TX_CFG};
constexpr Reg<MacTxDelay> kMacTxDelay{ETH_MAC_REGS_START + ETH_MAC_TX_DELAY};
// The TX interrupts that have fired since they were last cleared. Writing 1 to a bit clears it.
constexpr Reg<MacTxInterrupts> kMacTxInterrupts{ETH_MAC_REGS_START + ETH_MAC_TX_INT};
// The TX interrupts' current state, which follows each cause and needs no clearing.
constexpr Reg<MacTxInterrupts> kMacTxInterruptsRaw{ETH_MAC_REGS_START + ETH_MAC_TX_INT_RAW};
// The FIFO word count at which stamp_fifo_full sets.
constexpr Reg<> kMacTxStampFifoFullThreshold{ETH_MAC_REGS_START + ETH_MAC_TS_FIFO_FULL_THRESH};
// The TX timestamp FIFO's head entry, in the measured word order, which differs from the RSm410 programming guide's.
// kMacTxStampTagLo must be read first, because that read takes the entry off the FIFO and the other three registers
// then show its words.
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
FORCE_INLINE bool empty() { return !kMacTxInterruptsRaw.read().stamp_fifo_not_empty; }
__attribute__((noinline, cold)) inline void clear() {
    while (pop()) {
    }
}
}  // namespace tx_stamp_fifo

}  // namespace eth_ptp
