// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// A clock sync port is the active eth core on one side of a link. Each round, a link's two ports exchange bursts of
// stamped frames. The host fits a line to many rounds' stamps to get the offset and rate between the two chips'
// refclks.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "hostdev/dev_msgs.h"
#include "hostdev/streaming_profiler_common.h"
#include "internal/ethernet/dataflow_api.h"
#include "internal/ethernet/eth_ptp.hpp"
#include "tt_metal/impl/streaming_profiler/kernels/eth_clock.hpp"

namespace link_sync {

static_assert(
    kernel_profiler::kEthRefclkHz == eth_ptp::kRefclkHz &&
    kernel_profiler::kRefclkTicksPerUpdate == eth_ptp::kRefclkTicksPerUpdate);

// A link's stamped frames use a TX queue and header row that nothing else uses. The Ethernet firmware uses header rows
// 0 to 2. The fabric routers send on TX queue 0, and when a router runs on two ERISCs, its receiver uses queue 1.
constexpr uint32_t kLinkTxq = 2;
constexpr uint32_t kLinkHeaderRow = 3;
// The RX TCAM row of the link's stamp rule, and the label the rule tags its stamps with. Both are arbitrary.
constexpr uint32_t kLinkTcamRow = 63;
constexpr uint32_t kLinkLabel = 0x15;
// Each port reads the arrival times of the stamped frames it receives from the RX timestamp FIFO, which records only
// the frames that a TCAM rule matches. Both ports send their stamped frames to this destination address and match it
// with that rule, so the FIFO holds only those frames' arrival times. No other frame is sent here, because the Ethernet
// firmware only sends to ff:ff:ff:ff:ff:ff, 01:00:00:00:00:00 and 02:00:00:00:00:00, and the fabric routers only to
// ff:ff:ff:ff:ff:ff and 01:00:00:00:00:01.
constexpr uint64_t kStampFrameDestination = 0x0211'2233'4455ull;
constexpr eth_ptp::RxTcamNonIpMatch kStampMatch = eth_ptp::rx_tcam_match_destination(kStampFrameDestination);
constexpr bool kSyncCheck = get_named_compile_time_arg_val("LINK_SYNC_CHECK") != 0;

constexpr uint32_t kTripsPerRound = 128;
constexpr uint32_t kBurstFrames = 4;
constexpr uint32_t kBurstsPerRound = kTripsPerRound / kBurstFrames;
constexpr uint32_t kFrameBytes = 32;
static_assert(kTripsPerRound % kBurstFrames == 0);

// One stamped frame of a burst. The transmitter sends it with awaiting_echo set, and the receiver clears that and
// echoes the frame back. awaiting_echo is the frame's last word, so once it changes in L1, the rest of the frame has
// landed.
struct LinkFrame {
    eth_ptp::FrameStampSlot stamp;
    uint32_t round;
    uint32_t pad[2];
    uint32_t awaiting_echo;
};
static_assert(sizeof(LinkFrame) == kFrameBytes && kFrameBytes % 16 == 0);
static_assert(
    offsetof(kernel_profiler::LinkSyncL1, slots) == 0 &&
    kBurstFrames * kFrameBytes == sizeof(kernel_profiler::LinkSyncL1::slots));

// One port's egress or ingress stamps in a round, kept as the first stamp, the sum of each stamp's offset from it, and
// their count. The sum is 64-bit because a round's 128 offsets from its first stamp can add up to more than 2^32 ns
// once the round spans about 34 ms, and on a loaded router a round can spread over tens of milliseconds.
struct StampSum {
    uint32_t count = 0;
    uint64_t first = 0;
    uint64_t sum_from_first = 0;
    FORCE_INLINE void reset() {
        count = 0;
        sum_from_first = 0;
    }
    FORCE_INLINE void add(uint64_t stamp) {
        if (count == 0) {
            first = stamp;
        }
        sum_from_first += stamp - first;
        count++;
    }
};

// Tracks the number of AICLK cycles per refclk update (four ticks), rounded to the nearest cycle, by re-reading the
// refclk every kRemeasureCycles. reference_wall is 64-bit because a port's first frame can come minutes after start(),
// long enough for a 32-bit wall-clock difference to wrap.
struct UpdatePeriod {
    static constexpr uint32_t kRemeasureCycles = 1u << 20;
    static constexpr uint32_t kStartMeasureTicks = 1000;
    // The longest span remeasure() measures over. It skips a longer one, so measure() never sees more than 32 bits of
    // cycles. The router can go that long without stepping during a fabric pause, and the period then keeps its old
    // value until the next remeasure.
    static constexpr uint32_t kMaxMeasureCycles = 1u << 31;
    uint64_t reference_wall = 0;
    uint32_t cycles_per_update = 0, reference_refclk = 0;
    void start() {
        const eth_ptp::ClocksLo first = eth_ptp::await_refclk_update();
        while (eth_ptp::kRefclkLo.read() - first.refclk < kStartMeasureTicks) {
        }
        const eth_ptp::ClocksLo edge = eth_ptp::await_refclk_update();
        cycles_per_update = measure(edge.wall - first.wall, edge.refclk - first.refclk);
        take_reference();
    }
    FORCE_INLINE void remeasure_if_due() {
        if (eth_ptp::kWallClockLo.read() - static_cast<uint32_t>(reference_wall) >= kRemeasureCycles) {
            remeasure();
        }
    }

private:
    static FORCE_INLINE uint32_t measure(uint32_t cycles, uint32_t refclk_ticks) {
        const uint32_t updates = refclk_ticks / eth_ptp::kRefclkTicksPerUpdate;
        return (cycles + updates / 2) / updates;
    }
    FORCE_INLINE void take_reference() {
        const eth_ptp::Instant now = eth_ptp::read_instant();
        reference_wall = now.wall;
        reference_refclk = static_cast<uint32_t>(now.refclk);
    }
    __attribute__((noinline)) void remeasure() {
        const uint64_t prev_wall = reference_wall;
        const uint32_t prev_refclk = reference_refclk;
        take_reference();
        if (reference_wall - prev_wall < kMaxMeasureCycles) {
            cycles_per_update = measure(reference_wall - prev_wall, reference_refclk - prev_refclk);
        }
    }
};

// What TransmitterPort and ReceiverPort share. Every member starts at zero, so the firmware has no initial values to
// copy for a port when it loads the kernel. start() sets the nonzero ones.
struct PortBase {
    eth_ptp::TxHeaderRow<kLinkTxq, kLinkHeaderRow> header;
    eth_ptp::RxStampRule<kLinkTcamRow, kLinkLabel> rule;
    volatile kernel_profiler::LinkSyncL1* l1 = nullptr;
    uint32_t round = 0;
    UpdatePeriod period;
    uint32_t random_state = 0;
    StampSum egress, ingress;
    uint32_t ring_tail = 0;
    bool stopped = false;

    // The frame slots and the ctl and done words must be zero when the port starts, because a stale awaiting_echo would
    // stall the link's clock sync for good.
    void start() {
        l1 = reinterpret_cast<volatile kernel_profiler::LinkSyncL1*>(
            get_named_compile_time_arg_val("LINK_SYNC_L1_ADDR"));
        // When the eth relay launches, it starts reading this port's ring at the tail it finds in this word.
        control_vector()[kernel_profiler::SPSC_LINK_SYNC_TAIL] = ring_tail;
        eth_ptp::restart_ptp_timer();
        rule.install(kStampMatch);
        header.install(kStampFrameDestination);
        // The queue stays armed until stop(), so the MAC stamps every frame and no stamp slot needs clearing.
        eth_ptp::txq_arm_in_frame(kLinkTxq);
        period.start();
        random_state = eth_ptp::kWallClockLo.read() | 1u;
    }
    void stop() {
        eth_ptp::txq_disarm(kLinkTxq);
        header.uninstall();
        rule.uninstall();
        control_vector()[kernel_profiler::SPSC_LINK_SYNC_DONE] = kernel_profiler::kResidentDoneWord;
        stopped = true;
    }
    static FORCE_INLINE kernel_profiler::LinkSyncCtl ctl() {
        return static_cast<kernel_profiler::LinkSyncCtl>(control_vector()[kernel_profiler::SPSC_LINK_SYNC_CTL]);
    }

protected:
    // A burst's frame i uses slot i, at the same L1 address on both ports, so an echo lands on the frame it answers.
    FORCE_INLINE volatile LinkFrame& frame(uint32_t slot) const {
        return reinterpret_cast<volatile LinkFrame*>(l1->slots)[slot];
    }
    // Adds the burst's egress stamps, which its frames carry, and its ingress stamps, which are in the RX timestamp
    // FIFO. The FIFO doesn't say which frame a stamp belongs to, so the ports make sure it holds one burst's stamps at
    // a time. The transmitter only sends a burst once the previous one has been fully echoed, and the receiver only
    // takes a burst's stamps once all its frames have arrived. A frame's stamp is in the FIFO before the frame itself
    // is visible, and only the link's rule records stamps. The FIFO therefore holds exactly this burst's stamps, in
    // order, unless a frame was resent and stamped twice, in which case the burst is dropped.
    __attribute__((noinline)) void take_burst() {
        if (!eth_ptp::rx_stamp_fifo::holds_exactly<kBurstFrames>()) {
            eth_ptp::rx_stamp_fifo::flush();
            return;
        }
#pragma GCC unroll 1
        for (uint32_t i = 0; i < kBurstFrames; i++) {
            egress.add(eth_ptp::frame_stamp_ns(frame(i).stamp));
            ingress.add(eth_ptp::rx_stamp_fifo::pop());
        }
    }
    // Waits a uniformly random number of cycles, up to one refclk update period, then sends the slot's frame unless the
    // queue is busy, and returns how many frames it sent, 0 or 1. A stamp is the send time rounded down to a whole
    // refclk tick. The random delay spans a whole number of ticks, so the amount rounded off is uniform over a tick
    // whatever the code's timing, and over a round's frames it averages to half a tick at both ports, which cancels in
    // the offset between them.
    __attribute__((noinline)) uint32_t send_dithered(uint32_t slot) {
        period.remeasure_if_due();
        const uint32_t cycles = eth_clock::draw(random_state, period.cycles_per_update);
        const uint32_t start = eth_ptp::kWallClockLo.read();
        while (eth_ptp::kWallClockLo.read() - start < cycles) {
        }
        // If the queue is busy, leave the frame for a later step. Waiting on a queue that a link-level resend keeps
        // busy would stop the router serving the fabric.
        if (internal_::eth_txq_is_busy(kLinkTxq)) {
            return 0;
        }
        const uint32_t word_addr = reinterpret_cast<uintptr_t>(&frame(slot)) >> 4;
        internal_::eth_send_packet_unsafe(kLinkTxq, word_addr, word_addr, kFrameBytes >> 4);
        return 1;
    }
    static FORCE_INLINE volatile uint32_t* control_vector() {
        return reinterpret_cast<volatile uint32_t*>(GET_MAILBOX_ADDRESS_DEV(profiler.control_vector));
    }
    // Writes one role's stamp sum for the round to the ring. The host computes the average, because a 64-bit divide
    // would cost the router far more code size.
    __attribute__((noinline)) void record(const StampSum& sum, kernel_profiler::SyncRole role) {
        auto& slot = const_cast<kernel_profiler::SyncLinkRecord&>(
            l1->ring[ring_tail % kernel_profiler::kLinkSyncRingRecords].link);
        slot.meta = kernel_profiler::SyncMeta{.role = role, .kind = kernel_profiler::SyncKind::Link};
        slot.round = round;
        slot.first_ns = sum.first;
        slot.sum_from_first_ns = sum.sum_from_first;
        slot.count = sum.count;
        std::atomic_thread_fence(std::memory_order_release);
        control_vector()[kernel_profiler::SPSC_LINK_SYNC_TAIL] = ++ring_tail;
    }
    // Records the round's egress and ingress sums. A round with no stamps is not recorded, and neither is a round the
    // ring has no room for, because a port never waits for the eth relay to make room.
    __attribute__((noinline)) void close_round(
        kernel_profiler::SyncRole egress_role, kernel_profiler::SyncRole ingress_role) {
        if (egress.count != 0 &&
            ring_tail - control_vector()[kernel_profiler::SPSC_LINK_SYNC_HEAD] <=
                kernel_profiler::kLinkSyncRingRecords - kernel_profiler::kLinkSyncRecordsPerRound) {
            record(egress, egress_role);
            record(ingress, ingress_role);
        }
    }
    FORCE_INLINE void open_round(
        uint32_t next, kernel_profiler::SyncRole egress_role, kernel_profiler::SyncRole ingress_role) {
        close_round(egress_role, ingress_role);
        round = next;
        egress.reset();
        ingress.reset();
    }
};

struct TransmitterPort : PortBase {
    static constexpr uint32_t kBurstTicks =
        (kSyncCheck ? kernel_profiler::kLinkSyncCheckRoundTicks : kernel_profiler::kLinkSyncRoundTicks) /
        kBurstsPerRound;
    // The furthest ahead a burst is ever scheduled. next_burst_refclk keeps only the low word, because it is never more
    // than this far ahead.
    static constexpr uint32_t kMaxLeadTicks = 1u << 20;
    static_assert(kBurstTicks < kMaxLeadTicks);
    uint32_t next_burst_refclk = 0;
    uint32_t frames_to_send = 0, round_bursts_sent = 0;

    void start() {
        PortBase::start();
        resync();
    }
    // Returns whether a frame is waiting to be sent or the next burst is due. next_burst_refclk is never more than
    // kMaxLeadTicks ahead, so a burst that appears further ahead is overdue. After a pause of any length, this test
    // holds a burst back by at most kMaxLeadTicks, whereas testing the sign of the difference could hold it back 2^31
    // ticks.
    FORCE_INLINE bool due() const {
        return frames_to_send != 0 || next_burst_refclk - eth_ptp::kRefclkLo.read() > kMaxLeadTicks;
    }
    FORCE_INLINE void serve() {
        if (frames_to_send != 0) {
            frames_to_send -= send_dithered(kBurstFrames - frames_to_send);
            return;
        }
        invalidate_l1_cache();
        if (ctl() != kernel_profiler::LinkSyncCtl::Run) {
            round_bursts_sent = 0;
            resync();
            return;
        }
        if (!echoed()) {
            return;
        }
        burst();
    }

private:
    FORCE_INLINE void resync() { next_burst_refclk = eth_ptp::kRefclkLo.read() + kBurstTicks; }
    FORCE_INLINE bool echoed() const {
        for (uint32_t i = 0; i < kBurstFrames; i++) {
            if (frame(i).awaiting_echo) {
                return false;
            }
        }
        return true;
    }
    __attribute__((noinline)) void burst() {
        const uint32_t now = eth_ptp::kRefclkLo.read();
        take_burst();
        if (round_bursts_sent == 0) {
            open_round(round + 1, kernel_profiler::SyncRole::ReturnEgress, kernel_profiler::SyncRole::ReturnIngress);
        }
        if (++round_bursts_sent == kBurstsPerRound) {
            round_bursts_sent = 0;
        }
#pragma GCC unroll 1
        for (uint32_t i = 0; i < kBurstFrames; i++) {
            volatile LinkFrame& sent = frame(i);
            sent.round = round;
            sent.awaiting_echo = 1;
        }
        frames_to_send = kBurstFrames;
        next_burst_refclk += kBurstTicks;
        if (next_burst_refclk - now > kMaxLeadTicks) {
            next_burst_refclk = now;
        }
    }
};

struct ReceiverPort : PortBase {
    uint32_t frames_taken = 0, frames_echoed = 0;

    FORCE_INLINE bool due() const {
        invalidate_l1_cache();
        return frames_echoed != frames_taken || frame(frames_taken).awaiting_echo != 0;
    }
    FORCE_INLINE void serve() {
        if (frames_echoed == frames_taken) {
            take();
        }
        echo();
    }

private:
    // Takes each frame as it arrives, and the burst's stamps with its last frame, before echoing it. The earlier
    // frames' slots still hold the transmitter's egress stamps, because the MAC writes an echo's stamp only on the wire
    // and the transmitter sends nothing more until every echo is in.
    __attribute__((noinline)) void take() {
        volatile LinkFrame& taken = frame(frames_taken);
        if (frames_taken == 0 && taken.round != round) {
            open_round(
                taken.round, kernel_profiler::SyncRole::ForwardEgress, kernel_profiler::SyncRole::ForwardIngress);
        }
        if (frames_taken == kBurstFrames - 1) {
            take_burst();
        }
        taken.awaiting_echo = 0;
        frames_taken++;
    }
    __attribute__((noinline)) void echo() {
        frames_echoed += send_dithered(frames_echoed);
        if (frames_echoed == kBurstFrames) {
            frames_taken = 0;
            frames_echoed = 0;
        }
    }
};

template <kernel_profiler::LinkSyncRole Role>
using Port = std::conditional_t<Role == kernel_profiler::LinkSyncRole::Transmitter, TransmitterPort, ReceiverPort>;

}  // namespace link_sync
