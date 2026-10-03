// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

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

namespace tt::tt_metal::link_sync {

static_assert(kernel_profiler::kEthRefclkHz == eth_ptp::kRefclkHz);

// A link's stamped frames take a TX queue and header row nothing else uses: the firmware holds header rows 0 to 2
// (eth_ptp.hpp), and the fabric routers send on TX queue 0 and, when two ERISCs run a router, its receiver on queue 1.
constexpr uint32_t kLinkTxq = 2;
constexpr uint32_t kLinkHeaderRow = 3;
constexpr uint32_t kLinkTcamRow = 63;
constexpr uint32_t kLinkLabel = 0x15;
// The stamp frames' destination. The RX rule matches only its middle four bytes, 0xA5, which no firmware frame has
// (eth_ptp.hpp) and which read the same in either byte order.
constexpr uint64_t kStampFrameDa = 0x02A5'A5A5'A5A5ull;
// A 1 in the mask means don't care, so the rule compares only the four 0xA5 bytes.
constexpr eth_ptp::RxTcamNonIpPattern kStampRowValues{.da = {0xA5A5'A500u, 0x0000'00A5u}};
constexpr eth_ptp::RxTcamNonIpPattern kStampRowMask{
    .sa = {~0u, ~0u, ~0u, ~0u},
    .da = {0x0000'00FFu, 0xFFFF'FF00u, ~0u, ~0u},
    .addr_flags = {.augmented_da = 0xF, .augmented_sa = 0xF},
    .ethertype = {.value = 0xFFFF, .augmented = 0xF},
    .priority = {.pcp = 7}};
#if defined(PROFILE_STREAMING_SYNC_CHECK)
constexpr bool kSyncCheck = true;
#else
constexpr bool kSyncCheck = false;
#endif

// The stamp FIFO doesn't say which frame a stamp is for, so the ends take turns: the sender sends a burst only once the
// last one is fully echoed, and the receiver takes a burst's stamps only once all its frames are in. Each take then
// finds exactly that burst's stamps, in order.
constexpr uint32_t kTripsPerRound = 128;
constexpr uint32_t kBurstFrames = 4;
constexpr uint32_t kBurstsPerRound = kTripsPerRound / kBurstFrames;
constexpr uint32_t kFrameBytes = 96;
static_assert(kTripsPerRound % kBurstFrames == 0 && kBurstFrames >= 2);

// A frame in L1. The MAC writes its egress stamp into `stamp`. The keys come last, so a frame whose key shows has
// landed everything before it.
struct LinkFrame {
    eth_ptp::FrameStampSlot stamp;
    uint32_t round;
    uint32_t rsvd0;
    int64_t ptp_minus_refclk_ns;  // the sending end's
    uint32_t rsvd1[14];
    uint32_t key;
    uint32_t echo_key;
};
static_assert(sizeof(LinkFrame) == kFrameBytes && kFrameBytes % 16 == 0);
static_assert(eth_ptp::kFrameStampOffsetBytes + eth_ptp::kFrameStampBytes <= offsetof(LinkFrame, round));
static_assert(
    offsetof(kernel_profiler::LinkSyncL1, slots) == 0 &&
    kBurstFrames * kFrameBytes == sizeof(kernel_profiler::LinkSyncL1::slots));

constexpr uint32_t kTripBits = 9;
constexpr uint32_t frame_key(uint32_t round, uint32_t trip) { return (round << kTripBits) | (trip + 1); }
static_assert(kTripsPerRound < (1u << kTripBits));
// 64-bit because a round's 128 offsets from its first stamp sum past 2^32 ns once the round spans ~34 ms, and on a
// loaded router a round's stamps can spread over tens of milliseconds.
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

struct RoundStamps {
    StampSum egress, ingress;
    int64_t peer_ptp_minus_refclk_ns = 0;
    // The peer's egress stamp on `frame`; also keeps the peer's PTP offset it carries.
    __attribute__((noinline)) uint64_t take_peer_stamp(const volatile LinkFrame& frame) {
        peer_ptp_minus_refclk_ns = frame.ptp_minus_refclk_ns;
        return eth_ptp::frame_stamp_ns(frame.stamp);
    }
    // A frame's stamp is in the FIFO before the frame is visible, only the link's rule records stamps, and nothing else
    // of ours is in flight to this end, so the FIFO holds this burst's stamps in order unless a frame was resent, which
    // stamps it twice.
    __attribute__((noinline)) void take_burst(const uint64_t* egress_stamps) {
        namespace rx_stamp_fifo = eth_ptp::rx_stamp_fifo;
        if (!rx_stamp_fifo::holds_exactly<kBurstFrames>()) {
            rx_stamp_fifo::flush();
            return;
        }
#pragma GCC unroll 1
        for (uint32_t i = 0; i < kBurstFrames; i++) {
            egress.add(egress_stamps[i]);
            ingress.add(rx_stamp_fifo::pop());
        }
    }
};

// AICLK cycles per refclk update (four ticks), in sixteenths. It follows AICLK from a refclk reading every
// kRemeasureCycles. The reference wall clock is 64-bit because a link end's first frame can come minutes after start()
// on a large mesh, past where 32-bit wall and refclk differences alias.
struct UpdatePeriod {
    static constexpr uint32_t kRemeasureCycles = 1u << 20;
    static constexpr uint32_t kStartMeasureTicks = 1000;
    // A longer span overflows the << 4 into sixteenths. The router can go that long without stepping in a fabric pause,
    // and then the period stays until the next remeasure.
    static constexpr uint32_t kMaxMeasureCycles = 1u << (32 - 4);
    uint64_t ref_wall = 0;
    uint32_t cycles16 = 0, ref_refclk = 0;
    void start() {
        const eth_ptp::ClocksLo first = eth_ptp::await_refclk_update();
        while (eth_ptp::kRefclkLo.read() - first.refclk < kStartMeasureTicks) {
        }
        const eth_ptp::ClocksLo edge = eth_ptp::await_refclk_update();
        cycles16 = ((edge.wall - first.wall) << 4) / ((edge.refclk - first.refclk) / eth_ptp::kRefclkTicksPerUpdate);
        rereference();
    }
    FORCE_INLINE void refresh() {
        if (eth_ptp::kWallClockLo.read() - static_cast<uint32_t>(ref_wall) >= kRemeasureCycles) {
            remeasure();
        }
    }

private:
    FORCE_INLINE void rereference() {
        const eth_ptp::Instant now = eth_ptp::read_instant();
        ref_wall = now.wall;
        ref_refclk = static_cast<uint32_t>(now.refclk);
    }
    __attribute__((noinline)) void remeasure() {
        const uint64_t prev_wall = ref_wall;
        const uint32_t prev_refclk = ref_refclk;
        rereference();
        if (ref_wall - prev_wall < kMaxMeasureCycles) {
            cycles16 = (static_cast<uint32_t>(ref_wall - prev_wall) << 4) /
                       ((ref_refclk - prev_refclk) / eth_ptp::kRefclkTicksPerUpdate);
        }
    }
};

// Every member is zero-initialised, so an end has no .data for the firmware to copy; start() sets the rest.
struct EndBase {
    int64_t ptp_minus_refclk_ns = 0;
    eth_ptp::TxHeaderRow<kLinkTxq, kLinkHeaderRow> header;
    eth_ptp::RxStampRule<kLinkTcamRow, kLinkLabel> rule;
    volatile kernel_profiler::LinkSyncL1* l1 = nullptr;
    uint32_t round = 0;
    bool in_round = false;
    UpdatePeriod period;
    uint32_t walk = 0;
    RoundStamps stamps;
    uint32_t records = 0;

    // The host zeroes the slots, ctl and done before the end starts: a stale key would stall the link for good.
    void open(uint32_t link_l1) {
        l1 = reinterpret_cast<volatile kernel_profiler::LinkSyncL1*>(link_l1);
        ptp_minus_refclk_ns = eth_ptp::restart_ptp_timer();
        rule.install(kStampRowValues, kStampRowMask);
        header.install(kStampFrameDa);
        // Armed until stop(), so every frame the end sends is stamped and no slot needs clearing.
        eth_ptp::txq_arm_in_frame(kLinkTxq);
    }
    void start() {
        period.start();
        walk = eth_ptp::kWallClockLo.read() | 1u;
    }
    void stop() {
        eth_ptp::txq_disarm(kLinkTxq);
        header.restore();
        rule.remove();
    }

protected:
    static constexpr uint32_t kMaxDitherCycles = 128;
    // A burst's frame i uses slot i, at the same L1 address on both ends, so an echo lands on the frame it answers.
    FORCE_INLINE volatile LinkFrame& frame(uint32_t slot) const {
        return reinterpret_cast<volatile LinkFrame*>(l1->slots)[slot];
    }
    // Waits a uniformly random number of cycles within one refclk update period, then issues the slot's frame. The
    // stamps move in whole ticks, so a delay uniform over a whole number of ticks makes each stamp's rounding
    // independent of when the router got to the frame, and averaging a round's frames removes it.
    __attribute__((noinline)) bool send_dithered(uint32_t slot) {
        period.refresh();
        const uint32_t cycles = eth_clock::draw(walk, (period.cycles16 + 8) >> 4);
        dither_wait(cycles / 2);
        dither_wait(cycles - cycles / 2);
        // If the queue is busy, leave the frame for a later step. Waiting on a queue that a link-level resend keeps
        // busy would stop the router serving the fabric.
        if (internal_::eth_txq_is_busy(kLinkTxq)) {
            return false;
        }
        const uint32_t addr = reinterpret_cast<uintptr_t>(&frame(slot)) >> 4;
        internal_::eth_send_packet_unsafe(kLinkTxq, addr, addr, kFrameBytes >> 4);
        return true;
    }
    // Half the longest wait, run twice: a sled of the full length overflows the active-eth kernel config buffer when
    // the router also carries eth zones.
    __attribute__((noinline)) static void dither_wait(uint32_t cycles) { eth_clock::nops<kMaxDitherCycles / 2>(cycles); }
    static FORCE_INLINE volatile uint32_t* control_vector() {
        return reinterpret_cast<volatile uint32_t*>(GET_MAILBOX_ADDRESS_DEV(profiler.control_vector));
    }
    // The host does the average, since a 64-bit divide is a library routine in the ERISC's text.
    __attribute__((noinline)) void record(
        const StampSum& sum, int64_t stamping_end_offset_ns, kernel_profiler::SyncRole role) {
        volatile kernel_profiler::SyncLinkRecord& slot = l1->ring[records % kernel_profiler::kLinkSyncRingRecords].link;
        slot.meta =
            kernel_profiler::word_of(kernel_profiler::SyncMeta{.role = role, .kind = kernel_profiler::SyncKind::Link});
        slot.round = round;
        slot.first = static_cast<uint64_t>(static_cast<int64_t>(sum.first) - stamping_end_offset_ns);
        slot.sum_from_first_ns = sum.sum_from_first;
        slot.count = sum.count;
        std::atomic_thread_fence(std::memory_order_release);
        control_vector()[kernel_profiler::SPSC_LINK_SYNC_TAIL] = ++records;
    }
    // A round the ring has no room for isn't recorded: an end never waits for the eth relay.
    __attribute__((noinline)) void close_round(
        kernel_profiler::SyncRole egress_role, kernel_profiler::SyncRole ingress_role) {
        if (stamps.egress.count != 0 && records - control_vector()[kernel_profiler::SPSC_LINK_SYNC_HEAD] <=
                                            kernel_profiler::kLinkSyncRingRecords - 2) {
            record(stamps.egress, stamps.peer_ptp_minus_refclk_ns, egress_role);
            record(stamps.ingress, ptp_minus_refclk_ns, ingress_role);
        }
    }
    FORCE_INLINE void open_round(
        uint32_t next, kernel_profiler::SyncRole egress_role, kernel_profiler::SyncRole ingress_role) {
        if (in_round) {
            close_round(egress_role, ingress_role);
        }
        in_round = true;
        round = next;
        stamps.egress.reset();
        stamps.ingress.reset();
    }
};

struct SenderLink : EndBase {
    static constexpr uint32_t kBurstTicks =
        (kSyncCheck ? kernel_profiler::kLinkSyncCheckPaceTicks : kernel_profiler::kLinkSyncPaceTicks) / kBurstsPerRound;
    // Low words only: a burst is never scheduled more than kMaxLeadCycles ahead.
    static constexpr uint32_t kMaxLeadCycles = 1u << 24;
    uint32_t next_burst_refclk = 0, next_burst_wall = 0;
    uint32_t burst_first_trip = 0, burst_frames_sent = 0, round_bursts_sent = 0;

    void start() {
        EndBase::start();
        burst_frames_sent = kBurstFrames;
        resync();
    }
    // Anything further ahead than kMaxLeadCycles is overdue, however long the router went without stepping (a fabric
    // pause can outlast the 2^31 cycles a signed difference would allow).
    FORCE_INLINE bool due() const {
        return burst_frames_sent != kBurstFrames || next_burst_wall - eth_ptp::kWallClockLo.read() > kMaxLeadCycles;
    }
    FORCE_INLINE void serve() {
        if (burst_frames_sent != kBurstFrames) {
            send_next();
            return;
        }
        invalidate_l1_cache();
        if (l1->ctl != kernel_profiler::LinkSyncCtl::Run) {
            round_bursts_sent = 0;
            resync();
            return;
        }
        if (in_round && !echoed()) {
            return;
        }
        burst();
    }

private:
    FORCE_INLINE void resync() {
        const eth_ptp::ClocksLo now = eth_ptp::read_clocks_lo();
        next_burst_refclk = now.refclk + kBurstTicks;
        schedule(now);
    }
    FORCE_INLINE void schedule(const eth_ptp::ClocksLo& now) {
        next_burst_wall =
            now.wall + (((next_burst_refclk - now.refclk) * (period.cycles16 / eth_ptp::kRefclkTicksPerUpdate)) >> 4);
    }
    FORCE_INLINE bool echoed() const {
        for (uint32_t i = 0; i < kBurstFrames; i++) {
            if (frame(i).echo_key != frame_key(round, burst_first_trip + i)) {
                return false;
            }
        }
        return true;
    }
    __attribute__((noinline)) void send_next() { burst_frames_sent += send_dithered(burst_frames_sent); }
    __attribute__((noinline)) void burst() {
        const eth_ptp::ClocksLo now = eth_ptp::read_clocks_lo();
        if (in_round) {
            uint64_t egress_stamps[kBurstFrames];
#pragma GCC unroll 1
            for (uint32_t i = 0; i < kBurstFrames; i++) {
                egress_stamps[i] = stamps.take_peer_stamp(frame(i));
            }
            stamps.take_burst(egress_stamps);
        }
        if (round_bursts_sent == 0) {
            open_round(
                in_round ? round + 1 : round,
                kernel_profiler::SyncRole::ReturnEgress,
                kernel_profiler::SyncRole::ReturnIngress);
        }
        burst_first_trip = round_bursts_sent * kBurstFrames;
        if (++round_bursts_sent == kBurstsPerRound) {
            round_bursts_sent = 0;
        }
#pragma GCC unroll 1
        for (uint32_t i = 0; i < kBurstFrames; i++) {
            volatile LinkFrame& sent = frame(i);
            sent.ptp_minus_refclk_ns = ptp_minus_refclk_ns;
            sent.echo_key = 0;
            sent.round = round;
            sent.key = frame_key(round, burst_first_trip + i);
        }
        burst_frames_sent = 0;
        next_burst_refclk += kBurstTicks;
        if (static_cast<int32_t>(next_burst_refclk - now.refclk) < 0) {
            next_burst_refclk = now.refclk;
        }
        schedule(now);
    }
};

struct ReceiverLink : EndBase {
    uint32_t frames_taken = 0, frames_echoed = 0;
    uint64_t egress_stamps[kBurstFrames] = {};

    // Take a burst's ingress stamps when its last frame arrives, before that frame's echo; the sender sends nothing
    // more until every echo is in.
    FORCE_INLINE bool due() const {
        invalidate_l1_cache();
        return frames_echoed != frames_taken || frame(frames_taken).key != 0;
    }
    FORCE_INLINE void serve() {
        if (frames_echoed == frames_taken) {
            take();
        }
        echo();
    }

private:
    __attribute__((noinline)) void take() {
        volatile LinkFrame& taken = frame(frames_taken);
        if (frames_taken == 0 && (!in_round || taken.round != round)) {
            open_round(
                taken.round, kernel_profiler::SyncRole::ForwardEgress, kernel_profiler::SyncRole::ForwardIngress);
        }
        egress_stamps[frames_taken] = stamps.take_peer_stamp(taken);
        if (frames_taken == kBurstFrames - 1) {
            stamps.take_burst(egress_stamps);
        }
        taken.ptp_minus_refclk_ns = ptp_minus_refclk_ns;
        taken.echo_key = taken.key;
        taken.key = 0;
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

template <bool Sender>
using LinkEnd = std::conditional_t<Sender, SenderLink, ReceiverLink>;

// Out of line: inlined, the receiver's check makes the router's main loop spill.
template <bool Sender>
struct RouterEnd {
    static inline LinkEnd<Sender> end;
    __attribute__((noipa, cold)) static void start(uint32_t l1) {
        end.open(l1);
        end.start();
    }
    __attribute__((noinline)) static bool due() { return end.due(); }
    __attribute__((noinline)) static void serve() { end.serve(); }
    // Every 16th loop keeps the link's cost to the router small. The sync check's 10x pace needs every loop: at 16 its
    // bursts outlast their slots and too few rounds are left to hold out.
    static constexpr uint32_t kRouterStepLoops = kSyncCheck ? 1 : 16;
    static FORCE_INLINE void step(uint32_t iter) {
        if ((iter & (kRouterStepLoops - 1)) == 0 && due()) {
            serve();
        }
    }
    __attribute__((noipa, cold)) static void stop() { end.stop(); }
};

struct NoLinkEnd {
    static void start(uint32_t) {}
    static void step(uint32_t) {}
    static void stop() {}
};

template <kernel_profiler::LinkSyncRole Role>
using RouterHook = std::conditional_t<
    Role == kernel_profiler::LinkSyncRole::None,
    NoLinkEnd,
    RouterEnd<Role == kernel_profiler::LinkSyncRole::Sender>>;

}  // namespace tt::tt_metal::link_sync
