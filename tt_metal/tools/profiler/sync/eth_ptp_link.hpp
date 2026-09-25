// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The streaming profiler's link half on the tile's 1588 hardware (internal/ethernet/eth_ptp.hpp): the session every
// sync kernel opens, how a round's frames are exchanged and its stamps accumulated and reported, and the sync
// records the two ends leave in their mailbox for the pusher (hostdev/streaming_profiler_common.h).

#pragma once

#include <cstdint>
#include <type_traits>

#include "hostdev/streaming_profiler_common.h"

namespace tt::tt_metal::eth_ptp {

// The end a fabric router hosts (fabric_erisc_router.cpp), by its LINK_SYNC_ROLE: 0 is no end and compiles to
// nothing, as does any role on a part without the timestamping hardware; 1 sends, 2 echoes (below).
template <uint32_t Role, bool DataCache>
struct RouterHook {
    void start(uint32_t) {}
    void step() {}
    void stop() {}
};

}  // namespace tt::tt_metal::eth_ptp

#if defined(ARCH_BLACKHOLE)

#include "hostdev/dev_msgs.h"
#include "internal/ethernet/eth_ptp.hpp"

namespace tt::tt_metal::eth_ptp {

constexpr uint32_t kLinkTxq = 2;        // fabric routers send on queue 0
constexpr uint32_t kLinkHeaderRow = 3;  // firmware programs rows 0..2
constexpr uint32_t kLinkTcamRow = 63;
constexpr uint32_t kLinkLabel = 0x15;
using LinkQueue = TxQueue<kLinkTxq>;
using LinkHeaderRow = TxHeaderRow<kLinkTxq, kLinkHeaderRow>;
using LinkRule = RxStampRule<kLinkTcamRow, kLinkLabel>;

// A round is kTripsPerRound exchanges spread over the round's period in bursts of kBurstFrames, each frame sent at a
// step of its own, so a core lends the link one frame's hand-offs at a time: a core that held a whole burst stalled
// every fabric flow through it, which the channels' buffers do not absorb (a full ring of unicasts lost 9.5 % to
// 1.3 us holds). The link relation is a line in the refclk domain, so the stamps average to the same offset however
// they are spread. Every stamp is quantised to the timer's 20 ns
// tick and a round's mean gains from its frames sitting at different phases of it, so the link's trip k (frame j of
// round r is trip r * kTripsPerRound + j) is issued (k * kFramePhaseStep mod kCombPhases) / kCombPhases of an update
// period past a refclk update: the ERISC sees the refclk move every four ticks, each move on a tick edge, so the comb
// covers four ticks, hence each tick, exactly in the refclk domain whatever AICLK does, and the mean's rounding noise
// falls to ~0.6 ns per round. A frame carries its own egress stamp
// (TxQueue::arm_in_frame), so each side pairs the peer's egress stamp, read from a frame it received, with its own
// ingress stamp of that frame and averages the pairs (HwRound): a round counts if any frame produced both, the peer's
// timer ran, and the peer's PTP offset (carried in every frame, to put its stamps in its refclk domain) held for the
// whole round. The two sides' pairs need not be the same frames: the link's relation is a line, so each side's
// averages are one instant of it (the host's LinkSolver).
//
// The classifier's FIFO says nothing of which frame an ingress stamp belongs to, so the ends take turns: the sender
// issues a burst only once every frame of the last one has been echoed, and the receiver echoes a burst only once all
// its frames are in. Each end takes a burst's stamps when all its frames are in and nothing of the other end's can be
// in flight towards it, so the FIFO holds exactly those stamps, in the frames' order (take_stamps).
constexpr uint32_t kTripsPerRound = 96;
constexpr uint32_t kBurstFrames = 4;
constexpr uint32_t kBurstsPerRound = kTripsPerRound / kBurstFrames;
// A frame's payload: the MAC's egress stamp field first (eth_ptp.hpp kFrameStampField), then the sync word (bytes_sent
// the frame's key, receiver_ack the key an echo answers, reserved_2 the round) and the sending end's PTP offset and
// timer flag. Frames of 16 to 128 bytes hand off in the same time.
constexpr uint32_t kFrameBytes = 96;
constexpr uint32_t kWordSync = 4, kWordOffsetLo = 8, kWordOffsetHi = 9, kWordTimerOk = 10;
// The L1 both ends own, the same addresses on both (kernel_profiler::kLinkSyncL1Bytes): the frame slots, then the
// control and diagnostic words.
constexpr uint32_t kSlotsBytes = kBurstFrames * kFrameBytes;
constexpr uint32_t kCtlOffset = kernel_profiler::kLinkSyncCtlOffset;
constexpr uint32_t kCombPhases = 256;
constexpr uint32_t kFramePhaseStep = 157;  // odd, so the kCombPhases phases are a permutation
// The control word at the diagnostics' base, the host's: rounds are issued only while it reads kCtlRun, and a
// resident kernel exits on kCtlStop. Set once the profiler's consumer and trackers are up, so no round predates
// the clock coverage that places it.
constexpr uint32_t kCtlRun = kernel_profiler::kLinkSyncCtlRun, kCtlStop = kernel_profiler::kLinkSyncCtlStop;
constexpr uint32_t kHwUnitsPerNs = kernel_profiler::kLinkSyncStampUnitsPerNs;
constexpr uint32_t kPaceTicks = kernel_profiler::kLinkSyncPaceTicks;
static_assert(kTripsPerRound % kBurstFrames == 0 && kBurstFrames >= 2 && (kFramePhaseStep & 1) == 1);
static_assert(kFrameBytes % 16 == 0 && 4 * (kWordTimerOk + 1) <= kFrameBytes);
static_assert(
    kFrameStampField + 10 <= 4 * kWordSync && 4 * kWordSync + sizeof(eth_channel_sync_t) <= 4 * kWordOffsetLo);
static_assert(kSlotsBytes <= kCtlOffset);
static_assert(kCtlOffset + 8 + 13 * sizeof(uint32_t) <= kernel_profiler::kLinkSyncRingOffset);  // StopDiag::write

// Frame j of a round rides in slot j % kBurstFrames, the same L1 address on both ends, so the receiver's echo lands
// on the frame it answers. The sync word carries the round and the trip: bytes_sent in the frame, which the receiver
// clears once it has taken it, and receiver_ack in the echo; reserved_2 carries the round's full number.
FORCE_INLINE uint32_t frame_at(uint32_t base, uint32_t j) { return base + (j % kBurstFrames) * kFrameBytes; }
FORCE_INLINE volatile uint32_t* words(uint32_t frame) { return reinterpret_cast<volatile uint32_t*>(frame); }
FORCE_INLINE volatile eth_channel_sync_t* sync_word(uint32_t frame) {
    return reinterpret_cast<volatile eth_channel_sync_t*>(frame + 4 * kWordSync);
}
constexpr uint32_t kTripMask = 0x1FF;
constexpr uint32_t frame_key(uint32_t round, uint32_t j) { return (round << 9) | (j + 1); }
static_assert(kTripsPerRound <= kTripMask);
// Trip k's place in an update period, in wall cycles, p16 being wall cycles per update times 16.
FORCE_INLINE uint32_t frame_phase_cycles(uint32_t k, uint32_t p16) {
    static_assert(kCombPhases * 16 == 1u << 12);
    return (((k * kFramePhaseStep) & (kCombPhases - 1)) * p16) >> 12;
}
// The next refclk update the ERISC sees, from reads back to back: its count and the wall read between the two refclk
// reads that differ. False if none came within the spins (a dead refclk).
FORCE_INLINE bool next_update(uint32_t& wall, uint32_t& refclk) {
    uint32_t prev = rd(kPtpCfrLo);
    for (uint32_t spin = 0; spin < 1024; spin++) {
        const uint32_t w = rd(kWallClockLo);
        const uint32_t r = rd(kPtpCfrLo);
        if (r != prev) {
            wall = w;
            refclk = r;
            return true;
        }
        prev = r;
    }
    return false;
}

// Waits to a wall-clock target with a delay loop calibrated once, so a frame sits at its cycle wherever its wait
// began: a poll on the wall clock exits a read's latency late by an amount set by where the loop was, and that is a
// function of when the wait began, so polling alone bent the grid by an AICLK-dependent amount.
struct Pacer {
    uint32_t iter16 = 0;  // wall cycles per delay-loop turn, x16
    // One copy: the loop's cycles per turn depend on its placement, and the calibration must time the same
    // instructions until() spins.
    __attribute__((noinline)) static void turns(uint32_t n) {
        for (uint32_t i = n; i != 0; i--) {
            asm volatile("");
        }
    }
    void calibrate() {
        const uint32_t a = rd(kWallClockLo);
        turns(4096);
        const uint32_t b = rd(kWallClockLo);
        iter16 = ((b - a) * 16u) / 4096u;
    }
    FORCE_INLINE void until(uint32_t target) const {
        const int32_t rem = static_cast<int32_t>(target - rd(kWallClockLo)) - 16;
        if (rem > 0) {
            turns((static_cast<uint32_t>(rem) * 16u) / iter16);
        }
        while (static_cast<int32_t>(rd(kWallClockLo) - target) < 0) {
        }
    }
};

// One side's sum of one stamp kind over a round, relative to its first. At a loaded router's slow steps a round's
// stamps spread over tens of milliseconds, so the sum of 256 offsets from the first passes 32 bits (~33 ms of spread).
struct StampSum {
    uint32_t n = 0;
    uint64_t base = 0;
    uint64_t rel = 0;  // sum of (stamp - base), ns
    FORCE_INLINE void reset() {
        n = 0;
        rel = 0;
    }
    FORCE_INLINE void add(uint64_t ts) {
        if (n == 0) {
            base = ts;
        }
        rel += ts - base;
        n++;
    }
    // The average in kHwUnitsPerNs of the refclk domain: PTP64NS minus the timer's offset from the CFR count, the
    // offset taken off in 64ths of a ns and the one rounding done last, so no part of it settles on every stamp of the
    // session. The sum is divided by the count in 16-bit digits, each a 32-bit divide (the remainder stays under the
    // count, n <= kTripsPerRound): a 64-bit divide is a routine in the ERISC's text.
    __attribute__((noinline)) uint64_t q(int64_t ptp_offset_64) const {
        static_assert(64 % kHwUnitsPerNs == 0 && kTripsPerRound <= (1u << 15));
        constexpr int64_t kPerUnit = 64 / kHwUnitsPerNs;
        const uint64_t x = rel * 64 + n / 2;
        uint64_t mean64 = 0;
        uint32_t r = 0;
        for (int32_t shift = 48; shift >= 0; shift -= 16) {
            const uint32_t cur = (r << 16) | static_cast<uint32_t>((x >> shift) & 0xFFFFu);
            mean64 = (mean64 << 16) | (cur / n);
            r = cur % n;
        }
        const int64_t avg64 = static_cast<int64_t>(base) * 64 - ptp_offset_64 + static_cast<int64_t>(mean64);
        return static_cast<uint64_t>((avg64 + kPerUnit / 2) / kPerUnit);
    }
};
// This end's offset and timer flag into a frame it is about to send, and its stamp field cleared: the slot still holds
// the stamp of the last frame that came in through it, so a frame the MAC did not stamp reads as zero, not as that.
FORCE_INLINE void carry_offset(volatile uint32_t* w, const PtpTimer& timer) {
    w[kWordOffsetLo] = static_cast<uint32_t>(timer.offset_64);
    w[kWordOffsetHi] = static_cast<uint32_t>(static_cast<uint64_t>(timer.offset_64) >> 32);
    w[kWordTimerOk] = timer.ok ? 1u : 0u;
    w[kFrameStampHiWord] = 0;
    w[kFrameStampHiWord + 1] = 0;
}

// What each end leaves past its control word for the host's log (streaming_profiler_device.cpp reads it back): +8
// rounds, +12 the timer word (1 ran, 2 never acknowledged its rate, the PTP offset in the bits above), +16 wall cycles
// inside steps that sent or took frames and +24 wall cycles of the run (two words each), +32 refclk ticks of the run
// (two words), +40 the longest such step in wall cycles, then the counts: +44 rounds not recorded, +48 frames left for
// a later step (the queue was busy at their phase), +52 bursts whose ingress stamps did not match their frames, +56
// frames that came in without an egress stamp.
struct StopDiag {
    uint32_t rounds = 0, timer = 0;
    uint64_t hold = 0, span_wall = 0, span_refclk = 0;
    uint32_t hold_max = 0;
    uint32_t drop[4] = {};
    FORCE_INLINE void note_hold(uint32_t cycles) {
        hold += cycles;
        hold_max = cycles > hold_max ? cycles : hold_max;
    }
    FORCE_INLINE void note_round(bool recorded) {
        rounds++;
        drop[0] += !recorded;
    }
    void write(uint32_t stop_addr) const {
        volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stop_addr + 8);
        const uint32_t words[13] = {
            rounds,
            timer,
            static_cast<uint32_t>(hold),
            static_cast<uint32_t>(hold >> 32),
            static_cast<uint32_t>(span_wall),
            static_cast<uint32_t>(span_wall >> 32),
            static_cast<uint32_t>(span_refclk),
            static_cast<uint32_t>(span_refclk >> 32),
            hold_max,
            drop[0],
            drop[1],
            drop[2],
            drop[3]};
        for (uint32_t i = 0; i < 13; i++) {
            w[i] = words[i];
        }
    }
};

// One side's hardware round: the peer's egress stamps, read from the frames it sent this side, paired with their
// ingress stamps here, and the peer's PTP offset they came with.
struct HwRound {
    StampSum tx, rx;
    int64_t peer_offset_64 = 0;
    bool peer_ok = false, peer_seen = false;
    FORCE_INLINE void begin() {
        tx.reset();
        rx.reset();
        peer_seen = false;
        peer_ok = true;
    }
    // A frame from the peer: the offset and timer flag it carried, and its egress stamp, 0 if the MAC did not stamp it.
    __attribute__((noinline)) uint64_t peer_frame(const volatile uint32_t* w) {
        const int64_t off = static_cast<int64_t>((static_cast<uint64_t>(w[kWordOffsetHi]) << 32) | w[kWordOffsetLo]);
        peer_ok = peer_ok && w[kWordTimerOk] == 1 && (!peer_seen || off == peer_offset_64);
        peer_offset_64 = off;
        peer_seen = true;
        return frame_stamp(w);
    }
    __attribute__((noinline)) void pair(const uint64_t* tx_ts, const uint64_t* rx_ts, StopDiag& diag) {
        for (uint32_t m = 0; m < kBurstFrames; m++) {
            if (tx_ts[m] == 0) {
                diag.drop[3]++;
                continue;
            }
            tx.add(tx_ts[m]);
            rx.add(rx_ts[m]);
        }
    }
    bool usable() const { return tx.n != 0 && peer_ok; }
};
// The ingress stamps of a burst whose frames are all in, oldest first. A frame's stamp is in the FIFO before the
// frame is visible, and nothing else of ours can be in flight towards this end, so the FIFO holds one stamp per
// frame, unless a frame came in twice (a Go-back-N resend is stamped on its way in, then dropped as a duplicate) or
// the FIFO filled. Only this pops it, so a fill since the last take still shows as full. Any other count leaves the
// burst out and empties the FIFO.
__attribute__((noinline)) inline bool take_stamps(uint64_t* rx, StopDiag& diag) {
    constexpr RxStampFifo fifo{};
    bool ok = fifo.holds_exactly(kBurstFrames);
    for (uint32_t i = 0; ok && i < kBurstFrames; i++) {
        RxStampFifo::Entry e{};
        ok = fifo.pop(e) && e.valid && e.label == kLinkLabel;
        rx[i] = e.ts;
    }
    if (!ok) {
        fifo.flush();
        diag.drop[2]++;
    }
    return ok;
}
// The queue stays armed in-frame for the whole session (EndBase::open), so every frame it sends carries its egress
// stamp, and its keepalives take the stamp in their padding. A frame goes only onto a free queue and otherwise waits
// for a later step: a router waiting on a queue a link-level resend keeps busy stops serving the fabric.
FORCE_INLINE bool issue(uint32_t frame) {
    if (internal_::eth_txq_is_busy(kLinkTxq)) {
        return false;
    }
    internal_::eth_send_packet_unsafe(kLinkTxq, frame >> 4, frame >> 4, kFrameBytes >> 4);
    return true;
}

// Each frame's place: a comb phase past the refclk update it waits for, the update period in wall cycles measured
// from the update the end's previous frame waited for, over the updates between them, so it holds to a fraction of a
// cycle and follows AICLK with no estimate of it. The comb starts kLeadCycles past the update, so no phase is closer to
// it than a frame can be issued.
struct Grid {
    static constexpr uint32_t kLeadCycles = 64;
    uint32_t p16 = 0;                         // wall cycles per update, x16
    uint32_t edge_wall = 0, edge_refclk = 0;  // the last update a frame waited for
    Pacer pacer;
    // The first period from updates ~20 us apart.
    void start() {
        pacer.calibrate();
        uint32_t w0 = 0, r0 = 0, w1 = 0, r1 = 0;
        if (next_update(w0, r0)) {
            while (rd(kPtpCfrLo) - r0 < 1000u) {
            }
            if (next_update(w1, r1)) {
                p16 = ((w1 - w0) << 4) / ((r1 - r0) / 4u);
            }
        }
        edge_wall = w1;
        edge_refclk = r1;
    }
    // Wall cycles per refclk tick, x16.
    uint32_t c16() const { return p16 >> 2; }
    // Frame j of round `round`: false if the queue was busy at its phase, which leaves it for a later step. A gap of
    // 2^27 cycles or more since the last update (a pause) keeps the period it had.
    __attribute__((noinline)) bool send(uint32_t base, uint32_t round, uint32_t j, StopDiag& diag) {
        uint32_t w = 0, r = 0;
        if (next_update(w, r)) {
            const uint32_t cycles = w - edge_wall, updates = (r - edge_refclk) / 4u;
            if (updates != 0 && cycles < (1u << 27)) {
                p16 = (cycles << 4) / updates;
            }
            edge_wall = w;
            edge_refclk = r;
        } else {
            w = rd(kWallClockLo);
        }
        pacer.until(w + kLeadCycles + frame_phase_cycles(round * kTripsPerRound + j, p16));
        const bool went = issue(frame_at(base, j));
        diag.drop[1] += !went;
        return went;
    }
};

// The records: one per stamp average, this core's refclk-domain reading with the round's number and the stamp's role,
// so the host pairs the two ends by identity and fits refclk against refclk: DVFS on either chip's wall clock cannot
// enter the link solve. They go to the ring at the end of this core's link L1 (hostdev kLinkSyncRingOffset), its count
// published in this core's profiler control vector for the pusher's sweep.
namespace link {
constexpr uint32_t kRoleT0 = kernel_profiler::kSyncRoleT0;
constexpr uint32_t kRoleT1 = kernel_profiler::kSyncRoleT1;
constexpr uint32_t kRoleT1B = kernel_profiler::kSyncRoleT1B;
constexpr uint32_t kRoleT2 = kernel_profiler::kSyncRoleT2;
struct Ring {
    uint32_t base = 0, n = 0;
    volatile uint32_t* tail = nullptr;
    void open(uint32_t l1) {
        base = l1 + kernel_profiler::kLinkSyncRingOffset;
        n = 0;
        tail = reinterpret_cast<volatile uint32_t*>(GET_MAILBOX_ADDRESS_DEV(profiler.control_vector)) +
               kernel_profiler::SPSC_LINK_SYNC_TAIL;
        *tail = 0;
    }
    void record_hw(uint64_t value, uint32_t round, uint32_t role) {
        volatile uint32_t* r = reinterpret_cast<volatile uint32_t*>(
            base + (n % kernel_profiler::kLinkSyncRingRecords) * kernel_profiler::kSyncRecordWords * 4);
        r[kernel_profiler::SYNC_META] = (kernel_profiler::kSyncKindLink << 8) | role;
        r[kernel_profiler::SYNC_ROUND] = round;
        r[kernel_profiler::SYNC_VALUE_LO] = static_cast<uint32_t>(value);
        r[kernel_profiler::SYNC_VALUE_HI] = static_cast<uint32_t>(value >> 32);
        asm volatile("fence" ::: "memory");
        *tail = ++n;
    }
};
}  // namespace link

// Both ends clear the slots before the handshake: a frame that lands before its receiver's start() is then kept, and
// a burst whose frame was lost would hold the link for good.
inline void clear_slots(uint32_t base) {
    for (uint32_t j = 0; j < kBurstFrames; j++) {
        volatile eth_channel_sync_t* s = sync_word(frame_at(base, j));
        s->bytes_sent = 0;
        s->receiver_ack = 0;
        s->reserved_2 = 0;
    }
}

// What both ends of a link hold, driven by whoever owns the core -- a resident kernel or the fabric router: open()
// before the link handshake, start() once the peer is up, step() as often as the core can spare, stop() at teardown.
// A step returns at once when nothing is due, and otherwise sends, or reads and echoes, one frame, so it holds the core
// for a frame's hand-offs at most. The tile's stamping blocks are borrowed from open() to stop(). Every member is
// zero-initialised, so an end costs no .data for the firmware to copy: start() sets the rest.
struct EndBase {
    PtpTimer timer;
    LinkHeaderRow header;
    LinkRule rule;
    uint32_t slot_base = 0, diag_addr = 0, round = 0;
    bool started = false;
    Instant start_at{};
    Grid grid;
    StopDiag diag;
    HwRound rnd;
    link::Ring ring;

    void open(uint32_t l1) {
        slot_base = l1;
        clear_slots(l1);
        timer.start();
        rule.install();
        header.install();
        LinkQueue{}.arm_in_frame();
    }
    void stop() {
        LinkQueue{}.disarm();
        header.restore();
        rule.remove();
        write_diag();
    }

protected:
    void begin(uint32_t l1, uint32_t ctl) {
        diag_addr = ctl;
        ring.open(l1);
        start_at = read_instant();
        grid.start();
    }
    // Rewritten at every round's close, so a host that cannot stop this end (a router) still reads the current
    // figures. The PTP offset rides in the timer word: a tick multiple, so its low two bits are free.
    __attribute__((noinline)) void write_diag() {
        const Instant now = read_instant();
        diag.timer = (timer.ok ? 1u : 2u) | (static_cast<uint32_t>(timer.offset_64 >> 6) & ~3u);
        diag.span_wall = now.wall() - start_at.wall();
        diag.span_refclk = now.refclk - start_at.refclk;
        diag.write(diag_addr);
    }
    __attribute__((noinline)) void close_round(uint32_t tx_role, uint32_t rx_role) {
        const bool recorded = timer.ok && rnd.usable();
        if (recorded) {
            ring.record_hw(rnd.tx.q(rnd.peer_offset_64), round, tx_role);
            ring.record_hw(rnd.rx.q(timer.offset_64), round, rx_role);
        }
        diag.note_round(recorded);
        write_diag();
    }
};

// DataCache says whether the core runs with its L1 data cache on, in which case a step invalidates before polling
// what the peer or the host wrote; a router runs with it off and skips the fence.
template <bool DataCache = true>
struct SenderLink : EndBase {
    uint32_t burst_ticks = 0;
    // The earliest refclk tick of the next burst, and the bursts issued: a round is kBurstsPerRound of them.
    uint64_t slot_cfr = 0, bursts = 0;
    // The next slot as a wall cycle, set from the burst's own reading: an idle step then costs one wall-clock read
    // where the refclk costs two, and nothing a stamp depends on is timed by it.
    uint32_t slot_wall = 0;
    // The burst in flight, from trip out_j0: out_sent of its frames issued, their echoes awaited once all are.
    uint32_t out_j0 = 0, out_sent = 0, next_j = 0;

    void start(uint32_t l1, uint32_t ctl) {
        begin(l1, ctl);
        burst_ticks = kPaceTicks / kBurstsPerRound;
        out_sent = kBurstFrames;
        slot_cfr = start_at.refclk;
        schedule(start_at);
    }
    FORCE_INLINE void schedule(const Instant& now) {
        const int64_t ticks = static_cast<int64_t>(slot_cfr - now.refclk);
        slot_wall = now.wall_lo + ((static_cast<uint32_t>(ticks < 0 ? 0 : ticks) * grid.c16()) >> 4);
    }
    FORCE_INLINE void step() {
        if (out_sent != kBurstFrames) {
            send_next();
            return;
        }
        const uint32_t w = rd(kWallClockLo);
        if (static_cast<int32_t>(w - slot_wall) < 0) {
            return;
        }
        if constexpr (DataCache) {
            invalidate_l1_cache();
        }
        if (rd(diag_addr) != kCtlRun) {
            // A round in progress closes at the next burst rather than spanning the pause.
            bursts -= bursts % kBurstsPerRound;
            const Instant now = read_instant();
            slot_cfr = now.refclk;
            schedule(now);
            return;
        }
        if (started && !echoed()) {
            return;
        }
        burst();
    }

private:
    FORCE_INLINE bool echoed() const {
        for (uint32_t i = 0; i < kBurstFrames; i++) {
            if (sync_word(frame_at(slot_base, i))->receiver_ack != frame_key(round, out_j0 + i)) {
                return false;
            }
        }
        return true;
    }
    __attribute__((noinline)) void send_next() {
        const uint32_t w = rd(kWallClockLo);
        out_sent += grid.send(slot_base, round, out_j0 + out_sent, diag);
        diag.note_hold(rd(kWallClockLo) - w);
    }
    // A slot, once all the last burst's echoes are in: its pairs (egress stamps from the echoes themselves, ingress
    // stamps here), the round's records at a round boundary, then the next burst's frames, which the steps after it
    // send. A round is a count of bursts, not of slots: a burst whose steps outlast its slot delays the next, and the
    // round still holds all its trips, so its phases cover the tick exactly. The next slot is a slot on, or now when
    // that is later.
    __attribute__((noinline)) void burst() {
        const Instant now = read_instant();
        if (started) {
            uint64_t tx[kBurstFrames], rx[kBurstFrames];
            for (uint32_t i = 0; i < kBurstFrames; i++) {
                tx[i] = rnd.peer_frame(words(frame_at(slot_base, i)));
            }
            if (take_stamps(rx, diag)) {
                rnd.pair(tx, rx, diag);
            }
        }
        if (bursts % kBurstsPerRound == 0) {
            if (started) {
                close_round(link::kRoleT1B, link::kRoleT2);
                round++;
            }
            rnd.begin();
            next_j = 0;
        }
        started = true;
        out_j0 = next_j;
        next_j += kBurstFrames;
        for (uint32_t i = 0; i < kBurstFrames; i++) {
            const uint32_t f = frame_at(slot_base, i);
            carry_offset(words(f), timer);
            volatile eth_channel_sync_t* s = sync_word(f);
            s->receiver_ack = 0;
            s->reserved_2 = round;
            s->bytes_sent = frame_key(round, out_j0 + i);
        }
        out_sent = 0;
        bursts++;
        slot_cfr += burst_ticks;
        if (static_cast<int64_t>(slot_cfr - now.refclk) < 0) {
            slot_cfr = now.refclk;
        }
        schedule(now);
        diag.note_hold(rd(kWallClockLo) - now.wall_lo);
    }
};

// The receiver follows the sender's cadence.
template <bool DataCache = true>
struct ReceiverLink : EndBase {
    // The burst coming in, from trip echo_j0: `taken` of its frames read, their egress stamps in tx, and `echoed` of
    // those echoed.
    uint32_t echo_j0 = 0, taken = 0, echoed = 0;
    uint64_t tx[kBurstFrames] = {};

    void start(uint32_t l1, uint32_t ctl) { begin(l1, ctl); }
    // Each frame is read and echoed at the step that finds it, the burst's ingress stamps taken with its last frame,
    // before that frame's echo: the sender issues nothing more until every echo is in. An echo the queue did not take
    // goes at the next step.
    FORCE_INLINE void step() {
        if constexpr (DataCache) {
            invalidate_l1_cache();
        }
        if (echoed != taken) {
            echo();
        } else if (sync_word(frame_at(slot_base, taken))->bytes_sent != 0) {
            take();
            echo();
        }
    }

private:
    // Frame `taken`: its egress stamp and the offset it carries, then its echo set up in the same slot, carrying its
    // key back and, once sent, this end's egress stamp. A burst's first frame names the round, the sender's; a new one
    // closes the previous. Its last frame takes the burst's ingress stamps and pairs them.
    __attribute__((noinline)) void take() {
        const uint32_t w = rd(kWallClockLo);
        const uint32_t f = frame_at(slot_base, taken);
        volatile eth_channel_sync_t* s = sync_word(f);
        if (taken == 0) {
            if (!started || s->reserved_2 != round) {
                if (started) {
                    close_round(link::kRoleT0, link::kRoleT1);
                }
                started = true;
                round = s->reserved_2;
                rnd.begin();
            }
            echo_j0 = (s->bytes_sent & kTripMask) - 1;
        }
        tx[taken] = rnd.peer_frame(words(f));
        if (taken == kBurstFrames - 1) {
            uint64_t rx[kBurstFrames];
            if (take_stamps(rx, diag)) {
                rnd.pair(tx, rx, diag);
            }
        }
        carry_offset(words(f), timer);
        s->receiver_ack = s->bytes_sent;
        s->bytes_sent = 0;
        taken++;
        diag.note_hold(rd(kWallClockLo) - w);
    }
    __attribute__((noinline)) void echo() {
        const uint32_t w = rd(kWallClockLo);
        echoed += grid.send(slot_base, round, echo_j0 + echoed, diag);
        if (echoed == kBurstFrames) {
            taken = 0;
            echoed = 0;
        }
        diag.note_hold(rd(kWallClockLo) - w);
    }
};

template <bool Sender, bool DataCache>
using LinkEnd = std::conditional_t<Sender, SenderLink<DataCache>, ReceiverLink<DataCache>>;

// Calls F(arg) with the caller-saved registers kept here instead of by the caller: the asm clobbers ra alone, so the
// code around the call site is register-allocated as if there were no call.
template <void (*F)(uint32_t)>
FORCE_INLINE void saved_call(uint32_t arg) {
    asm volatile(
        "addi sp, sp, -64\n\t"
        "sw t0, 0(sp)\n\t"
        "sw t1, 4(sp)\n\t"
        "sw t2, 8(sp)\n\t"
        "sw t3, 12(sp)\n\t"
        "sw t4, 16(sp)\n\t"
        "sw t5, 20(sp)\n\t"
        "sw t6, 24(sp)\n\t"
        "sw a0, 28(sp)\n\t"
        "sw a1, 32(sp)\n\t"
        "sw a2, 36(sp)\n\t"
        "sw a3, 40(sp)\n\t"
        "sw a4, 44(sp)\n\t"
        "sw a5, 48(sp)\n\t"
        "sw a6, 52(sp)\n\t"
        "sw a7, 56(sp)\n\t"
        "mv a0, %z[arg]\n\t"
        "call %[fn]\n\t"
        "lw t0, 0(sp)\n\t"
        "lw t1, 4(sp)\n\t"
        "lw t2, 8(sp)\n\t"
        "lw t3, 12(sp)\n\t"
        "lw t4, 16(sp)\n\t"
        "lw t5, 20(sp)\n\t"
        "lw t6, 24(sp)\n\t"
        "lw a0, 28(sp)\n\t"
        "lw a1, 32(sp)\n\t"
        "lw a2, 36(sp)\n\t"
        "lw a3, 40(sp)\n\t"
        "lw a4, 44(sp)\n\t"
        "lw a5, 48(sp)\n\t"
        "lw a6, 52(sp)\n\t"
        "lw a7, 56(sp)\n\t"
        "addi sp, sp, 64"
        :
        : [fn] "s"(F), [arg] "rJ"(arg)
        : "ra");
}

// A router's end over its LINK_SYNC_ADDR region, at the product's pace. The router reaches it only through saved_call
// and the bodies are noipa, so the router's own code compiles as it does with no end.
template <bool Sender, bool DataCache>
struct HostedEnd {
    static inline LinkEnd<Sender, DataCache> end;
    __attribute__((noipa)) static void start_body(uint32_t l1) {
        end.open(l1);
        end.start(l1, l1 + kCtlOffset);
    }
    __attribute__((noipa)) static void step_body(uint32_t) { end.step(); }
    __attribute__((noipa)) static void stop_body(uint32_t) { end.stop(); }
    FORCE_INLINE void start(uint32_t l1) { saved_call<&start_body>(l1); }
    FORCE_INLINE void step() { saved_call<&step_body>(0); }
    FORCE_INLINE void stop() { saved_call<&stop_body>(0); }
};
template <bool DataCache>
struct RouterHook<1, DataCache> : HostedEnd<true, DataCache> {};
template <bool DataCache>
struct RouterHook<2, DataCache> : HostedEnd<false, DataCache> {};

}  // namespace tt::tt_metal::eth_ptp

#endif  // ARCH_BLACKHOLE
