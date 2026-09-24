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
using LinkSession = StampSession<kLinkTxq, kLinkHeaderRow, kLinkTcamRow, kLinkLabel>;

// A round is kTripsPerRound exchanges spread over the round's period in bursts of kBurstFrames, each frame sent at a
// step of its own, so a core lends the link one frame's hand-offs at a time: a core that held a whole burst stalled
// every fabric flow through it, which the channels' buffers do not absorb (a full ring of unicasts lost 9.5 % to
// 1.3 us holds). The link relation is a line in the refclk domain, so the stamps average to the same offset however
// they are spread. Every stamp is quantised to the timer's 20 ns
// tick and a round's mean gains from its frames sitting at different phases of it, so frame j is issued
// (j * kFramePhaseStep mod 256) / 256 of a tick past its slot, the slots themselves on the refclk: an exact grid over
// the tick whatever the cadence or AICLK, and the mean's rounding noise falls to ~0.4 ns per round. The pacer lands
// a frame on a wall cycle, so the grid has as many distinct phases as the tick has cycles, and where that comb sits
// against the timer's edge would otherwise be fixed for the session; each burst shifts it by a pseudo-random part of
// a tick, so the comb's origin averages out over the round as its phases do. A frame carries its own egress stamp
// (stamps_arm_in_frame), so each side pairs the peer's egress stamp, read from a frame it received, with its own
// ingress stamp of that frame and averages the pairs (HwRound): a round counts if any frame produced both, the peer's
// timer ran, and the peer's PTP offset (carried in every frame, to put its stamps in its refclk domain) held for the
// whole round. The two sides' pairs need not be the same frames: the link's relation is a line, so each side's
// averages are one instant of it (the host's LinkSolver).
//
// The classifier's FIFO says nothing of which frame an ingress stamp belongs to, so the ends take turns: the sender
// issues a burst only once every frame of the last one has been echoed, and the receiver echoes a burst only once all
// its frames are in. Each end takes a burst's stamps when all its frames are in and nothing of the other end's can be
// in flight towards it, so the FIFO holds exactly those stamps, in the frames' order (take_stamps).
constexpr uint32_t kTripsPerRound = 256;
constexpr uint32_t kBurstFrames = 4;
constexpr uint32_t kBurstsPerRound = kTripsPerRound / kBurstFrames;
// A frame's payload: the sync word (bytes_sent the frame's key, receiver_ack the key an echo answers, reserved_2 the
// round), the sending end's PTP offset and timer flag, then the MAC's egress stamp field. Two WORD_CNT units against
// a keepalive's one, which is what counts the frames among the queue's hand-offs; 80 bytes is the first two-unit
// size, 96 leaves a unit's margin either way. Frames of 16 to 128 bytes hand off in the same time.
constexpr uint32_t kFrameBytes = 96;
constexpr uint32_t kWordOffsetLo = 4, kWordOffsetHi = 5, kWordTimerOk = 6;
// The L1 both ends own, the same addresses on both (kernel_profiler::kLinkSyncL1Bytes): where the peer's pilot lands,
// the frame slots, then the control and diagnostic words.
constexpr uint32_t kPilotOffset = 0;
constexpr uint32_t kSlotsOffset = kFrameBytes;
constexpr uint32_t kSlotsBytes = kBurstFrames * kFrameBytes;
constexpr uint32_t kCtlOffset = kernel_profiler::kLinkSyncCtlOffset;
constexpr uint32_t kFramePhaseStep = 157;  // odd, so the 256 phases are a permutation
// Polls for a hand-off on a queue the fabric may be loading (our frames wait behind its at the MAC), after which the
// frame is left for a later step: ~6 us, a bound on a step's hold.
constexpr uint32_t kHandoffSpins = 256;
// The control word at the diagnostics' base, the host's: rounds are issued only while it reads kCtlRun, and a
// resident kernel exits on kCtlStop. Set once the profiler's consumer and trackers are up, so no round predates
// the clock coverage that places it.
constexpr uint32_t kCtlRun = kernel_profiler::kLinkSyncCtlRun, kCtlStop = kernel_profiler::kLinkSyncCtlStop;
constexpr uint32_t kHwUnitsPerNs = kernel_profiler::kLinkSyncStampUnitsPerNs;
constexpr uint32_t kPaceTicks =
    kernel_profiler::kLinkSyncPaceTicks;  // the product's round period; tests pick their own
static_assert(kTripsPerRound % kBurstFrames == 0 && kBurstFrames >= 2 && (kFramePhaseStep & 1) == 1);
static_assert(kFrameBytes >= 80 && kFrameBytes % 16 == 0 && kFrameBytes >= sizeof(eth_channel_sync_t));
static_assert(sizeof(eth_channel_sync_t) <= 4 * kWordOffsetLo && 4 * (kWordTimerOk + 1) <= kFrameStampField);
static_assert(kFrameStampField + 10 <= kFrameBytes);
static_assert(kSlotsOffset + kSlotsBytes <= kCtlOffset);
static_assert(kCtlOffset + 8 + 14 * sizeof(uint32_t) <= kernel_profiler::kLinkSyncRingOffset);  // StopDiag::write

// Frame j of a round rides in slot j % kBurstFrames, the same L1 address on both ends, so the receiver's echo lands
// on the frame it answers. The sync word carries the round and the trip: bytes_sent in the frame, which the receiver
// clears once it has taken it, and receiver_ack in the echo; reserved_2 carries the round's full number.
inline __attribute__((always_inline)) volatile eth_channel_sync_t* slot(uint32_t base, uint32_t j) {
    return reinterpret_cast<volatile eth_channel_sync_t*>(base + kSlotsOffset + (j % kBurstFrames) * kFrameBytes);
}
inline __attribute__((always_inline)) volatile eth_channel_sync_t* pilot(uint32_t base) {
    return reinterpret_cast<volatile eth_channel_sync_t*>(base + kPilotOffset);
}
constexpr uint32_t kTripMask = 0x1FF;
constexpr uint32_t frame_key(uint32_t round, uint32_t j) { return (round << 9) | (j + 1); }
static_assert(kTripsPerRound <= kTripMask);
// Frame j's phase of the stamp tick in wall cycles, c16 being wall cycles per refclk tick times 16.
inline __attribute__((always_inline)) uint32_t frame_phase_cycles(uint32_t j, uint32_t c16) {
    return (((j * kFramePhaseStep) & (kTripsPerRound - 1)) * c16) >> 12;
}

// Waits to a wall-clock target with a delay loop calibrated once, so a frame sits at its cycle wherever its wait
// began: a poll on the wall clock exits a read's latency late by an amount set by where the loop was, and that is a
// function of when the wait began, so polling alone bent the grid by an AICLK-dependent amount.
struct Pacer {
    uint32_t iter16 = 32;  // wall cycles per delay-loop turn, x16
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
    __attribute__((always_inline)) void until(uint32_t target) const {
        const int32_t rem = static_cast<int32_t>(target - rd(kWallClockLo)) - 16;
        if (rem > 0) {
            turns((static_cast<uint32_t>(rem) * 16u) / iter16);
        }
        while (static_cast<int32_t>(rd(kWallClockLo) - target) < 0) {
        }
    }
};

// One side's sum of one stamp kind over a round, relative to its first so it stays in 32 bits and the average costs
// 32-bit divides, not the 64-bit routine a runtime count would otherwise pull into the ERISC's text.
struct StampSum {
    uint32_t n = 0;
    uint64_t base = 0;
    uint32_t rel = 0;  // sum of (stamp - base), ns: 256 stamps spread over a 10 ms round reach ~1.3e9
    __attribute__((always_inline)) void reset() {
        n = 0;
        rel = 0;
    }
    __attribute__((always_inline)) void add(uint64_t ts) {
        if (n == 0) {
            base = ts;
        }
        rel += static_cast<uint32_t>(ts - base);
        n++;
    }
    // The average in quarter-ns of the refclk domain: PTP64NS minus the timer's offset from the CFR count, the offset
    // taken off in 64ths of a ns and the one rounding done last, so no part of it settles on every stamp of the
    // session. The quotient and remainder of the ns sum are scaled separately: the sum itself times the units would
    // not fit.
    uint64_t q(int64_t ptp_offset_64) const {
        static_assert(64 % kHwUnitsPerNs == 0);
        constexpr int64_t kPerUnit = 64 / kHwUnitsPerNs;
        const uint32_t whole = rel / n;
        const uint32_t part = rel - whole * n;
        // part < n <= kTripsPerRound, so the remainder's share stays a 32-bit division: a 64-bit one is a routine.
        const int64_t avg64 = static_cast<int64_t>(base) * 64 - ptp_offset_64 + static_cast<int64_t>(whole) * 64 +
                              static_cast<int64_t>((part * 64u + n / 2) / n);
        return static_cast<uint64_t>((avg64 + kPerUnit / 2) / kPerUnit);
    }
};
// This end's offset and timer flag into a frame it is about to send, and its stamp field cleared: the slot still holds
// the stamp of the last frame that came in through it, so a frame the MAC did not stamp reads as zero, not as that.
template <typename Session>
__attribute__((always_inline)) inline void carry_offset(volatile eth_channel_sync_t* s, const Session& sess) {
    volatile uint32_t* w = reinterpret_cast<volatile uint32_t*>(s);
    w[kWordOffsetLo] = static_cast<uint32_t>(sess.ptp_offset_64);
    w[kWordOffsetHi] = static_cast<uint32_t>(static_cast<uint64_t>(sess.ptp_offset_64) >> 32);
    w[kWordTimerOk] = sess.timer_ok ? 1u : 0u;
    w[kFrameStampHiWord] = 0;
    w[kFrameStampHiWord + 1] = 0;
}

// What each end leaves past its control word for the host's log (streaming_profiler_device.cpp reads it back): +8
// rounds, +12 the timer word (0 no hardware path, 1 ran, 2 never acknowledged its rate), +16 wall cycles inside
// steps that sent or took frames and +24 wall cycles of the run (two words each), +32 refclk ticks of the run (two
// words), +40 the longest such step in wall cycles, then the counts: +44 rounds not recorded, +48 frames with queue
// units beyond their own (a keepalive or a resend), +52 frames whose pilot or itself did not hand off in time, +56
// bursts whose ingress stamps did not match their frames, +60 frames that came in without an egress stamp.
struct StopDiag {
    uint32_t rounds = 0, timer = 0;
    uint64_t hold = 0, span_wall = 0, span_refclk = 0;
    uint32_t hold_max = 0;
    uint32_t drop[5] = {};
    __attribute__((always_inline)) void note_hold(uint32_t cycles) {
        hold += cycles;
        hold_max = cycles > hold_max ? cycles : hold_max;
    }
    __attribute__((always_inline)) void note_round(bool recorded) {
        rounds++;
        drop[0] += !recorded;
    }
    void write(uint32_t stop_addr) const {
        volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stop_addr + 8);
        const uint32_t words[14] = {
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
            drop[3],
            drop[4]};
        for (uint32_t i = 0; i < 14; i++) {
            w[i] = words[i];
        }
    }
};

// One side's hardware round: the peer's egress stamps, read from the frames it sent this side, paired with their
// ingress stamps here, and the peer's PTP offset they came with.
struct HwRound {
    uint32_t id = 0;
    StampSum tx, rx;
    int64_t peer_offset_64 = 0;
    bool peer_ok = false, peer_seen = false;
    __attribute__((always_inline)) void begin(uint32_t round) {
        id = round;
        tx.reset();
        rx.reset();
        peer_seen = false;
        peer_ok = true;
    }
    // A frame from the peer: the offset and timer flag it carried, and its egress stamp, 0 if the MAC did not stamp it.
    __attribute__((always_inline)) uint64_t peer_frame(const volatile uint32_t* w) {
        const int64_t off = static_cast<int64_t>((static_cast<uint64_t>(w[kWordOffsetHi]) << 32) | w[kWordOffsetLo]);
        peer_ok = peer_ok && w[kWordTimerOk] == 1 && (!peer_seen || off == peer_offset_64);
        peer_offset_64 = off;
        peer_seen = true;
        return frame_stamp(w);
    }
    __attribute__((always_inline)) void pair(const uint64_t* tx_ts, const uint64_t* rx_ts, StopDiag& diag) {
        for (uint32_t m = 0; m < kBurstFrames; m++) {
            if (tx_ts[m] == 0) {
                diag.drop[4]++;
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
template <typename Session>
__attribute__((noinline)) bool take_stamps(const Session&, uint64_t* rx, StopDiag& diag) {
    bool ok = (raw::rd(kRxThStatus) & (kRxThStatusFull | kRxThStatusEntriesMask)) == kBurstFrames;
    for (uint32_t i = 0; ok && i < kBurstFrames; i++) {
        raw::RxStamp st{};
        ok = raw::rx_th_pop(st) && st.valid && st.label == Session::kLabel;
        rx[i] = st.rx_ts;
    }
    if (!ok) {
        raw::rx_th_flush();
        diag.drop[3]++;
    }
    return ok;
}
// The queue's own keepalives are kept out of the armed window. A keepalive is the queue's TT-link sequence update,
// carrying its acks for the other direction, and the MAC stamps whatever the queue hands off while armed: a keepalive
// stamped at byte 82 of a far shorter packet reaches the peer corrupt, which forces resends and, now and then, a
// spurious ack that leaves the peer's queue re-sending a window the peer never accepts. A keepalive is generated only
// after the keepalive timeout without a packet sent (documented; measured 8002 cycles), so a pilot frame sent just
// before arming leaves the queue none to generate while armed, the armed window being one frame's. arm_burst waits for
// the pilot's two WORD_CNT units (a keepalive that left just before it is one and cannot pass for it), then arms and
// anchors the counters; false if the pilot did not go. The pilot goes under the queue's boot header row, so the peer's
// classifier does not stamp it: ingress stamps stay the frames'.
struct Anchor {
    uint32_t start = 0, word = 0;
};
// A frame goes only onto a free queue, and the queue is given kHandoffSpins polls to free up: under fabric load a
// link-level resend on it can keep it busy for good, and a router waiting on it stops serving the fabric, which then
// deadlocks. No context switch while waiting: the core's owner may be a router, whose switches to base firmware are
// coordinated with the tile's other RISC.
__attribute__((always_inline)) inline bool txq_free() {
    for (uint32_t spin = 0; internal_::eth_txq_is_busy(kLinkTxq); spin++) {
        if (spin == kHandoffSpins) {
            return false;
        }
    }
    return true;
}
__attribute__((always_inline)) inline bool issue(volatile eth_channel_sync_t* s) {
    if (!txq_free()) {
        return false;
    }
    const uint32_t addr = reinterpret_cast<uint32_t>(s);
    internal_::eth_send_packet_unsafe(kLinkTxq, addr >> 4, addr >> 4, kFrameBytes >> 4);
    return true;
}
template <typename Session>
__attribute__((noinline)) inline bool arm_burst(const Session& s, uint32_t base, Anchor& at, StopDiag& diag) {
    const uint32_t units0 = raw::txq_word_cnt(Session::kTxq);
    tx_header_row_select(s, true);
    const bool went = issue(pilot(base)) && txq_free();
    tx_header_row_select(s, false);
    if (!went) {
        diag.drop[2]++;
        return false;
    }
    for (uint32_t spin = 0; raw::txq_word_cnt(Session::kTxq) - units0 < 2; spin++) {
        if (spin == kHandoffSpins) {
            diag.drop[2]++;
            return false;
        }
    }
    stamps_arm_in_frame(s);
    at.start = raw::txq_pkt_start_cnt(Session::kTxq);
    at.word = raw::txq_word_cnt(Session::kTxq);
    return true;
}
// Disarms once the armed frames have all started: ts_cmd is sticky and is sampled as the queue latches each frame's
// command, so a frame still queued when the polls run out goes without its egress stamp. WORD_CNT trails the starts
// while the MAC is busy with the fabric's queue, so it is read after them; units beyond two per frame are a keepalive
// or a link-level resend that went while armed.
template <typename Session>
__attribute__((noinline)) inline void finish_burst(
    const Session& s, const Anchor& at, uint32_t frames, StopDiag& diag) {
    for (uint32_t spin = 0; raw::txq_pkt_start_cnt(Session::kTxq) - at.start < frames; spin++) {
        if (spin == kHandoffSpins) {
            diag.drop[2]++;
            break;
        }
    }
    stamps_disarm(s);
    diag.drop[1] += raw::txq_word_cnt(Session::kTxq) - at.word > 2 * frames;
}

// The tick grid of an end's bursts, the receiver's echoes being a burst of its own. c16 is wall cycles per refclk
// tick, x16: the frames' phases of the tick are spun in wall cycles, and a grid scaled by the wrong AICLK covers more
// or less than the tick, which biases the stamps' rounding by stamp kind. AICLK is a PLL multiple of the refclk's
// crystal in steps of an eighth (6.25 MHz, measured: every run's slope is on that grid to 1e-9), so a rough ratio
// between two readings rounds to the exact value; one that rounds badly has a DVFS step inside it and the previous
// value stands. 1.25 GHz until measured.
struct Grid {
    uint32_t c16 = 400;
    uint32_t per_c16 = 0xFFFFFFFFu / 400;  // 2^32 / c16: send() multiplies where it would divide
    uint32_t walk = 1;                     // the bursts' comb origins, phase_walk's state
    uint32_t origin = 0;                   // this burst's comb origin, a wall cycle
    Pacer pacer;
    void start(uint32_t seed) {
        pacer.calibrate();
        walk = seed | 1u;
    }
    // Readings up to 2^24 wall cycles apart, so the ratio x256 stays in 32 bits.
    __attribute__((always_inline)) void rate(const Instant& a, const Instant& b) {
        const uint64_t wall = b.wall() - a.wall();
        if (a.refclk == 0 || b.refclk <= a.refclk || (wall >> 24) != 0) {
            return;
        }
        const uint32_t q8 = (static_cast<uint32_t>(wall) * 256u) /
                            static_cast<uint32_t>(b.refclk - a.refclk);  // ratio x256: a grid step is 32
        const uint32_t snapped = ((q8 + 16u) / 32u) * 32u;
        if (q8 + 12u >= snapped && q8 <= snapped + 12u && (snapped >> 4) != c16) {
            c16 = snapped >> 4;
            per_c16 = 0xFFFFFFFFu / c16;
        }
    }
    // A burst's comb origin, a pseudo-random part of a tick past now. walk's top 16 bits scale into [0, cycles per
    // tick): one multiply, no division routine in a router's text.
    __attribute__((always_inline)) void begin() {
        phase_walk(walk);
        origin = rd(kWallClockLo) + (((walk >> 16) * (c16 >> 4)) >> 16);
    }
    // Trip j under an arming of its own, on the first wall cycle past the arming that is a whole number of ticks from
    // the burst's origin plus the trip's phase: each frame of a burst goes at a step of its own, and the phases stay
    // exact however far apart the steps fall. A burst whose steps span 2^26 cycles or more restarts its comb, which
    // keeps the multiply's tick count within one of the exact one. False if the queue did not take the frame in time;
    // it is left for a later step.
    template <typename Session>
    __attribute__((noinline)) bool send(const Session& s, uint32_t base, uint32_t j, StopDiag& diag) {
        Anchor at;
        if (!arm_burst(s, base, at, diag)) {
            return false;
        }
        const uint32_t soon = rd(kWallClockLo) + 32;
        if (soon - origin >= (1u << 26)) {
            begin();
        }
        const uint32_t first = origin + frame_phase_cycles(j, c16);
        uint32_t target = first;
        const int32_t ahead = static_cast<int32_t>(soon - first);
        if (ahead > 0) {
            // floor(ahead * 16 / c16), or one under it, then the tick or two up to the first slot past soon
            uint32_t k = static_cast<uint32_t>((static_cast<uint64_t>(ahead) * per_c16) >> 28);
            for (uint32_t n = 0; n < 3; n++, k++) {
                target = first + static_cast<uint32_t>((static_cast<uint64_t>(k) * c16) >> 4);
                if (static_cast<int32_t>(target - soon) >= 0) {
                    break;
                }
            }
        }
        pacer.until(target);
        const bool went = issue(slot(base, j));
        finish_burst(s, at, went ? 1u : 0u, diag);
        return went;
    }
};

// The records: one per stamp average, this core's refclk-domain reading against its wall clock with the round's
// number and the stamp's role, so the host pairs the two ends by identity and fits refclk against refclk: DVFS on
// either chip's wall clock cannot enter the link solve. They go to the ring at the end of this core's link L1
// (hostdev kLinkSyncRingOffset), its count published in this core's profiler control vector for the pusher's sweep.
namespace link {
constexpr uint32_t kRoleT0 = kernel_profiler::kSyncRoleT0;
constexpr uint32_t kRoleT1 = kernel_profiler::kSyncRoleT1;
constexpr uint32_t kRoleT1B = kernel_profiler::kSyncRoleT1B;
constexpr uint32_t kRoleT2 = kernel_profiler::kSyncRoleT2;
// `Bracket`: the record's (wall, refclk) pair is read at a refclk update (read_bracketed, ~5 us of spinning), so the
// host's AICLK-to-AICLK check of the round reads the wall clock to a cycle; a router's end takes the plain pair.
template <bool Bracket>
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
        const Instant t = Bracket ? read_bracketed() : read_instant();
        volatile uint32_t* r = reinterpret_cast<volatile uint32_t*>(
            base + (n % kernel_profiler::kLinkSyncRingRecords) * kernel_profiler::kSyncRecordWords * 4);
        r[kernel_profiler::SYNC_META] =
            ((t.spins < 0xFFFFu ? t.spins : 0xFFFFu) << 16) | (kernel_profiler::kSyncKindLink << 8) | role;
        r[kernel_profiler::SYNC_ROUND] = round;
        r[kernel_profiler::SYNC_VALUE_LO] = static_cast<uint32_t>(value);
        r[kernel_profiler::SYNC_VALUE_HI] = static_cast<uint32_t>(value >> 32);
        r[kernel_profiler::SYNC_WALL_LO] = t.wall_lo;
        r[kernel_profiler::SYNC_WALL_HI] = t.wall_hi;
        r[kernel_profiler::SYNC_REF_LO] = static_cast<uint32_t>(t.refclk);
        r[kernel_profiler::SYNC_REF_HI] = static_cast<uint32_t>(t.refclk >> 32);
        asm volatile("fence" ::: "memory");
        *tail = ++n;
    }
};
}  // namespace link

// The two ends of a link, driven by whoever owns the core -- a resident kernel or the fabric router: open() before
// the link handshake, start() once the peer is up, step() as often as the core can spare, stop() at teardown. A step
// returns at once when nothing is due, and otherwise sends, or reads and echoes, one frame, so it holds the core for
// a frame's hand-offs at most. DataCache says whether the core runs with its L1 data cache on, in which case a step
// invalidates before polling what the peer or the host wrote; a router runs with it off and skips the fence. Both
// are constant-initialised: the ERISC runs no dynamic init.
constexpr uint32_t kRatioTicks = 1000;  // 20 us before a slot: read jitter of tens of cycles is under a tenth of a step

// Both ends clear the slots before the handshake: a frame that lands before its receiver's start() is then kept, and
// a burst whose frame was lost would hold the link for good.
inline void clear_slots(uint32_t base) {
    for (uint32_t j = 0; j < kBurstFrames; j++) {
        volatile eth_channel_sync_t* s = slot(base, j);
        s->bytes_sent = 0;
        s->receiver_ack = 0;
        s->src_id = 0;
        s->reserved_2 = 0;
    }
}

template <bool DataCache = true, bool Bracket = true>
struct SenderLink {
    LinkSession sess;
    uint32_t slot_base = 0, burst_ticks = 0;
    // The earliest refclk tick of the next burst, and the bursts issued: a round is kBurstsPerRound of them.
    uint64_t slot_cfr = 0, bursts = 0;
    // The next slot and its ratio sample as wall cycles, set from the burst's own reading: an idle step then costs
    // one wall-clock read where the refclk costs two, and nothing a stamp depends on is timed by it.
    uint32_t slot_wall = 0, pre_wall = 0;
    uint32_t next_round = 0, diag_addr = 0;
    Instant start_at{}, pre{};
    Grid grid;
    StopDiag diag;
    HwRound rnd;
    link::Ring<Bracket> ring;
    uint32_t round = 0;
    // The burst in flight, from trip out_j0: out_sent of its frames issued, their echoes awaited once all are.
    uint32_t out_j0 = 0, out_sent = kBurstFrames, next_j = 0;
    bool started = false;

    bool open(uint32_t l1) {
        slot_base = l1;
        clear_slots(l1);
        return sess.begin();
    }
    void start(uint32_t l1, uint32_t ctl, uint32_t pace_ticks) {
        diag_addr = ctl;
        ring.open(l1);
        burst_ticks = pace_ticks / kBurstsPerRound;
        start_at = read_instant();
        grid.start(start_at.wall_lo);
        slot_cfr = start_at.refclk + kRatioTicks;
        schedule(start_at);
    }
    __attribute__((always_inline)) void schedule(const Instant& now) {
        const int64_t ticks = static_cast<int64_t>(slot_cfr - now.refclk);
        slot_wall = now.wall_lo + ((static_cast<uint32_t>(ticks < 0 ? 0 : ticks) * grid.c16) >> 4);
        pre_wall = slot_wall - ((kRatioTicks * grid.c16) >> 4);
    }
    __attribute__((always_inline)) void step() {
        if (out_sent != kBurstFrames) {
            send_next();
            return;
        }
        const uint32_t w = rd(kWallClockLo);
        if (static_cast<int32_t>(w - slot_wall) < 0) {
            if (pre.refclk == 0 && static_cast<int32_t>(w - pre_wall) >= 0) {
                pre = read_instant();
            }
            return;
        }
        if constexpr (DataCache) {
            invalidate_l1_cache();
        }
        if (rd(diag_addr) != kCtlRun) {
            // Rounds resume on a fresh slot one ratio window ahead, so the first burst has its sample. A round in
            // progress closes at the next burst rather than spanning the pause.
            bursts -= bursts % kBurstsPerRound;
            const Instant now = read_instant();
            slot_cfr = now.refclk + kRatioTicks;
            schedule(now);
            pre = Instant{};
            return;
        }
        if (started && !echoed()) {
            return;
        }
        burst();
    }
    void stop() {
        sess.end();
        write_diag();
    }

private:
    // The diagnostics are rewritten at every round's close, so a host that cannot stop this end (a router) still
    // reads the current figures.
    void write_diag() {
        const Instant now = read_instant();
        // The session's PTP offset rides in the timer word: a tick multiple, so its low two bits are free.
        diag.timer = (sess.timer_ok ? 1u : 2u) | (static_cast<uint32_t>(sess.ptp_offset_64 >> 6) & ~3u);
        diag.span_wall = now.wall() - start_at.wall();
        diag.span_refclk = now.refclk - start_at.refclk;
        diag.write(diag_addr);
    }
    void close_round() {
        const bool recorded = sess.timer_ok && rnd.usable();
        if (recorded) {
            ring.record_hw(rnd.tx.q(rnd.peer_offset_64), round, link::kRoleT1B);
            ring.record_hw(rnd.rx.q(sess.ptp_offset_64), round, link::kRoleT2);
        }
        diag.note_round(recorded);
        write_diag();
    }
    __attribute__((always_inline)) bool echoed() const {
        for (uint32_t i = 0; i < kBurstFrames; i++) {
            if (slot(slot_base, i)->receiver_ack != frame_key(round, out_j0 + i)) {
                return false;
            }
        }
        return true;
    }
    __attribute__((noinline)) void send_next() {
        const uint32_t w = rd(kWallClockLo);
        out_sent += grid.send(sess, slot_base, out_j0 + out_sent, diag);
        diag.note_hold(rd(kWallClockLo) - w);
    }
    // A slot, once all the last burst's echoes are in: its pairs (egress stamps from the echoes themselves, ingress
    // stamps here), the round's records at a round boundary, then the next burst's frames, which the steps after it
    // send. A round is a count of bursts, not of slots: a burst whose steps outlast its slot delays the next, and the
    // round still holds all its trips, so its phases cover the tick exactly. The next slot is a slot on, or a ratio
    // window past this burst when that is later.
    __attribute__((noinline)) void burst() {
        const Instant now = read_instant();
        grid.rate(pre, now);
        pre = Instant{};
        if (started) {
            uint64_t tx[kBurstFrames], rx[kBurstFrames];
            for (uint32_t i = 0; i < kBurstFrames; i++) {
                tx[i] = rnd.peer_frame(reinterpret_cast<volatile uint32_t*>(slot(slot_base, i)));
            }
            if (take_stamps(sess, rx, diag)) {
                rnd.pair(tx, rx, diag);
            }
        }
        if (bursts % kBurstsPerRound == 0) {
            if (started) {
                close_round();
            }
            round = next_round++;
            rnd.begin(round);
            next_j = 0;
        }
        started = true;
        out_j0 = next_j;
        next_j += kBurstFrames;
        for (uint32_t i = 0; i < kBurstFrames; i++) {
            volatile eth_channel_sync_t* s = slot(slot_base, i);
            carry_offset(s, sess);
            s->receiver_ack = 0;
            s->reserved_2 = round;
            s->bytes_sent = frame_key(round, out_j0 + i);
        }
        grid.begin();
        out_sent = 0;
        bursts++;
        slot_cfr += burst_ticks;
        if (static_cast<int64_t>(slot_cfr - now.refclk) < static_cast<int64_t>(kRatioTicks)) {
            slot_cfr = now.refclk + kRatioTicks;
        }
        schedule(now);
        diag.note_hold(rd(kWallClockLo) - now.wall_lo);
    }
};

template <bool DataCache = true, bool Bracket = true>
struct ReceiverLink {
    LinkSession sess;
    uint32_t slot_base = 0, diag_addr = 0;
    uint32_t round = 0;
    // The burst coming in, from trip echo_j0: `taken` of its frames read, their egress stamps in tx, and `echoed` of
    // those echoed.
    uint32_t echo_j0 = 0, taken = 0, echoed = 0;
    uint64_t tx[kBurstFrames] = {};
    link::Ring<Bracket> ring;
    bool started = false;
    Instant start_at{}, last{};
    Grid grid;
    StopDiag diag;
    HwRound rnd;

    bool open(uint32_t l1) {
        slot_base = l1;
        clear_slots(l1);
        return sess.begin();
    }
    // The receiver follows the sender's cadence; pace_ticks is the sender's and is not used here.
    void start(uint32_t l1, uint32_t ctl, uint32_t = 0) {
        diag_addr = ctl;
        ring.open(l1);
        start_at = read_instant();
        grid.start(start_at.wall_lo);
    }
    // Each frame is read and echoed at the step that finds it, the burst's ingress stamps taken with its last frame,
    // before that frame's echo: the sender issues nothing more until every echo is in. An echo the queue did not take
    // goes at the next step.
    __attribute__((always_inline)) void step() {
        if constexpr (DataCache) {
            invalidate_l1_cache();
        }
        if (echoed != taken) {
            echo();
        } else if (slot(slot_base, taken)->bytes_sent != 0) {
            take();
            echo();
        }
    }
    void stop() {
        sess.end();
        write_diag();
    }

private:
    void write_diag() {
        const Instant now = read_instant();
        // The session's PTP offset rides in the timer word: a tick multiple, so its low two bits are free.
        diag.timer = (sess.timer_ok ? 1u : 2u) | (static_cast<uint32_t>(sess.ptp_offset_64 >> 6) & ~3u);
        diag.span_wall = now.wall() - start_at.wall();
        diag.span_refclk = now.refclk - start_at.refclk;
        diag.write(diag_addr);
    }
    void close_round() {
        const bool recorded = sess.timer_ok && rnd.usable();
        if (recorded) {
            ring.record_hw(rnd.tx.q(rnd.peer_offset_64), round, link::kRoleT0);
            ring.record_hw(rnd.rx.q(sess.ptp_offset_64), round, link::kRoleT1);
        }
        diag.note_round(recorded);
        write_diag();
    }
    // Frame `taken`: its egress stamp and the offset it carries, then its echo set up in the same slot, carrying its
    // key back and, once sent, this end's egress stamp. A burst's first frame measures the AICLK ratio since the last
    // and names the round, the sender's; a new one closes the previous. Its last frame takes the burst's ingress
    // stamps and pairs them.
    __attribute__((noinline)) void take() {
        const uint32_t w = rd(kWallClockLo);
        volatile eth_channel_sync_t* s = slot(slot_base, taken);
        if (taken == 0) {
            const Instant now = read_instant();
            grid.rate(last, now);
            last = now;
            if (!started || s->reserved_2 != round) {
                if (started) {
                    close_round();
                }
                started = true;
                round = s->reserved_2;
                rnd.begin(round);
            }
            echo_j0 = (s->bytes_sent & kTripMask) - 1;
            grid.begin();
        }
        tx[taken] = rnd.peer_frame(reinterpret_cast<volatile uint32_t*>(s));
        if (taken == kBurstFrames - 1) {
            uint64_t rx[kBurstFrames];
            if (take_stamps(sess, rx, diag)) {
                rnd.pair(tx, rx, diag);
            }
        }
        carry_offset(s, sess);
        s->receiver_ack = s->bytes_sent;
        s->bytes_sent = 0;
        taken++;
        diag.note_hold(rd(kWallClockLo) - w);
    }
    __attribute__((noinline)) void echo() {
        const uint32_t w = rd(kWallClockLo);
        echoed += grid.send(sess, slot_base, echo_j0 + echoed, diag);
        if (echoed == kBurstFrames) {
            taken = 0;
            echoed = 0;
        }
        diag.note_hold(rd(kWallClockLo) - w);
    }
};

// The end a core runs, chosen by role, with the one start(l1, ctl, pace_ticks) of both.
template <bool Sender, bool DataCache>
using LinkEnd = std::conditional_t<Sender, SenderLink<DataCache, false>, ReceiverLink<DataCache, false>>;

// Calls F(arg) with the caller-saved registers kept here instead of by the caller: the asm clobbers ra alone, so the
// code around the call site is register-allocated as if there were no call.
template <void (*F)(uint32_t)>
__attribute__((always_inline)) inline void saved_call(uint32_t arg) {
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
        end.start(l1, l1 + kCtlOffset, kPaceTicks);
    }
    __attribute__((noipa)) static void step_body(uint32_t) { end.step(); }
    __attribute__((noipa)) static void stop_body(uint32_t) { end.stop(); }
    __attribute__((always_inline)) void start(uint32_t l1) { saved_call<&start_body>(l1); }
    __attribute__((always_inline)) void step() { saved_call<&step_body>(0); }
    __attribute__((always_inline)) void stop() { saved_call<&stop_body>(0); }
};
template <bool DataCache>
struct RouterHook<1, DataCache> : HostedEnd<true, DataCache> {};
template <bool DataCache>
struct RouterHook<2, DataCache> : HostedEnd<false, DataCache> {};

}  // namespace tt::tt_metal::eth_ptp

#endif  // ARCH_BLACKHOLE
