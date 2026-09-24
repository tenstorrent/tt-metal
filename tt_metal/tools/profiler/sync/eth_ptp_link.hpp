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

// A round is kTripsPerRound exchanges spread over the round's period in bursts of kBurstFrames, so a core lends the
// link a few frames' worth of time at once rather than a round's: the link relation is a line in the refclk domain,
// so the stamps average to the same offset however they are spread. Every stamp is quantised to the timer's 20 ns
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
constexpr uint32_t kFrameTicks = 12;       // 240 ns between a burst's frames: a receiver takes a frame in ~150 cycles
constexpr uint32_t kFramePhaseStep = 157;  // odd, so the 256 phases are a permutation
// Waits on a link the fabric may be loading: our frames queue behind its at the MAC on the way out, and the frames
// of a burst then reach the receiver spread out. Both are bounds on a step's hold.
constexpr uint32_t kBurstStampSpins =
    256;  // polls for a burst's hand-offs, then for its stamps, before it is given up: ~6 us each
constexpr uint32_t kBurstFollowSpins = 256;  // polls for each of a burst's later frames once its first is in: ~6 us
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
// on the word the sender polls. The sync word carries the round and the trip, so a slot's stale content can match
// nothing the receiver waits for, and reserved_2 the round's full number.
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

// Waits to a wall-clock target with a delay loop calibrated once, so a burst's later frames sit at their cycle
// from the first: a poll on the wall clock exits a read's latency late by an amount set by where the loop was, and
// that is a function of the previous frame's phase, so polling alone bent the grid by an AICLK-dependent amount.
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
// bursts and +24 wall cycles of the run (two words each), +32 refclk ticks of the run (two words), +40 the longest
// burst in wall cycles, then the counts: +44 rounds not recorded, +48 bursts with queue units beyond their frames' (a
// keepalive or a resend), +52 bursts whose pilot or frames did not hand off in time, +56 frames whose ingress stamps
// could not be matched to them, +60 frames that came in without an egress stamp.
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
// ingress stamps here, and the peer's PTP offset they came with. The classifier's FIFO says nothing of which frame an
// ingress stamp belongs to, so a drain's stamps are credited, in arrival order, to the frames that came in since the
// last one only when there are exactly as many: a stamp is never paired with another frame's.
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
    // The egress stamps of the n frames that came in since the last drain, and the k ingress stamps it found.
    __attribute__((always_inline)) void pair(
        const uint64_t* tx_ts, const uint64_t* rx_ts, uint32_t n, uint32_t k, StopDiag& diag) {
        if (k != n) {
            diag.drop[3] += n;
            return;
        }
        for (uint32_t m = 0; m < n; m++) {
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
// The queue's own keepalives are kept out of the armed window. A keepalive is the queue's TT-link sequence update,
// carrying its acks for the other direction, and the MAC stamps whatever the queue hands off while armed: a keepalive
// stamped at byte 82 of a far shorter packet reaches the peer corrupt, which forces resends and, now and then, a
// spurious ack that leaves the peer's queue re-sending a window the peer never accepts. A keepalive is generated only
// after the keepalive timeout without a packet sent (documented; measured 8002 cycles), so a pilot frame sent just
// before arming leaves the queue none to generate while armed, provided the armed window has no idle stretch: the
// sender's frames go out back to back, and the receiver arms per echo since the frames it answers can come
// microseconds apart. arm_burst waits for the pilot's two WORD_CNT units (a keepalive that left just before it is
// one and cannot pass for it), then arms and anchors the counters; false, the frames counted lost, if the pilot did
// not go. The pilot goes under the queue's boot header row, so the peer's classifier does not stamp it: ingress
// stamps stay the frames'.
struct Anchor {
    uint32_t start = 0, word = 0;
};
// A frame goes only onto a free queue, and the queue is given kBurstStampSpins polls to free up: under fabric load
// a link-level resend on it can keep it busy for good, and a router waiting on it stops serving the fabric, which
// then deadlocks. No context switch while waiting: the core's owner may be a router, whose switches to base firmware
// are coordinated with the tile's other RISC.
__attribute__((always_inline)) inline bool txq_free() {
    for (uint32_t spin = 0; internal_::eth_txq_is_busy(kLinkTxq); spin++) {
        if (spin == kBurstStampSpins) {
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
        if (spin == kBurstStampSpins) {
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
        if (spin == kBurstStampSpins) {
            diag.drop[2]++;
            break;
        }
    }
    stamps_disarm(s);
    diag.drop[1] += raw::txq_word_cnt(Session::kTxq) - at.word > 2 * frames;
}

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
// returns at once when nothing is due; when a burst (sender) or a frame (receiver) is due it is handled whole, so a
// step holds the core for ~1 us at most. DataCache says whether the core runs with its L1 data cache on, in which case
// a step invalidates before polling what the peer or the host wrote; a router runs with it off and skips the fence.
// Both are constant-initialised: the ERISC runs no dynamic init.
constexpr uint32_t kRatioTicks = 1000;  // 20 us before a slot: read jitter of tens of cycles is under a tenth of a step

template <bool DataCache = true, bool Bracket = true>
struct SenderLink {
    LinkSession sess;
    uint32_t slot_base = 0, burst_ticks = 0;
    // Wall cycles per refclk tick, x16: the frames' phases of the tick are spun in wall cycles, and a grid scaled by
    // the wrong AICLK covers more or less than the tick, which biases the stamps' rounding by stamp kind. AICLK is a
    // PLL multiple of the refclk's crystal in steps of an eighth (6.25 MHz, measured: every run's slope is on that
    // grid to 1e-9), so a rough ratio over the kRatioTicks before each slot rounds to the exact value; one that
    // rounds badly has a DVFS step inside it and the previous value stands. 1.25 GHz until measured.
    uint32_t c16 = 400;
    uint64_t slot_cfr = 0, bursts = 0;
    // The next slot and its ratio sample as wall cycles, set from the burst's own reading: an idle step then costs
    // one wall-clock read where the refclk costs two, and nothing a stamp depends on is timed by it.
    uint32_t slot_wall = 0, pre_wall = 0;
    uint32_t next_round = 0, diag_addr = 0;
    uint32_t walk = 1;  // the bursts' comb origins, phase_walk's state
    Instant start_at{}, pre{};
    Pacer pacer;
    StopDiag diag;
    HwRound rnd;
    link::Ring<Bracket> ring;
    uint32_t round = 0;
    uint32_t sent_round = 0, sent_j0 = 0;  // the last burst sent, whose echoes the next burst reads
    bool sent = false;

    bool open() { return sess.begin(); }
    void start(uint32_t l1, uint32_t ctl, uint32_t pace_ticks) {
        slot_base = l1;
        diag_addr = ctl;
        ring.open(l1);
        burst_ticks = pace_ticks / kBurstsPerRound;
        for (uint32_t j = 0; j < kBurstFrames; j++) {
            volatile eth_channel_sync_t* s = slot(slot_base, j);
            s->bytes_sent = 0;
            s->receiver_ack = 0;
            s->src_id = 0;
            s->reserved_2 = 0;
        }
        pacer.calibrate();
        start_at = read_instant();
        walk = start_at.wall_lo | 1u;
        slot_cfr = start_at.refclk + kFrameTicks;
        schedule(start_at);
    }
    __attribute__((always_inline)) void schedule(const Instant& now) {
        const int64_t ticks = static_cast<int64_t>(slot_cfr - now.refclk);
        slot_wall = now.wall_lo + ((static_cast<uint32_t>(ticks < 0 ? 0 : ticks) * c16) >> 4);
        pre_wall = slot_wall - ((kRatioTicks * c16) >> 4);
    }
    __attribute__((always_inline)) void step() {
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
            // Rounds resume on a fresh slot, not a backlog of missed ones; one ratio window ahead, so the first
            // burst has its sample. A round in progress closes at the next burst rather than spanning the pause.
            bursts -= bursts % kBurstsPerRound;
            const Instant now = read_instant();
            slot_cfr = now.refclk + kRatioTicks;
            schedule(now);
            pre = Instant{};
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
    // A burst: the previous burst's echoes (their egress stamps from the echoes themselves, their ingress stamps
    // here), the round's records at a round boundary, then kBurstFrames frames kFrameTicks apart from the first, each
    // at its phase of the tick, all in wall cycles from one reading (the slot wait's exit shifts the whole burst
    // alike, which the grid does not mind). The echoes come back after the burst, so none arrives while ours go out.
    __attribute__((noinline)) void burst() {
        const Instant now = read_instant();
        const uint32_t hold0 = now.wall_lo;
        if (pre.refclk != 0 && now.refclk > pre.refclk) {
            const uint32_t q8 = (static_cast<uint32_t>(now.wall() - pre.wall()) * 256u) /
                                static_cast<uint32_t>(now.refclk - pre.refclk);  // ratio x256: a grid step is 32
            const uint32_t snapped = ((q8 + 16u) / 32u) * 32u;
            if (q8 + 12u >= snapped && q8 <= snapped + 12u) {
                c16 = snapped >> 4;
            }
        }
        pre = Instant{};
        uint64_t tx[kBurstFrames], rx[kBurstFrames];
        uint32_t n = 0, k = 0;
        if (sent) {
            for (uint32_t i = 0; i < kBurstFrames; i++) {
                volatile eth_channel_sync_t* e = slot(slot_base, i);
                if (e->receiver_ack == frame_key(sent_round, sent_j0 + i)) {
                    tx[n++] = rnd.peer_frame(reinterpret_cast<volatile uint32_t*>(e));
                }
            }
            sent = false;
        }
        rx_stamps_drain(sess, [&](uint64_t ts) {
            if (k < kBurstFrames) {
                rx[k] = ts;
            }
            k++;
        });
        rnd.pair(tx, rx, n, k, diag);
        const uint32_t j0 = static_cast<uint32_t>(bursts % kBurstsPerRound) * kBurstFrames;
        if (j0 == 0) {
            if (bursts != 0) {
                close_round();
            }
            round = next_round++;
            rnd.begin(round);
        }
        Anchor at;
        if (arm_burst(sess, slot_base, at, diag)) {
            const uint32_t spacing = (kFrameTicks * c16) >> 4;
            const uint32_t phase0 = frame_phase_cycles(j0, c16);
            phase_walk(walk);
            // walk's top 16 bits scale into [0, cycles per tick): one multiply, no division routine in a router's text.
            const uint32_t w0 = rd(kWallClockLo) + 32 + phase0 + (((walk >> 16) * (c16 >> 4)) >> 16);
            for (uint32_t i = 0; i < kBurstFrames; i++) {
                const uint32_t j = j0 + i;
                volatile eth_channel_sync_t* s = slot(slot_base, i);
                carry_offset(s, sess);
                s->receiver_ack = 0;
                pacer.until(w0 + i * spacing + frame_phase_cycles(j, c16) - phase0);
                s->reserved_2 = round;
                s->bytes_sent = frame_key(round, j);
                if (!issue(s)) {
                    break;
                }
            }
            finish_burst(sess, at, kBurstFrames, diag);
            sent = true;
            sent_round = round;
            sent_j0 = j0;
        }
        diag.note_hold(rd(kWallClockLo) - hold0);
        bursts++;
        slot_cfr += burst_ticks;
        schedule(now);
    }
};

template <bool DataCache = true, bool Bracket = true>
struct ReceiverLink {
    LinkSession sess;
    uint32_t slot_base = 0, diag_addr = 0;
    uint32_t round = 0, expect = 0;
    link::Ring<Bracket> ring;
    bool started = false, mid_burst = false;
    Instant start_at{};
    StopDiag diag;
    HwRound rnd;
    // The burst in progress: its frames' egress stamps and the ingress stamps drained for it so far.
    uint64_t btx[kBurstFrames] = {}, brx[kBurstFrames] = {};
    uint32_t bn = 0, bk = 0;

    bool open() { return sess.begin(); }
    // The receiver follows the sender's cadence; pace_ticks is the sender's and is not used here.
    void start(uint32_t l1, uint32_t ctl, uint32_t = 0) {
        slot_base = l1;
        diag_addr = ctl;
        ring.open(l1);
        for (uint32_t j = 0; j < kBurstFrames; j++) {
            volatile eth_channel_sync_t* s = slot(slot_base, j);
            s->bytes_sent = 0;
            s->receiver_ack = 0;
            s->src_id = 0;
            s->reserved_2 = 0;
        }
        start_at = read_instant();
    }
    // Every frame waiting: the next in order, or a round's first frame in slot 0 if the order broke. A burst's
    // frames are echoed within one step, waiting up to kBurstFollowSpins for each of the rest to land once the
    // first has: the owner of the core may come by rarely, and a burst split across steps would outlast the cover
    // of its pilot.
    __attribute__((always_inline)) void step() {
        uint32_t spins = 0;
        if constexpr (DataCache) {
            invalidate_l1_cache();
        }
        for (;;) {
            volatile eth_channel_sync_t* s = slot(slot_base, expect);
            uint32_t key = s->bytes_sent;
            if (key == 0) {
                volatile eth_channel_sync_t* first = slot(slot_base, 0);
                if (s != first) {
                    key = first->bytes_sent;
                    if ((key & kTripMask) == 1) {
                        s = first;
                    }
                }
                if (key == 0 || s != first || (key & kTripMask) != 1) {
                    if (mid_burst && spins < kBurstFollowSpins) {
                        spins++;
                        if constexpr (DataCache) {
                            invalidate_l1_cache();
                        }
                        continue;
                    }
                    return;
                }
            }
            mid_burst = frame(s, key);
            spins = 0;
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
    __attribute__((always_inline)) void drain_burst() {
        rx_stamps_drain(sess, [&](uint64_t ts) {
            if (bk < kBurstFrames) {
                brx[bk] = ts;
            }
            bk++;
        });
    }
    // A frame's ingress stamp reaches the FIFO after its payload is visible here, so the burst's stamps are waited
    // for until there are as many as frames came in, and only then credited to them.
    __attribute__((noinline)) void settle_burst() {
        drain_burst();
        for (uint32_t spin = 0; bk < bn && spin < kBurstStampSpins; spin++) {
            drain_burst();
        }
        rnd.pair(btx, brx, bn, bk, diag);
        bn = 0;
        bk = 0;
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
    // A frame: its egress stamp from the frame itself, its ingress stamp, and its echo from the same slot, which
    // carries its key back and this end's egress stamp, armed on its own behind a pilot and handed off before the next
    // echo is issued. The round's number is the sender's, read from the frame; a new one closes the previous. True
    // while the burst has frames to come.
    __attribute__((noinline)) bool frame(volatile eth_channel_sync_t* s, uint32_t key) {
        const Instant now = read_instant();
        const uint32_t j = (key & kTripMask) - 1;
        const uint32_t i = j % kBurstFrames;
        if ((i == 0 && bn != 0) || bn == kBurstFrames) {
            settle_burst();  // the previous burst's last frame never came
        }
        if (j == 0) {
            if (started) {
                close_round();
            }
            started = true;
            round = s->reserved_2;
            rnd.begin(round);
        } else if (key != frame_key(round, j)) {
            s->bytes_sent = 0;  // a frame of a round already given up
            return false;
        }
        btx[bn++] = rnd.peer_frame(reinterpret_cast<volatile uint32_t*>(s));
        drain_burst();
        carry_offset(s, sess);
        s->receiver_ack = key;
        s->bytes_sent = 0;
        Anchor at;
        const bool armed = arm_burst(sess, slot_base, at, diag);
        issue(s);
        if (armed) {
            finish_burst(sess, at, 1, diag);
        }
        if (i == kBurstFrames - 1) {
            settle_burst();
        }
        expect = j + 1;
        diag.note_hold(rd(kWallClockLo) - now.wall_lo);
        return i != kBurstFrames - 1;
    }
};

// The end a core runs, chosen by role, with the one start(l1, ctl, pace_ticks) of both.
template <bool Sender, bool DataCache>
using LinkEnd = std::conditional_t<Sender, SenderLink<DataCache, false>, ReceiverLink<DataCache, false>>;

// A router's end over its LINK_SYNC_ADDR region, at the product's pace.
template <bool Sender, bool DataCache>
struct HostedEnd {
    LinkEnd<Sender, DataCache> end;
    // All three always inline: as out-of-line members they shift the compiler's inlining of the router's own callees
    // and cost it 528 B of text, past the active-eth kernel-config budget.
    __attribute__((always_inline)) void start(uint32_t l1) {
        end.open();
        end.start(l1, l1 + kCtlOffset, kPaceTicks);
    }
    __attribute__((always_inline)) void step() { end.step(); }
    __attribute__((always_inline)) void stop() { end.stop(); }
};
template <bool DataCache>
struct RouterHook<1, DataCache> : HostedEnd<true, DataCache> {};
template <bool DataCache>
struct RouterHook<2, DataCache> : HostedEnd<false, DataCache> {};

}  // namespace tt::tt_metal::eth_ptp

#endif  // ARCH_BLACKHOLE
