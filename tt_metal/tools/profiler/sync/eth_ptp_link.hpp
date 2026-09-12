// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The streaming profiler's link half on the tile's 1588 hardware (internal/ethernet/eth_ptp.hpp): the session every
// sync kernel opens, how a round's frames are exchanged and its stamps accumulated and reported, and the PP_CLOCK
// records the two kernels write. Without the streaming profiler the record path compiles to nothing.

#pragma once

#include <cstdint>

#include "internal/ethernet/eth_ptp.hpp"
#if defined(PROFILE_KERNEL) && defined(PROFILE_STREAMING)
#include "tools/profiler/kernel_profiler.hpp"
#endif

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
// the tick whatever the cadence or AICLK, and the mean's rounding noise falls to ~0.4 ns per round. The stamp
// request is armed once per burst, and the queue's counters tell the frames' stamps from those of a keepalive the
// queue had waiting (arm_burst, collect_burst). Each side sums its stamps unpaired, so a round counts only if every
// frame produced both of a side's stamps. The round's first frame is exchanged alone before the next is issued, so
// its software stamps see an idle queue on both ends.
constexpr uint32_t kTripsPerRound = 256;
constexpr uint32_t kBurstFrames = 4;
constexpr uint32_t kBurstsPerRound = kTripsPerRound / kBurstFrames;
// A frame's payload: the sync word first, the rest padding. Two WORD_CNT units against a keepalive's one, which is
// what counts the frames among the queue's hand-offs; 80 bytes is the first two-unit size, 96 leaves a unit's margin
// either way. Frames of 16 to 128 bytes hand off in the same time.
constexpr uint32_t kFrameBytes = 96;
// The L1 both ends own, the same addresses on both (streaming_profiler_link_sync.hpp): where the peer's pilot lands,
// the frame slots, then the control and diagnostic words.
constexpr uint32_t kPilotOffset = 0;
constexpr uint32_t kSlotsOffset = kFrameBytes;
constexpr uint32_t kSlotsBytes = kBurstFrames * kFrameBytes;
constexpr uint32_t kCtlOffset = kSlotsOffset + kSlotsBytes;
constexpr uint32_t kFrameTicks = 12;       // 240 ns between a burst's frames: a receiver takes a frame in ~150 cycles
constexpr uint32_t kFramePhaseStep = 157;  // odd, so the 256 phases are a permutation
constexpr uint32_t kEchoSpins = 50'000;    // polls for the first frame's echo before the round is given up: ~1 ms
// Waits on a link the fabric may be loading: our frames queue behind its at the MAC on the way out, and the frames
// of a burst then reach the receiver spread out. Both are bounds on a step's hold.
constexpr uint32_t kBurstStampSpins =
    256;  // polls for a burst's hand-offs, then for its stamps, before it is given up: ~6 us each
constexpr uint32_t kBurstFollowSpins = 256;  // polls for each of a burst's later frames once its first is in: ~6 us
// The control word at the diagnostics' base, the host's: rounds are issued only while it reads kCtlRun, and a
// resident kernel exits on kCtlStop. Set once the profiler's consumer and trackers are up, so no round predates
// the clock coverage that places it.
constexpr uint32_t kCtlRun = 1, kCtlStop = 2;
constexpr uint32_t kHwUnitsPerNs = 4;
static_assert(kTripsPerRound % kBurstFrames == 0 && kBurstFrames >= 2 && (kFramePhaseStep & 1) == 1);
static_assert(kFrameBytes >= 80 && kFrameBytes % 16 == 0 && kFrameBytes >= sizeof(eth_channel_sync_t));
static_assert(kCtlOffset == 480);  // streaming_profiler_link_sync.hpp kCtlOffset

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
    // The average in quarter-ns of the refclk domain: PTP64NS minus the timer's offset from the CFR count. The
    // quotient and remainder of the ns sum are scaled separately: the sum itself times the units would not fit.
    uint64_t q(const LinkSession& s) const {
        const int64_t base_units = (static_cast<int64_t>(base) - s.ptp_offset_ns) * kHwUnitsPerNs;
        const uint32_t whole = rel / n;
        const uint32_t part = rel - whole * n;
        return static_cast<uint64_t>(base_units + whole * kHwUnitsPerNs + (part * kHwUnitsPerNs + n / 2) / n);
    }
};
// One side's hardware round: the egress stamps of the frames it sent and the ingress stamps of those it received.
struct HwRound {
    uint32_t id = 0;
    StampSum tx, rx;
    __attribute__((always_inline)) void begin(uint32_t round) {
        id = round;
        tx.reset();
        rx.reset();
    }
    bool complete(uint32_t frames) const { return tx.n == frames && rx.n == frames; }
};

// What each end leaves past its control word for the host's log (streaming_profiler_device.cpp reads it back): +8
// rounds, +12 the timer word (0 no hardware path, 1 ran, 2 never acknowledged its rate), +16 wall cycles inside
// bursts and +24 wall cycles of the run (two words each), +32 refclk ticks of the run (two words), +40 the longest
// burst in wall cycles, then the counts: +44 rounds dropped, +48 bursts with a hand-off beyond the frames (none, by
// the pilot's cover), +52 bursts whose frames did not all hand off or stamp in time, +56 rounds with the ingress
// count off, +60 waits for a frame or echo given up.
struct StopDiag {
    uint32_t rounds = 0, timer = 0;
    uint64_t hold = 0, span_wall = 0, span_refclk = 0;
    uint32_t hold_max = 0;
    uint32_t drop[5] = {};
    __attribute__((always_inline)) void note_hold(uint32_t cycles) {
        hold += cycles;
        hold_max = cycles > hold_max ? cycles : hold_max;
    }
    __attribute__((always_inline)) void note_round(const HwRound& r, bool ok, bool waits_given_up) {
        rounds++;
        drop[0] += !(ok && r.complete(kTripsPerRound));
        drop[3] += r.rx.n != kTripsPerRound;
        drop[4] += waits_given_up;
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

// A burst's egress stamps come out of the queue's counters, not out of the tag alone: PKT_START_CNT counts each
// hand-off as the tag is sampled, WORD_CNT a keepalive as one unit and a frame as two, so once the units minus the
// hand-offs since arming reach kBurstFrames every frame has been handed off, and the entries under the tag number
// the hand-offs. The queue's own keepalives are kept out of the armed window altogether: a keepalive is generated
// only after the keepalive timeout without a packet sent (documented; measured 8002 cycles), so a pilot frame sent
// just before arming leaves the queue no keepalive to generate for many times a burst's length. Without it a
// keepalive generated while the request sat armed either went ahead of the first frame (counted, tellable) or, when
// it fell on the first frame's command, merged into that frame and left its stamp request behind: a second entry
// under the tag for the first frame, one burst in ~1000, that no counter sees. arm_burst waits for the pilot's two
// units (a keepalive that left just before it is one and cannot pass for it), then arms and anchors the counters;
// false, the burst counted lost, if the pilot did not go. The pilot goes under the queue's boot header row, so the
// peer's classifier does not stamp it: a burst's ingress stamps stay its frames'.
struct Anchor {
    uint32_t start = 0, word = 0, tag_lo = 0;
};
// No context switch while the queue is busy: the core's owner may be a router, whose switches to base firmware
// are coordinated with the tile's other RISC.
__attribute__((always_inline)) inline void issue(volatile eth_channel_sync_t* s) {
    const uint32_t addr = reinterpret_cast<uint32_t>(s);
    internal_::eth_send_packet<false>(kLinkTxq, addr >> 4, addr >> 4, kFrameBytes >> 4);
}
template <typename Session>
__attribute__((noinline)) inline bool arm_burst(
    const Session& s, uint32_t base, uint64_t tag, Anchor& at, StopDiag& diag) {
    const uint32_t units0 = raw::txq_word_cnt(Session::kTxq);
    tx_header_row_select(s, true);
    issue(pilot(base));
    while (internal_::eth_txq_is_busy(Session::kTxq)) {
    }
    tx_header_row_select(s, false);
    for (uint32_t spin = 0; raw::txq_word_cnt(Session::kTxq) - units0 < 2; spin++) {
        if (spin == kBurstStampSpins) {
            diag.drop[2]++;
            return false;
        }
    }
    stamps_arm(s, tag);
    at.start = raw::txq_pkt_start_cnt(Session::kTxq);
    at.word = raw::txq_word_cnt(Session::kTxq);
    at.tag_lo = static_cast<uint32_t>(tag);
    return true;
}
// The frames' egress stamps added to `into` once they have all been handed off and stamped; false, with the burst
// counted as lost, if that did not happen within the polls allowed. WORD_CNT is read before PKT_START_CNT so a
// hand-off between the two reads shows one frame too few, never one too many. The hand-offs beyond the frames are
// counted (none, by the pilot's cover); their entries, ahead of the frames', are passed over.
template <typename Session>
__attribute__((noinline)) inline bool collect_burst(
    const Session& s, const Anchor& at, StampSum& into, StopDiag& diag) {
    uint32_t sent = 0;
    for (uint32_t spin = 0;; spin++) {
        const uint32_t units = raw::txq_word_cnt(Session::kTxq) - at.word;
        sent = raw::txq_pkt_start_cnt(Session::kTxq) - at.start;
        if (units - sent == kBurstFrames) {
            break;
        }
        if (spin == kBurstStampSpins) {
            stamps_disarm(s);
            diag.drop[2]++;
            return false;
        }
    }
    diag.drop[1] += sent != kBurstFrames;
    uint64_t got[kBurstFrames] = {};
    uint32_t n = 0;
    for (uint32_t spin = 0; n < sent && spin <= kBurstStampSpins; spin++) {
        tx_stamps_drain(at.tag_lo, [&](uint64_t ts) {
            got[n % kBurstFrames] = ts;
            n++;
        });
    }
    stamps_disarm(s);
    if (n != sent) {
        diag.drop[2]++;
        return false;
    }
    for (uint32_t i = 0; i < kBurstFrames; i++) {
        into.add(got[(n + i) % kBurstFrames]);
    }
    return true;
}

// The records: one PP_CLOCK per stamp, this core's refclk against its wall clock with the round's number and the
// stamp's role (spsc_packet.h), so the host pairs the two ends by identity and fits refclk against refclk: DVFS on
// either chip's wall clock cannot enter the link solve.
namespace link {
#if defined(PROFILE_KERNEL) && defined(PROFILE_STREAMING)
constexpr uint32_t kRoleT0 = kernel_profiler::ppfmt::CLOCK_ROLE_T0;
constexpr uint32_t kRoleT1 = kernel_profiler::ppfmt::CLOCK_ROLE_T1;
constexpr uint32_t kRoleT1B = kernel_profiler::ppfmt::CLOCK_ROLE_T1B;
constexpr uint32_t kRoleT2 = kernel_profiler::ppfmt::CLOCK_ROLE_T2;
// Room for `records` clock records in this core's ring, without waiting: a producer on an eth core must not stall,
// and a round a side has no room for is one the host never completes; nothing behind it shifts.
inline __attribute__((always_inline)) bool room(uint32_t records) {
    return kernel_profiler::ring_has_room(records * kernel_profiler::CLOCK_RECORD_WORDS);
}
// A software stamp, read as an Instant at the event and recorded whenever the trip's work is done.
inline __attribute__((always_inline)) void record_sw(const Instant& t, uint32_t round, uint32_t role) {
    kernel_profiler::ring_write_clock(
        kernel_profiler::ppfmt::CLOCK_LINK_REFCLK, t.refclk, t.wall_lo, t.wall_hi, round, role);
}
// A hardware stamp average, placed at the wall clock of its recording.
inline __attribute__((always_inline)) void record_hw(uint64_t value, uint32_t round, uint32_t role) {
    const Instant t = read_instant();
    kernel_profiler::ring_write_clock(kernel_profiler::ppfmt::CLOCK_LINK_PTP, value, t.wall_lo, t.wall_hi, round, role);
}
#else
constexpr uint32_t kRoleT0 = 0, kRoleT1 = 0, kRoleT1B = 0, kRoleT2 = 0;
inline bool room(uint32_t) { return false; }
inline void record_sw(const Instant&, uint32_t, uint32_t) {}
inline void record_hw(uint64_t, uint32_t, uint32_t) {}
#endif
}  // namespace link

// The two ends of a link, driven by whoever owns the core -- a resident kernel or the fabric router: open() before
// the link handshake, start() once the peer is up, step() as often as the core can spare, stop() at teardown. A step
// returns at once when nothing is due; when a burst (sender) or a frame (receiver) is due it is handled whole, so a
// step holds the core for ~1 us at most. SwStamps adds the round's first trip as software stamps, a second stream
// the host checks the hardware one against; it costs the sender a wait for that trip's echo, ~1 us once per round,
// so a router leaves it off. DataCache says whether the core runs with its L1 data cache on, in which case a step
// invalidates before polling what the peer or the host wrote; a router runs with it off and skips the fence. Both
// are constant-initialised: the ERISC runs no dynamic init.
constexpr uint32_t kRatioTicks = 1000;  // 20 us before a slot: read jitter of tens of cycles is under a tenth of a step

template <bool SwStamps, bool DataCache = true>
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
    uint32_t next_round = 0, diag_addr = 0;
    Instant start_at{}, pre{};
    Pacer pacer;
    StopDiag diag;
    HwRound rnd;
    uint32_t round = 0;
    bool emit = false, ok = false, gave_up = false;
    Instant t0{}, t2{};

    bool open() { return sess.begin(); }
    void start(uint32_t l1, uint32_t pace_ticks, uint32_t diag) {
        slot_base = l1;
        diag_addr = diag;
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
        slot_cfr = start_at.refclk + kFrameTicks;
    }
    __attribute__((always_inline)) void step() {
        if constexpr (DataCache) {
            invalidate_l1_cache();
        }
        if (rd(diag_addr) != kCtlRun) {
            slot_cfr = read_cfr() + kFrameTicks;  // rounds resume on a fresh slot, not a backlog of missed ones
            return;
        }
        const uint64_t cfr = read_cfr();
        if (cfr < slot_cfr) {
            if (pre.refclk == 0 && cfr + kRatioTicks >= slot_cfr) {
                pre = read_instant();
            }
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
        diag.timer = sess.timer_ok ? 1u : 2u;
        diag.span_wall = now.wall() - start_at.wall();
        diag.span_refclk = now.refclk - start_at.refclk;
        diag.write(diag_addr);
    }
    void close_round() {
        if (emit) {
            if constexpr (SwStamps) {
                link::record_sw(t0, round, link::kRoleT0);
                link::record_sw(t2, round, link::kRoleT2);
            }
            if (ok && sess.timer_ok && rnd.complete(kTripsPerRound)) {
                link::record_hw(rnd.tx.q(sess), round, link::kRoleT0);
                link::record_hw(rnd.rx.q(sess), round, link::kRoleT2);
            }
        }
        diag.note_round(rnd, ok, gave_up);
        write_diag();
    }
    // A burst: the previous burst's echo stamps, the round's records at a round boundary, then kBurstFrames frames
    // kFrameTicks apart from the first, each at its phase of the tick, all in wall cycles from one reading (the slot
    // wait's exit shifts the whole burst alike, which the grid does not mind), and their egress stamps. The echoes
    // come back after the burst, so none arrives while a frame of ours is being stamped.
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
        rx_stamps_drain(sess, [&](uint64_t ts) { rnd.rx.add(ts); });
        const uint32_t j0 = static_cast<uint32_t>(bursts % kBurstsPerRound) * kBurstFrames;
        if (j0 == 0) {
            if (bursts != 0) {
                close_round();
            }
            round = next_round++;
            rnd.begin(round);
            emit = link::room(SwStamps ? 4 : 2);
            ok = true;
            gave_up = false;
        }
        const uint64_t tag = 0x5000'0000'0000'0000ull | bursts;
        Anchor at;
        if (!arm_burst(sess, slot_base, tag, at, diag)) {
            ok = false;
        } else {
            const uint32_t spacing = (kFrameTicks * c16) >> 4;
            const uint32_t phase0 = frame_phase_cycles(j0, c16);
            const uint32_t w0 = rd(kWallClockLo) + 32 + phase0;
            for (uint32_t i = 0; i < kBurstFrames; i++) {
                const uint32_t j = j0 + i;
                volatile eth_channel_sync_t* s = slot(slot_base, i);
                pacer.until(w0 + i * spacing + frame_phase_cycles(j, c16) - phase0);
                s->reserved_2 = round;
                s->bytes_sent = frame_key(round, j);
                if constexpr (SwStamps) {
                    if (j == 0) {
                        t0 = read_instant();
                    }
                }
                issue(s);
            }
            if (!collect_burst(sess, at, rnd.tx, diag)) {
                ok = false;
            }
            if constexpr (SwStamps) {
                // The receiver echoes a burst once its last frame is in, so the first frame's echo follows the burst.
                if (j0 == 0) {
                    volatile eth_channel_sync_t* s = slot(slot_base, 0);
                    for (uint32_t spin = 0; s->bytes_sent != 0; spin++) {
                        if (spin == kEchoSpins) {
                            ok = false;
                            gave_up = true;
                            break;
                        }
                        invalidate_l1_cache();
                    }
                    t2 = read_instant();
                }
            }
        }
        diag.note_hold(rd(kWallClockLo) - hold0);
        bursts++;
        slot_cfr += burst_ticks;
    }
};

template <bool SwStamps, bool DataCache = true>
struct ReceiverLink {
    LinkSession sess;
    uint32_t slot_base = 0, diag_addr = 0;
    uint32_t round = 0, expect = 0;
    bool started = false, emit = false, ok = false, mid_burst = false, armed = false;
    Anchor at;
    Instant start_at{}, t1{}, t1b{};
    StopDiag diag;
    HwRound rnd;

    bool open() { return sess.begin(); }
    void start(uint32_t l1, uint32_t diag) {
        slot_base = l1;
        diag_addr = diag;
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
        diag.timer = sess.timer_ok ? 1u : 2u;
        diag.span_wall = now.wall() - start_at.wall();
        diag.span_refclk = now.refclk - start_at.refclk;
        diag.write(diag_addr);
    }
    void close_round() {
        if (emit) {
            if constexpr (SwStamps) {
                link::record_sw(t1, round, link::kRoleT1);
                link::record_sw(t1b, round, link::kRoleT1B);
            }
            if (ok && sess.timer_ok && rnd.complete(kTripsPerRound)) {
                link::record_hw(rnd.rx.q(sess), round, link::kRoleT1);
                link::record_hw(rnd.tx.q(sess), round, link::kRoleT1B);
            }
        }
        diag.note_round(rnd, ok, false);
        write_diag();
    }
    // A frame: its ingress stamp, its echo from the same slot, and after a burst's last echo the burst's egress
    // stamps. The round's number is the sender's, read from the frame; a new one closes the previous. An echo waits
    // for the previous one's hand-off, so the queue never holds two of ours: four issued back to back left the last
    // unfinished for microseconds, one burst in ~1000. True while the burst has frames to come.
    __attribute__((noinline)) bool frame(volatile eth_channel_sync_t* s, uint32_t key) {
        const Instant now = read_instant();
        const uint32_t j = (key & kTripMask) - 1;
        if (j == 0) {
            if (started) {
                close_round();
            }
            started = true;
            round = s->reserved_2;
            rnd.begin(round);
            emit = link::room(SwStamps ? 4 : 2);
            ok = true;
            t1 = now;
        } else if (key != frame_key(round, j)) {
            s->bytes_sent = 0;  // a frame of a round already given up
            return false;
        } else if (j != expect) {
            ok = false;
        }
        rx_stamps_drain(sess, [&](uint64_t ts) { rnd.rx.add(ts); });
        const uint32_t i = j % kBurstFrames;
        if (i == 0) {
            const uint64_t tag = 0x5200'0000'0000'0000ull | (round * kBurstsPerRound + j / kBurstFrames);
            armed = arm_burst(sess, slot_base, tag, at, diag);
            ok = ok && armed;
        } else if (armed) {
            for (uint32_t spin = 0; raw::txq_pkt_start_cnt(LinkSession::kTxq) - at.start < i; spin++) {
                if (spin == kBurstStampSpins) {
                    armed = false;
                    ok = false;
                    diag.drop[2]++;
                    stamps_disarm(sess);
                    break;
                }
            }
        }
        s->bytes_sent = 0;
        if constexpr (SwStamps) {
            if (j == 0) {
                t1b = read_instant();
            }
        }
        issue(s);
        if (i == kBurstFrames - 1 && armed) {
            if (!collect_burst(sess, at, rnd.tx, diag)) {
                ok = false;
            }
        }
        expect = j + 1;
        diag.note_hold(rd(kWallClockLo) - now.wall_lo);
        return i != kBurstFrames - 1;
    }
};

}  // namespace tt::tt_metal::eth_ptp
