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
// the tick whatever the cadence or AICLK, and the mean's rounding noise falls to ~0.4 ns per round. Within a burst
// the queue is never idle for its keepalive timeout, so the stamp request is armed once per burst, under a tag the
// burst's frames then leave behind (collect_burst). Each side sums its stamps unpaired, so a round counts only if
// every frame produced both of a side's stamps. The round's first frame is exchanged
// alone before the next is issued, so its software stamps see an idle queue on both ends.
constexpr uint32_t kTripsPerRound = 256;
constexpr uint32_t kBurstFrames = 4;
constexpr uint32_t kBurstsPerRound = kTripsPerRound / kBurstFrames;
constexpr uint32_t kFrameTicks = 12;       // 240 ns between a burst's frames: a receiver takes a frame in ~150 cycles
constexpr uint32_t kFramePhaseStep = 157;  // odd, so the 256 phases are a permutation
constexpr uint32_t kEchoSpins = 50'000;    // polls for the first frame's echo before the round is given up: ~1 ms
constexpr uint32_t kBurstStampSpins = 64;  // drains for a burst's egress stamps before it is given up: ~2 us
constexpr uint32_t kHwUnitsPerNs = 4;
static_assert(kTripsPerRound % kBurstFrames == 0 && kBurstFrames >= 2 && (kFramePhaseStep & 1) == 1);

// Frame j of a round rides in slot j % kBurstFrames, the same L1 address on both ends, so the receiver's echo lands
// on the word the sender polls. The sync word carries the round and the trip, so a slot's stale content can match
// nothing the receiver waits for, and reserved_2 the round's full number.
inline __attribute__((always_inline)) volatile eth_channel_sync_t* slot(uint32_t base, uint32_t j) {
    return reinterpret_cast<volatile eth_channel_sync_t*>(base + (j % kBurstFrames) * sizeof(eth_channel_sync_t));
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
    static void turns(uint32_t n) {
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

// The egress stamps of a burst's frames, added to `into`. The request is armed under kGapTag before the burst and
// retagged with the burst's tag right after the first frame's command, so a keepalive the idle queue stamped ahead
// of the frames carries kGapTag and the frames the burst's; this waits for the frames' count under it, clears the
// request, and discards the rest. False if more turned up or they did not come.
constexpr uint64_t kGapTag = 0x4A4A'4A4A'FFFF'FFFFull;
template <typename Session>
__attribute__((noinline)) inline bool collect_burst(const Session& s, uint32_t tag_lo, StampSum& into) {
    uint64_t got[kBurstFrames] = {};
    uint32_t n = 0;
    const auto take = [&](uint64_t ts) {
        if (n < kBurstFrames) {
            got[n] = ts;
        }
        n++;
    };
    bool ok = true;
    for (uint32_t spin = 0; ok && n < kBurstFrames; spin++) {
        tx_stamps_drain(tag_lo, take);
        ok = n <= kBurstFrames && spin < kBurstStampSpins;
    }
    stamps_disarm(s);
    if (!ok) {
        return false;
    }
    for (uint32_t i = 0; i < kBurstFrames; i++) {
        into.add(got[i]);
    }
    return true;
}

// What each end leaves past its stop word for the host's log at stop (streaming_profiler_device.cpp reads it back):
// +8 rounds, +12 the timer word (0 no hardware path, 1 ran, 2 never acknowledged its rate), +16 wall cycles inside
// bursts and +24 wall cycles of the run (two words each), +32 refclk ticks of the run (two words), +40 the longest
// burst in wall cycles, +44 rounds dropped for a stamp count off the frame count, then egress stamps over, under,
// ingress stamps off, and waits given up.
struct StopDiag {
    uint32_t rounds = 0, timer = 0;
    uint64_t hold = 0, span_wall = 0, span_refclk = 0;
    uint32_t hold_max = 0;
    uint32_t drop[5] = {};
    __attribute__((always_inline)) void note_hold(uint32_t cycles) {
        hold += cycles;
        hold_max = cycles > hold_max ? cycles : hold_max;
    }
    __attribute__((always_inline)) void note_round(const HwRound& r, bool ok) {
        rounds++;
        if (!r.complete(kTripsPerRound)) {
            drop[0]++;
            drop[1] += r.tx.n > kTripsPerRound;
            drop[2] += r.tx.n < kTripsPerRound;
            drop[3] += r.rx.n != kTripsPerRound;
        }
        drop[4] += !ok;
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

}  // namespace tt::tt_metal::eth_ptp
