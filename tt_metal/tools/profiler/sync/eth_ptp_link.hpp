// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The streaming profiler's link half on the tile's 1588 hardware (internal/ethernet/eth_ptp.hpp): the session every
// sync kernel opens, how a round's stamps are accumulated and reported, and the PP_CLOCK records the two kernels
// write. Without the streaming profiler the record path compiles to nothing.

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

// A round is kTripsPerRound back-to-back exchanges whose stamps are averaged on each side: every stamp is quantised
// to the timer's 20 ns tick, the trips sit at different phases of it, so the mean's rounding noise falls by the
// square root of the count (8.8 -> ~0.4 ns per round at 256), and the averages are reported in quarter-ns units
// so that gain reaches the host whole. A round of 256 trips takes ~320 us of the link's 1 ms cadence.
constexpr uint32_t kTripsPerRound = 256;
constexpr uint32_t kHwUnitsPerNs = 4;

// One side's hardware round: its two stamps accumulated over the trips that produced both, each relative to the
// round's first so the sums stay in 32 bits and the average costs a 32-bit divide, not the 64-bit routine that a
// runtime count would otherwise pull into the ERISC's text. A trip missing a stamp leaves the averages rather than
// taking the round with it; a round keeping fewer than half its trips is dropped. The two sides may then average
// over sets that differ by a trip, which moves a midpoint by microseconds of the round's span and, at ppm drift,
// picoseconds of offset.
struct HwRound {
    uint32_t id = 0;
    bool emit = false;  // this side had ring room for the round's records and a running timer
    uint32_t n = 0;
    uint64_t base_a = 0, base_b = 0;  // the first trip's stamps
    uint32_t rel_a = 0, rel_b = 0;    // sums of (stamp - base), ns: 256 trips of a 320 us round fit with room
    __attribute__((always_inline)) void begin(uint32_t round, bool ok) {
        id = round;
        emit = ok;
        n = 0;
        rel_a = 0;
        rel_b = 0;
    }
    __attribute__((always_inline)) void add(uint64_t a, uint64_t b) {
        if (a == 0 || b == 0) {
            return;
        }
        if (n == 0) {
            base_a = a;
            base_b = b;
        }
        rel_a += static_cast<uint32_t>(a - base_a);
        rel_b += static_cast<uint32_t>(b - base_b);
        n++;
    }
    bool complete() const { return emit && n >= kTripsPerRound / 2; }
    // The average in quarter-ns of the refclk domain: PTP64NS minus the timer's offset from the CFR count.
    uint64_t q_a(const LinkSession& s) const { return q(s, base_a, rel_a); }
    uint64_t q_b(const LinkSession& s) const { return q(s, base_b, rel_b); }

private:
    uint64_t q(const LinkSession& s, uint64_t base, uint32_t rel) const {
        const int64_t base_units = (static_cast<int64_t>(base) - s.ptp_offset_ns) * kHwUnitsPerNs;
        return static_cast<uint64_t>(base_units + (rel * kHwUnitsPerNs + n / 2) / n);
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
