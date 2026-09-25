// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The Blackhole Ethernet tile's clocks: the ERISC wall clock, which counts AI clock cycles, and the PTP timer's two
// counters, the reference count (50 MHz ticks since power-on) and the PTP time in ns, which runs once PtpTimer::start()
// in eth_ptp.hpp has started it. Reading the low half of any of them latches its high half for the next read. These
// work from idle and active Ethernet kernels alike; the timestamping hardware is in eth_ptp.hpp.

#pragma once

#include <cstdint>

#include "hostdev/streaming_profiler_common.h"
#include "internal/ethernet/tt_eth_ss_regs.h"
#include "internal/risc_attribs.h"

namespace tt::tt_metal::eth_ptp {

constexpr uint32_t kRefclkHz = kernel_profiler::kEthRefclkHz;
constexpr uint32_t kNsPerRefclkTick = 1'000'000'000u / kRefclkHz;  // 20
constexpr uint32_t kPtiRefclk = kNsPerRefclkTick << 16;            // the timer's per-tick increment, 8.16 fixed point

FORCE_INLINE uint32_t rd(uint32_t addr) { return *reinterpret_cast<volatile uint32_t*>(addr); }
FORCE_INLINE void wr(uint32_t addr, uint32_t v) { *reinterpret_cast<volatile uint32_t*>(addr) = v; }

template <typename T>
constexpr uint32_t bits(T v) {
    static_assert(sizeof(T) == sizeof(uint32_t));
    return __builtin_bit_cast(uint32_t, v);
}

// A memory-mapped register at a fixed address, read and written as T: uint32_t, or a struct of uint32_t bit-fields
// laid out from bit 0. Every bit of such a struct has a name, reserved ones included, so that a value built with
// designated initializers has all its other bits zero.
template <typename T = uint32_t>
struct Reg {
    static_assert(sizeof(T) == sizeof(uint32_t));
    uint32_t addr;
    FORCE_INLINE T read() const { return __builtin_bit_cast(T, rd(addr)); }
    FORCE_INLINE void write(T v) const { wr(addr, bits(v)); }
};

// The high word that goes with the low word is WALL_CLOCK_1_AT, latched by the low word's read; WALL_CLOCK_1 is live.
constexpr Reg<> kWallClockLo{ETH_RISC_REGS_START + ETH_RISC_WALL_CLOCK_0};
constexpr Reg<> kWallClockHi{ETH_RISC_REGS_START + ETH_RISC_WALL_CLOCK_1_AT};

struct PtpUpdateStat {
    uint32_t pti_pending : 1;
    uint32_t timestamp_pending : 1;
    uint32_t rsvd0 : 6;
    uint32_t pti_ack : 1;
    uint32_t timestamp_ack : 1;
    uint32_t rsvd1 : 22;
};

constexpr Reg<> kPtpTimerCtrl{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_CTRL};
constexpr Reg<> kPtpFutureCfrLo{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_FUTURE_CFR_LO};
constexpr Reg<> kPtpFutureCfrHi{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_FUTURE_CFR_HI};
constexpr Reg<> kPtpFuturePti{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_FUTURE_PTI};
constexpr Reg<> kPtpFutureTimestampLo{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_FUTURE_TIMESTAMP_LO};
constexpr Reg<> kPtpFutureTimestampHi{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_FUTURE_TIMESTAMP_HI};
constexpr Reg<> kPtpUpdatePti{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_UPDATE_PTI};
constexpr Reg<> kPtpUpdateTimestamp{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_UPDATE_TIMESTAMP};
constexpr Reg<PtpUpdateStat> kPtpUpdateStat{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_UPDATE_STAT};
constexpr Reg<> kPtpPtiStat{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_PTI_STAT};
constexpr Reg<> kPtpCfrLo{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_CFR_LO};
constexpr Reg<> kPtpCfrHi{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_CFR_HI};
constexpr Reg<> kPtp64nsLo{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_64NS_LO};
constexpr Reg<> kPtp64nsHi{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_64NS_HI};

// One step of a xorshift32 walk, any nonzero state.
FORCE_INLINE uint32_t xorshift(uint32_t& x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

// LO first: the read latches HI.
FORCE_INLINE uint64_t read_cfr() {
    const uint32_t lo = kPtpCfrLo.read();
    const uint32_t hi = kPtpCfrHi.read();
    return (static_cast<uint64_t>(hi) << 32) | lo;
}
FORCE_INLINE uint64_t read_ptp64ns() {
    const uint32_t lo = kPtp64nsLo.read();
    const uint32_t hi = kPtp64nsHi.read();
    return (static_cast<uint64_t>(hi) << 32) | lo;
}

// The wall clock and the refclk read as one instant. The two LO reads go back to back, so one register read is the
// whole skew between the clocks; each LO latches its HI, and the ERISC is the only reader of these registers, so
// nothing re-latches before the HIs are collected.
struct Instant {
    uint32_t wall_lo, wall_hi;
    uint64_t refclk;
    uint32_t spins = 0;  // read_bracketed: iterations to the caught update plus one; 0 for a plain instant
    uint64_t wall() const { return (static_cast<uint64_t>(wall_hi) << 32) | wall_lo; }
};
// The wall clock's low read latches its high word for only a few cycles, and the refclk read between the two takes
// longer, so a low word that wrapped before the high read tore the pair by 2^32 (once per 20 s run across 8 chips).
// A high word read ahead of the instant equal to the one read after it means no wrap fell inside the window; the
// stamped reads themselves keep their order and spacing.
FORCE_INLINE Instant read_instant() {
    Instant t;
    for (;;) {
        const uint32_t hi0 = kWallClockHi.read();
        t.wall_lo = kWallClockLo.read();
        const uint32_t rlo = kPtpCfrLo.read();
        t.wall_hi = kWallClockHi.read();
        const uint32_t rhi = kPtpCfrHi.read();
        if (t.wall_hi == hi0) {
            t.refclk = (static_cast<uint64_t>(rhi) << 32) | rlo;
            return t;
        }
    }
}
// The instant of one refclk update, the way the clock pusher samples (eth_clock_pusher.cpp): a wall read between two
// refclk reads that differ is within a cycle of the update, with no read-latency term. The ERISC sees the refclk
// move every four ticks and the three reads span a couple of cycles, so a spin catches one in ~50, about 5 us.
// The spin is bounded only so a dead refclk cannot hold the core; then the plain instant stands.
// Each iteration is padded by a pseudo-random 0-15 turns of a nop loop (phase_walk): with a fixed iteration length, at
// an AICLK where the 80 ns between updates is a whole number of iterations, the update lands at the same phase of every
// one and, outside the bracket, is never caught -- whole half seconds without a sample at 1237.5 and 1306.25 MHz; a
// short regular walk locked the same way at 1350 MHz. `x` is the walk's state, any nonzero seed.
FORCE_INLINE void phase_walk(uint32_t& x) {
    // A loop even at -O3, where complete unrolling makes a 180-byte nop sled of every copy.
#pragma GCC unroll 1
    for (uint32_t d = xorshift(x) & 15u; d != 0; d--) {
        asm volatile("nop");
    }
}
// One attempt at a bracketed pair: true when a refclk update fell between the refclk reads around the wall read, `t`
// (a read_instant, for the high words) then holding that wall read and the new refclk.
FORCE_INLINE bool try_bracket(Instant& t, uint32_t& x) {
    phase_walk(x);
    const uint32_t ra = kPtpCfrLo.read();
    const uint32_t w = kWallClockLo.read();
    const uint32_t rb = kPtpCfrLo.read();
    if (ra == rb) {
        return false;
    }
    t.wall_hi += w < t.wall_lo;
    t.wall_lo = w;
    const uint32_t r_hi = static_cast<uint32_t>(t.refclk >> 32) + (rb < static_cast<uint32_t>(t.refclk));
    t.refclk = (static_cast<uint64_t>(r_hi) << 32) | rb;
    return true;
}
FORCE_INLINE Instant read_bracketed() {
    Instant t = read_instant();
    uint32_t x = t.wall_lo | 1u;
    for (uint32_t spin = 0; spin < 65536u; spin++) {
        if (try_bracket(t, x)) {
            t.spins = spin + 1;
            return t;
        }
    }
    return t;
}
// The very next refclk update, waited for with no gap between reads: its count and the wall read just before the
// refclk read that changed. It returns within one update (80 ns), where read_bracketed takes ~5 us, but its wall read
// sits early in the bracket rather than centred, so it paces frames and does not sample the clock. False if no update
// came within the spins (a dead refclk).
FORCE_INLINE bool next_refclk_update(uint32_t& wall, uint32_t& refclk) {
    uint32_t prev = kPtpCfrLo.read();
    for (uint32_t spin = 0; spin < 1024; spin++) {
        const uint32_t w = kWallClockLo.read();
        const uint32_t r = kPtpCfrLo.read();
        if (r != prev) {
            wall = w;
            refclk = r;
            return true;
        }
        prev = r;
    }
    return false;
}

}  // namespace tt::tt_metal::eth_ptp
