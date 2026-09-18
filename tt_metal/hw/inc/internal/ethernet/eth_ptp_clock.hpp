// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The Blackhole Ethernet tile's clocks: the ERISC wall clock (AICLK ticks) and the eth_ctrl PTP timer's two counters,
// CFR (refclk ticks, free-running from power-on) and PTP64NS (ns, running once ptp_timer_start() in eth_ptp.hpp
// enables it). Reading a counter's LO half captures its HI half (tt_ptp_timer.sv; for the wall clock the captured
// half is WALL_CLOCK_1_AT, WALL_CLOCK_1 being live). Usable from idle and active eth kernels alike; the stamping
// hardware is eth_ptp.hpp.

#pragma once

#include <cstdint>

#include "hostdev/streaming_profiler_common.h"
#include "internal/ethernet/tt_eth_ss_regs.h"

namespace tt::tt_metal::eth_ptp {

constexpr uint32_t kRefclkHz = kernel_profiler::kEthRefclkHz;
constexpr uint32_t kNsPerRefclkTick = 1'000'000'000u / kRefclkHz;  // 20
constexpr uint32_t kPtiRefclk = kNsPerRefclkTick << 16;            // the timer's per-tick increment, 8.16 fixed point

// ERISC wall clock (AICLK ticks). Reading LO latches HI into WALL_CLOCK_1_AT.
constexpr uint32_t kWallClockLo = ETH_RISC_REGS_START + ETH_RISC_WALL_CLOCK_0;
constexpr uint32_t kWallClockHi = ETH_RISC_REGS_START + ETH_RISC_WALL_CLOCK_1_AT;

constexpr uint32_t kPtpTimerCtrl = ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_CTRL;
constexpr uint32_t kPtpFutureCfrLo = ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_FUTURE_CFR_LO;
constexpr uint32_t kPtpFutureCfrHi = ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_FUTURE_CFR_HI;
constexpr uint32_t kPtpFuturePti = ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_FUTURE_PTI;
constexpr uint32_t kPtpFutureTimestampLo = ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_FUTURE_TIMESTAMP_LO;
constexpr uint32_t kPtpFutureTimestampHi = ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_FUTURE_TIMESTAMP_HI;
constexpr uint32_t kPtpUpdatePti = ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_UPDATE_PTI;
constexpr uint32_t kPtpUpdateTimestamp = ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_UPDATE_TIMESTAMP;
constexpr uint32_t kPtpUpdateStat = ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_UPDATE_STAT;
constexpr uint32_t kPtpPtiStat = ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_PTI_STAT;
constexpr uint32_t kPtpCfrLo = ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_CFR_LO;
constexpr uint32_t kPtpCfrHi = ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_CFR_HI;
constexpr uint32_t kPtp64nsLo = ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_64NS_LO;
constexpr uint32_t kPtp64nsHi = ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_64NS_HI;
constexpr uint32_t kUpdateStatPtiAck = 1u << 8;
constexpr uint32_t kUpdateStatTsAck = 1u << 9;

inline __attribute__((always_inline)) uint32_t rd(uint32_t addr) { return *reinterpret_cast<volatile uint32_t*>(addr); }
inline __attribute__((always_inline)) void wr(uint32_t addr, uint32_t v) {
    *reinterpret_cast<volatile uint32_t*>(addr) = v;
}

// LO first: the read latches HI.
inline __attribute__((always_inline)) uint64_t read_cfr() {
    const uint32_t lo = rd(kPtpCfrLo);
    const uint32_t hi = rd(kPtpCfrHi);
    return (static_cast<uint64_t>(hi) << 32) | lo;
}
inline __attribute__((always_inline)) uint64_t read_ptp64ns() {
    const uint32_t lo = rd(kPtp64nsLo);
    const uint32_t hi = rd(kPtp64nsHi);
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
inline __attribute__((always_inline)) Instant read_instant() {
    Instant t;
    for (;;) {
        const uint32_t hi0 = rd(kWallClockHi);
        t.wall_lo = rd(kWallClockLo);
        const uint32_t rlo = rd(kPtpCfrLo);
        t.wall_hi = rd(kWallClockHi);
        const uint32_t rhi = rd(kPtpCfrHi);
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
// Each iteration is padded by a pseudo-random 0-15 cycles (phase_walk): with a fixed iteration length, at an AICLK
// where the 80 ns between updates is a whole number of iterations, the update lands at the same phase of every one
// and, outside the bracket, is never caught -- whole half seconds without a sample at 1237.5 and 1306.25 MHz; a
// short regular walk locked the same way at 1350 MHz. `x` is the walk's state, any nonzero seed.
inline __attribute__((always_inline)) void phase_walk(uint32_t& x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    for (uint32_t d = x & 15u; d != 0; d--) {
        asm volatile("nop");
    }
}
inline __attribute__((always_inline)) Instant read_bracketed() {
    Instant t = read_instant();
    uint32_t x = t.wall_lo | 1u;
    for (uint32_t spin = 0; spin < 65536u; spin++) {
        phase_walk(x);
        const uint32_t ra = rd(kPtpCfrLo);
        const uint32_t w = rd(kWallClockLo);
        const uint32_t rb = rd(kPtpCfrLo);
        if (ra != rb) {
            t.wall_hi += w < t.wall_lo;
            t.wall_lo = w;
            const uint32_t r_hi = static_cast<uint32_t>(t.refclk >> 32) + (rb < static_cast<uint32_t>(t.refclk));
            t.refclk = (static_cast<uint64_t>(r_hi) << 32) | rb;
            t.spins = spin + 1;
            return t;
        }
    }
    return t;
}

}  // namespace tt::tt_metal::eth_ptp
