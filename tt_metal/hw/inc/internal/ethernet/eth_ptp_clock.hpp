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

#include "internal/ethernet/tt_eth_ss_regs.h"

namespace tt::tt_metal::eth_ptp {

constexpr uint32_t kRefclkHz = 50'000'000;
constexpr uint32_t kPtiRefclk = (1'000'000'000u / kRefclkHz) << 16;  // 20.0 ns per tick in the timer's 8.16 fixed point

// ERISC wall clock (AICLK ticks). Reading LO latches HI into WALL_CLOCK_1_AT.
constexpr uint32_t kWallClockLo = ETH_RISC_REGS_START + ETH_RISC_WALL_CLOCK_0;
constexpr uint32_t kWallClockHi = ETH_RISC_REGS_START + ETH_RISC_WALL_CLOCK_1_AT;

// eth_ctrl.ptp_timer_a
constexpr uint32_t kPtpTimerBase = 0xFFB98800;
constexpr uint32_t kPtpTimerCtrl = kPtpTimerBase + 0x00;  // [0] timer_en
constexpr uint32_t kPtpFutureCfrLo = kPtpTimerBase + 0x04;
constexpr uint32_t kPtpFutureCfrHi = kPtpTimerBase + 0x08;
constexpr uint32_t kPtpFuturePti = kPtpTimerBase + 0x0C;  // [23:0] = 8 integer ns bits . 16 fractional
constexpr uint32_t kPtpFutureTimestampLo = kPtpTimerBase + 0x10;
constexpr uint32_t kPtpFutureTimestampHi = kPtpTimerBase + 0x14;
constexpr uint32_t kPtpUpdatePti = kPtpTimerBase + 0x20;        // [0]
constexpr uint32_t kPtpUpdateTimestamp = kPtpTimerBase + 0x24;  // [0]
constexpr uint32_t kPtpUpdateStat = kPtpTimerBase + 0x40;       // [0] pti pending [1] ts pending [8] pti ack [9] ts ack
constexpr uint32_t kPtpPtiStat = kPtpTimerBase + 0x44;          // [23:0] per-tick increment in use
constexpr uint32_t kPtpCfrLo = kPtpTimerBase + 0x50;            // the refclk count; reading LO latches HI
constexpr uint32_t kPtpCfrHi = kPtpTimerBase + 0x54;
constexpr uint32_t kPtp64nsLo = kPtpTimerBase + 0x60;  // the PTP time in ns; reading LO latches HI
constexpr uint32_t kPtp64nsHi = kPtpTimerBase + 0x64;
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
    uint64_t wall() const { return (static_cast<uint64_t>(wall_hi) << 32) | wall_lo; }
};
inline __attribute__((always_inline)) Instant read_instant() {
    Instant t;
    t.wall_lo = rd(kWallClockLo);
    const uint32_t rlo = rd(kPtpCfrLo);
    t.wall_hi = rd(kWallClockHi);
    const uint32_t rhi = rd(kPtpCfrHi);
    t.refclk = (static_cast<uint64_t>(rhi) << 32) | rlo;
    return t;
}

}  // namespace tt::tt_metal::eth_ptp
