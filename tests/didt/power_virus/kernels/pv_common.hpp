// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Shared by the power-virus host test and its kernels: the per-Tensix L1
// control block, the active-ETH link kernel layout and the throttle scheme.
// The host part is constants only; the kernel part needs a kernel build.

#pragma once

#include <cstdint>

namespace pv {

// -- Tensix L1 control block (offsets from the control base) -----------------
// The host writes the control block before launch and sets STOP to end the run.
constexpr uint32_t kStop = 0;         // host writes 1 to stop every generator on the core
constexpr uint32_t kGo = 16;          // 1 = start; preset by host, or set by the BRISC super-sync barrier
constexpr uint32_t kReady = 32;       // TRISC unpack writes 1 once initialized (super-sync waits on it)
constexpr uint32_t kFinal = 48;       // TRISC loop index to stop at, agreed by all three TRISCs
constexpr uint32_t kFpuRecord = 64;   // [elapsed_lo, elapsed_hi, loops_lo, loops_hi]
constexpr uint32_t kDramRecord = 80;  // [elapsed_lo, elapsed_hi, xfers_lo, xfers_hi]
constexpr uint32_t kNocRecord = 96;   // [elapsed_lo, elapsed_hi, xfers_lo, xfers_hi]
constexpr uint32_t kControlBytes = 128;

constexpr uint32_t kFinalUnset = 0xFFFFFFFFu;

// -- Active-ETH L1 layout of the link kernel (offsets from unreserved base) ---
constexpr uint32_t kEthHandshake = 0;  // written by peer
constexpr uint32_t kEthTiming = 16;    // timing staging, persisted to DRAM at the end
constexpr uint32_t kEthFinIn = 32;     // written by peer: [magic, peer_sent]
constexpr uint32_t kEthFinOut = 48;    // staging for our FIN
constexpr uint32_t kEthStop = 64;      // host writes 1 to stop
constexpr uint32_t kEthHeaderBytes = 128;

// Throttle time unit, in wall-clock ticks.
constexpr uint32_t kIdleUnitTicks = 512;

// -- Throttle modes. Every generator takes [mode, a, b, duty_period, duty_active]
// as compile-time args. All periods are in units of kIdleUnitTicks.
//   kModeFull   – no throttle; a, b unused
//   kModeSmooth – steady partial load: after each unit of work that took t
//                 ticks, wait t × a / b (a = 100 - pct, b = pct). TRISC instead
//                 pads every tile with a fixed number of NOPs (a = NOP count).
//   kModePhase  – phase-locked: run while (clock / 512) % a < b (a power of
//                 two). Every core on the chip switches together.
//   kModeBurst  – a units of work, then idle b × 512 ticks
//   kModeRate   – steady partial load for generators that share a resource
//                 (DRAM) or cannot time their own work (Ethernet): start one
//                 unit of work every a ticks. The host derives a from the
//                 generator's full rate at 1.35 GHz, so the rate scales with
//                 AICLK when the chip throttles.
// Duty cycle, on top of any mode: run only while (clock / 512) % duty_period
// < duty_active (duty_period a power of two; 0 = no duty cycle).
constexpr uint32_t kModeFull = 0;
constexpr uint32_t kModeSmooth = 1;
constexpr uint32_t kModePhase = 2;
constexpr uint32_t kModeBurst = 3;
constexpr uint32_t kModeRate = 4;

// Rate mode: at most this many units of work are issued back to back to catch
// up after a stall, so the load stays even.
constexpr uint32_t kRateMaxCatchUp = 4;

}  // namespace pv

#ifdef KERNEL_BUILD
namespace pv {

// Throttle decisions for one generator; `now` is the low wall-clock word.
template <uint32_t mode, uint32_t a, uint32_t b, uint32_t duty_period, uint32_t duty_active>
struct Throttle {
    static_assert(mode != kModePhase || (a != 0 && (a & (a - 1)) == 0), "phase period must be a power of two");
    static_assert(duty_period == 0 || (duty_period & (duty_period - 1)) == 0, "duty period must be a power of two");
    static_assert(mode != kModeSmooth || b != 0, "smooth mode needs pct > 0");

    // True if anything ever gates on the clock (phase or duty cycle).
    static constexpr bool kGated = duty_period != 0 || mode == kModePhase;

    // 2^32 / 512 is a multiple of every power-of-two period, so the low clock
    // word gives a consistent phase across wrap-around.
    static inline __attribute__((always_inline)) bool open(uint32_t now) {
        const uint32_t unit = now / kIdleUnitTicks;
        if constexpr (duty_period != 0) {
            if ((unit & (duty_period - 1)) >= duty_active) {
                return false;
            }
        }
        if constexpr (mode == kModePhase) {
            if ((unit & (a - 1)) >= b) {
                return false;
            }
        }
        return true;
    }

    // Ticks to idle after a unit of work that took work_ticks.
    static inline __attribute__((always_inline)) uint32_t pause_after(uint32_t work_ticks) {
        if constexpr (mode == kModeSmooth) {
            return work_ticks * a / b;
        } else if constexpr (mode == kModeBurst) {
            return b * kIdleUnitTicks;
        } else {
            return 0;
        }
    }
};

}  // namespace pv
#endif
