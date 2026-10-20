// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "internal/ethernet/tt_eth_api.h"
#include "internal/risc_attribs.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/telemetry/fabric_phase_profile_types.hpp"

#include <cstddef>
#include <cstdint>

// Core clock cycle counter (mcycle). A single CSR read, unlike the wall-clock register, which costs tens of cycles.
FORCE_INLINE uint32_t fabric_core_cycles() {
    uint32_t cycles;
    asm volatile("csrr %0, mcycle" : "=r"(cycles));
    return cycles;
}

// Ethernet TX queue register offsets, indexed by FabricTxqRegister.
constexpr uint32_t fabric_txq_register_offsets[TXQ_REGISTER_COUNT] = {
    ETH_TXQ_TRANSFER_CNT,
    ETH_TXQ_PKT_START_CNT,
    ETH_TXQ_PKT_END_CNT,
    ETH_TXQ_WORD_CNT,
    ETH_TXQ_RESEND_CNT,
    ETH_TXQ_CTRL,
    ETH_TXQ_MAX_PKT_SIZE_BYTES,
    ETH_TXQ_BURST_LEN,
    ETH_TXQ_REMOTE_SEQ_TIMEOUT,
    ETH_TXQ_LOCAL_SEQ_UPDATE_TIMEOUT,
    ETH_TXQ_MIN_PACKET_SIZE_WORDS,
    ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD};

// Reads the registers of every TX queue in TXQ_MASK into `dst`, the FabricPhaseProfile::txq array viewed as 32-bit
// words (each value is a uint64: low word, then a zero high word).
template <uint32_t TXQ_MASK>
FORCE_INLINE void sample_txq_registers(volatile uint32_t* dst) {
    for (uint32_t q = 0; q < PHASE_PROFILE_TXQS && q < NUM_ETH_QUEUES; q++) {
        if ((TXQ_MASK >> q) & 1) {
            for (uint32_t r = 0; r < TXQ_REGISTER_COUNT; r++) {
                dst[2 * (q * TXQ_REGISTER_COUNT + r)] = eth_txq_reg_read(q, fabric_txq_register_offsets[r]);
                dst[2 * (q * TXQ_REGISTER_COUNT + r) + 1] = 0;
            }
        }
    }
}

// Device side of the experimental per-ERISC phase profiler (see fabric_phase_profile_types.hpp).
//
// The hot path updates 32-bit counters in ERISC local data memory; flush() folds them into 64-bit running totals
// and copies those to L1 once per outer main-loop iteration. All durations are core clock cycles.
//
// open<P>() / close<P>(start) bracket one occurrence of phase P. close always counts the occurrence (level >= 1);
// both read the cycle counter only when P is timed (level >= 2 and bit P of PHASE_MASK).
// TXQ_MASK selects the Ethernet TX queues whose registers flush() samples (0: none).
template <
    uint32_t MODE,
    size_t BUFFER_ADDR,
    uint32_t ERISC_ID,
    uint32_t PHASE_MASK,
    bool ITER_TIMING,
    uint32_t TXQ_MASK>
struct FabricPhaseProfiler {
    static constexpr bool ENABLED = MODE >= 1;

    template <FabricPhase P>
    static constexpr bool timed = MODE >= 2 && ((PHASE_MASK >> P) & 1) != 0;

    struct Window {
        uint32_t iters;
        uint32_t busy;
        uint32_t blocked;
        uint32_t idle;
        uint32_t gap;
        uint32_t pkts;
        uint32_t events[EVENT_COUNT];
        uint32_t phase_ticks[PHASE_COUNT];
        uint32_t phase_count[PHASE_COUNT];
    };

    FabricPhaseProfile totals;
    Window window;
    uint32_t last_ts;
    uint32_t idle_streak;
    uint32_t idle_iters;
    bool in_gap;
    bool blocked;
    bool progress;
    // Set when any iteration of the current window is outside an idle gap. Windows spent entirely in a gap (between
    // tests) are dropped at flush so they do not inflate per-packet iteration and visit counts.
    bool window_active;

    static FORCE_INLINE uint32_t now() { return fabric_core_cycles(); }

    FORCE_INLINE void clear_window() {
        uint32_t* words = reinterpret_cast<uint32_t*>(&window);
        for (uint32_t i = 0; i < sizeof(Window) / sizeof(uint32_t); i++) {
            words[i] = 0;
        }
    }

    // Measures what one empty timed open/close pair costs: the cycles it shows inside the phase it brackets, and
    // the cycles it adds to any enclosing timer (or to the loop when nothing encloses it).
    FORCE_INLINE void calibrate() {
        constexpr uint32_t N = 64;
        uint32_t inner = 0;
        const uint32_t start = now();
        for (uint32_t i = 0; i < N; i++) {
            const uint32_t t = now();
            const uint32_t t_end = now();
            inner += t_end - t;
            window.phase_ticks[0] += t_end - t;
            window.phase_count[0]++;
            asm volatile("" ::: "memory");  // keep the window updates in memory, as at real call sites
        }
        const uint32_t full = now() - start;
        // Undo the calibration's own window updates.
        window.phase_ticks[0] -= inner;
        window.phase_count[0] -= N;
        totals.timer_cal = (static_cast<uint64_t>(full * 100 / N) << 32) | (inner * 100 / N);
    }

    template <uint32_t N_TX_CHANNELS, uint32_t N_RX_CHANNELS>
    FORCE_INLINE void init() {
        uint32_t* words = reinterpret_cast<uint32_t*>(&totals);
        for (uint32_t i = 0; i < sizeof(FabricPhaseProfile) / sizeof(uint32_t); i++) {
            words[i] = 0;
        }
        clear_window();
        totals.info = N_TX_CHANNELS | (N_RX_CHANNELS << 8) | (static_cast<uint64_t>(PHASE_MASK & 0xFFFF) << 16) |
                      (static_cast<uint64_t>(MODE) << 32);
        totals.txq_mask = TXQ_MASK;
        if constexpr (MODE >= 2) {
            calibrate();
        }
        last_ts = now();
        idle_streak = 0;
        idle_iters = 0;
        window_active = false;
        in_gap = false;
        blocked = false;
        progress = false;
        flush();
    }

    FORCE_INLINE void mark_progress() { progress = true; }
    FORCE_INLINE void mark_blocked() { blocked = true; }
    FORCE_INLINE void count_event(FabricPhaseEvent event) { window.events[event]++; }
    FORCE_INLINE void count_pkt() { window.pkts++; }

    // Attributes the time since the previous call to busy, blocked or idle (with ITER_TIMING), and counts it.
    FORCE_INLINE void end_iteration() {
        window.iters++;
        if constexpr (!ITER_TIMING) {
            if (progress || blocked) {
                idle_iters = 0;
            } else if (idle_iters < PHASE_PROFILE_IDLE_GAP_ITERS) {
                idle_iters++;
            }
            in_gap = idle_iters >= PHASE_PROFILE_IDLE_GAP_ITERS;
        } else {
            const uint32_t t = now();
            const uint32_t delta = t - last_ts;
            last_ts = t;
            if (progress) {
                window.busy += delta;
                window.idle += idle_streak;
                idle_streak = 0;
                in_gap = false;
            } else if (blocked) {
                window.blocked += delta;
                window.idle += idle_streak;
                idle_streak = 0;
                in_gap = false;
            } else if (in_gap) {
                window.gap += delta;
            } else {
                idle_streak += delta;
                if (idle_streak >= PHASE_PROFILE_IDLE_GAP_TICKS) {
                    window.gap += idle_streak;
                    idle_streak = 0;
                    in_gap = true;
                }
            }
        }
        window_active |= !in_gap;
        blocked = false;
        progress = false;
    }

    // Time spent here and outside the inner loop (termination checks, telemetry) is charged to the next iteration.
    FORCE_INLINE void flush() {
#if defined(COMPILE_FOR_DRISC)
        totals.wall_ts = get_timestamp();
#else
        totals.wall_ts = eth_read_wall_clock();
#endif
        totals.cycle_ts += now() - static_cast<uint32_t>(totals.cycle_ts);
        totals.gap_ticks += window.gap;
        if (window_active) {
            totals.iters += window.iters;
            totals.busy_ticks += window.busy;
            totals.blocked_ticks += window.blocked;
            totals.idle_ticks += window.idle;
            totals.pkts += window.pkts;
            for (uint32_t i = 0; i < EVENT_COUNT; i++) {
                totals.events[i] += window.events[i];
            }
            for (uint32_t i = 0; i < PHASE_COUNT; i++) {
                totals.phase_ticks[i] += window.phase_ticks[i];
                totals.phase_count[i] += window.phase_count[i];
            }
        }
        window_active = false;
        clear_window();
        if constexpr (TXQ_MASK != 0) {
            sample_txq_registers<TXQ_MASK>(reinterpret_cast<volatile uint32_t*>(&totals.txq[0][0]));
        }
        volatile tt_l1_ptr uint32_t* dst =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(BUFFER_ADDR + ERISC_ID * sizeof(FabricPhaseProfile));
        const uint32_t* src = reinterpret_cast<const uint32_t*>(&totals);
        for (uint32_t i = 0; i < sizeof(FabricPhaseProfile) / sizeof(uint32_t); i++) {
            dst[i] = src[i];
        }
    }

    template <FabricPhase P>
    FORCE_INLINE uint32_t open() {
        if constexpr (timed<P>) {
            return now();
        } else {
            return 0;
        }
    }

    template <FabricPhase P>
    FORCE_INLINE void close(uint32_t start) {
        if constexpr (timed<P>) {
            window.phase_ticks[P] += now() - start;
        }
        window.phase_count[P]++;
    }
};

// Disabled profiler: every hot-path hook compiles away. With TXQ_MASK set, init() clears this ERISC's block and
// flush() samples the TX queue registers into it every PHASE_PROFILE_TXQ_SAMPLE_PERIOD outer loop iterations;
// nothing else is recorded.
template <size_t BUFFER_ADDR, uint32_t ERISC_ID, uint32_t PHASE_MASK, bool ITER_TIMING, uint32_t TXQ_MASK>
struct FabricPhaseProfiler<0, BUFFER_ADDR, ERISC_ID, PHASE_MASK, ITER_TIMING, TXQ_MASK> {
    static constexpr bool ENABLED = false;
    uint32_t txq_flushes;

    static FORCE_INLINE volatile tt_l1_ptr uint32_t* block() {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(BUFFER_ADDR + ERISC_ID * sizeof(FabricPhaseProfile));
    }

    template <uint32_t N_TX_CHANNELS, uint32_t N_RX_CHANNELS>
    FORCE_INLINE void init() {
        if constexpr (TXQ_MASK != 0) {
            for (uint32_t i = 0; i < sizeof(FabricPhaseProfile) / sizeof(uint32_t); i++) {
                block()[i] = 0;
            }
            block()[offsetof(FabricPhaseProfile, txq_mask) / sizeof(uint32_t)] = TXQ_MASK;
            txq_flushes = 0;
            sample_txq_registers<TXQ_MASK>(block() + offsetof(FabricPhaseProfile, txq) / sizeof(uint32_t));
        }
    }
    FORCE_INLINE void mark_progress() {}
    FORCE_INLINE void mark_blocked() {}
    FORCE_INLINE void count_event(FabricPhaseEvent) {}
    FORCE_INLINE void count_pkt() {}
    FORCE_INLINE void end_iteration() {}
    FORCE_INLINE void flush() {
        if constexpr (TXQ_MASK != 0) {
            if (++txq_flushes == PHASE_PROFILE_TXQ_SAMPLE_PERIOD) {
                txq_flushes = 0;
                sample_txq_registers<TXQ_MASK>(block() + offsetof(FabricPhaseProfile, txq) / sizeof(uint32_t));
            }
        }
    }
    template <FabricPhase P>
    FORCE_INLINE uint32_t open() {
        return 0;
    }
    template <FabricPhase P>
    FORCE_INLINE void close(uint32_t) {}
};

// Spins for `cycles` core clock cycles; used to inject extra per-packet work for criticality experiments.
template <uint32_t cycles>
FORCE_INLINE void fabric_experiment_spin() {
    if constexpr (cycles > 0) {
        const uint32_t start = fabric_core_cycles();
        while (fabric_core_cycles() - start < cycles) {
        }
    }
}
