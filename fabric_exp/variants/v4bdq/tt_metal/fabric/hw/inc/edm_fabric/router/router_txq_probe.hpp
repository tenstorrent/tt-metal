// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// TX-queue probe (variants/v4bdq only; attribution, never mainline; V4_BOUNDS.md §C). Measures, on the sending ERISC:
//   - per send: whether the post-send status read (the post-send group) still saw the TX queue busy, and how many
//     busy polls the spin before the packets-received update then took (the router's own flow, unchanged but for a
//     counter decision taken before the data command);
//   - every 61st send (61: not a divisor of any burst or ring length): the queue's busy time after the data command,
//     polled right after the command until it reads not busy (the post-send group then runs after the wait, so the
//     sampled send is perturbed), and after the packets-received register command;
//   - pre-send yields (the queue busy before a send).
// Results go into this ERISC's phase-profile block, which the builder allocates only with TT_FABRIC_TXQ_COUNTERS=1
// (the host diffs it per test into results/<label>_phase.csv); without it every hook compiles away. Slots (k = index
// into events[], phase_ticks[], phase_count[] as one array; 64-bit, low word first), router_rt.hpp's layout:
//   0 (unused; sends = 61 x sampled + < 61)   1 normal sends whose post-send status read saw the queue busy   2 busy polls in those sends' spins
//   3 pre-send yields   4 sampled sends   5 sampled: data busy cycles (sum)   6 sampled: data busy max
//   7 sampled: register-command busy cycles (sum)   8 sampled: register busy max
//   9 sampled: one more poll right after the data wait, on the now idle queue (cycles, sum: the poll's own cost)
//   10-17 sampled data busy histogram, 16-cycle buckets (the last open)

#include "router_common.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/telemetry/fabric_phase_profile_types.hpp"

#include <cstddef>

namespace fabric_router::txq_probe {

#if defined(FABRIC_TXQ_PROBE_ADDR)
constexpr uint32_t buffer = FABRIC_TXQ_PROBE_ADDR;  // offline compile checks: the logged args have no profile buffer
#else
constexpr uint32_t buffer = NAMED_CT_ARG("PHASE_PROFILE_BUFFER_ADDR");
#endif
constexpr bool on = buffer != 0;
constexpr uint32_t block = buffer + MY_ERISC_ID * sizeof(FabricPhaseProfile);
constexpr uint32_t slots_addr = block + offsetof(FabricPhaseProfile, events);
// The block's packet count (low word): the host reports a block only when it changed over the test; 61 per sample.
constexpr uint32_t pkts_addr = block + offsetof(FabricPhaseProfile, pkts);
static_assert(offsetof(FabricPhaseProfile, phase_ticks) == offsetof(FabricPhaseProfile, events) + 8 * EVENT_COUNT);
static_assert(offsetof(FabricPhaseProfile, phase_count) == offsetof(FabricPhaseProfile, phase_ticks) + 8 * PHASE_COUNT);
static_assert(EVENT_COUNT + 2 * PHASE_COUNT >= 18);
constexpr uint32_t SAMPLE_EVERY = 61;

FORCE_INLINE volatile tt_l1_ptr uint32_t* slot(uint32_t k) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slots_addr + 8 * k);
}
FORCE_INLINE void add(uint32_t k, uint32_t v) {
    volatile tt_l1_ptr uint32_t* p = slot(k);
    const uint32_t low = p[0] + v;
    p[0] = low;
    if (low < v) {
        p[1] = p[1] + 1;
    }
}
FORCE_INLINE void max_into(uint32_t k, uint32_t v) {
    if (v > slot(k)[0]) {
        slot(k)[0] = v;
    }
}
FORCE_INLINE uint32_t cycles() {
    uint32_t c;
    asm volatile("csrr %0, mcycle" : "=r"(c));
    return c;
}

inline uint32_t countdown = SAMPLE_EVERY;

// Cycles until queue q reads not busy, from now (the poll is eth_txq_is_busy's: a CMD read, then STATUS).
FORCE_INLINE uint32_t busy_time(uint8_t q) {
    const uint32_t t0 = cycles();
    while (internal_::eth_txq_is_busy(q)) {
    }
    return cycles() - t0;
}

inline void init(uint8_t q) {
    if constexpr (on) {
        auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(block);
        for (uint32_t i = 0; i < sizeof(FabricPhaseProfile) / sizeof(uint32_t); i++) {
            words[i] = 0;
        }
        countdown = SAMPLE_EVERY;
    }
}

// Taken before the data command: whether this send is sampled.
FORCE_INLINE bool sample_due() {
    if constexpr (on) {
        const uint32_t n = countdown - 1;
        const bool due = n == 0;
        countdown = due ? SAMPLE_EVERY : n;
        return due;
    } else {
        return false;
    }
}

// Sampled send, right after the data command: wait until the queue is free, timed.
__attribute__((noinline)) inline void time_data_command(uint8_t q) {
    const uint32_t t = busy_time(q);
    const uint32_t idle = busy_time(q);
    add(4, 1);
    add(5, t);
    add(9, idle);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pkts_addr) =
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pkts_addr) + SAMPLE_EVERY;
    max_into(6, t);
    const uint32_t b = t / 16 < 7 ? t / 16 : 7;
    add(10 + b, 1);
}

// Sampled send, right after the packets-received register command.
__attribute__((noinline)) inline void time_register_command(uint8_t q) {
    const uint32_t t = busy_time(q);
    add(7, t);
    max_into(8, t);
}

// On a normal send whose post-send status read saw the queue busy, the busy
// polls its spin took.
__attribute__((noinline)) inline void count_busy_send(uint32_t polls) {
    add(1, 1);
    add(2, polls);
}
__attribute__((noinline)) inline void count_yield() { add(3, 1); }

}  // namespace fabric_router::txq_probe
