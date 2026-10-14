// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared layout of the per-core stats row (uint32 words). Mirrored in late_atomic_bench.py.
#pragma once

#include <cstdint>

#include "c_tensix_core.h"

// Program A (late_atomic_a.cpp)
constexpr uint32_t S_A_RUNS = 1;
constexpr uint32_t S_A_BAR_MAX = 2;    // control only: max cycles in A's atomic barrier (= RTT of A's last atomic)
constexpr uint32_t S_A_BAR_SUM16 = 3;  // control only: sum of (barrier cycles >> 4)
constexpr uint32_t S_A_BAR_LATE = 4;   // control only: runs whose barrier took > LATE_THRESH cycles
constexpr uint32_t S_A_EXIT_TS = 5;    // wall clock (low 32) when A returned

// Program B (late_atomic_b.cpp)
constexpr uint32_t S_B_RUNS = 6;
constexpr uint32_t S_B_D0_NZ = 7;     // runs where NIU atomic-resp counter != sw counter at kernel_main entry
constexpr uint32_t S_B_D1_NZ = 8;     // same, after PROBE_WAIT cycles (the key metric)
constexpr uint32_t S_B_DMAX = 9;      // max (hw - sw) seen
constexpr uint32_t S_B_STATE = 10;    // 1 = inside the barrier action, 2 = action done
constexpr uint32_t S_B_GAP_MIN = 11;  // cycles from A exit to B kernel_main (includes go-signal wait)
constexpr uint32_t S_B_GAP_MAX = 12;
constexpr uint32_t S_B_RESID_NZ = 13;    // barrier returned, yet hw != sw afterwards (passed early)
constexpr uint32_t S_B_FIRST_HIT = 14;   // B run index of the first d1 != 0
constexpr uint32_t S_B_ENTRY_HW = 15;    // NIU_MST_ATOMIC_RESP_RECEIVED before the action (last run)
constexpr uint32_t S_B_ENTRY_SW = 16;    // noc_nonposted_atomics_acked before the action (last run)
constexpr uint32_t S_B_ACTION_MAX = 17;  // max cycles spent in the action
constexpr uint32_t S_B_LAST_D1 = 18;

// Semaphore words (targets only; nobody waits on them)
constexpr uint32_t S_SEM_FAR = 28;    // A's atomics land here on the far core
constexpr uint32_t S_SEM_PEER = 29;   // B's own atomics land here (self / peers)
constexpr uint32_t S_SEM_FLOOD = 30;  // flood atomics land here

constexpr uint32_t B_STATE_IN_ACTION = 1;
constexpr uint32_t B_STATE_DONE = 2;

FORCE_INLINE void spin_cycles(uint32_t n) {
    const uint64_t end = c_tensix_core::read_wall_clock() + n;
    while (c_tensix_core::read_wall_clock() < end) {
    }
}

FORCE_INLINE uint32_t mix32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}
