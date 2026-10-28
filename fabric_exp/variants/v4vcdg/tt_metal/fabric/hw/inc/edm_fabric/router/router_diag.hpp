// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Diagnostic counters for the completion-credit study (v4gidg / v4vcdg only; never mainline). They are written into
// this ERISC's block of the phase-profile buffer, which the builder allocates only with TT_FABRIC_TXQ_COUNTERS=1; the
// test harness reads the blocks before and after each test and writes the differences to <label>_phase.csv
// (consolidation/dg_report.py decodes them). Without that setting the buffer address is 0 and every counter compiles
// away. Each value is the low word of the block's 64-bit field (the high word stays 0).
//
// Slots (k = index into events[], phase_ticks[], phase_count[] taken as one array; pkts is the block's own field):
//   E1 (receiver)  pkts  slots credited
//                  0-15  credits by (n - 1) + 4 * (pass without a packet) + 8 * (first slot not group-aligned)
//                  16    credit frames sent       17  frames put off (TX queue busy)
//                  18    proofs that failed       19  cycles in proofs (every attempt)
//                  20    passes without a packet while a partial group waits
//   E0 (sender)    pkts  worker slots credited back
//                  0     passes where the worker channel had a packet and the far ring too few free slots
//                  1-5   worker credits of 1, 2, 3, 4, 5+ slots
//                  6     cycles in credit_worker      7  credit_worker calls

#include "router_common.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/telemetry/fabric_phase_profile_types.hpp"

#include <cstddef>

namespace fabric_router::diag {

#if defined(FABRIC_DIAG_ADDR)
constexpr uint32_t buffer = FABRIC_DIAG_ADDR;  // offline builds: the logged args have no profile buffer
#else
constexpr uint32_t buffer = NAMED_CT_ARG("PHASE_PROFILE_BUFFER_ADDR");
#endif
constexpr bool on = buffer != 0;
constexpr uint32_t block = buffer + MY_ERISC_ID * sizeof(FabricPhaseProfile);
constexpr uint32_t pkts_addr = block + offsetof(FabricPhaseProfile, pkts);
constexpr uint32_t slots_addr = block + offsetof(FabricPhaseProfile, events);
static_assert(offsetof(FabricPhaseProfile, phase_ticks) == offsetof(FabricPhaseProfile, events) + 8 * EVENT_COUNT);
static_assert(offsetof(FabricPhaseProfile, phase_count) == offsetof(FabricPhaseProfile, phase_ticks) + 8 * PHASE_COUNT);
constexpr uint32_t num_slots = EVENT_COUNT + 2 * PHASE_COUNT;

enum Slot : uint32_t {
    E1_FRAMES = 16,
    E1_FRAMES_PUT_OFF = 17,
    E1_PROOFS_FAILED = 18,
    E1_PROOF_CYCLES = 19,
    E1_IDLE_PARTIAL = 20,
    E0_BLOCKED = 0,
    E0_CREDIT_1 = 1,
    E0_CREDIT_CYCLES = 6,
    E0_CREDIT_CALLS = 7,
};

FORCE_INLINE void add_at(uint32_t addr, uint32_t v) {
    auto* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    *p = *p + v;
}

template <uint32_t K>
FORCE_INLINE void add(uint32_t v = 1) {
    static_assert(K < num_slots);
    if constexpr (on) {
        add_at(slots_addr + 8 * K, v);
    }
}

FORCE_INLINE uint32_t cycles() {
    uint32_t c = 0;
    if constexpr (on) {
        asm volatile("csrr %0, mcycle" : "=r"(c));
    }
    return c;
}

inline uint32_t worker_returned = 0;  // the worker's count at its last credit

inline void init() {
    if constexpr (on) {
        auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(block);
        for (uint32_t i = 0; i < sizeof(FabricPhaseProfile) / sizeof(uint32_t); i++) {
            words[i] = 0;
        }
        worker_returned = 0;
    }
}

// E1: n slots credited.
FORCE_INLINE void credited(uint32_t n, bool progress, bool unaligned) {
    if constexpr (on) {
        add_at(pkts_addr, n);
        add_at(slots_addr + 8 * ((n - 1) + (progress ? 0 : 4) + (unaligned ? 8 : 0)), 1);
    }
}

// E1: a proof that started at `start` (cycles()) ended; `left` says whether it proved the slots free.
FORCE_INLINE void proved(uint32_t start, bool left) {
    if constexpr (on) {
        add<E1_PROOF_CYCLES>(cycles() - start);
        if (!left) {
            add<E1_PROOFS_FAILED>();
        }
    }
}

// E0: the worker's absolute count went to `returned`; the credit took `spent` cycles.
FORCE_INLINE void worker_credit(uint32_t returned, uint32_t spent) {
    if constexpr (on) {
        const uint32_t n = returned - worker_returned;
        worker_returned = returned;
        add_at(pkts_addr, n);
        add_at(slots_addr + 8 * (E0_CREDIT_1 + (n < 5 ? n - 1 : 4)), 1);
        add<E0_CREDIT_CYCLES>(spent);
        add<E0_CREDIT_CALLS>();
    }
}

}  // namespace fabric_router::diag
