// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// ttnvtop on-chip perf-counter sampler (Phase 2.1.a).
//
// Lives in brisc's idle-wait loop. Every ~100k AICLK cycles, snapshots
// WALL_CLOCK + FPU_OUT_L/H into a fixed L1 ring at MEM_UTIL_SAMPLER_BASE.
// Host tools (ttnvtop-collector) read the ring via UMD bulk L1 reads and
// diff successive entries.
//
// Cost per idle-loop iteration when not due: 4 instructions (two MMIO
// reads for WALL_CLOCK, an elapsed-cycles compare). On a sample tick:
// ~20 instructions, <30 cycles. At 100 us period and 1 GHz, overhead is
// ~0.03% of AICLK.
//
// No interrupts, no CLINT — polling from brisc's main loop. The ring lives
// at a fixed address outside mailboxes_t, so adding it does not change
// MEM_MAILBOX_SIZE or mailboxes_t layout.
//
// Requires: MEM_UTIL_SAMPLER_BASE/SIZE from dev_mem_map.h; RISCV_DEBUG_REG_*
// from tensix.h; reg_read from risc_common.h.

#pragma once

#include <cstdint>

// ring header + ring[63] of 16-byte entries = 16 + 63*16 = 1024 B = MEM_UTIL_SAMPLER_SIZE
constexpr uint32_t UTIL_SAMPLER_MAGIC = 0x53555454u;  // 'TTUS' little-endian
constexpr uint32_t UTIL_SAMPLER_VERSION = 1u;
constexpr uint32_t UTIL_SAMPLER_RING_SIZE = 63u;
constexpr uint32_t UTIL_SAMPLER_DEFAULT_PERIOD_CYCLES = 100000u;

struct util_sampler_entry_t {
    uint32_t wall_clock_l;
    uint32_t wall_clock_h;
    uint32_t fpu_out_l;
    uint32_t fpu_out_h;
};
static_assert(sizeof(util_sampler_entry_t) == 16);

struct util_sampler_msg_t {
    volatile uint32_t magic;
    volatile uint32_t version;
    volatile uint32_t head;
    volatile uint32_t period_cycles;
    volatile util_sampler_entry_t ring[UTIL_SAMPLER_RING_SIZE];
};
static_assert(sizeof(util_sampler_msg_t) == 16 + UTIL_SAMPLER_RING_SIZE * sizeof(util_sampler_entry_t));
static_assert(sizeof(util_sampler_msg_t) == MEM_UTIL_SAMPLER_SIZE, "util_sampler_msg_t size must equal reservation");

namespace ttnvtop_sampler {

// Namespace-scope global: C++ function-local statics would need __cxa_guard_*
// runtime support that isn't present on brisc firmware.
inline uint64_t g_last_sample_wall = 0;

inline volatile util_sampler_msg_t* ring() {
    return reinterpret_cast<volatile util_sampler_msg_t*>(static_cast<uintptr_t>(MEM_UTIL_SAMPLER_BASE));
}

inline void init() {
    auto* s = ring();
    s->magic = UTIL_SAMPLER_MAGIC;
    s->version = UTIL_SAMPLER_VERSION;
    s->head = 0u;
    s->period_cycles = UTIL_SAMPLER_DEFAULT_PERIOD_CYCLES;
    for (uint32_t i = 0; i < UTIL_SAMPLER_RING_SIZE; ++i) {
        s->ring[i].wall_clock_l = 0u;
        s->ring[i].wall_clock_h = 0u;
        s->ring[i].fpu_out_l = 0u;
        s->ring[i].fpu_out_h = 0u;
    }
    g_last_sample_wall = 0;
}

inline void maybe_tick() {
    // WALL_CLOCK_L first — hw latches the high half on the low read to avoid
    // tearing across the 32-bit boundary.
    const uint32_t wall_l = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
    const uint32_t wall_h = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);
    const uint64_t wall_now = (static_cast<uint64_t>(wall_h) << 32) | wall_l;

    auto* s = ring();
    const uint32_t period = s->period_cycles;
    if (period != 0u && (wall_now - g_last_sample_wall) < period) {
        return;
    }
    g_last_sample_wall = wall_now;

    const uint32_t fpu_out_l = reg_read(RISCV_DEBUG_REG_PERF_CNT_OUT_L_FPU);
    const uint32_t fpu_out_h = reg_read(RISCV_DEBUG_REG_PERF_CNT_OUT_H_FPU);

    const uint32_t head = s->head;
    const uint32_t slot = head % UTIL_SAMPLER_RING_SIZE;
    s->ring[slot].wall_clock_l = wall_l;
    s->ring[slot].wall_clock_h = wall_h;
    s->ring[slot].fpu_out_l = fpu_out_l;
    s->ring[slot].fpu_out_h = fpu_out_h;
    // Head written last so a racing host reader never sees a half-written slot.
    s->head = head + 1;
}

}  // namespace ttnvtop_sampler
