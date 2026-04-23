// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// ttnvtop on-chip perf-counter sampler (Phase 2.1).
//
// Lives in brisc's idle-wait loop. Every ~100k AICLK cycles (configurable),
// snapshots WALL_CLOCK + FPU_OUT_L/H into a fixed L1 ring. Host (ttnvtop-
// collector) reads the ring in bulk and diffs successive entries to derive
// per-core compute%.
//
// Cost per idle-loop iteration when sampler is not yet due: 4 instructions
// (two mmio reads for WALL_CLOCK_{L,H} and an elapsed-cycles compare).
// Cost on a sample tick: ~20 instructions (four extra mmio reads, four
// stores, head advance).
//
// No interrupts, no CLINT — just polling from the firmware's main loop.
// Requires: mailboxes_t must contain a util_sampler_msg_t member named
// util_sampler (see tt_metal/hw/inc/hostdev/dev_msgs.h).

#pragma once

#include <cstdint>

// Magic value the host uses to detect that firmware populated the ring.
// 'TTUS' little-endian. If host reads anything else the firmware isn't
// running this sampler and it should fall back to direct-register polling.
constexpr uint32_t UTIL_SAMPLER_MAGIC = 0x53555454u;
constexpr uint32_t UTIL_SAMPLER_VERSION = 1u;

// Default sampler period in AICLK cycles. ~100 us at 1 GHz. Host may
// rewrite mailboxes->util_sampler.period_cycles at runtime to tune.
constexpr uint32_t UTIL_SAMPLER_DEFAULT_PERIOD_CYCLES = 100000u;

// Forward-declare the brisc-visible mailbox struct / reg_read. Both are
// defined by whatever firmware translation unit includes this header after
// dev_msgs.h and risc_common.h.
namespace ttnvtop_sampler {

// Called once during brisc boot after dev_msgs are visible. Populates the
// magic/version and zero-initialises the ring. Idempotent if called twice.
template <typename MailboxesT>
inline void init(MailboxesT* mailboxes) {
    auto& s = mailboxes->util_sampler;
    s.magic = UTIL_SAMPLER_MAGIC;
    s.version = UTIL_SAMPLER_VERSION;
    s.head = 0u;
    if (s.period_cycles == 0u) {
        s.period_cycles = UTIL_SAMPLER_DEFAULT_PERIOD_CYCLES;
    }
    for (uint32_t i = 0; i < util_sampler_ring_size; ++i) {
        s.ring[i].wall_clock_l = 0u;
        s.ring[i].wall_clock_h = 0u;
        s.ring[i].fpu_out_l = 0u;
        s.ring[i].fpu_out_h = 0u;
    }
}

// Called from brisc's idle-wait loop. Reads WALL_CLOCK_L (which latches H
// on WH/BH), compares elapsed cycles to the period, and if due snapshots
// the perf counters into the next ring slot.
//
// The static `last_sample` tracks the most recent sample time; a single
// global per-core variable is fine because this is called only from brisc.
template <typename MailboxesT>
inline void maybe_tick(MailboxesT* mailboxes) {
    // WALL_CLOCK_L must be read first — the hardware latches the high half
    // atomically on the low-half read to avoid tearing across the 32-bit
    // boundary. See c_tensix_core::read_wall_clock() for the same pattern.
    const uint32_t wall_l = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
    const uint32_t wall_h = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);
    const uint64_t wall_now = (static_cast<uint64_t>(wall_h) << 32) | wall_l;

    static uint64_t last_sample_wall = 0;
    const uint32_t period = mailboxes->util_sampler.period_cycles;
    if (period != 0u && (wall_now - last_sample_wall) < period) {
        return;
    }
    last_sample_wall = wall_now;

    const uint32_t fpu_out_l = reg_read(RISCV_DEBUG_REG_PERF_CNT_OUT_L_FPU);
    const uint32_t fpu_out_h = reg_read(RISCV_DEBUG_REG_PERF_CNT_OUT_H_FPU);

    const uint32_t head = mailboxes->util_sampler.head;
    const uint32_t slot = head & (util_sampler_ring_size - 1);
    auto& entry = mailboxes->util_sampler.ring[slot];
    entry.wall_clock_l = wall_l;
    entry.wall_clock_h = wall_h;
    entry.fpu_out_l = fpu_out_l;
    entry.fpu_out_h = fpu_out_h;
    // Head written last so a racing host reader never sees a half-written slot.
    mailboxes->util_sampler.head = head + 1;
}

}  // namespace ttnvtop_sampler
