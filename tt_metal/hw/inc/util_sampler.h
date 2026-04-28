// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// ttnvtop on-chip perf-counter sampler (Phase 2.1.a + 2.1.c schema bump).
//
// Lives in brisc's idle-wait loop today (Phase 2.1.a). Every ~100k AICLK
// cycles, snapshots WALL_CLOCK_L + FPU_OUT_H + the currently-running
// kernel_id into a fixed L1 ring at MEM_UTIL_SAMPLER_BASE. The MATH-thread
// (TRISC1) hook lands in a follow-up LLK submodule PR (Phase 2.1.c
// firmware) — this header reserves the metadata bytes (math_fidelity,
// counter_sel, producer_riscv, flags) it will populate.
//
// Host tools (ttnvtop-collector) read the ring via UMD bulk L1 reads,
// detect new entries via the monotonic `head` counter, and diff successive
// entries with wrap-aware delta arithmetic on `wall_clock_l` (32 bits,
// wraps every ~4.3 s at 1 GHz — handled host-side per core).
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
// from tensix.h; reg_read from risc_common.h (BRISC firmware) or
// ckernel.h (LLK math thread). The arch headers are pulled in
// transparently below so callers don't need to remember the include order.

#pragma once

#include <cstdint>

// Pull in MEM_UTIL_SAMPLER_BASE / MEM_UTIL_SAMPLER_SIZE for the layout
// static_asserts below. dev_mem_map.h is on every Tensix RISC-V firmware
// include path (firmware + jitted-kernel builds alike).
#include "dev_mem_map.h"

// ring header (32 B) + ring[62] of 16-byte entries = 32 + 62*16 = 1024 B = MEM_UTIL_SAMPLER_SIZE
//
// Phase 2.1.c bumped the header from 16 B (4 u32) to 32 B by adding
// `current_kernel_id` (stashed by trisc1 firmware on kernel start; read by
// maybe_tick_with_kernel_id() on each sample tick) plus 12 B of reserved pad
// to keep the header a 16 B multiple. Ring capacity dropped 63 -> 62 to
// conserve the fixed 1 KiB reservation.
constexpr uint32_t UTIL_SAMPLER_MAGIC = 0x53555454u;  // 'TTUS' little-endian
constexpr uint32_t UTIL_SAMPLER_VERSION = 2u;         // v2: kernel_id + metadata bytes (Phase 2.1.c)
constexpr uint32_t UTIL_SAMPLER_RING_SIZE = 62u;
// 1 ms at 1 GHz. Phase 2.1.c.i: bumped from 100,000 (100 µs) because the host
// drain at 50 Hz × 62 slots × 64 cores = ~198k/sec drainable per chip, while
// 100µs sampling on 64 cores × 2 producer threads (BRISC + TRISC1) generated
// ~1.28M samples/sec/chip → ~84% structural sample loss. 1 ms cuts that to
// ~128k/sec/chip with no loss while keeping per-program TIME% attribution
// far finer than the host-poll's ~10 ms cadence. Override via the on-chip
// period_cycles field if a workload genuinely needs sub-ms resolution.
constexpr uint32_t UTIL_SAMPLER_DEFAULT_PERIOD_CYCLES = 1000000u;

// producer_riscv values — debug only, lets the host collector tell which
// firmware loop wrote a given sample.
constexpr uint8_t UTIL_SAMPLER_PRODUCER_BRISC = 0u;
constexpr uint8_t UTIL_SAMPLER_PRODUCER_TRISC1 = 1u;

// counter_sel matches the PERF_CNT_FPU1[16:8] mux that produced fpu_count.
// 0 = FPU_INSTRUCTION, 1 = SFPU_INSTRUCTION. Reserved values are pass-through.
constexpr uint8_t UTIL_SAMPLER_COUNTER_FPU = 0u;
constexpr uint8_t UTIL_SAMPLER_COUNTER_SFPU = 1u;

// math_fidelity: 0 = unset (Phase 2.1.a producer), 1 = LoFi (16 cyc/tile),
// 2 = HiFi2 (32), 4 = HiFi4 (64). The TRISC1 hook (Phase 2.1.c LLK PR) will
// populate this from the live tile op's fidelity.
constexpr uint8_t UTIL_SAMPLER_MATH_FIDELITY_UNSET = 0u;
constexpr uint8_t UTIL_SAMPLER_MATH_FIDELITY_LOFI = 1u;
constexpr uint8_t UTIL_SAMPLER_MATH_FIDELITY_HIFI2 = 2u;
constexpr uint8_t UTIL_SAMPLER_MATH_FIDELITY_HIFI4 = 4u;

// flags bit 0: kernel-start marker. The TRISC1 LLK hook will set this on
// the first tick after a math_hw_configure so the host can attribute
// cycles to the new kernel without ambiguity at slot boundaries.
constexpr uint8_t UTIL_SAMPLER_FLAG_KERNEL_START = 1u << 0;

struct util_sampler_entry_t {
    uint32_t wall_clock_l;   // RISCV_DEBUG_REG_WALL_CLOCK_L snapshot. Wraps every ~4.3s at 1 GHz.
    uint32_t kernel_id;      // host_assigned_id (full u32 from launch_msg) at sample time. 0 = no program.
    uint32_t fpu_count;      // RISCV_DEBUG_REG_PERF_CNT_OUT_H_FPU snapshot (whichever counter_sel is current).
    uint8_t math_fidelity;   // 0 = unset; 1 = LoFi; 2 = HiFi2; 4 = HiFi4.
    uint8_t counter_sel;     // 0 = FPU_INSTRUCTION; 1 = SFPU_INSTRUCTION.
    uint8_t producer_riscv;  // 0 = BRISC (Phase 2.1.a); 1 = TRISC1 (Phase 2.1.c). Debug only.
    uint8_t flags;           // bit 0: kernel_start marker (set by TRISC1 hook A). Other bits reserved.
};
static_assert(sizeof(util_sampler_entry_t) == 16);

struct util_sampler_msg_t {
    volatile uint32_t magic;              // 'TTUS'
    volatile uint32_t version;            // 2 (was 1 in Phase 2.1.a)
    volatile uint32_t head;               // monotonic counter
    volatile uint32_t period_cycles;      // host-tunable; default 100_000 (100 us at 1 GHz)
    volatile uint32_t current_kernel_id;  // stashed by trisc1 firmware on kernel start
                                          // (host_assigned_id from launch_msg); read by
                                          // maybe_tick_with_kernel_id() to attribute each
                                          // sample. 0 = no program currently bound.
    volatile uint32_t next_due_wall_l;    // Phase 2.1.c.i: persistent sample-deadline
                                          // (RISCV_DEBUG_REG_WALL_CLOCK_L target). Lives
                                          // in L1 instead of a function-local static so
                                          // it survives across LLK kernel TUs. Without
                                          // this, every JIT-built kernel re-instantiates
                                          // the static at 0 and Hook B fires on its
                                          // first wait_for_dest call, flooding the ring
                                          // with O(kernels/sec) samples instead of
                                          // O(1/period_us). Single writer per core
                                          // (TRISC1's maybe_tick_with_kernel_id).
    volatile uint32_t reserved[2];        // pad to 32 B so ring[] stays 16 B aligned.
    volatile util_sampler_entry_t ring[UTIL_SAMPLER_RING_SIZE];
};
static_assert(sizeof(util_sampler_msg_t) == 32 + UTIL_SAMPLER_RING_SIZE * sizeof(util_sampler_entry_t));
static_assert(sizeof(util_sampler_msg_t) == 1024, "v2 must keep size at 1024 to match MEM_UTIL_SAMPLER_SIZE");
static_assert(sizeof(util_sampler_msg_t) == MEM_UTIL_SAMPLER_SIZE, "util_sampler_msg_t size must equal reservation");

namespace ttnvtop_sampler {

// Local volatile-MMIO read. The header is consumed by both BRISC firmware
// (which has `::reg_read` from internal/.../risc_common.h) and TRISC math
// (which has `ckernel::reg_read`). Rather than #if-dispatch by COMPILE_FOR_*,
// inline the same one-line MMIO load here — both vendor implementations are
// byte-identical (volatile uint32_t* cast + read).
inline __attribute__((always_inline)) uint32_t sampler_reg_read(uint32_t addr) {
    volatile uint32_t* p = reinterpret_cast<volatile uint32_t*>(addr);
    return *p;
}

// Namespace-scope global: C++ function-local statics would need __cxa_guard_*
// runtime support that isn't present on brisc firmware.
inline uint64_t g_last_sample_wall = 0;

inline volatile util_sampler_msg_t* ring() {
    return reinterpret_cast<volatile util_sampler_msg_t*>(static_cast<uintptr_t>(MEM_UTIL_SAMPLER_BASE));
}

inline void init() {
    // Tight u32-store loop over the entire 1024 B reservation (256 u32s).
    // Compiler emits ~6 RV32 instructions; the previously-explicit per-field
    // ring loop unrolled enough to push brisc.elf over its 0x1800 region
    // budget. Host-side reader gates on `head`, so untouched ring slots are
    // safely ignored regardless of contents — but a clean zero is still the
    // least surprising starting state for the magic/version/period header.
    volatile uint32_t* p = reinterpret_cast<volatile uint32_t*>(static_cast<uintptr_t>(MEM_UTIL_SAMPLER_BASE));
    for (uint32_t i = 0; i < (MEM_UTIL_SAMPLER_SIZE / 4u); ++i) {
        p[i] = 0u;
    }
    auto* s = ring();
    s->magic = UTIL_SAMPLER_MAGIC;
    s->version = UTIL_SAMPLER_VERSION;
    s->period_cycles = UTIL_SAMPLER_DEFAULT_PERIOD_CYCLES;
    g_last_sample_wall = 0;
}

// Stash the current kernel's host_assigned_id. Called by trisc1 firmware just
// before invoking the jitted kernel — the LLK math thread then picks it up
// via maybe_tick_with_kernel_id() and stamps it onto each sample. Single
// writer (trisc1 firmware between kernel launches), single reader (LLK
// running on trisc1) so no fence/atomic is required; the L1 location is
// also visible to the host collector for debug.
//
// Phase 2.1.c.iii (kernel-boundary forced fire): when entering a NEW kernel
// (kernel_id != 0), also reset next_due_wall_l = 0 so the very first
// _llk_math_wait_for_dest_available_ call by this kernel passes Hook B's
// deadline check and samples the ring. Without this reset, kernels whose
// total wait_for_dest activity completes in << period will run entirely
// between two scheduled fires and be missed (probabilistic, scaling with
// kernel_duration / period). With the reset, every kernel that calls
// wait_for_dest_available_ at least once is captured with 100% probability.
//
// Cost: ~1 extra Hook B fire per kernel start. For Llama-class workloads
// (~1k kernels/sec/core), this adds ~64k samples/sec/chip on top of the
// period-driven rate of ~128k/sec/chip — well within the host drain
// capacity of ~200k/sec/chip. No schema bump, no new L1.
//
// We don't reset on kernel_id == 0 (between-kernels idle) so the next
// fire is still period-throttled in the gap.
inline void set_current_kernel_id(uint32_t kernel_id) {
    auto* s = ring();
    s->current_kernel_id = kernel_id;
    if (kernel_id != 0u) {
        s->next_due_wall_l = 0u;
    }
}

// DEPRECATED (Phase 2.1.c): brisc-side idle-loop sampler from Phase 2.1.a.
// Superseded by maybe_tick_with_kernel_id() running on trisc1 from the LLK
// math hook. Kept only for backward compat; brisc no longer invokes it
// (single-writer invariant for the L1 ring). New callers must use
// maybe_tick_with_kernel_id().
inline void maybe_tick() {
    // WALL_CLOCK_L first — hw latches the high half on the low read to avoid
    // tearing across the 32-bit boundary. We only persist the low half in v2
    // (host reconstructs a 64-bit timeline via wrap detection) — the high
    // read still serves as the latch trigger.
    const uint32_t wall_l = sampler_reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
    const uint32_t wall_h = sampler_reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);
    const uint64_t wall_now = (static_cast<uint64_t>(wall_h) << 32) | wall_l;

    auto* s = ring();
    const uint32_t period = s->period_cycles;
    if (period != 0u && (wall_now - g_last_sample_wall) < period) {
        return;
    }
    g_last_sample_wall = wall_now;

    // OUT_H reflects whichever counter the host has muxed via PERF_CNT_FPU1
    // (FPU_INSTRUCTION or SFPU_INSTRUCTION). For Phase 2.1.a brisc producer,
    // we don't know the live mux setting — log counter_sel=0 (FPU) as a
    // best-effort default; the host's own host-side counter_sel state
    // tracks the truth and overrides this for attribution.
    const uint32_t fpu_count = sampler_reg_read(RISCV_DEBUG_REG_PERF_CNT_OUT_H_FPU);

    // TODO(Phase 2.1.c LLK PR): the TRISC1 hook will pass the live
    // host_assigned_id from the launch slot here. brisc-side we only see
    // mailboxes->launch[mailboxes->launch_msg_rd_ptr].kernel_config
    // .host_assigned_id, but reading it would couple this header to
    // mailboxes_t — defer to the LLK hook which already sees fidelity at
    // the right call site.
    const uint32_t kernel_id = 0u;

    const uint32_t head = s->head;
    const uint32_t slot = head % UTIL_SAMPLER_RING_SIZE;
    s->ring[slot].wall_clock_l = wall_l;
    s->ring[slot].kernel_id = kernel_id;
    s->ring[slot].fpu_count = fpu_count;
    s->ring[slot].math_fidelity = UTIL_SAMPLER_MATH_FIDELITY_UNSET;
    s->ring[slot].counter_sel = UTIL_SAMPLER_COUNTER_FPU;
    s->ring[slot].producer_riscv = UTIL_SAMPLER_PRODUCER_BRISC;
    s->ring[slot].flags = 0u;
    // Head written last so a racing host reader never sees a half-written slot.
    s->head = head + 1;
}

// Phase 2.1.c TRISC1/LLK producer. Called from _llk_math_wait_for_dest_available_
// (a tile-rate hot path). Fast path is ~5 instructions: WALL_CLOCK_L read,
// signed-delta deadline compare, early return. Cached deadline avoids a
// second MMIO read on the slow path. Single writer for the ring (trisc1).
inline void maybe_tick_with_kernel_id() {
    // Phase 2.1.c.i: deadline lives in L1 ring header (`next_due_wall_l`)
    // instead of a function-local static. JIT-built kernels are each their
    // own translation unit — a static would be re-instantiated at 0 per
    // kernel, causing Hook B to fire on the first wait_for_dest of every
    // new kernel (O(kernels/sec) sample rate, ring overflow, ~90% lost
    // observed empirically). The L1 location persists across kernel TUs
    // and across the BRISC↔TRISC1 producer split.
    auto* s = ring();
    const uint32_t wall_l = sampler_reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
    const uint32_t deadline = s->next_due_wall_l;
    if (static_cast<int32_t>(wall_l - deadline) < 0) {
        return;
    }

    const uint32_t period = s->period_cycles;
    if (period == 0u) {
        return;
    }
    s->next_due_wall_l = wall_l + period;

    // Slow path: write a sample. counter_sel/math_fidelity stamps land in a
    // future hook (Phase 2.1.d / .e). For now they're zeroed — the host
    // tracks counter_sel mux state independently.
    const uint32_t fpu_out_h = sampler_reg_read(RISCV_DEBUG_REG_PERF_CNT_OUT_H_FPU);
    const uint32_t kid = s->current_kernel_id;  // stashed by set_current_kernel_id

    const uint32_t head = s->head;
    const uint32_t slot = head % UTIL_SAMPLER_RING_SIZE;
    s->ring[slot].wall_clock_l = wall_l;
    s->ring[slot].kernel_id = kid;
    s->ring[slot].fpu_count = fpu_out_h;
    s->ring[slot].math_fidelity = UTIL_SAMPLER_MATH_FIDELITY_UNSET;
    s->ring[slot].counter_sel = UTIL_SAMPLER_COUNTER_FPU;
    s->ring[slot].producer_riscv = UTIL_SAMPLER_PRODUCER_TRISC1;
    s->ring[slot].flags = 0u;
    // Head written last so a racing host reader never sees a half-written slot.
    s->head = head + 1;
}

// Phase 2.1.c.ii: TRISC1/LLK first-fire-only producer. Called from
// _llk_math_dest_section_done_ to catch persistent matmul kernels that
// hold the dest buffer open across many tiles and never invoke
// _llk_math_wait_for_dest_available_. Only fires when next_due_wall_l == 0
// (Hook A resets it on every kernel start). After the first fire on any
// math hook for the current kernel, the deadline becomes non-zero and
// this function early-returns — so the cost on kernels that DO call
// wait_for_dest_available_ (most of them) is one volatile-read per
// dest_section_done call. Without this gate, doubling the sample
// source doubles the per-chip ring producer rate, overflows the host
// drain budget, and reduces attribution coverage instead of raising it.
inline void maybe_tick_with_kernel_id_first_only() {
    auto* s = ring();
    if (s->next_due_wall_l != 0u) {
        return;  // already fired at least once for this kernel
    }
    maybe_tick_with_kernel_id();
}

// Phase 2.1.c.iv: unconditional kernel-start sample. Writes one ring entry
// at the moment trisc.cc transfers control to the JIT-compiled kernel.
// Guarantees that every kernel that ever runs on TRISC1 — including ones
// that never call _llk_math_wait_for_dest_available_ or
// _llk_math_dest_section_done_ (pure SFPU paths, unpack/pack-only,
// kernels that complete before any tile is written, debug/test kernels)
// — has at least one ring entry attributed to its host_assigned_id.
//
// Bypasses the deadline gate by design: the goal is presence capture, not
// rate-limited sampling. After this fires, next_due_wall_l is set to
// wall + period, so subsequent Hook B / Hook B' calls inside this kernel
// are throttled normally.
//
// Cost: one ring write per kernel launch. At Llama's ~1k kernels/sec/core
// × 64 cores × 2 chips = ~128k extra entries/sec total — well within the
// host drain budget. The kernel_id written is the encoded host_assigned_id
// stashed by set_current_kernel_id, decoded host-side like every other
// sample.
inline void force_kernel_start_sample() {
    auto* s = ring();
    const uint32_t kid = s->current_kernel_id;
    if (kid == 0u) {
        return;  // no kernel bound — nothing to attribute
    }
    const uint32_t wall_l = sampler_reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
    const uint32_t period = s->period_cycles;
    if (period != 0u) {
        s->next_due_wall_l = wall_l + period;
    }
    const uint32_t fpu_out_h = sampler_reg_read(RISCV_DEBUG_REG_PERF_CNT_OUT_H_FPU);

    const uint32_t head = s->head;
    const uint32_t slot = head % UTIL_SAMPLER_RING_SIZE;
    s->ring[slot].wall_clock_l = wall_l;
    s->ring[slot].kernel_id = kid;
    s->ring[slot].fpu_count = fpu_out_h;
    s->ring[slot].math_fidelity = UTIL_SAMPLER_MATH_FIDELITY_UNSET;
    s->ring[slot].counter_sel = UTIL_SAMPLER_COUNTER_FPU;
    s->ring[slot].producer_riscv = UTIL_SAMPLER_PRODUCER_TRISC1;
    s->ring[slot].flags = UTIL_SAMPLER_FLAG_KERNEL_START;
    s->head = head + 1;
}

}  // namespace ttnvtop_sampler
