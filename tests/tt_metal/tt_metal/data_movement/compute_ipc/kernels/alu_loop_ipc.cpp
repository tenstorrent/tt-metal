// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Isolated ALU-only IPC probe: measures instructions-retired-per-cycle for a fixed sequence of
// independent integer ALU ops, with zero NoC/memory traffic, so the result reflects only the core's
// issue/execute pipeline -- not cache, prefetcher, or uncached-memory behavior. Intended to be run on
// a single Quasar DM and a single BH BRISC to compare raw core IPC in isolation from the
// memory-system effects that the FD copy benchmarks cannot separate out on their own.
//
// The instruction count per loop iteration is fixed and known from the asm below (12 independent ALU
// ops + 1 decrement + 1 branch = 14), so instret is a correctness check as well as a measurement: the
// host asserts the measured delta equals exactly 14*iterations + 3 (see run_alu_pass()'s comment for
// the "+3"). A second pass at 2x the iteration count gives a cheap linearity sanity check without a
// full marginal-cost sweep, which isn't needed here since (unlike NoC/iDMA) there is no per-call
// fixed cost to separate out.
//
// A second probe, run_load_use_pass(), targets a specific hypothesis from the FD prefetcher
// disassembly: process_relay_inline_noflush_cmd's command-header decode reads through
// uncached_l1_ptr<CQPrefetchCmd>(cmd_ptr) (cq_common.hpp), which adds MEM_L1_UNCACHED_BASE on Quasar
// and is a no-op on BH/WH. The decoded length feeds a shift/or almost immediately -- a tight
// load-use dependency, not an independent op -- so this probe replicates that shape (load, then
// immediately consume the result) against both the cached and uncached view of the same address, to
// see whether the uncached alias costs more cycles per load-use on Quasar specifically.
//
// Arch portability: rdcycle/rdinstret are Zicntr CSRs implemented identically on tt-1xx (BRISC) and
// tt-2xx (Quasar DM); deltas are kept in uint32_t so the same source works on RV32 and RV64 without
// needing the *h high-word CSRs (see clock_calibration_example.cpp for the precedent).

#include "api/dataflow/dataflow_api.h"
#include "api/debug/device_print.h"
#include "experimental/kernel_args.h"
#include <cstdint>
#if defined(ARCH_QUASAR)
// L2_INVALIDATE_ADDR (via overlay_addresses.h) for the invalidate-then-cache pass. Explicit tt-2xx
// path rather than the bare "risc_common.h" both arches resolve, so this can never pull the tt-1xx one.
#include "internal/tt-2xx/risc_common.h"
#endif

// Iteration counts are sized for EMULATOR runtime, not statistical power. The signal here is huge
// (cached ~5 vs uncached ~40 cyc/iter) and the emulator is deterministic, so reruns are bit-identical
// and extra samples buy nothing. What they must stay large enough for is to swamp the one-time
// pipeline/branch-predictor warmup, measured at 18-50 cycles per pass -- at these counts that is
// <=1% of any pass. The uncached passes dominate wall time (~40-160 cyc per iteration vs ~1 for ALU),
// so they get the smallest counts.
constexpr uint32_t kOpsPerIter = 14;  // 12 ALU ops + addi (decrement) + bnez (branch)
constexpr uint32_t kIterationsPass0 = 5000;
constexpr uint32_t kIterationsPass1 = 10000;  // 2x pass0, for the linearity sanity check
constexpr uint32_t kLoadUseOpsPerIter = 4;    // lw + add (use) + addi (decrement) + bnez (branch)
constexpr uint32_t kLoadUseIterations = 5000;

// Four load-use pairs per iteration, modelling the real command-header decode: in
// process_relay_inline_noflush_cmd, cmd->relay_inline.length is reconstructed from 4 lbu byte-loads,
// each consumed by a shift/or 1-2 instructions later. All four offsets sit inside ONE 64 B cache
// line, so a cached variant takes one miss then three hits -- the pattern an invalidate-then-cache
// strategy would produce in cq_prefetch.cpp. Requires load_src_addr to be 64 B aligned.
constexpr uint32_t kMultiLoadOpsPerIter = 10;  // 4x(lw+add) + addi + bnez
constexpr uint32_t kInvalLoadOpsPerIter = 13;  // fence + sd + fence + 4x(lw+add) + addi + bnez
constexpr uint32_t kMultiLoadIterations = 2000;

// Bound on the full-cache-invalidate completion poll. Large enough that a legitimately slow
// invalidation still completes, small enough that a misprogrammed ready-bit mask cannot hang the
// emulator run and take every other pass's results down with it.
constexpr uint32_t kFullInvalMaxPoll = 10000;
// The full-invalidate pass wipes the whole cache every iteration, so it is the most expensive pass
// here; it gets its own smaller count.
constexpr uint32_t kFullInvalIterations = 500;

struct AluPassResult {
    uint32_t cycles;
    uint32_t instret;
};

#if defined(ARCH_QUASAR)
// Rocket HPM counters (mhpmcounter3..6 at 0xB03..0xB06, selected by mhpmevent3..6 at 0x323..0x326).
// Selector = event set in bits[7:0], one-hot event mask in bits[N:8]. The event table these encodings
// come from is UNVERIFIED against this build's RTL, which is what these passes are here to check: each
// one has a known answer, so a counter that disagrees identifies a wrong encoding rather than a
// surprising microarchitecture. Read outside the timed asm blocks -- at thousands of iterations per
// pass the few cycles of skew are immaterial, unlike the short brackets in the FD benchmark.
// A cache miss and an uncached access both increment dcache_miss, but they are distinguishable: a miss
// blocks the D$ and does not interlock, an uncached access interlocks and does not block the D$. So
// dcache_miss decomposes as long_lat/32 (uncached loads) + dcache_blocked/60 (real misses) + uncached
// stores, and replay independently counts the first two terms. Earlier runs also validated INSN_LOAD
// (exact) and STALL_LOAD_USE (~1 cyc/cached load, ~2 for a miss); both are dropped as answered.
struct HpmSnapshot {
    uint32_t dcache_miss;     // events
    uint32_t dcache_blocked;  // cycles -- real cache-miss cost
    uint32_t long_lat;        // cycles -- uncached-access cost
    uint32_t replay;          // events -- one per miss and per uncached access
};

FORCE_INLINE void hpm_program() {
    asm volatile("csrw 0x325, %0" ::"r"(0x201));    // set 1 bit 1: long-latency interlock
    asm volatile("csrw 0x326, %0" ::"r"(0x10001));  // set 1 bit 8: replay
    asm volatile("csrw 0x323, %0" ::"r"(0x202));    // set 2 bit 1: D$ miss
    asm volatile("csrw 0x324, %0" ::"r"(0x1001));   // set 1 bit 4: D$ blocked
}

FORCE_INLINE HpmSnapshot hpm_read() {
    HpmSnapshot s;
    asm volatile("csrr %0, 0xB03" : "=r"(s.dcache_miss));
    asm volatile("csrr %0, 0xB04" : "=r"(s.dcache_blocked));
    asm volatile("csrr %0, 0xB05" : "=r"(s.long_lat));
    asm volatile("csrr %0, 0xB06" : "=r"(s.replay));
    return s;
}

FORCE_INLINE HpmSnapshot hpm_delta(const HpmSnapshot& a, const HpmSnapshot& b) {
    return {
        b.dcache_miss - a.dcache_miss,
        b.dcache_blocked - a.dcache_blocked,
        b.long_lat - a.long_lat,
        b.replay - a.replay};
}
#endif

// Self-contained: both CSR pairs are read inside the same asm block as the loop, so no
// compiler-inserted instruction (spill, register move) can land inside the timed window. Clobbers
// only caller-saved temporaries (t0-t6, a2-a7) -- none are callee-saved, so no prologue/epilogue
// save/restore instructions are needed around the block either.
FORCE_INLINE AluPassResult run_alu_pass(uint32_t iterations) {
    uint32_t cycle_start, cycle_end, instret_start, instret_end;
    asm volatile(
        "rdcycle   %[c0]\n"
        "rdinstret %[i0]\n"
        "mv        a2, %[iters]\n"
        "1:\n"
        "add  t0, t0, t1\n"
        "add  t1, t1, t2\n"
        "add  t2, t2, t3\n"
        "add  t3, t3, t4\n"
        "add  t4, t4, t5\n"
        "add  t5, t5, t6\n"
        "xor  t6, t6, a3\n"
        "xor  a3, a3, a4\n"
        "xor  a4, a4, a5\n"
        "sll  a5, a5, 1\n"
        "srl  a6, a6, 1\n"
        "add  a7, a7, t0\n"
        "addi a2, a2, -1\n"
        "bnez a2, 1b\n"
        "rdcycle   %[c1]\n"
        "rdinstret %[i1]\n"
        : [c0] "=&r"(cycle_start), [i0] "=&r"(instret_start), [c1] "=r"(cycle_end), [i1] "=r"(instret_end)
        : [iters] "r"(iterations)
        : "t0", "t1", "t2", "t3", "t4", "t5", "t6", "a2", "a3", "a4", "a5", "a6", "a7", "memory");
    return {.cycles = cycle_end - cycle_start, .instret = instret_end - instret_start};
}

// Same self-contained-timing discipline as run_alu_pass(), but the loop body is a load whose result
// feeds the very next instruction -- a tight load-use dependency, not an independent op -- replicating
// the shape found in process_relay_inline_noflush_cmd's command-header decode. `addr` selects which
// view of the same underlying L1 buffer to read: pass the plain address for the cached path, or
// addr + MEM_L1_UNCACHED_BASE for the uncached-alias path (Quasar only).
FORCE_INLINE AluPassResult run_load_use_pass(uint32_t iterations, uint32_t addr) {
    uint32_t cycle_start, cycle_end, instret_start, instret_end;
    asm volatile(
        "rdcycle   %[c0]\n"
        "rdinstret %[i0]\n"
        "mv        a2, %[iters]\n"
        "1:\n"
        "lw   t0, 0(%[addr])\n"
        "add  t1, t1, t0\n"
        "addi a2, a2, -1\n"
        "bnez a2, 1b\n"
        "rdcycle   %[c1]\n"
        "rdinstret %[i1]\n"
        : [c0] "=&r"(cycle_start), [i0] "=&r"(instret_start), [c1] "=r"(cycle_end), [i1] "=r"(instret_end)
        : [iters] "r"(iterations), [addr] "r"(addr)
        : "t0", "t1", "a2", "memory");
    return {.cycles = cycle_end - cycle_start, .instret = instret_end - instret_start};
}

// Four load-use pairs against one cache line, no cache maintenance. Baseline for isolating the
// invalidate cost below (subtract this from run_invalidate_load_use_pass) and, when handed the
// uncached alias, the figure the invalidate-then-cache strategy has to beat.
FORCE_INLINE AluPassResult run_multi_load_use_pass(uint32_t iterations, uintptr_t addr) {
    uint32_t cycle_start, cycle_end, instret_start, instret_end;
    asm volatile(
        "rdcycle   %[c0]\n"
        "rdinstret %[i0]\n"
        "mv        a2, %[iters]\n"
        "1:\n"
        "lw   t0, 0(%[addr])\n"
        "add  t1, t1, t0\n"
        "lw   t0, 4(%[addr])\n"
        "add  t1, t1, t0\n"
        "lw   t0, 8(%[addr])\n"
        "add  t1, t1, t0\n"
        "lw   t0, 12(%[addr])\n"
        "add  t1, t1, t0\n"
        "addi a2, a2, -1\n"
        "bnez a2, 1b\n"
        "rdcycle   %[c1]\n"
        "rdinstret %[i1]\n"
        : [c0] "=&r"(cycle_start), [i0] "=&r"(instret_start), [c1] "=r"(cycle_end), [i1] "=r"(instret_end)
        : [iters] "r"(iterations), [addr] "r"(addr)
        : "t0", "t1", "a2", "memory");
    return {.cycles = cycle_end - cycle_start, .instret = instret_end - instret_start};
}

#if defined(ARCH_QUASAR)
// One L2 line invalidate, then four CACHED load-use pairs against that line. The invalidate sequence
// is written out inline rather than calling invalidate_l2_cache_line() so the whole loop stays in one
// asm block with a known instruction count -- it replicates that helper exactly (fence; store addr to
// the INVALIDATE64 register; fence). This is the candidate replacement for reading command fields
// through the uncached alias: pay one invalidate per line, then read the fields cached.
FORCE_INLINE AluPassResult run_invalidate_load_use_pass(uint32_t iterations, uintptr_t addr, uintptr_t inv_reg) {
    uint32_t cycle_start, cycle_end, instret_start, instret_end;
    asm volatile(
        "rdcycle   %[c0]\n"
        "rdinstret %[i0]\n"
        "mv        a2, %[iters]\n"
        "1:\n"
        "fence\n"
        "sd   %[addr], 0(%[invreg])\n"
        "fence\n"
        "lw   t0, 0(%[addr])\n"
        "add  t1, t1, t0\n"
        "lw   t0, 4(%[addr])\n"
        "add  t1, t1, t0\n"
        "lw   t0, 8(%[addr])\n"
        "add  t1, t1, t0\n"
        "lw   t0, 12(%[addr])\n"
        "add  t1, t1, t0\n"
        "addi a2, a2, -1\n"
        "bnez a2, 1b\n"
        "rdcycle   %[c1]\n"
        "rdinstret %[i1]\n"
        : [c0] "=&r"(cycle_start), [i0] "=&r"(instret_start), [c1] "=r"(cycle_end), [i1] "=r"(instret_end)
        : [iters] "r"(iterations), [addr] "r"(addr), [invreg] "r"(inv_reg)
        : "t0", "t1", "a2", "memory");
    return {.cycles = cycle_end - cycle_start, .instret = instret_end - instret_start};
}

// The line invalidate with NO loads following it, so the line is already invalid every iteration and
// nothing is refetched. Splits the ~102 cyc measured by run_invalidate_load_use_pass into the
// primitive's own cost (this pass) versus the cold refetch from TL1 (the remainder). The two have
// opposite implications: a slow primitive could be replaced, a slow refetch cannot.
FORCE_INLINE AluPassResult run_invalidate_only_pass(uint32_t iterations, uintptr_t addr, uintptr_t inv_reg) {
    uint32_t cycle_start, cycle_end, instret_start, instret_end;
    asm volatile(
        "rdcycle   %[c0]\n"
        "rdinstret %[i0]\n"
        "mv        a2, %[iters]\n"
        "1:\n"
        "fence\n"
        "sd   %[addr], 0(%[invreg])\n"
        "fence\n"
        "addi a2, a2, -1\n"
        "bnez a2, 1b\n"
        "rdcycle   %[c1]\n"
        "rdinstret %[i1]\n"
        : [c0] "=&r"(cycle_start), [i0] "=&r"(instret_start), [c1] "=r"(cycle_end), [i1] "=r"(instret_end)
        : [iters] "r"(iterations), [addr] "r"(addr), [invreg] "r"(inv_reg)
        : "a2", "memory");
    return {.cycles = cycle_end - cycle_start, .instret = instret_end - instret_start};
}

// FULL-cache invalidate (L2 wipe + whole L1 D$ discard), then the same four cached load-use pairs, for
// a like-for-like comparison against the per-line variant above. Replicates invalidate_cache_all()
// minus the I$ discard (we are not reloading code), i.e. invalidate_l2_cache() + invalidate_l1_dcache(0).
//
// Two deliberate deviations from the helper, both necessary here:
//  - It writes ALL EIGHT DM ready bits (0xFF) rather than just this hart's. invalidate_l2_cache()'s own
//    comment requires either every DM core to call it or one core to write the others' bits; only one DM
//    runs in this benchmark, so writing 0xFF is what lets the hardware proceed.
//  - The completion poll is BOUNDED (kFullInvalMaxPoll). An unbounded poll would hang the emulator run
//    and lose every other pass's results with it, since the host only reads them after the kernel exits.
//    Instret is therefore NOT exactly predictable for this pass -- it varies with poll iterations, and
//    the host reports it rather than asserting on it. A suspiciously large instret means the poll was
//    spinning; instret at the floor means it cleared immediately.
FORCE_INLINE AluPassResult
run_full_invalidate_load_use_pass(uint32_t iterations, uintptr_t addr, uintptr_t full_inv_reg, uint32_t max_poll) {
    uint32_t cycle_start, cycle_end, instret_start, instret_end;
    asm volatile(
        "rdcycle   %[c0]\n"
        "rdinstret %[i0]\n"
        "mv        a2, %[iters]\n"
        "1:\n"
        "fence\n"
        "li   t2, 0xFF\n"  // all 8 DM ready bits; see comment above
        "sd   t2, 0(%[fullinv])\n"
        "mv   t3, %[maxpoll]\n"
        "2:\n"
        "ld   t2, 0(%[fullinv])\n"
        "beqz t2, 3f\n"  // hardware cleared it: invalidation complete
        "addi t3, t3, -1\n"
        "bnez t3, 2b\n"
        "3:\n"
        "tt.cache.cdiscard.d.l1 x0\n"  // whole L1 D$ discard (L2 is inclusive, so do both)
        "fence\n"
        "lw   t0, 0(%[addr])\n"
        "add  t1, t1, t0\n"
        "lw   t0, 4(%[addr])\n"
        "add  t1, t1, t0\n"
        "lw   t0, 8(%[addr])\n"
        "add  t1, t1, t0\n"
        "lw   t0, 12(%[addr])\n"
        "add  t1, t1, t0\n"
        "addi a2, a2, -1\n"
        "bnez a2, 1b\n"
        "rdcycle   %[c1]\n"
        "rdinstret %[i1]\n"
        : [c0] "=&r"(cycle_start), [i0] "=&r"(instret_start), [c1] "=r"(cycle_end), [i1] "=r"(instret_end)
        : [iters] "r"(iterations), [addr] "r"(addr), [fullinv] "r"(full_inv_reg), [maxpoll] "r"(max_poll)
        : "t0", "t1", "t2", "t3", "a2", "memory");
    return {.cycles = cycle_end - cycle_start, .instret = instret_end - instret_start};
}
#endif

// Quasar overlay->NoC writes need a software flush or the uncached alias for the host's NoC read to
// see them; tt-1xx L1 is plain NoC-visible memory, so a direct write is enough there.
FORCE_INLINE void write_result(uint32_t result_addr, uint32_t idx, uint32_t value) {
#if defined(ARCH_QUASAR)
    volatile uint32_t* results = reinterpret_cast<volatile uint32_t*>(result_addr + MEM_L1_UNCACHED_BASE);
#else
    volatile uint32_t* results = reinterpret_cast<volatile uint32_t*>(result_addr);
#endif
    results[idx] = value;
}

#if defined(ARCH_QUASAR)
// Four consecutive words per pass, in HpmSnapshot field order.
FORCE_INLINE void write_hpm(uint32_t result_addr, uint32_t idx, const HpmSnapshot& s) {
    write_result(result_addr, idx + 0, s.dcache_miss);
    write_result(result_addr, idx + 1, s.dcache_blocked);
    write_result(result_addr, idx + 2, s.long_lat);
    write_result(result_addr, idx + 3, s.replay);
}
#endif

void kernel_main() {
    constexpr uint32_t result_addr = get_arg(args::result_addr);
    constexpr uint32_t load_src_addr = get_arg(args::load_src_addr);

#if defined(ARCH_QUASAR)
    hpm_program();
    const HpmSnapshot hpm_t0 = hpm_read();
#endif
    const AluPassResult pass0 = run_alu_pass(kIterationsPass0);
#if defined(ARCH_QUASAR)
    const HpmSnapshot hpm_t1 = hpm_read();
#endif
    const AluPassResult pass1 = run_alu_pass(kIterationsPass1);
#if defined(ARCH_QUASAR)
    // Snapshots bracket ONE pass each. Anything between a pair -- another pass, a write_result (which
    // stores through the uncached alias), a DEVICE_PRINT -- lands in that pass's counts.
    const HpmSnapshot hpm_t1b = hpm_read();
#endif
    const AluPassResult cached_pass = run_load_use_pass(kLoadUseIterations, load_src_addr);
#if defined(ARCH_QUASAR)
    const HpmSnapshot hpm_t2 = hpm_read();
    // The null test: run_alu_pass has no loads at all, so every one of these must be 0. A nonzero value
    // means that counter is not the event its encoding claims.
    write_hpm(result_addr, 27, hpm_delta(hpm_t0, hpm_t1));
    write_hpm(result_addr, 31, hpm_delta(hpm_t1b, hpm_t2));
#endif

    write_result(result_addr, 0, kIterationsPass0);
    write_result(result_addr, 1, pass0.cycles);
    write_result(result_addr, 2, pass0.instret);
    write_result(result_addr, 3, kIterationsPass1);
    write_result(result_addr, 4, pass1.cycles);
    write_result(result_addr, 5, pass1.instret);
    write_result(result_addr, 6, kLoadUseIterations);
    write_result(result_addr, 7, cached_pass.cycles);
    write_result(result_addr, 8, cached_pass.instret);

    DEVICE_PRINT(
        "ALU_IPC pass0 iters={} cycles={} instret={} pass1 iters={} cycles={} instret={} load_use_cached "
        "iters={} cycles={} instret={}\n",
        kIterationsPass0,
        pass0.cycles,
        pass0.instret,
        kIterationsPass1,
        pass1.cycles,
        pass1.instret,
        kLoadUseIterations,
        cached_pass.cycles,
        cached_pass.instret);

    // 4-load baseline, cached, no cache maintenance. Runs on both arches.
    const AluPassResult multi_cached = run_multi_load_use_pass(kMultiLoadIterations, load_src_addr);
    write_result(result_addr, 12, kMultiLoadIterations);
    write_result(result_addr, 13, multi_cached.cycles);
    write_result(result_addr, 14, multi_cached.instret);
    DEVICE_PRINT(
        "ALU_IPC multi_load_cached iters={} cycles={} instret={}\n",
        kMultiLoadIterations,
        multi_cached.cycles,
        multi_cached.instret);

#if defined(ARCH_QUASAR)
    const HpmSnapshot hpm_t2b = hpm_read();
    const AluPassResult uncached_pass = run_load_use_pass(kLoadUseIterations, load_src_addr + MEM_L1_UNCACHED_BASE);
    const HpmSnapshot hpm_t3 = hpm_read();
    // insn_load must be exactly kLoadUseIterations here and in the cached block, which validates the
    // set-0 encoding. The interesting unknown is dcache_miss: ~0 means uncached accesses bypass the miss
    // counter, ~kLoadUseIterations means they are counted as misses -- which would explain the ~312
    // "misses" per FD command without any of them being real cache misses.
    write_hpm(result_addr, 35, hpm_delta(hpm_t2b, hpm_t3));
    write_result(result_addr, 9, kLoadUseIterations);
    write_result(result_addr, 10, uncached_pass.cycles);
    write_result(result_addr, 11, uncached_pass.instret);
    DEVICE_PRINT(
        "ALU_IPC load_use_uncached iters={} cycles={} instret={}\n",
        kLoadUseIterations,
        uncached_pass.cycles,
        uncached_pass.instret);

    // The figure the invalidate-then-cache strategy must beat: 4 field reads done the way FD does
    // them today, straight through the uncached alias.
    const AluPassResult multi_uncached =
        run_multi_load_use_pass(kMultiLoadIterations, load_src_addr + MEM_L1_UNCACHED_BASE);
    write_result(result_addr, 15, kMultiLoadIterations);
    write_result(result_addr, 16, multi_uncached.cycles);
    write_result(result_addr, 17, multi_uncached.instret);

    const HpmSnapshot hpm_t3b = hpm_read();
    // The candidate: one L2 line invalidate, then the same 4 reads cached.
    const AluPassResult inval_cached =
        run_invalidate_load_use_pass(kMultiLoadIterations, load_src_addr, L2_INVALIDATE_ADDR);
    // Read before the write_results below, whose uncached stores would otherwise land in this pass.
    // Each iteration does one MMIO invalidate store plus one line refill, so dcache_miss should read
    // 2 * kMultiLoadIterations.
    const HpmSnapshot hpm_t4 = hpm_read();
    write_result(result_addr, 18, kMultiLoadIterations);
    write_result(result_addr, 19, inval_cached.cycles);
    write_result(result_addr, 20, inval_cached.instret);
    write_hpm(result_addr, 39, hpm_delta(hpm_t3b, hpm_t4));

    // Same invalidate with no loads after it, to split the above into primitive vs refetch cost.
    const AluPassResult inval_only = run_invalidate_only_pass(kMultiLoadIterations, load_src_addr, L2_INVALIDATE_ADDR);
    write_result(result_addr, 21, kMultiLoadIterations);
    write_result(result_addr, 22, inval_only.cycles);
    write_result(result_addr, 23, inval_only.instret);

    // Full-cache invalidate instead of per-line, same 4 cached reads, for a like-for-like comparison.
    const AluPassResult full_inval = run_full_invalidate_load_use_pass(
        kFullInvalIterations, load_src_addr, L2_FULL_INVALIDATE_ADDR, kFullInvalMaxPoll);
    write_result(result_addr, 24, kFullInvalIterations);
    write_result(result_addr, 25, full_inval.cycles);
    write_result(result_addr, 26, full_inval.instret);

    DEVICE_PRINT(
        "ALU_IPC multi_load_uncached iters={} cycles={} instret={} inval_then_cached iters={} cycles={} instret={} "
        "inval_only iters={} cycles={} instret={} full_inval iters={} cycles={} instret={}\n",
        kMultiLoadIterations,
        multi_uncached.cycles,
        multi_uncached.instret,
        kMultiLoadIterations,
        inval_cached.cycles,
        inval_cached.instret,
        kMultiLoadIterations,
        inval_only.cycles,
        inval_only.instret,
        kFullInvalIterations,
        full_inval.cycles,
        full_inval.instret);
#endif
}
