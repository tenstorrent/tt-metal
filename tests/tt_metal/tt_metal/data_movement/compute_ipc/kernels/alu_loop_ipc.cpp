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

constexpr uint32_t kOpsPerIter = 14;  // 12 ALU ops + addi (decrement) + bnez (branch)
constexpr uint32_t kIterationsPass0 = 100000;
constexpr uint32_t kIterationsPass1 = 200000;  // 2x pass0, for the linearity sanity check
constexpr uint32_t kLoadUseOpsPerIter = 4;     // lw + add (use) + addi (decrement) + bnez (branch)
constexpr uint32_t kLoadUseIterations = 100000;

struct AluPassResult {
    uint32_t cycles;
    uint32_t instret;
};

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

void kernel_main() {
    constexpr uint32_t result_addr = get_arg(args::result_addr);
    constexpr uint32_t load_src_addr = get_arg(args::load_src_addr);

    const AluPassResult pass0 = run_alu_pass(kIterationsPass0);
    const AluPassResult pass1 = run_alu_pass(kIterationsPass1);
    const AluPassResult cached_pass = run_load_use_pass(kLoadUseIterations, load_src_addr);

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

#if defined(ARCH_QUASAR)
    const AluPassResult uncached_pass = run_load_use_pass(kLoadUseIterations, load_src_addr + MEM_L1_UNCACHED_BASE);
    write_result(result_addr, 9, kLoadUseIterations);
    write_result(result_addr, 10, uncached_pass.cycles);
    write_result(result_addr, 11, uncached_pass.instret);
    DEVICE_PRINT(
        "ALU_IPC load_use_uncached iters={} cycles={} instret={}\n",
        kLoadUseIterations,
        uncached_pass.cycles,
        uncached_pass.instret);
#endif
}
