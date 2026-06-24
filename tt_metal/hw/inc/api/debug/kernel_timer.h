// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Lightweight, std::chrono-like cycle timer for profiling device kernels on Quasar when tracy /
// the normal kernel profiler is unavailable (e.g. on the RTL emulator).
//
// Cycle source: the RISC-V `rdcycle` CSR, read through the single kernel_now() seam.
//
//   WHY rdcycle and not WALL_CLOCK? The 64-bit tile WALL_CLOCK registers (get_timestamp() in
//   risc_common.h) are what silicon profiling uses and are cross-core comparable, BUT reading them
//   HANGS the Quasar RTL emulator (the DM core waits forever for the TRISCs to complete — verified
//   2026-06-24). rdcycle is the per-core RISC-V cycle counter, is emulator-proven (the
//   tests/.../quasar_core_shift reference kernel uses it for exactly this reason), and is all we
//   need for per-kernel elapsed-cycle deltas.
//
//   Trade-offs vs WALL_CLOCK: rdcycle is PER-CORE (each DM/TRISC has its own counter, so timestamps
//   are NOT comparable across cores — only deltas on the same core are meaningful), and it is 32-bit
//   on the TRISC (tt-qsr32) / 64-bit on the DM (tt-qsr64). We work in 32-bit and rely on wrap-safe
//   modular subtraction, which is correct for any single interval shorter than 2^32 cycles.
//
// Usage (explicit):
//     KernelTimer t;
//     t.start();
//     ... work ...
//     uint32_t cycles = t.stop();
//     kernel_timer_write(timer_l1_addr, /*slot=*/1, cycles);
//
// Usage (RAII / scoped, chrono- or tracy-ZoneScoped-like):
//     { ScopedKernelTimer _t(timer_l1_addr, /*slot=*/1); ... work ... }   // writes on scope exit
//
// Results are written to L1 through the uncached alias (MEM_L1_UNCACHED_BASE) so the host reads the
// freshly-written value rather than a stale cache line. Each slot is 8 bytes (two uint32: the cycle
// count in word 0, and word 1 = 0 so the host can read a uniform 64-bit slot) at
// timer_l1_addr + slot * 8.

#include <cstdint>

#include "internal/risc_attribs.h"  // tt_l1_ptr
#include "dev_mem_map.h"            // MEM_L1_UNCACHED_BASE
#include "api/debug/dprint.h"       // DEVICE_PRINT (emulator readback path; WALL_CLOCK profiler unavailable)

// Single seam for the cycle source. rdcycle is the per-core RISC-V cycle CSR and works on the
// emulator (WALL_CLOCK does not). Returns the low 32 bits of the cycle counter; the "memory" clobber
// is a compiler barrier so surrounding work is not reordered across the read.
inline uint32_t kernel_now() {
    uint32_t c;
    asm volatile("rdcycle %0" : "=r"(c)::"memory");
    return c;
}

// Explicit start/stop timer. stop() returns elapsed cycles; 32-bit modular subtraction is wrap-safe
// for any interval shorter than 2^32 cycles.
struct KernelTimer {
    uint32_t t0 = 0;

    inline void start() { t0 = kernel_now(); }

    inline uint32_t stop() const { return kernel_now() - t0; }
};

// Write a cycle count to L1 (uncached alias) at timer_l1_addr + slot*8. Word 0 = cycles, word 1 = 0
// so the host always reads a uniform 8-byte slot. Also DEVICE_PRINT the result: on the emulator,
// where there is no host L1-read binding from pytest, the DPRINT is the practical readback path
// (captured via TT_METAL_DPRINT_CORES=all); the L1 slot is for host-side gtest readback on silicon.
inline void kernel_timer_write(uint32_t timer_l1_addr, uint32_t slot, uint32_t cycles) {
    volatile tt_l1_ptr uint32_t* p =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MEM_L1_UNCACHED_BASE + timer_l1_addr + slot * 8);
    p[0] = cycles;
    p[1] = 0;
    DEVICE_PRINT("[kernel-timer] slot={} cycles={}\n", slot, cycles);
}

// RAII scoped timer: records start at construction, writes elapsed cycles to (timer_l1_addr, slot)
// at destruction. Times the enclosing lexical scope.
struct ScopedKernelTimer {
    KernelTimer timer;
    uint32_t l1_addr;
    uint32_t slot;

    inline ScopedKernelTimer(uint32_t timer_l1_addr, uint32_t slot_) : l1_addr(timer_l1_addr), slot(slot_) {
        timer.start();
    }

    inline ~ScopedKernelTimer() { kernel_timer_write(l1_addr, slot, timer.stop()); }
};
