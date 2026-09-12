// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Hello world for Quasar DM: freedom-metal + tt-metal risc_common.h X280 cache
// ops in one binary (-mcpu=tt-qsr64-rocc). Build/verify only — no Quasar silicon.
// See README.md.

#include <stdint.h>
#include <stdio.h>

// --- freedom-e-sdk / freedom-metal ------------------------------------------
#include <metal/cache.h>
#include <metal/cpu.h>

// --- tt-metal ---------------------------------------------------------------
#include "dev_mem_map.h"  // tt_metal/hw/inc/internal/tt-2xx/quasar/

// --- this demo --------------------------------------------------------------
#include "quasar_tty.h"
#include "x280_cache_tt.h"

int main(void);

// 64B-aligned handoff buffer for both cache flush APIs (line-sized).
static volatile char handoff[128] __attribute__((aligned(64)));

static void report_toolchain(void) {
    printf("[1] toolchain and ISA\n");
    printf("  XLEN                                = %d\n", (int)__riscv_xlen);
    printf("  sizeof(void*)                       = %u bytes\n", (unsigned)sizeof(void*));
#ifdef __riscv_atomic
    printf("  A (atomics)                         = yes\n");
#else
    printf("  A (atomics)                         = no\n");
#endif
#ifdef __riscv_flen
    printf("  F/D (hardware float)                = yes (FLEN=%d)\n", (int)__riscv_flen);
#else
    printf("  F/D (hardware float)                = no  -- soft-float only\n");
#endif
#ifdef __riscv_compressed
    printf("  C (compressed)                      = yes\n");
#else
    printf("  C (compressed)                      = no\n");
#endif
#ifdef __riscv_zba
    printf("  Zba/Zbb/Zbs (bitmanip)              = yes\n");
#endif
    printf("\n");
}

static void report_freedom_metal(void) {
    printf("[2] freedom-metal (freedom-e-sdk)\n");

    const int hartid = metal_cpu_get_current_hartid();
    struct metal_cpu* cpu = metal_cpu_get(hartid);

    printf("  metal_cpu_get_current_hartid()      = %d\n", hartid);
    printf("  metal_cpu_get_num_harts()           = %d\n", metal_cpu_get_num_harts());
    printf("  metal_dcache_l1_available(hart)     = %d\n", metal_dcache_l1_available(hartid));
    printf("  metal_icache_l1_available(hart)     = %d\n", metal_icache_l1_available(hartid));

    if (cpu != NULL) {
        printf("  metal_cpu_get_timebase()            = %llu Hz\n", metal_cpu_get_timebase(cpu));
        printf("  metal_cpu_get_timer()               = %llu ticks\n", metal_cpu_get_timer(cpu));
    } else {
        printf("  metal_cpu_get()                     = NULL (no cpu node in BSP)\n");
    }
    printf("\n");
}

static void report_memory_map(void) {
    printf("[3] Quasar memory map (tt-metal dev_mem_map.h)\n");
    printf("  MEM_L1_BASE                         = 0x%08lx\n", (unsigned long)MEM_L1_BASE);
    printf("  MEM_L1_SIZE                         = %lu KB\n", (unsigned long)MEM_L1_SIZE / 1024);
    printf("  MEM_L1_UNCACHED_BASE                = 0x%08lx\n", (unsigned long)MEM_L1_UNCACHED_BASE);
    printf("  MEM_KERNEL_BASE  (link base)        = 0x%08lx\n", (unsigned long)MEM_KERNEL_BASE);
    printf("  MEM_DM_KERNEL_SIZE (window)         = %lu KB\n", (unsigned long)MEM_DM_KERNEL_SIZE / 1024);
    printf("  NUM_DM_CORES                        = %d\n", (int)NUM_DM_CORES);
    printf("\n");
    printf("  &main                               = 0x%08lx\n", (unsigned long)(uintptr_t)&main);
    printf("  &handoff                            = 0x%08lx\n", (unsigned long)(uintptr_t)handoff);
    printf("  console buffer (TL1)                = 0x%08lx\n", (unsigned long)quasar_tty_address());
    printf("\n");
}

// The integration point. Both codebases implement the same X280 cache
// maintenance; here they operate on the same address, one after the other.
//
//   freedom-metal  src/cache.c        .insn i 0x73, 0, x0, addr, -0x40
//   tt-metal       risc_common.h      tt.cache.cflush.d.l1 addr
//
// Those assemble to the same instruction -- CFLUSH.D.L1 from the X280 core
// manual. build.sh checks the encodings in the linked objects.
static void cache_handoff(void) {
    printf("[4] cache handoff: freedom-metal <-> tt-metal\n");

    const int hartid = metal_cpu_get_current_hartid();
    const uintptr_t addr = (uintptr_t)handoff;

    // Produce data in the DM core's private, write-back L1 D$.
    const char msg[] = "Hello, World from a Quasar DM core (SiFive X280 derivative)";
    for (unsigned i = 0; i < sizeof(msg); ++i) {
        handoff[i] = msg[i];
    }

    // Path A: push it out with SiFive's own HAL.
    metal_dcache_l1_flush(hartid, addr);
    printf("  metal_dcache_l1_flush(hart, addr)   -> L1 D$ line written back\n");

    // Path B: push it out with tt-metal's Quasar DM primitives, then take it all
    // the way to TL1 where the NoC and the host can see it.
    tt_x280_flush_l1_dcache(addr);
    tt_x280_flush_l2_cache_range(addr, sizeof(handoff));
    printf("  tt_x280_flush_l1_dcache(addr)       -> L1 D$ line written back\n");
    printf("  tt_x280_flush_l2_cache_range(...)   -> L2 drained to TL1 (%u bytes)\n", (unsigned)sizeof(handoff));

    // Instruction cache, both sides. FENCE.I on the tt-metal side.
    tt_x280_invalidate_l1_icache();
    printf("  tt_x280_invalidate_l1_icache()      -> FENCE.I\n");

    printf("\n");
    printf("  handoff buffer now reads: \"%s\"\n", (const char*)handoff);
    printf("\n");
}

int main(void) {
    quasar_tty_init();

    printf("\n");
    printf("=====================================================================\n");
    printf(" hello_x280 -- freedom-e-sdk + tt-metal on a Quasar DM core\n");
    printf("=====================================================================\n");
    printf("\n");

    report_toolchain();
    report_freedom_metal();
    report_memory_map();
    cache_handoff();

    printf("---------------------------------------------------------------------\n");
    printf(" Hello, World!\n");
    printf("\n");
    printf(" This binary links SiFive freedom-metal (from freedom-e-sdk) against\n");
    printf(" tt-metal's X280 cache code, for -mcpu=tt-qsr64-rocc, at tt-metal's\n");
    printf(" own DM kernel link address. %u bytes of console output are sitting\n", (unsigned)quasar_tty_length());
    printf(" in Tensix L1 at 0x%08lx.\n", (unsigned long)quasar_tty_address());
    printf("---------------------------------------------------------------------\n");
    printf("\n");

    // Final flush so nothing is stranded in L1 D$ / L2 if the core is halted
    // right after main returns.
    tt_x280_flush_l2_cache_full();
    return 0;
}
