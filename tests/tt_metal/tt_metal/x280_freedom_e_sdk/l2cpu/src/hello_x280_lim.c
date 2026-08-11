// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// hello_x280_lim -- "Hello, World" for a SiFive X280 hart in a Blackhole L2CPU
// tile, built from SiFive's freedom-e-sdk.
//
// What this is
// ------------
// A Blackhole chip has four L2CPU tiles, each an X280 cluster of 4 harts
// (rv64gcv, VLEN=512), each tile with 2 MB of L3/LIM, a DDR front port and NOC
// paths. tt-llm-engine's x280/ directory already runs bare-metal firmware there
// with a hand-rolled boot stub, a hand-rolled linker script and a stock
// riscv64-unknown-elf toolchain.
//
// This program replaces the hand-rolled boot stub and linker script with
// freedom-metal's, keeps everything else, and prints. It is the same core, the
// same toolchain (tt-llm-engine's own vendored riscv64-unknown-elf-gcc), the
// same ISA (-march=rv64gc -mabi=lp64d), and the same load address
// (X280_ACTIVE_FW_LOAD_ADDR = 0x08001000).
//
// Where the output goes
// ---------------------
// An L2CPU tile has no UART. printf lands in a LIM block at a fixed address
// (see x280_lim_console.h) that the host reads back over the NOC, and the
// program finishes by writing the same sentinel tt-llm-engine's host loader
// already polls. See ../README.md for the run path -- and for why running it on
// a Galaxy chassis needs care.

#include <stdint.h>
#include <stdio.h>

// --- freedom-e-sdk / freedom-metal ------------------------------------------
#include <metal/cpu.h>

// --- this demo --------------------------------------------------------------
#include "x280_lim_console.h"

__attribute__((noreturn)) void x280_cease(void);

// Symbols freedom-metal's linker script defines; used to report where we landed.
// _enter is the image's first instruction, so it doubles as .text start.
extern char _enter[];
extern char metal_segment_bss_target_end[];
extern char metal_segment_stack_begin[];
extern char metal_segment_stack_end[];
extern char metal_segment_heap_target_end[];

#define READ_CSR(name)                                      \
    ({                                                      \
        uintptr_t __v;                                      \
        __asm__ __volatile__("csrr %0, " name : "=r"(__v)); \
        __v;                                                \
    })

static void report_core(void) {
    printf("[1] core identity (CSRs)\n");
    printf("  mhartid                             = %lu\n", (unsigned long)READ_CSR("mhartid"));
    printf("  mvendorid                           = 0x%lx\n", (unsigned long)READ_CSR("mvendorid"));
    printf("  marchid                             = 0x%lx\n", (unsigned long)READ_CSR("marchid"));
    printf("  mimpid                              = 0x%lx\n", (unsigned long)READ_CSR("mimpid"));
    printf("  misa                                = 0x%lx\n", (unsigned long)READ_CSR("misa"));

    // vlenb is CSR 0xc22. Reading it traps as an illegal instruction unless
    // mstatus.VS has been taken out of Off -- which freedom-metal's _enter does
    // not do, and src/x280_bringup.c's __metal_before_start hook does. So a
    // sane number here is a live check that the hook ran.
    const unsigned long vlenb = (unsigned long)READ_CSR("0xc22");
    printf("  vlenb (CSR 0xc22)                   = %lu bytes -> VLEN=%lu\n", vlenb, vlenb * 8);
    printf(
        "  mstatus.VS                          = %lu (0=Off; nonzero means the\n",
        (unsigned long)((READ_CSR("mstatus") >> 9) & 0x3));
    printf("                                         __metal_before_start hook ran)\n");
    printf("\n");
}

static void report_freedom_metal(void) {
    printf("[2] freedom-metal (freedom-e-sdk)\n");

    const int hartid = metal_cpu_get_current_hartid();
    struct metal_cpu* cpu = metal_cpu_get(hartid);

    printf("  metal_cpu_get_current_hartid()      = %d\n", hartid);
    printf("  metal_cpu_get_num_harts()           = %d\n", metal_cpu_get_num_harts());
    if (cpu != NULL) {
        printf("  metal_cpu_get_timebase()            = %llu Hz\n", metal_cpu_get_timebase(cpu));
        printf("  metal_cpu_get_timer()               = %llu ticks\n", metal_cpu_get_timer(cpu));
    } else {
        printf("  metal_cpu_get()                     = NULL\n");
    }
    printf("\n");
}

static void report_lim(void) {
    printf("[3] LIM layout (tt-llm-engine x280/include/x280.h)\n");
    printf("  X280_LIM_BASE                       = 0x%08lx\n", X280_LIM_BASE);
    printf("  X280_ACTIVE_FW_LOAD_ADDR            = 0x%08lx\n", X280_ACTIVE_FW_LOAD_ADDR);
    printf("  X280_ACTIVE_FW_REGION_END           = 0x%08lx\n", X280_ACTIVE_FW_REGION_END);
    printf("\n");
    printf("  _enter (image first instruction)    = 0x%08lx\n", (unsigned long)(uintptr_t)_enter);
    printf("  .bss end                            = 0x%08lx\n", (unsigned long)(uintptr_t)metal_segment_bss_target_end);
    printf(
        "  heap end                            = 0x%08lx\n", (unsigned long)(uintptr_t)metal_segment_heap_target_end);
    printf(
        "  stack                               = 0x%08lx .. 0x%08lx\n",
        (unsigned long)(uintptr_t)metal_segment_stack_begin,
        (unsigned long)(uintptr_t)metal_segment_stack_end);
    printf(
        "  console block                       = 0x%08lx (magic 0x%016llx)\n",
        X280_CONSOLE_ADDR,
        (unsigned long long)X280_CONSOLE_MAGIC);
    printf("  sentinel                            = 0x%08lx\n", (unsigned long)(uintptr_t)X280_SENTINEL_ADDR);
    printf("\n");
}

static void report_float(void) {
    // The X280 has F and D, and -mabi=lp64d passes doubles in FP registers. The
    // Quasar DM port next door had neither, so its newlib was soft-float only;
    // doing real FP here is a cheap way to show this target is the full rv64gc
    // core freedom-e-sdk's rv64 BSPs already assume.
    printf("[4] hardware floating point (rv64gc / lp64d)\n");
    volatile double a = 1.0;
    volatile double b = 3.0;
    const double q = a / b;
    printf("  1.0 / 3.0                           = %.10f\n", q);
    printf("  sizeof(double)                      = %u\n", (unsigned)sizeof(double));
    printf("\n");
}

int main(void) {
    x280_console_init();

    printf("\n");
    printf("=====================================================================\n");
    printf(" hello_x280_lim -- freedom-e-sdk on a Blackhole L2CPU SiFive X280\n");
    printf("=====================================================================\n");
    printf("\n");

    report_core();
    report_freedom_metal();
    report_lim();
    report_float();

    printf("---------------------------------------------------------------------\n");
    printf(" Hello, World!\n");
    printf("\n");
    printf(" Booted by freedom-metal's _enter, linked by freedom-metal's linker\n");
    printf(" script, libc from freedom-e-sdk's newlib, running on a stock SiFive\n");
    printf(" X280 inside Blackhole. %u bytes of console output are in LIM at\n", (unsigned)x280_console_length());
    printf(" 0x%08lx.\n", X280_CONSOLE_ADDR);
    printf("---------------------------------------------------------------------\n");
    printf("\n");

    // Tell the host we got here, using the sentinel its loader already polls.
    *X280_SENTINEL_ADDR = X280_SENTINEL_VALUE;

    // Retire the hart rather than spinning, so the tile is quiet afterwards.
    x280_cease();
}
