// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Hello world for a Blackhole L2CPU X280 via freedom-e-sdk.
// Same toolchain/ISA/load addr as tt-llm-engine x280/; boot/link via freedom-metal.
// No UART: printf → LIM console (x280_lim_console.h); host polls sentinel over NOC.
// See ../README.md for build/run and Galaxy hazards.

#include <stdint.h>
#include <stdio.h>

#include <metal/cpu.h>

#include "x280_lim_console.h"

__attribute__((noreturn)) void x280_cease(void);

// freedom-metal linker symbols; _enter is the image's first instruction.
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

    // misa bit 21 = V. Skip vlenb if absent (illegal insn on U54/qemu).
    const uintptr_t misa = READ_CSR("misa");
    const int has_v = (misa >> 21) & 1;
    if (has_v) {
        // Also traps if mstatus.VS still Off — nonzero vlenb means bringup hook ran.
        const unsigned long vlenb = (unsigned long)READ_CSR("0xc22");
        printf("  misa.V (vector)                     = yes\n");
        printf("  vlenb (CSR 0xc22)                   = %lu bytes -> VLEN=%lu\n", vlenb, vlenb * 8);
    } else {
        printf("  misa.V (vector)                     = no (not an X280 -- emulated core)\n");
        printf("  vlenb (CSR 0xc22)                   = skipped, V absent\n");
    }
    printf(
        "  mstatus.VS                          = %lu (0=Off. On an X280 the\n",
        (unsigned long)((READ_CSR("mstatus") >> 9) & 0x3));
    printf("                                         __metal_before_start hook makes\n");
    printf("                                         this nonzero; on a core without\n");
    printf("                                         V the field is hardwired to 0.)\n");
#ifdef X280_QEMU
    printf("  build flavor                        = X280_QEMU (emulation harness)\n");
#else
    printf("  build flavor                        = hardware (CEASE, no UART mirror)\n");
#endif
    printf("\n");
}

static void report_freedom_metal(void) {
    printf("[2] freedom-metal (freedom-e-sdk)\n");

    const int hartid = metal_cpu_get_current_hartid();
    struct metal_cpu* cpu = metal_cpu_get(hartid);

    printf("  metal_cpu_get_current_hartid()      = %d\n", hartid);
    printf("  metal_cpu_get_num_harts()           = %d\n", metal_cpu_get_num_harts());
    if (cpu != NULL) {
        printf("  metal_cpu_get_timebase()            = %lu Hz\n", (unsigned long)metal_cpu_get_timebase(cpu));
        printf("  metal_cpu_get_timer()               = %lu ticks\n", (unsigned long)metal_cpu_get_timer(cpu));
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
        "  console block                       = 0x%08lx (magic 0x%016lx)\n",
        X280_CONSOLE_ADDR,
        (unsigned long)X280_CONSOLE_MAGIC);
    printf("  sentinel                            = 0x%08lx\n", (unsigned long)(uintptr_t)X280_SENTINEL_ADDR);
    printf("\n");
}

static void report_float(void) {
    // Smoke-test hardware F/D under -mabi=lp64d (Quasar DM is soft-float only).
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

    // Host loader already polls this sentinel.
    *X280_SENTINEL_ADDR = X280_SENTINEL_VALUE;

    x280_cease();
}
