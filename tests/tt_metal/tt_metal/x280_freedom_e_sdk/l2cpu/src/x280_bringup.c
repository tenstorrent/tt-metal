// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// X280 bring-up gaps freedom-metal's _enter does not cover (weak hooks only):
//   __metal_before_start  — enable mstatus.FS and mstatus.VS (both Off at reset)
//   __metal_after_main    — CEASE instead of spinning
// Chicken-bit clear, stacks, .bss/.data are already done by freedom-metal.

#include <stdint.h>

// After SP is valid, before .data/.bss — CSR writes only, no globals.
void __metal_before_start(void) {
    // mstatus.FS = Dirty [14:13]=0b11. Required for -mabi=lp64d (FP spills trap if Off).
    __asm__ __volatile__("csrs mstatus, %0" ::"r"((uintptr_t)(3u << 13)));

    // mstatus.VS = Initial [10:9]=0b01. Required before any vector insn / vlenb read.
    // Same step as tt-llm-engine x280/boot/entry.S (SiFive coreip_21G3.04.00 §5.8).
    __asm__ __volatile__("csrs mstatus, %0" ::"r"((uintptr_t)(1u << 9)));
}

// SiFive CEASE; metal_shutdown() needs sifive,test0 which L2CPU lacks.
// Skipped under X280_QEMU (generic U54 has no CEASE).
__attribute__((noreturn)) void x280_cease(void) {
#ifndef X280_QEMU
    __asm__ __volatile__(".word 0x30500073");  // CEASE
#endif
    for (;;) {
        __asm__ __volatile__("wfi");
    }
}

void __metal_after_main(void) { x280_cease(); }
