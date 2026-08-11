// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The X280-specific bits freedom-metal's boot path does not cover.
//
// freedom-metal's src/entry.S already performs almost everything
// tt-llm-engine's x280/boot/entry.S does, because both are booting a SiFive
// core:
//
//   x280/boot/entry.S step        freedom-metal _enter
//   ----------------------        --------------------
//   1. init gp                    yes (.option norelax; la gp, __global_pointer$)
//   2. set mtvec                  yes (early_trap_vector)
//   3. clear Feature Disable      yes -- csrwi 0x7C1, 0, gated on
//      CSR 0x7c1                       __metal_chicken_bit (set by our BSP)
//   3b. mstatus.VS = Initial      NO   <-- this file
//   4. per-hart stack pointer     yes (sp -= hartid * __stack_size)
//   5. zero .bss                  yes (and copies .data, which x280's
//                                      entry.S calls a "Level 3 / future" item)
//   6. call main(hartid)          yes, via _start -> __libc_init_array -> main
//   7. spin if main returns       yes (__metal_after_main) <-- this file
//                                      improves it to CEASE
//
// So the port needs two hooks, both weak symbols freedom-metal's entry.S
// already calls if they exist. No patch to freedom-metal.

#include <stdint.h>

// Called by freedom-metal's _enter after the stack pointer is valid but BEFORE
// .data is copied and .bss is zeroed. Touching globals here is not safe; CSR
// writes are.
void __metal_before_start(void) {
    // mstatus.VS = Initial (bits [10:9] = 0b01). At reset VS = Off, and any
    // vector instruction -- including a read of vlenb -- traps as an illegal
    // instruction. X280 is rv64gcv with VLEN=512.
    //
    // Source: SiFive coreip_21G3.04.00 manual section 5.8, and the identical
    // step in tt-llm-engine's x280/boot/entry.S.
    __asm__ __volatile__("csrs mstatus, %0" ::"r"((uintptr_t)(1u << 9)));
}

// CEASE: SiFive custom instruction that retires the hart. tt-llm-engine's
// entry.S uses the same raw encoding in its trap handler; freedom-metal has no
// notion of it (its metal_shutdown() drives a sifive,test0 block, which an
// L2CPU tile does not have), so spell it out.
__attribute__((noreturn)) void x280_cease(void) {
    __asm__ __volatile__(".word 0x30500073");  // CEASE
    for (;;) {
        __asm__ __volatile__("wfi");
    }
}

// Called by freedom-metal's _enter if main() ever returns. Upstream spins here;
// halting the hart is both tidier and what the host-side loader expects to see.
void __metal_after_main(void) { x280_cease(); }
