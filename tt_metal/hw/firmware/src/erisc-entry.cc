// SPDX-FileCopyrightText: © 2023-2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "eth_l1_address_map.h"

void Application();

[[gnu::noreturn]] void erisc_exit();

// This is a bespoke setjmp/longjmp implementation. We do not use
// regular setjmp/longjmp as that uses a 304 byte buffer. We only need
// enough to save the callee-save registers (13). Making this function
// naked allows us to place the jmp buffer at SP, which means we do
// not need to record a separate offset between sp and the jmp buffer.
// The function relies on optimization to avoid unexpected register
// usage.

extern "C" __attribute__((section("erisc_l1_code.0"), naked, optimize("Os"))) void ApplicationHandler(void) {
    // Save callee saves.
    __asm__ volatile(
        "addi sp, sp, -13 * 4\n\t"
        "sw ra, 0 * 4(sp)\n\t"
        "sw s0, 1 * 4(sp)\n\t"
        "sw s1, 2 * 4(sp)\n\t"
        "sw s2, 3 * 4(sp)\n\t"
        "sw s3, 4 * 4(sp)\n\t"
        "sw s4, 5 * 4(sp)\n\t"
        "sw s5, 6 * 4(sp)\n\t"
        "sw s6, 7 * 4(sp)\n\t"
        "sw s7, 8 * 4(sp)\n\t"
        "sw s8, 9 * 4(sp)\n\t"
        "sw s9, 10 * 4(sp)\n\t"
        "sw s10, 11 * 4(sp)\n\t"
        "sw s11, 12 * 4(sp)\n\t" ::
            : "memory");

    // Record sp in the save slot.
    uint32_t *slot = reinterpret_cast<uint32_t *>(eth_l1_mem::address_map::ERISC_MEM_MAILBOX_STACK_SAVE);
    __asm__ volatile("sw sp, %[save_slot]\n\t" : [ save_slot ] "=m"(*slot));

    Application();

    erisc_exit();
}
