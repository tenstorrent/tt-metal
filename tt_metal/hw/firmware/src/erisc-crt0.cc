// SPDX-FileCopyrightText: © 2023-2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

void Application();

static uint32_t *stack;
[[gnu::noreturn]] void (*erisc_exit)();

static [[gnu::noreturn]] void erisc_do_exit() {
    // Restore sp from the save slot.
    __asm__ volatile("lw sp, %[sp]\n\t" : : [sp] "m"(stack));

    // Restore callee saves.
    __asm__ volatile(
        "lw ra, 0 * 4(sp)\n\t"
        "lw s0, 1 * 4(sp)\n\t"
        "lw s1, 2 * 4(sp)\n\t"
        "lw s2, 3 * 4(sp)\n\t"
        "lw s3, 4 * 4(sp)\n\t"
        "lw s4, 5 * 4(sp)\n\t"
        "lw s5, 6 * 4(sp)\n\t"
        "lw s6, 7 * 4(sp)\n\t"
        "lw s7, 8 * 4(sp)\n\t"
        "lw s8, 9 * 4(sp)\n\t"
        "lw s9, 10 * 4(sp)\n\t"
        "lw s10, 11 * 4(sp)\n\t"
        "lw s11, 12 * 4(sp)\n\t"
        "addi sp, sp, 4 * 13\n\t");

    // And we're done
    __asm__ volatile("ret");
}

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
    __asm__ volatile("sw sp, %[sp]\n\t" : [sp] "=m"(stack));

    erisc_exit = erisc_do_exit;

    Application();

    erisc_do_exit();
}
