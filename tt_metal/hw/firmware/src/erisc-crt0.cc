// SPDX-FileCopyrightText: © 2023-2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

void Application();

static void *stack;  // saved stack pointer

static void do_erisc_exit();

// Pointer to exit routine, (so it may be called from a kernel).  USED
// attribute is needed to keep this as an symbol that kernels may
// use. (Because we LTO the firmware, it would otherwise look
// removable.)
[[gnu::noreturn, gnu::used]] void (*erisc_exit)() = do_erisc_exit;

// This is a bespoke setjmp/longjmp implementation. We do not use
// regular setjmp/longjmp as that uses a 304 byte buffer. We only need
// enough to save the callee-save registers (13). Making this function
// naked allows us to place the jmp buffer at SP, which means we do
// not need to record a separate offset between sp and the jmp buffer.
// The function relies on optimization to avoid unexpected register
// usage.

extern "C" [[gnu::section(".start"), gnu::naked, gnu::optimize("Os")]] void _start(void) {
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

    Application();

    do_erisc_exit();
}

// This is not marked noreturn, because it does actually return --
// just not to where it came from!
static void do_erisc_exit() {
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
}
