// SPDX-FileCopyrightText: © 2023-2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "eth_l1_address_map.h"

// This is a bespoke setjmp/longjmp implementation. We do not use
// regular setjmp/longjmp as that uses a 304 byte buffer. We only need
// enough to save the callee-save registers (13). Making this function
// naked allows us to place the jmp buffer at SP, which means we do
// not need to record a separate offset between sp and the jmp buffer.
// The function relies on optimization to avoid unexpected register
// usage.

void erisc_exit() {
    // Restore sp from the save slot.
    uint32_t *slot = reinterpret_cast<uint32_t *>(eth_l1_mem::address_map::ERISC_MEM_MAILBOX_STACK_SAVE);
    __asm__ volatile("lw sp, %[save_slot]\n\t" : : [ save_slot ] "m"(*slot));

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
