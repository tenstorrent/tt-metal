// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Bare-metal entry for the x280 echo firmware. All four harts of the tile are
// released together; hart 0 runs fw_main, harts 1-3 park in wfi.
//
// A trap handler records mcause/count in the LIM mailbox and skips the faulting
// instruction (all instructions are 4 bytes: built without the C extension), so
// an unsupported cbo.* opcode degrades gracefully instead of killing the hart.

// Mailbox in the UNCACHED GDDR alias (x280 physical) so stores are instantly
// NOC-visible. Firmware .text executes from the cached GDDR alias (see dram.ld).
#define MBOX 0x30100000
#define MBOX_TRAP_COUNT  (MBOX + 0x18)
#define MBOX_MCAUSE_LAST (MBOX + 0x20)
#define MBOX_T5_SAVE     (MBOX + 0x38)

#define MBOX_BOOT_MARKER (MBOX + 0x30)

    .section .text.start, "ax"
    .globl _start
_start:
    // Stamp a marker into LIM as the very first thing any hart does, before any
    // branch or stack setup. LIM is uncached, so this is instantly NOC-visible.
    // If a boot leaves heartbeat=0 but this marker set, the hart DID execute
    // (distinguishes "unclocked/never-ran" from "ran but wedged").
    li      t0, MBOX_BOOT_MARKER
    li      t1, 0xB0071E55
    sd      t1, 0(t0)

    csrr    t0, mhartid
    bnez    t0, park
    la      t1, trap_handler
    csrw    mtvec, t1
    la      sp, _stack_top
    call    fw_main
park:
    wfi
    j       park

    .align  4
trap_handler:
    csrw    mscratch, t6
    li      t6, MBOX_T5_SAVE
    sd      t5, 0(t6)

    li      t5, MBOX_TRAP_COUNT
    ld      t6, 0(t5)
    addi    t6, t6, 1
    sd      t6, 0(t5)

    li      t5, MBOX_MCAUSE_LAST
    csrr    t6, mcause
    sd      t6, 0(t5)

    csrr    t6, mepc
    addi    t6, t6, 4
    csrw    mepc, t6

    li      t6, MBOX_T5_SAVE
    ld      t5, 0(t6)
    csrr    t6, mscratch
    mret
