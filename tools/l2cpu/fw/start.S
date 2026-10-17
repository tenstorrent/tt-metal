// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
//
// SPDX-License-Identifier: Apache-2.0

/*
 * start.S: entry for every hart (the loader points all reset vectors, and the resident park loop, at the image
 * entry), the trap vector, and the guarded-access helpers used by the mailbox.
 *
 * Position independent: every address is formed pc-relative (lla), so the same flat binary runs at any load
 * address. Region base: the loader's boot record (l2cpu_boot.h) when present, else image base - L2CPU_OFF_FW;
 * QEMU: fixed. Release epoch: the resident page's boot_epoch when present (restart), else ASM_RELEASE_MAGIC.
 *
 * Entry state (measured on the chip): M-mode, MMU off, mstatus.MIE = 0, FS = VS = 0, mnstatus.NMIE = 1;
 * a0/a1 undefined.
 */
#include "fw_asm.h"
#include "l2cpu_boot.h"

    .section .text.entry, "ax"
    .globl _start
_start:
    csrw    mie, zero
    csrci   mstatus, 8                  /* MIE = 0: interrupts only wake wfi, never trap */
    csrr    s0, mhartid
    li      t1, ASM_FW_MAX_HARTS
    bgeu    s0, t1, .Lpark_nostack

    .option push
    .option norelax
    lla     gp, __global_pointer$
    .option pop

    /* s1 = region base = _start - OFF_FW */
    lla     s1, _start
    li      t1, L2CPU_OFF_FW
    sub     s1, s1, t1
#if defined(L2CPU_PLATFORM_BH)
    /* restartable images may run from another slot: take the region from the loader's scratch record when its
     * magic is present (l2cpu_boot.h boot record; 32-bit accesses) */
    li      t1, L2CPU_BOOT_RECORD_PA
    lwu     t2, 0x08(t1)
    lwu     t3, 0x0C(t1)
    slli    t3, t3, 32
    or      t2, t2, t3
    li      t3, L2CPU_BOOT_RECORD_MAGIC
    bne     t2, t3, 1f
    lwu     t2, 0x10(t1)
    lwu     t3, 0x14(t1)
    slli    t3, t3, 32
    or      s1, t2, t3
1:
    csrsi   0x353, 8                    /* mnstatus.NMIE = 1 (already 1 out of reset and from the park loop) */
    li      t1, (1 << 17)
    csrc    mstatus, t1                 /* MPRV = 0 */
#else
    li      s1, ASM_QEMU_REGION_BASE /* QEMU: fixed region (images may run from either slot) */
#endif
    /* s2 = boot epoch from the resident page if present, else the legacy release magic */
    li      s2, ASM_RELEASE_MAGIC
    li      t1, L2CPU_OFF_RESIDENT + L2CPU_RES_MAGIC
    add     t1, s1, t1                  /* t1 = resident control block (offsets relative to RES_MAGIC) */
    lwu     t2, 0(t1)
    li      t3, L2CPU_RES_MAGIC_LO
    bne     t2, t3, 2f
    lwu     t2, 4(t1)
    li      t3, L2CPU_RES_MAGIC_HI
    bne     t2, t3, 2f
    lwu     s2, L2CPU_RES_BOOT_EPOCH - L2CPU_RES_MAGIC(t1)
2:

    /* sp = region + OFF_STACKS + (hart + 1) * STACK_SIZE */
    li      t1, L2CPU_OFF_STACKS
    add     t1, s1, t1
    addi    t2, s0, 1
    li      t3, L2CPU_STACK_SIZE
    mul     t2, t2, t3
    add     sp, t1, t2

    /* mscratch = region + OFF_TRAP_STACKS + (hart + 1) * TRAP_STACK_SIZE */
    li      t1, L2CPU_OFF_TRAP_STACKS
    add     t1, s1, t1
    addi    t2, s0, 1
    li      t3, L2CPU_TRAP_STACK_SIZE
    mul     t2, t2, t3
    add     t1, t1, t2
    csrw    mscratch, t1

    /* tp = &hl[hart] */
    lla     t1, hl
    li      t3, ASM_HL_SIZE
    mul     t2, s0, t3
    add     tp, t1, t2

    lla     t1, trap_entry
    csrw    mtvec, t1                   /* direct mode (trap_entry is 4-byte aligned) */

    /* FS = VS = Initial so FP and vector code may run (sampling library) */
    li      t1, (1 << 13) | (1 << 9)
    csrs    mstatus, t1
    csrw    fcsr, zero

#if defined(L2CPU_PLATFORM_BH) && L2CPU_BH_HWINIT
    csrw    0x7c1, zero                 /* clear feature-disable chicken bits (each hart, ISA README) */
#endif

    bnez    s0, .Lwait_release

    /* hart 0: zero .bss once, then release the others */
    lla     t1, _bss_start
    lla     t2, _bss_end
1:  bgeu    t1, t2, 2f
    sd      zero, 0(t1)
    addi    t1, t1, 8
    j       1b
2:  fence   rw, rw
    lla     t1, release_flag
    mv      t2, s2                      /* epoch-tagged: a stale value from an earlier boot never matches */
    sw      t2, 0(t1)
    fence   rw, rw
    j       .Lmain

.Lwait_release:
    /* bounded: hart 0 releases within microseconds; after L2CPU_BOOT_GATE_SPINS polls (~1 s on the chip) give up
     * and park in the resident error park (kind ERROR, mcause 0) so the host sees a hart that did not boot */
    lla     t1, release_flag
    mv      t2, s2
    li      t4, L2CPU_BOOT_GATE_SPINS
3:  lw      t3, 0(t1)
    beq     t3, t2, 7f
    addi    t4, t4, -1
    bnez    t4, 3b
    li      t1, L2CPU_OFF_RESIDENT + L2CPU_RES_MAGIC
    add     t1, s1, t1
    lwu     t2, 0(t1)
    li      t3, L2CPU_RES_MAGIC_LO
    bne     t2, t3, .Lpark_nostack
    csrw    mcause, zero
    csrw    mepc, zero
    li      t1, L2CPU_OFF_RESIDENT + L2CPU_RES_ERROR_PARK
    add     t1, s1, t1
    jr      t1
7:  fence   rw, rw

.Lmain:
    mv      a0, s0
    mv      a1, s1
    mv      a2, s2
    call    fw_main
    j       fw_park

.Lpark_nostack:
    csrw    mie, zero
4:  wfi
    j       4b

    .globl fw_park
fw_park:
    lla     t0, g_resident              /* resident page present: park there (restartable, RNMI-reachable) */
    ld      t0, 0(t0)
    beqz    t0, 6f
    addi    t0, t0, L2CPU_RES_ERROR_PARK
    jr      t0
6:
    csrw    mie, zero
5:  wfi
    j       5b

/* ---------------------------------------------------------------------------------------------- */
/* Trap vector: switch to the per-hart trap stack (mscratch), save everything, call trap_handler.  */
/* Frame: x[0] = mepc, x[1..31] = registers (x[2] = interrupted sp), then mcause, mtval, mstatus.  */
/* ---------------------------------------------------------------------------------------------- */
    .text
    .balign 64
    .globl trap_entry
trap_entry:
    csrrw   sp, mscratch, sp
    addi    sp, sp, -ASM_FRAME_SIZE
    sd      x1, 1*8(sp)
    sd      x3, 3*8(sp)
    sd      x4, 4*8(sp)
    sd      x5, 5*8(sp)
    sd      x6, 6*8(sp)
    sd      x7, 7*8(sp)
    sd      x8, 8*8(sp)
    sd      x9, 9*8(sp)
    sd      x10, 10*8(sp)
    sd      x11, 11*8(sp)
    sd      x12, 12*8(sp)
    sd      x13, 13*8(sp)
    sd      x14, 14*8(sp)
    sd      x15, 15*8(sp)
    sd      x16, 16*8(sp)
    sd      x17, 17*8(sp)
    sd      x18, 18*8(sp)
    sd      x19, 19*8(sp)
    sd      x20, 20*8(sp)
    sd      x21, 21*8(sp)
    sd      x22, 22*8(sp)
    sd      x23, 23*8(sp)
    sd      x24, 24*8(sp)
    sd      x25, 25*8(sp)
    sd      x26, 26*8(sp)
    sd      x27, 27*8(sp)
    sd      x28, 28*8(sp)
    sd      x29, 29*8(sp)
    sd      x30, 30*8(sp)
    sd      x31, 31*8(sp)
    csrr    t0, mscratch
    sd      t0, 2*8(sp)
    csrr    t0, mepc
    sd      t0, 0*8(sp)
    csrr    t0, mcause
    sd      t0, 32*8(sp)
    csrr    t0, mtval
    sd      t0, 33*8(sp)
    csrr    t0, mstatus
    sd      t0, 34*8(sp)
    mv      a0, sp
    call    trap_handler
    ld      t0, 0*8(sp)
    csrw    mepc, t0
    ld      x1, 1*8(sp)
    ld      x3, 3*8(sp)
    ld      x4, 4*8(sp)
    ld      x6, 6*8(sp)
    ld      x7, 7*8(sp)
    ld      x8, 8*8(sp)
    ld      x9, 9*8(sp)
    ld      x10, 10*8(sp)
    ld      x11, 11*8(sp)
    ld      x12, 12*8(sp)
    ld      x13, 13*8(sp)
    ld      x14, 14*8(sp)
    ld      x15, 15*8(sp)
    ld      x16, 16*8(sp)
    ld      x17, 17*8(sp)
    ld      x18, 18*8(sp)
    ld      x19, 19*8(sp)
    ld      x20, 20*8(sp)
    ld      x21, 21*8(sp)
    ld      x22, 22*8(sp)
    ld      x23, 23*8(sp)
    ld      x24, 24*8(sp)
    ld      x25, 25*8(sp)
    ld      x26, 26*8(sp)
    ld      x27, 27*8(sp)
    ld      x28, 28*8(sp)
    ld      x29, 29*8(sp)
    ld      x30, 30*8(sp)
    ld      x31, 31*8(sp)
    ld      x5, 5*8(sp)
    addi    sp, sp, ASM_FRAME_SIZE
    csrrw   sp, mscratch, sp
    mret

/* ---------------------------------------------------------------------------------------------- */
/* Guarded accesses: return {a0 = value, a1 = 0} or, if the access traps, {a0 = ?, a1 = mcause}.   */
/* The trap handler sees hl[hart].recover_pc (0(tp)) != 0, writes mcause into the frame's a1 and   */
/* resumes at recover_pc.                                                                          */
/* ---------------------------------------------------------------------------------------------- */
    .globl safe_ld64
safe_ld64:
    lla     t0, 1f
    sd      t0, 0(tp)
    li      a1, 0
    ld      a0, 0(a0)
1:  sd      zero, 0(tp)
    ret

    .globl safe_lw32
safe_lw32:
    lla     t0, 1f
    sd      t0, 0(tp)
    li      a1, 0
    lwu     a0, 0(a0)
1:  sd      zero, 0(tp)
    ret

    .globl safe_sd64
safe_sd64:
    lla     t0, 1f
    sd      t0, 0(tp)
    mv      t1, a1
    li      a1, 0
    sd      t1, 0(a0)
1:  sd      zero, 0(tp)
    ret

    .globl safe_sw32
safe_sw32:
    lla     t0, 1f
    sd      t0, 0(tp)
    mv      t1, a1
    li      a1, 0
    sw      t1, 0(a0)
1:  sd      zero, 0(tp)
    ret

    .data
    .balign 64
    .globl release_flag
release_flag:
    .word   0
    .balign 64
