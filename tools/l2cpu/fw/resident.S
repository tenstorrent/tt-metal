// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
//
// SPDX-License-Identifier: Apache-2.0

/*
 * resident.S: the resident page (l2cpu_boot.h). Position independent, no stack, assembled into a flat blob that the
 * host writes at region + L2CPU_OFF_RESIDENT once per chip-reset epoch, before the release. No image writes it.
 *
 * Entry points (fixed offsets, .org):
 *   +0x000 soft_park   : an image parks a hart (L1). Clobbers registers.
 *   +0x040 error_park  : an image's fatal-error park (kind ERROR).
 *   +0x100 + 0x40*h    : RNMI trap handler of hart h (BH): COUNT (resume) or PARK.
 *   +0x200 + 0x40*h    : RNMI exception trap handler of hart h (a fault while NMIE = 0): park.
 * Park loop: record state PARKED, wait for the hart's trigger bit to clear (BH), NMIE := 1 (BH), spin until
 * go_epoch differs from the epoch seen at park time, state LEAVING, fence.i, jump to `entry`.
 *
 * mn* CSRs (x280 numbering, ISA RNMIs.md): 0x350 mnscratch, 0x351 mnepc, 0x352 mncause, 0x353 mnstatus.
 * mnret = .word 0x70200073 (binutils 2.42 rejects the mnemonic). QEMU 8.2 has none of them: the QEMU blob
 * (no L2CPU_PLATFORM_BH) has the L1 paths only.
 */
#include "l2cpu_boot.h"

#define REC(h) (L2CPU_RES_REC + L2CPU_REC_SIZE * (h))
#define MNRET .word 0x70200073

    .section .text.resident, "ax"
    .option norvc
    .globl res_base
res_base:

/* ---- +0x000 soft park: t1 = hart, t0 = record ---- */
    .org L2CPU_RES_SOFT_PARK
soft_park:
    csrr    t1, mhartid
    lla     t0, res_base + L2CPU_RES_REC
    slli    t2, t1, 8                       /* * L2CPU_REC_SIZE (0x100) */
    add     t0, t0, t2
    li      t2, L2CPU_KIND_SOFT
    sw      t2, L2CPU_REC_KIND(t0)
    j       park_common

/* ---- +0x040 error park ---- */
    .org L2CPU_RES_ERROR_PARK
error_park:
    csrr    t1, mhartid
    lla     t0, res_base + L2CPU_RES_REC
    slli    t2, t1, 8
    add     t0, t0, t2
    li      t2, L2CPU_KIND_ERROR
    sw      t2, L2CPU_REC_KIND(t0)
    csrr    t2, mcause
    sd      t2, L2CPU_REC_MCAUSE(t0)
    csrr    t2, mepc
    sd      t2, L2CPU_REC_MEPC(t0)
    j       park_common

/* ---- +0x100: RNMI trap handlers (BH) ---- */
#if defined(L2CPU_PLATFORM_BH)
#define RNMI_ENTRY(h)                               \
    .org L2CPU_RES_RNMI_ENTRY + L2CPU_RES_STRIDE * (h); \
    csrw    0x350, t0;                              \
    lla     t0, res_base + REC(h);                  \
    sd      t1, L2CPU_REC_SAVE + 0(t0);             \
    sd      t2, L2CPU_REC_SAVE + 8(t0);             \
    sd      t3, L2CPU_REC_SAVE + 16(t0);            \
    li      t1, h;                                  \
    j       rnmi_common
#define RNMI_EXC(h)                                 \
    .org L2CPU_RES_RNMI_EXC + L2CPU_RES_STRIDE * (h); \
    csrw    0x350, t0;                              \
    lla     t0, res_base + REC(h);                  \
    li      t1, h;                                  \
    li      t2, L2CPU_KIND_RNMI_EXC;                \
    sw      t2, L2CPU_REC_KIND(t0);                 \
    j       rnmi_exc_common
#else
#define RNMI_ENTRY(h) .org L2CPU_RES_RNMI_ENTRY + L2CPU_RES_STRIDE * (h); j soft_park
#define RNMI_EXC(h) .org L2CPU_RES_RNMI_EXC + L2CPU_RES_STRIDE * (h); j soft_park
#endif
    RNMI_ENTRY(0)
    RNMI_ENTRY(1)
    RNMI_ENTRY(2)
    RNMI_ENTRY(3)
    RNMI_EXC(0)
    RNMI_EXC(1)
    RNMI_EXC(2)
    RNMI_EXC(3)

    .org 0x300
#if defined(L2CPU_PLATFORM_BH)
/* t0 = record, t1 = hart; t1..t3 saved, t0 in mnscratch */
rnmi_common:
    li      t2, L2CPU_STATE_IN_RNMI
    sw      t2, L2CPU_REC_STATE(t0)
    lw      t2, L2CPU_REC_RNMI_COUNT(t0)
    addi    t2, t2, 1
    sw      t2, L2CPU_REC_RNMI_COUNT(t0)
    csrr    t2, 0x352
    sd      t2, L2CPU_REC_MNCAUSE(t0)
    csrr    t2, 0x351
    sd      t2, L2CPU_REC_MNEPC(t0)
    csrr    t2, 0x353
    sd      t2, L2CPU_REC_MNSTATUS(t0)
    csrr    t2, mcause
    sd      t2, L2CPU_REC_MCAUSE(t0)
    csrr    t2, mepc
    sd      t2, L2CPU_REC_MEPC(t0)
    csrr    t2, mtval
    sd      t2, L2CPU_REC_MTVAL(t0)
    csrr    t2, mstatus
    sd      t2, L2CPU_REC_MSTATUS(t0)
    fence   rw, rw
    lla     t2, res_base + L2CPU_RES_RNMI_MODE
    lw      t2, 0(t2)
    li      t3, L2CPU_RNMI_PARK
    beq     t2, t3, 2f
    /* COUNT: wait until the host cleared our trigger bit, then resume */
    li      t2, L2CPU_RNMI_TRIGGER
3:  lw      t3, 0(t2)
    srl     t3, t3, t1
    andi    t3, t3, 1
    bnez    t3, 3b
1:  li      t2, L2CPU_STATE_RUNNING
    sw      t2, L2CPU_REC_STATE(t0)
    fence   rw, rw
    ld      t1, L2CPU_REC_SAVE + 0(t0)
    ld      t2, L2CPU_REC_SAVE + 8(t0)
    ld      t3, L2CPU_REC_SAVE + 16(t0)
    csrr    t0, 0x350
    MNRET
2:  li      t2, L2CPU_KIND_RNMI
    sw      t2, L2CPU_REC_KIND(t0)
    j       park_common

rnmi_exc_common:
    lw      t2, L2CPU_REC_RNMI_COUNT(t0)
    addi    t2, t2, 1
    sw      t2, L2CPU_REC_RNMI_COUNT(t0)
    csrr    t2, 0x352
    sd      t2, L2CPU_REC_MNCAUSE(t0)
    csrr    t2, 0x351
    sd      t2, L2CPU_REC_MNEPC(t0)
    csrr    t2, mcause
    sd      t2, L2CPU_REC_MCAUSE(t0)
    csrr    t2, mepc
    sd      t2, L2CPU_REC_MEPC(t0)
    csrr    t2, mtval
    sd      t2, L2CPU_REC_MTVAL(t0)
    j       park_common
#endif

/* t0 = record, t1 = hart */
park_common:
    csrw    mie, zero
    csrci   mstatus, 8                      /* MIE = 0 */
    li      t2, (1 << 17)
    csrc    mstatus, t2                     /* MPRV = 0 */
    csrr    t2, mstatus
    sd      t2, L2CPU_REC_MSTATUS(t0)
    lla     t3, res_base + L2CPU_RES_GO_EPOCH
    lw      t2, 0(t3)
    sw      t2, L2CPU_REC_PARKED_EPOCH(t0)
    lw      t2, L2CPU_REC_PARK_COUNT(t0)
    addi    t2, t2, 1
    sw      t2, L2CPU_REC_PARK_COUNT(t0)
    fence   rw, rw
    li      t2, L2CPU_STATE_PARKED
    sw      t2, L2CPU_REC_STATE(t0)
    fence   rw, rw
#if defined(L2CPU_PLATFORM_BH)
    li      t2, L2CPU_RNMI_TRIGGER          /* level trigger: wait until the host cleared our bit */
4:  lw      t4, 0(t2)
    srl     t4, t4, t1
    andi    t4, t4, 1
    bnez    t4, 4b
    csrsi   0x353, 8                        /* NMIE = 1: a new RNMI can take this hart again */
#endif
    lw      t4, L2CPU_REC_PARKED_EPOCH(t0)
5:  lw      t2, 0(t3)                       /* go_epoch */
    beq     t2, t4, 5b
    fence   rw, rw
    li      t2, L2CPU_STATE_LEAVING
    sw      t2, L2CPU_REC_STATE(t0)
    fence   rw, rw
    fence.i
    lla     t2, res_base + L2CPU_RES_ENTRY
    ld      t2, 0(t2)
    jr      t2

/* ---- control words with their initial values (host may patch entry / epoch before the release) ---- */
    .org L2CPU_RES_MAGIC
    .word   L2CPU_RES_MAGIC_LO
    .word   L2CPU_RES_MAGIC_HI
    .word   L2CPU_RES_VER
    .org L2CPU_RES_RNMI_MODE
    .word   L2CPU_RNMI_PARK
    .org L2CPU_RES_GO_EPOCH
    .word   0
    .org L2CPU_RES_ENTRY
    .dword  0
    .word   L2CPU_BOOT_COLD
    .word   1
    .org L2CPU_RES_SIZE - 4
    .word   0
