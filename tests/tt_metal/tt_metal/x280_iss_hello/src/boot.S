/*
 * SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * X280-cluster boot for the ISS: park harts 1..3, enable mstatus.FS and
 * mstatus.VS (both Off at reset; required before FP / RVV), then call main.
 */

    .section .text.start, "ax"
    .globl _start
    .globl tohost
    .globl fromhost

_start:
    csrr  t0, mhartid
    bnez  t0, park

    /* mstatus.FS = Dirty [14:13]=11, mstatus.VS = Initial [10:9]=01 */
    li    t1, (3 << 13) | (1 << 9)
    csrs  mstatus, t1

    la    sp, _stack_top
    la    gp, __global_pointer$
    call  main

    /* HTIF exit: (code << 1) | 1. main returns 0 on success. */
    slli  a0, a0, 1
    ori   a0, a0, 1
    la    t0, tohost
    sd    a0, 0(t0)
1:  wfi
    j     1b

park:
    wfi
    j     park

    .section .tohost, "aw", @progbits
    .align 6
tohost:
    .dword 0
    .align 6
fromhost:
    .dword 0
