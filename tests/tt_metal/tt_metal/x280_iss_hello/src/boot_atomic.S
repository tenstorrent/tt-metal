/*
 * SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Hart 0 publishes the host config block, then enters main(0).
 * Harts 1..3 park. 1-hart phases do not need helpers.
 */

    .section .text.start, "ax"
    .globl _start
    .globl tohost
    .globl fromhost
    .globl iss_publish_config

_start:
    csrr  t0, mhartid
    bnez  t0, park

    li    t1, (3 << 13) | (1 << 9)
    csrs  mstatus, t1

    la    sp, _stack_top
    la    gp, __global_pointer$
    call  iss_publish_config
    li    a0, 0
    call  main

    slli  a0, a0, 1
    ori   a0, a0, 1
    la    t1, tohost
    sd    a0, 0(t1)
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
