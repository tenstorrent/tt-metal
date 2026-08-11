/*
 * SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Reset trampoline for running the LIM firmware under qemu-system-riscv64
 * -machine sifive_u.
 *
 * Why this exists
 * ---------------
 * qemu's sifive_u mask ROM at 0x1000 jumps to a fixed address -- the DRAM base
 * 0x80000000 -- regardless of the entry point of whatever you pass to -kernel.
 * The firmware we want to run lives in L2 LIM at 0x08001000, which is where
 * Blackhole's host loader puts it too (X280_ACTIVE_FW_LOAD_ADDR).
 *
 * So this three-instruction stub is linked at the DRAM base, gets control from
 * the mask ROM, and jumps into LIM. That is a fair analogue of the real thing:
 * on Blackhole the host writes the image into LIM over the NOC and then releases
 * the L2CPU from reset. Here qemu writes the image into LIM with
 * `-device loader` and the trampoline performs the "release".
 *
 * It also parks every hart except hart 1. qemu's sifive_u is heterogeneous the
 * way a real FU740 is: hart 0 is an E51 (rv64imac -- no FP), harts 1..n are U54s
 * (rv64gc). The firmware is built -mabi=lp64d and uses hardware doubles, so it
 * must run on a U54. A real Blackhole L2CPU tile is 4 identical X280 harts and
 * needs no such filtering.
 */

.section .text.start, "ax"
.global _start

_start:
    csrr  t0, mhartid
    li    t1, 1
    bne   t0, t1, park          /* only hart 1 (a U54, has F/D) continues */

    li    t0, 0x08001000        /* X280_ACTIVE_FW_LOAD_ADDR */
    jr    t0

park:
    wfi
    j     park
