/*
 * SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * sifive_u mask ROM jumps to DRAM 0x80000000; firmware lives in LIM at
 * 0x08001000. This stub is linked at DRAM base and jumps into LIM (analogue of
 * host loading LIM then releasing reset). Parks all harts except 1: sifive_u
 * hart 0 is an E51 (no FP); firmware needs a U54 for -mabi=lp64d. Blackhole
 * X280 harts are identical and need no such filter.
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
