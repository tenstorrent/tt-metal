/* SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
 * SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Test responder for the Tensix <-> L2CPU link (x280 harts 0-3 of one L2CPU tile). Position independent: the image
 * is loaded at channel_base + L2CPU_LINK_IMAGE_OFFSET and finds the channel from its own pc. Hart 0 serves, harts 1-3 park.
 */
    .section .text.entry, "ax"
    .globl _start
_start:
    csrw    0x7c1, zero          /* clear the feature-disable bits (x280 entry requirement) */
    csrw    mie, zero
    csrr    t0, mhartid
    bnez    t0, park
    lla     sp, __stack_top
    lla     a0, _start
    call    responder_main
park:
    wfi
    j       park
