// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
//
// SPDX-License-Identifier: Apache-2.0

/* QEMU virt bare-metal entry for the sampling bit-exactness test. All harts enter here (M-mode). */
    .section .text.init
    .globl _start
_start:
    csrr  t0, mhartid
    li    t1, (3<<13)|(3<<9)      /* mstatus.FS = VS = dirty: enable FP and V */
    csrs  mstatus, t1
    csrw  fcsr, zero              /* frm = RNE, flags clear */
    la    sp, _stack_top          /* 64 KiB stack per hart */
    slli  t1, t0, 16
    sub   sp, sp, t1
    bnez  t0, 3f
    la    t2, _bss_start
    la    t3, _bss_end
1:  bgeu  t2, t3, 2f
    sd    zero, 0(t2)
    addi  t2, t2, 8
    j     1b
2:  call  main
    j     4f
3:  mv    a0, t0
    call  secondary_main
4:  wfi
    j     4b
