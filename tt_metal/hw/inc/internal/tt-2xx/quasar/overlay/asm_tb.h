// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef _CHIPYARD_ASM_H
#define _CHIPYARD_ASM_H

#if __riscv_xlen == 64
#define LREG ld
#define SREG sd
#define REGBYTES 8
#else
#define LREG lw
#define SREG sw
#define REGBYTES 4
#endif

#ifdef __ASSEMBLY__

/* Signal barrier release */
.macro BARRIER_PASS flag li t0, -1 fence w, w sw t0, \flag,
    t1.endm

        /* Wait at barrier */
        .altmacro.macro BARRIER_WAIT flag LOCAL L1,
    L2 L1 : auipc t1, % pcrel_hi(\flag) L2 : lw t0, % pcrel_lo(L1)(t1) beqz t0, L2 fence r, r.endm

#endif /* __ASSEMBLY__ */

#endif /* _CHIPYARD_ASM_H */
