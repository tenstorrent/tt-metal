// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
//
// SPDX-License-Identifier: Apache-2.0

/* fw_asm.h: runtime constants shared by start.S and C (the region layout itself comes from l2cpu_boot.h). */
#ifndef FW_ASM_H
#define FW_ASM_H

#define ASM_FW_MAX_HARTS 5              /* 4 harts + the QEMU test driver hart */
#define ASM_HL_SIZE 64                  /* sizeof(hart_local_t) */
#define ASM_RELEASE_MAGIC 0x52454C45    /* "RELE": release epoch without a resident page */
#define ASM_QEMU_REGION_BASE 0x80000000 /* = L2CPU_QEMU_REGION_BASE (static-asserted in main.c) */
#define ASM_FRAME_SIZE 288              /* 32 regs + mcause + mtval + mstatus = 35 * 8 = 280, rounded to 16 */

#endif
