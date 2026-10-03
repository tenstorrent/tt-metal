// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
//
// SPDX-License-Identifier: Apache-2.0

/* trap.c: machine-mode trap handler. Interrupts never trap (mstatus.MIE = 0, they only end wfi);
 * an exception either resumes at a guarded-access recovery point or records itself and parks the hart. */
#include "fw.h"
#include "platform.h"

#define REG_A1 11

void trap_handler(trap_frame_t* f) {
    hart_local_t* me = self();
    uint32_t h = me->hartid;
    uint64_t cause = f->mcause;

    if (cause >> 63) {
        /* Should not happen with MIE = 0; ignore but count. */
        return;
    }
    if (me->recover_pc) {
        /* Guarded access (mailbox peek/poke etc.): report mcause in a1 and resume. */
        f->x[REG_A1] = cause;
        f->x[0] = me->recover_pc;
        me->recover_pc = 0;
        return;
    }

    me->trap_depth++;
    if (h < L2CPU_NHARTS && g_hdr) {
        l2cpu_hart_state_t* hs = &g_hdr->hart_state[h];
        wr64(&hs->mcause, cause);
        wr64(&hs->mepc, f->x[0]);
        wr64(&hs->mtval, f->mtval);
        wr32(&hs->trap_count, rd32(&hs->trap_count) + 1);
        fence();
    }
    if (me->trap_depth == 1) {
        fw_log(
            "TRAP mcause=0x%lx mepc=0x%lx mtval=0x%lx sp=0x%lx ra=0x%lx", cause, f->x[0], f->mtval, f->x[2], f->x[1]);
    }
#if L2CPU_TEST
    if (h >= L2CPU_NHARTS) {
        plat_finish(3); /* the QEMU test driver itself trapped */
    }
#endif
    if (h < L2CPU_NHARTS) {
        fw_error_park(L2CPU_ERR_TRAP, cause, 0);
    }
    fw_park();
}
