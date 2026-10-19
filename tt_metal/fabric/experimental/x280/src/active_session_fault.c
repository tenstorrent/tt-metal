// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Test-only active-session fault image.
 */
#include "x280_hart_protocol.h"

#include "rt/x280_idle.h"

#ifndef X280_ACTIVE_FAULT_HART
#error "X280_ACTIVE_FAULT_HART must select the single faulting hart"
#endif

int main(uint64_t hartid) {
    /* The harness already published RUNNING for every hart, so the selected
     * one only has to trap. The host then sees a session whose harts disagree:
     * three terminal, one committed only to a trap record -- entry.S's
     * _trap_normal CEASEs the faulting hart, so it reaches no harness step
     * after main. */
    if (hartid == X280_ACTIVE_FAULT_HART) {
        __asm__ volatile("ebreak");
        __builtin_unreachable();
    }

    return 0;
}
