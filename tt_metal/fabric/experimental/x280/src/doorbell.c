// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * doorbell.c - test image: hart 0 parks in WFI on the MSI catcher doorbell.
 *
 * Hart 0 clears every PLIC enable word, enables source 6 (the MSI catcher) for
 * its own M-mode context, arms, and executes WFI with mie.MEIE and mie.MTIE set
 * and mstatus.MIE clear. A NOC write to the catcher wakes it without a trap; it
 * claims the source, pops the FIFO, completes the claim and records the wake in
 * the block (x280_hw_probe.h). The CLINT timer wakes it every
 * X280_DOORBELL_POLL_TICKS so that it can see STOP (obligation 1); those wakes
 * are counted apart. On STOP it disables what it enabled and returns. Harts 1-3
 * return at once.
 *
 * Used by X280DoorbellHw (tests/device/test_x280_doorbell_hw.cpp).
 */
#include "x280_hart_protocol.h"
#include "x280_hw_probe.h"

#include "rt/x280_idle.h"
#include "rt/x280_plic.h"

/* 10 ms of the 50 MHz mtime: how long a STOP can wait for hart 0. */
#define X280_DOORBELL_POLL_TICKS (X280_MTIME_HZ / 100u)
#define X280_DOORBELL_MAX_POPS 64u

static inline volatile uint64_t* word_at(uint64_t block, uint32_t offset) {
    return (volatile uint64_t*)(uintptr_t)(block + offset);
}

static void drain_catcher(void) {
    for (uint32_t n = 0; n < X280_DOORBELL_MAX_POPS && x280_msi_catcher_nonempty(); n++) {
        (void)x280_msi_catcher_pop();
    }
}

int main(uint64_t hartid) {
    if (hartid != 0) {
        return 0;
    }
    const uint64_t run_value = x280_hart_control(0)->run_value;
    const uint64_t block = x280_hart_args(0);
    if (block == 0 || x280_hart_args_bytes() < X280_PROBE_BLOCK_BYTES ||
        *word_at(block, X280_PROBE_RUN_VALUE) != run_value) {
        x280_hart_set_result(0, X280_PROBE_RESULT_BAD_ARGS);
        return 0;
    }

    const uint32_t ctx = X280_PLIC_M_CONTEXT(0);
    drain_catcher();
    x280_plic_disable_all();
    *word_at(block, X280_PROBE_DB_ENABLE) = x280_plic_enable_only(ctx, X280_MSI_PLIC_SOURCE);

    x280_csr_clear_mstatus(X280_MSTATUS_MIE);
    x280_csr_write_mie(X280_MIP_MEIP | X280_MIP_MTIP);
    x280_fence_ow();
    *word_at(block, X280_PROBE_DB_ARMED) = run_value;
    x280_fence_ow();

    uint64_t wakes = 0;
    uint64_t timer_wakes = 0;
    uint64_t other_wakes = 0;
    while (!x280_hart_stop_requested(0, run_value)) {
        x280_set_mtimecmp(0, x280_mtime() + X280_DOORBELL_POLL_TICKS);
        x280_wfi();
        const uint64_t mip = x280_csr_read_mip();
        if (mip & X280_MIP_MEIP) {
            const uint32_t source = x280_plic_claim(ctx);
            uint32_t value = 0;
            uint32_t pops = 0;
            if (source == X280_MSI_PLIC_SOURCE) {
                while (pops < X280_DOORBELL_MAX_POPS && x280_msi_catcher_nonempty()) {
                    value = x280_msi_catcher_pop();
                    pops++;
                }
            }
            if (source != 0) {
                x280_plic_complete(ctx, source);
            }
            *word_at(block, X280_PROBE_DB_SOURCE) = source;
            *word_at(block, X280_PROBE_DB_VALUE) = value;
            *word_at(block, X280_PROBE_DB_POPS) = pops;
            x280_fence_ow();
            if (source == X280_MSI_PLIC_SOURCE) {
                *word_at(block, X280_PROBE_DB_WAKES) = ++wakes;
            } else {
                *word_at(block, X280_PROBE_DB_OTHER_WAKES) = ++other_wakes;
            }
        } else if (mip & X280_MIP_MTIP) {
            *word_at(block, X280_PROBE_DB_TIMER_WAKES) = ++timer_wakes;
        } else {
            *word_at(block, X280_PROBE_DB_OTHER_WAKES) = ++other_wakes;
        }
        x280_fence_ow();
    }

    /* Leave nothing enabled: the next image inherits the PLIC and the timer. */
    x280_csr_write_mie(0);
    x280_set_mtimecmp(0, UINT64_MAX);
    x280_plic_disable_all();
    drain_catcher();
    x280_hart_set_result(0, wakes);
    return 0;
}
