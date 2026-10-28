// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * x280_active_start.c - the host<->hart lifecycle every active image runs.
 *
 * entry.S calls this instead of main when assembled with
 * X280_ACTIVE_HARNESS; the image's own main becomes the middle step:
 *
 *     _start (entry_active.o) -> x280_active_start(hartid) -> main(hartid)
 *
 * The five obligations this leaves to the image are in README.md § 3.
 * The load-bearing one is that main returns once
 * x280_hart_stop_requested(hartid, run_value) is true, on every hart.
 *
 * No harness state, so this object lands nothing in .bss and x280.ld's
 * X280_REQUIRE_NO_BSS assertion covers it unchanged.
 */
#include "x280_hart_protocol.h"

#include "rt/x280_idle.h"

/* The image's entry symbol, reached without declaring a function named main
 * in this translation unit. An image with no main is then an undefined-symbol
 * link error rather than a hart that silently does nothing. */
extern int x280_image_main(uint64_t hartid) __asm__("main");

__attribute__((noreturn)) void x280_active_start(uint64_t hartid) {
    const uint64_t run_value = x280_hart_control(hartid)->run_value;

    x280_hart_set_stage(hartid, X280_HART_STAGE_ENTERED);
    x280_hart_publish_running(hartid, run_value);

    x280_hart_set_stage(hartid, X280_HART_STAGE_MAIN);
    (void)x280_image_main(hartid);

    x280_hart_set_stage(hartid, X280_HART_STAGE_POSTED);
    x280_hart_commit_terminal(hartid, run_value);

    /* Unconditional, and it cannot deadlock a hart whose main already returned
     * because of STOP: the command word is latched, so stop_requested is still
     * true when we reach it. Holding here is what keeps a self-completing
     * image observable in RUNNING_ACTIVE_FW long enough for the host to read
     * its result. */
    x280_hart_set_stage(hartid, X280_HART_STAGE_STOPPING);
    x280_hart_wait_for_stop(hartid, run_value);

    x280_hart_set_stage(hartid, X280_HART_STAGE_RETURNING);
    x280_hart_return_to_idle(hartid, run_value);
}
