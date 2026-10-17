// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * The core_sync probe kernel's runtime-argument slots that every role reads
 * (tests/device/kernels/core_sync_probe.cpp). Slot 0 carries
 * CORE_SYNC_PROBE_LAYOUT_COOKIE (ubench_layout_cookie.h); a role's own
 * arguments start at CORE_SYNC_PROBE_ARG_ROLE_BASE, in the order the kernel's
 * header lists. The host packs and the kernel unpacks these, so both read them
 * from here.
 */
#ifndef X280_CORE_SYNC_PROBE_ARGS_H
#define X280_CORE_SYNC_PROBE_ARGS_H

#include "ubench_layout_cookie.h"

#define CORE_SYNC_PROBE_LAYOUT_COOKIE UBENCH_LAYOUT_COOKIE('C', 'S', 'P', 1)

enum core_sync_probe_arg_slot {
    CORE_SYNC_PROBE_ARG_LAYOUT_COOKIE = UBENCH_LAYOUT_COOKIE_SLOT,
    CORE_SYNC_PROBE_ARG_RUN_NONCE = 1,
    CORE_SYNC_PROBE_ARG_DIRECTOR_READY_BASE = 2,
    CORE_SYNC_PROBE_ARG_RESULT_L1_ADDR = 3,
    /* On the director, the counter every worker increments. */
    CORE_SYNC_PROBE_ARG_ARRIVAL_SEMAPHORE_ID = 4,
    CORE_SYNC_PROBE_ARG_ROLE_BASE = 5
};

/* Words of the probe's result record at CORE_SYNC_PROBE_ARG_RESULT_L1_ADDR.
 * The probe writes the run nonce last, so a record whose nonce word is not
 * this run's was never written by it: a refused probe (layout cookie
 * mismatch) writes none of the record. */
enum core_sync_probe_result_word {
    CORE_SYNC_PROBE_RESULT_OUTCOME = 0,
    CORE_SYNC_PROBE_RESULT_EXIT_LO = 1,
    CORE_SYNC_PROBE_RESULT_EXIT_HI = 2,
    CORE_SYNC_PROBE_RESULT_RUN_NONCE = 3,
    CORE_SYNC_PROBE_RESULT_WORDS = 4
};

#endif /* X280_CORE_SYNC_PROBE_ARGS_H */
