// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DM producer kernel for BenchmarkCaseSixDebugImplicitSync.
//
// Single 1Sx1A DFB (logical id 0): DM4 → Neo0, implicit_sync enabled.
//
// Quasar launches this kernel on num_threads_per_cluster DM harts, but only DM4
// produces. finish() -> handle_final_credits() calls sync_threads(get_num_threads()),
// so we force num_sw_threads=1 on the producer hart to avoid waiting for DMs that
// returned early without calling finish().

#include "api/kernel_thread_globals.h"
#include "dfb_implicit_read_helper.h"

void kernel_main() {
    uint32_t dm_id;
    asm volatile("csrr %0, mhartid" : "=r"(dm_id));

    if (dm_id != 4) {
        return;
    }

    num_sw_threads = 1;

    Noc noc;
    DataflowBuffer dfb(0);
    dfb_issue_implicit_read(noc, dfb);
    dfb.finish();
}
