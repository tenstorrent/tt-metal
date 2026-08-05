// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 host-API version of watcher_pause.cpp.
// Compiled only for TENSIX cores (BRISC / NCRISC / TRISC / DM). Ethernet callers
// continue to use watcher_pause.cpp via the legacy host API.

#include <cstdint>
#include "api/debug/pause.h"
#include "experimental/kernel_args.h"

#ifdef COMPILE_FOR_TRISC
#include "api/compute/common.h"
#else
#include "api/dataflow/dataflow_api.h"
#endif

void kernel_main() {
    uint32_t wait_cycles = get_arg(args::wait_cycles);

    // Triscs can't do a riscv_wait; only data-movement cores wait.
#ifndef COMPILE_FOR_TRISC
    riscv_wait(wait_cycles);
#endif
    PAUSE();
}
