// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/debug/pause.h"
/*
 * A test for the watcher pausing feature.
*/
#ifdef COMPILE_FOR_TRISC
#include "api/compute/common.h"
#else
#include "api/dataflow/dataflow_api.h"
#endif

#ifdef ARCH_QUASAR
#include "experimental/kernel_args.h"
#endif

void kernel_main() {
#ifdef ARCH_QUASAR
    uint32_t wait_cycles = get_arg(args::wait_cycles);
#else
    uint32_t wait_cycles = get_common_arg_val<uint32_t>(0);
#endif

    // Do a wait followed by a pause, triscs can't wait.
#ifndef COMPILE_FOR_TRISC
    riscv_wait(wait_cycles);
#endif
    PAUSE();
}
