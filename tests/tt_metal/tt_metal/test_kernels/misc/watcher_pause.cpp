// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
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

void kernel_main() {
    uint32_t wait_cycles = get_common_arg_val<uint32_t>(0);

    // Do a wait followed by a pause, triscs can't wait.
#ifndef COMPILE_FOR_TRISC
    riscv_wait(wait_cycles);
#endif
    PAUSE();
}
