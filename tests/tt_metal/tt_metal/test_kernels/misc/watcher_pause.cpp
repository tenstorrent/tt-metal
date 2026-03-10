// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/debug/pause.h"
/*
 * A test for the watcher pausing feature.
*/
#if !defined(COMPILE_FOR_BRISC) && !defined(COMPILE_FOR_NCRISC) && !defined(COMPILE_FOR_ERISC) && \
    !defined(COMPILE_FOR_IDLE_ERISC) && !defined(COMPILE_FOR_AERISC)
#include "api/compute/common.h"
#endif

void kernel_main() {
    uint32_t wait_cycles = get_arg_val<uint32_t>(0);
#if defined(COMPILE_FOR_IDLE_ERISC) || defined(COMPILE_FOR_AERISC)
    wait_cycles = 0x5f5e1000U;
#endif

    // Do a wait followed by a pause, triscs can't wait.
#ifndef COMPILE_FOR_TRISC
    riscv_wait(wait_cycles);
#endif
    PAUSE();
}
