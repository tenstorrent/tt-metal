// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/debug/pause.h"
/*
 * A test for the watcher pausing feature.
*/
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t wait_cycles = get_common_arg_val<uint32_t>(0);

    // Do a wait followed by a pause, triscs can't wait.
    riscv_wait(wait_cycles);
    PAUSE();
}
