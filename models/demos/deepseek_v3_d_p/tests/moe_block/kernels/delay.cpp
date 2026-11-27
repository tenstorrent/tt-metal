// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Skew injector: spins for arg0 wall-clock cycles (one core of the chip it is launched on).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t cycles = get_arg_val<uint32_t>(0);
    if (cycles > 0) {
        riscv_wait(cycles);
    }
}
