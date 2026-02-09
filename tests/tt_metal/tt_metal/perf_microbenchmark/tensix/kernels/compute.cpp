// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    for (int i = 0; i < LOOP_COUNT; i++) {
        DeviceZoneScopedN("TEST-FULL");
// Max unroll size
#pragma GCC unroll 65534
        for (int j = 0; j < LOOP_SIZE; j++) {
            TTI_NOP;
        }
    }
}
