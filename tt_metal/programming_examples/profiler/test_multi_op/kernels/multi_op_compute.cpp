// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    for (int i = 0; i < 200; i++) {
// Max unroll size
#pragma GCC unroll 65534
        for (int j = 0; j < 200; j++) {
            asm("nop");
        }
    }
}
