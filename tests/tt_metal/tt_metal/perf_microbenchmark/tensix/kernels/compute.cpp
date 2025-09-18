// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace NAMESPACE {
void MAIN {
    for (int i = 0; i < LOOP_COUNT; i++) {
        DeviceZoneScopedN("TEST-FULL");
// Max unroll size
#pragma GCC unroll 65534
        for (int j = 0; j < LOOP_SIZE; j++) {
            TTI_NOP;
        }
    }
}
}  // namespace NAMESPACE
