// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api.h"
#define DISPATCH_KERNEL 1
#include "tools/profiler/kernel_profiler.hpp"
#include "hostdevcommon/profiler_common.h"

namespace NAMESPACE {
void MAIN {
#if defined(COMPILE_FOR_TRISC) && COMPILE_FOR_TRISC == 0

    // Main loop: runs until dispatch_s BRISC signals terminate
    while (kernel_profiler::profiler_control_buffer[kernel_profiler::DISPATCH_TRISC_TERMINATE] == 0) {
        // TODO: Add stream monitoring logic here
    }

#endif
}
}  // namespace NAMESPACE
