// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#if defined(COMPILE_FOR_TRISC)

#if COMPILE_FOR_TRISC == 0
#include "tt_metal/impl/dispatch/kernels/cq_realtime_profiler_dispatch_subordinate.hpp"
#endif
#if COMPILE_FOR_TRISC == 1 && defined(DISPATCH_TELEMETRY_DISABLED) && !DISPATCH_TELEMETRY_DISABLED
#include "tt_metal/impl/dispatch/kernels/cq_telemetry_dispatch_subordinate.hpp"
#endif

void kernel_main() {
#if COMPILE_FOR_TRISC == 0
    dispatch_subordinate_realtime_profiler();
#elif COMPILE_FOR_TRISC == 1 && defined(DISPATCH_TELEMETRY_DISABLED) && !DISPATCH_TELEMETRY_DISABLED
    dispatch_subordinate_telemetry();
#endif
}

#endif
