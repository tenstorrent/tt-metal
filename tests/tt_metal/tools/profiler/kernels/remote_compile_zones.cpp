// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

#ifndef PROFILE_KERNEL
#error "This regression requires TT_METAL_DEVICE_PROFILER=1"
#endif

void kernel_main() {
    DeviceZoneScopedN("REMOTE-COMPILE-OUTER");
    {
        DeviceZoneScopedN("REMOTE-COMPILE-INNER");
    }
}
