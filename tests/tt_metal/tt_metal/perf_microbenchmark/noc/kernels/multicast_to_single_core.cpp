// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "tt_metal/tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    uint32_t src = 150*1024;
    uint64_t destination = get_noc_multicast_addr(WORKER_NOC_X, WORKER_NOC_Y, WORKER_NOC_X, WORKER_NOC_Y, src);

    {
        DeviceZoneScopedN("NOC-LATENCY");
        noc_async_write_multicast(src, destination, 1024, 1);
        noc_async_write_barrier();
    }
}
