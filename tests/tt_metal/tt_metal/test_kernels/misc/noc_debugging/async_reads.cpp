// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "api/dataflow/endpoints.h"

void kernel_main() {
    Noc noc;
    CoreLocalMem<uint32_t> local_buffer(L1_BUFFER_ADDR);
    UnicastEndpoint unicast_endpoint;

    constexpr uint32_t num_bytes = 64;

    for (uint32_t i = 0; i < NUM_ITERATIONS; ++i) {
        noc.async_read(
            unicast_endpoint,
            local_buffer,
            num_bytes,
            {
                .noc_x = OTHER_CORE_X,
                .noc_y = OTHER_CORE_Y,
                .addr = i,
            },
            {});
#if defined(USE_READ_BARRIER)
        noc.async_read_barrier();
#endif
    }

    noc.async_read_barrier();
}
