// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "experimental/noc.h"
#include "experimental/core_local_mem.h"
#include "experimental/endpoints.h"

void kernel_main() {
    experimental::Noc noc;
    experimental::CoreLocalMem<uint32_t> local_buffer(L1_BUFFER_ADDR);
    experimental::UnicastEndpoint unicast_endpoint;

    constexpr uint32_t num_bytes = 64;

    for (uint32_t i = 0; i < NUM_ITERATIONS; ++i) {
        noc.async_write(
            local_buffer,
            unicast_endpoint,
            num_bytes,
            {},
            {
                .noc_x = OTHER_CORE_X,
                .noc_y = OTHER_CORE_Y,
                .addr = DST_ADDR,
            });
#if defined(USE_WRITE_BARRIER)
        noc.async_write_barrier();
#endif
    }

    noc.async_write_barrier();
}
