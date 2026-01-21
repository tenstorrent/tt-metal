// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "experimental/noc.h"
#include "experimental/endpoints.h"

void kernel_main() {
    experimental::Noc noc;
    experimental::UnicastEndpoint unicast_endpoint;

    constexpr uint32_t num_bytes = 64;
    constexpr uint32_t num_iterations = 5000;

    for (uint32_t i = 0; i < num_iterations; ++i) {
        noc.async_read(
            unicast_endpoint,
            unicast_endpoint,
            num_bytes,
            {
                .noc_x = OTHER_CORE_X,
                .noc_y = OTHER_CORE_Y,
                .addr = SRC_ADDR,
            },
            {
                .addr = DST_ADDR,
            });
#if defined(USE_READ_BARRIER)
        noc.async_read_barrier();
#endif
        noc.async_write(
            unicast_endpoint,
            unicast_endpoint,
            num_bytes,
            {
                .addr = SRC_ADDR,
            },
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
    noc.async_read_barrier();
}
