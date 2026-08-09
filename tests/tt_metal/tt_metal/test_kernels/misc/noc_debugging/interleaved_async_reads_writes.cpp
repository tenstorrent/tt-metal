// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"

void kernel_main() {
    Noc noc;
    UnicastEndpoint unicast_endpoint;

    constexpr uint32_t num_bytes = 64;
    // 10 matches async_reads.cpp / async_writes.cpp: both issues this test looks for (write-after-write
    // from one source address, read-after-read to one destination) are already tripped by the second
    // iteration, so more buys no detection. Note that going past 4096 would additionally cross the
    // NOC counter wrap the host tracker compares with wrap_ge() -- that is deliberately not bought
    // here, because it costs ~32s in a single dispatch and trips CI's hang detector. Cover the
    // wrapping arithmetic with a host-side unit test instead.
    constexpr uint32_t num_iterations = 10;

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
