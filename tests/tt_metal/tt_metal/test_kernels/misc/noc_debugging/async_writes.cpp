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

#if defined(USE_TRID_BARRIER)
    // The per-trid barrier counter can be in a non-zero state from prior activity, so reset it before issuing
    // trid-tagged writes; otherwise noc_async_write_barrier_with_trid may never observe the writes drain and hang.
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
#endif

    for (uint32_t i = 0; i < NUM_ITERATIONS; ++i) {
#if defined(USE_TRID)
        // Transaction-id write: an ordinary non-posted write tagged with a trid.
        noc.async_write<NocOptions::TXN_ID>(
            local_buffer,
            unicast_endpoint,
            num_bytes,
            {},
            {
                .noc_x = OTHER_CORE_X,
                .noc_y = OTHER_CORE_Y,
                .addr = DST_ADDR,
            },
            {.trid = 1});
#else
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
#endif
#if defined(USE_WRITE_BARRIER)
        noc.async_write_barrier();
#elif defined(USE_TRID_BARRIER)
        noc.async_write_barrier<NocOptions::TXN_ID>({.trid = 1});
#elif defined(USE_FULL_BARRIER)
        noc.async_full_barrier();
#endif
    }

#if defined(USE_FULL_BARRIER)
    noc.async_full_barrier();
#elif defined(USE_TRID_BARRIER)
    noc.async_write_barrier<NocOptions::TXN_ID>({.trid = 1});
#else
    noc.async_write_barrier();
#endif
}
