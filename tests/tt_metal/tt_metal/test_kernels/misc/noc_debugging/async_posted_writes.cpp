// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "api/dataflow/endpoints.h"

void kernel_main() {
    // Repeatedly issue posted (fire-and-forget) writes from the same source address. Posted writes are drained by
    // a posted-writes flush (noc_async_posted_writes_flushed), not a regular write barrier. Without an in-loop
    // posted flush the source is reused before the prior posted write has departed, which the tool reports as a
    // missing write barrier/flush. An in-loop posted flush clears them.
    Noc noc;
    CoreLocalMem<uint32_t> local_buffer(L1_BUFFER_ADDR);
    UnicastEndpoint unicast_endpoint;

    constexpr uint32_t num_bytes = 64;

    for (uint32_t i = 0; i < NUM_ITERATIONS; ++i) {
        noc.async_write<NocOptions::POSTED>(
            local_buffer,
            unicast_endpoint,
            num_bytes,
            {},
            {
                .noc_x = OTHER_CORE_X,
                .noc_y = OTHER_CORE_Y,
                .addr = DST_ADDR,
            });
#if defined(USE_POSTED_FLUSH)
        noc.async_writes_flushed<NocOptions::POSTED>();
#endif
    }

    noc.async_writes_flushed<NocOptions::POSTED>();
}
