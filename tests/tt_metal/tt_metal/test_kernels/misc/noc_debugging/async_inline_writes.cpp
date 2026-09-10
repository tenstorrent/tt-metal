// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"

void kernel_main() {
    // Repeatedly issue inline dword writes (4-byte immediate value, no L1 source buffer) to another core.
    // Inline writes are released by a normal write barrier. Without one they remain outstanding at kernel end,
    // which the NOC debug tool reports as an unflushed-write issue. A write barrier drains them.
    Noc noc;
    UnicastEndpoint unicast_endpoint;

    for (uint32_t i = 0; i < NUM_ITERATIONS; ++i) {
        noc.inline_dw_write<NocOptions::INLINE_L1>(
            unicast_endpoint,
            0xDEADBEEF,
            {
                .noc_x = OTHER_CORE_X,
                .noc_y = OTHER_CORE_Y,
                .addr = DST_ADDR,
            });
    }

#if defined(USE_WRITE_BARRIER)
    noc.async_write_barrier();
#endif
}
