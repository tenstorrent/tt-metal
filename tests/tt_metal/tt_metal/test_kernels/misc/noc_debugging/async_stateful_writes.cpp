// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Program the write state once, then issue stateful writes that reuse it. Every write uses the same source
    // address, so without an in-loop barrier the tool flags a missing write barrier (source reused before flush).
    const uint64_t dst_noc_addr = get_noc_addr(OTHER_CORE_X, OTHER_CORE_Y, DST_ADDR);
    constexpr uint32_t num_bytes = 64;

    noc_async_write_one_packet_set_state(dst_noc_addr, num_bytes);

    for (uint32_t i = 0; i < NUM_ITERATIONS; ++i) {
        noc_async_write_one_packet_with_state(L1_BUFFER_ADDR, DST_ADDR);
#if defined(USE_WRITE_BARRIER)
        noc_async_write_barrier();
#endif
    }

    noc_async_write_barrier();
}
