// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Program the read state once, then issue stateful reads that reuse it. Every read lands at the same local
    // address, so without an in-loop barrier the tool flags a missing read barrier (destination reused before the
    // read has been confirmed complete).
    const uint64_t src_noc_addr = get_noc_addr(OTHER_CORE_X, OTHER_CORE_Y, SRC_ADDR);
    constexpr uint32_t num_bytes = 64;

    noc_async_read_one_packet_set_state(src_noc_addr, num_bytes);

    for (uint32_t i = 0; i < NUM_ITERATIONS; ++i) {
        noc_async_read_one_packet_with_state(SRC_ADDR, L1_BUFFER_ADDR);
#if defined(USE_READ_BARRIER)
        noc_async_read_barrier();
#endif
    }

    noc_async_read_barrier();
}
