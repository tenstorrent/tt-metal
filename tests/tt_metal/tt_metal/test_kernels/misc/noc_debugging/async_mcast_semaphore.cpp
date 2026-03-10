// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t semaphore_id = 0;
    uint32_t semaphore_addr = get_semaphore(semaphore_id);

    const uint64_t multicast_noc_addr =
        get_noc_multicast_addr(MCAST_START_X, MCAST_START_Y, MCAST_END_X, MCAST_END_Y, 0);

    volatile tt_l1_ptr uint32_t* semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_addr);

    noc_async_write_multicast_loopback_src(
        L1_BUFFER_ADDR, multicast_noc_addr | L1_BUFFER_ADDR, WRITE_SIZE, NUM_DEST_CORES, false);

#if defined(USE_WRITE_MCAST_FLUSH)
    noc_async_writes_flushed();
#endif

    *semaphore_ptr = 1;

    noc_semaphore_set_multicast_loopback_src(
        semaphore_addr, multicast_noc_addr | semaphore_addr, NUM_DEST_CORES, false);

#if defined(USE_SEMAPHORE_MCAST_FLUSH)
    noc_async_writes_flushed();
#endif
}
