// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    Noc noc;
    Semaphore mcast_semaphore(sem::mcast_sem);

    const uint64_t multicast_noc_addr =
        get_noc_multicast_addr(MCAST_START_X, MCAST_START_Y, MCAST_END_X, MCAST_END_Y, 0);

    noc_async_write_multicast_loopback_src(
        L1_BUFFER_ADDR, multicast_noc_addr | L1_BUFFER_ADDR, WRITE_SIZE, NUM_DEST_CORES, false);

#if defined(USE_WRITE_MCAST_FLUSH)
    noc_async_writes_flushed();
#endif

    mcast_semaphore.set(1);

    mcast_semaphore.set_multicast<NocOptions::MCAST_INCL_SRC>(
        noc, MCAST_START_X, MCAST_START_Y, MCAST_END_X, MCAST_END_Y, NUM_DEST_CORES);

#if defined(USE_SEMAPHORE_MCAST_FLUSH)
    noc_async_writes_flushed();
#endif
}
