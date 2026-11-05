// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/semaphore.h"
#include "experimental/kernel_args.h"

#include "api/debug/dprint.h"

#include <cstdint>

void kernel_main() {
    // Runtime args
    const uint32_t core_0_x = get_arg(args::core_0_x);
    const uint32_t core_0_y = get_arg(args::core_0_y);
    Semaphore semaphore(sem::core_sync);

    // Dataflow buffer bindings
    DataflowBuffer src1_dfb(dfb::src1);

    // Constants
    constexpr uint32_t one_tile = 1;

    // Communication config
    const uint32_t input_data_tile_size_bytes = src1_dfb.get_entry_size();
    Noc noc;

    // Prepare for data transfer
    DPRINT("5. READER 1: Preparing for data transfer\n");
    src1_dfb.reserve_back(one_tile);  // Reserve space for incoming tile

    // Indicate readiness for data transfer to core 0
    semaphore.up(noc, core_0_x, core_0_y, 1);
    noc_async_atomic_barrier();

    // Wait until core 0 will finish data transfer
    semaphore.wait(1);  // Wait to get tile from core 0
    semaphore.set(0);   // Reset semaphore

    DPRINT("6. READER 1: Data received and stored in src1\n");

    // Push data to src1 DFB to writer1
    src1_dfb.push_back(one_tile);
}
