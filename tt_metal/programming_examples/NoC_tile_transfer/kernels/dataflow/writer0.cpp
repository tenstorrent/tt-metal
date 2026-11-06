// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/semaphore.h"
#include "experimental/kernel_args.h"

#include "api/debug/dprint.h"

void kernel_main() {
    // Runtime args
    const uint32_t core_1_x = get_arg(args::core_1_x);
    const uint32_t core_1_y = get_arg(args::core_1_y);
    Semaphore semaphore(sem::core_sync);

    // Dataflow buffer bindings
    DataflowBuffer src0_dfb(dfb::src0);
    DataflowBuffer src1_dfb(dfb::src1);

    // Constants
    constexpr uint32_t one_tile = 1;

    // Communication config
    const uint32_t input_data_tile_size_bytes = src0_dfb.get_entry_size();
    Noc noc;

    // Wait for data to be available in src0 DFB
    src0_dfb.wait_front(one_tile);
    DPRINT("3. WRITER 0: Data available in src0 CB\n");

    // Wait for core 1 to be ready
    semaphore.wait(1);

    // Prepare data for transfer
    const auto core1_data_ptr = src1_dfb.get_read_ptr();
    const uint64_t core1_noc_addr = get_noc_addr(core_1_x, core_1_y, core1_data_ptr);
    const auto data_l1_ptr = src0_dfb.get_write_ptr();

    // Send data to core 1
    noc_async_write(data_l1_ptr, core1_noc_addr, input_data_tile_size_bytes);
    noc_async_write_barrier();

    // Indicate finish writing to core 1
    semaphore.up(noc, core_1_x, core_1_y, 1);
    noc_async_atomic_barrier();
    semaphore.set(0);  // Reset semaphore

    src0_dfb.pop_front(one_tile);  // Remove data from local buffer
    DPRINT("4. WRITER 0: Data sent to core 1 from core 0\n");
}
