// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "debug/dprint.h"

#include <cstdint>

void kernel_main() {
    // Runtime args
    const uint32_t core_0_x = get_arg_val<uint32_t>(0);
    const uint32_t core_0_y = get_arg_val<uint32_t>(1);
    const uint32_t semaphore = get_semaphore(get_arg_val<uint32_t>(2));

    // Compile time args
    constexpr uint32_t src0_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t src1_cb_index = get_compile_time_arg_val(1);

    // Constants
    constexpr uint32_t one_tile = 1;

    // Communication config
    const uint32_t input_data_tile_size_bytes = get_tile_size(src1_cb_index);
    auto sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore);

    // Prepare for data transfer
    DPRINT << "5. READER 1: Preparing for data transfer" << ENDL();
    cb_reserve_back(src1_cb_index, one_tile);  // Reserve space for incoming tile

    // Indicate readiness for data transfer to core 0
    uint64_t sem_addr = get_noc_addr(core_0_x, core_0_y, semaphore);
    noc_semaphore_inc(sem_addr, 1);
    noc_async_atomic_barrier();

    // Wait until core 0 will finish data transfer
    noc_semaphore_wait(sem_ptr, 1);  // Wait to get tile from core 0
    noc_semaphore_set(sem_ptr, 0);   // Reset semaphore

    DPRINT << "6. READER 1: Data received and stored in src1" << ENDL();

    // Push data to src1 CB to writer1
    cb_push_back(src1_cb_index, one_tile);
}
