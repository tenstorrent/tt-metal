// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "debug/dprint.h"

void kernel_main() {
    // Runtime args
    const uint32_t core_1_x = get_arg_val<uint32_t>(0);
    const uint32_t core_1_y = get_arg_val<uint32_t>(1);
    const uint32_t semaphore = get_semaphore(get_arg_val<uint32_t>(2));

    // Compile time args
    constexpr uint32_t src0_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t src1_cb_index = get_compile_time_arg_val(1);

    // Constants
    constexpr uint32_t one_tile = 1;

    // Communication config
    const uint32_t input_data_tile_size_bytes = get_tile_size(src0_cb_index);
    uint64_t sem_addr = get_noc_addr(core_1_x, core_1_y, semaphore);
    auto sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore);

    // Wait for data to be available in src0 CB
    cb_wait_front(src0_cb_index, one_tile);
    DPRINT << "3. WRITER 0: Data available in src0 CB" << ENDL();

    // Wait for core 1 to be ready
    noc_semaphore_wait(sem_ptr, 1);

    // Prepare data for transfer
    const auto core1_data_ptr = get_read_ptr(src1_cb_index);
    const uint64_t core1_noc_addr = get_noc_addr(core_1_x, core_1_y, core1_data_ptr);
    const auto data_l1_ptr = get_write_ptr(src0_cb_index);

    // Send data to core 1
    noc_async_write(data_l1_ptr, core1_noc_addr, input_data_tile_size_bytes);
    noc_async_write_barrier();

    // Indicate finish writing to core 1
    noc_semaphore_inc(sem_addr, 1);
    noc_async_atomic_barrier();
    noc_semaphore_set(sem_ptr, 0);  // Reset semaphore

    cb_pop_front(src0_cb_index, one_tile);  // Remove data from local buffer
    DPRINT << "4. WRITER 0: Data sent to core 1 from core 0" << ENDL();
}
