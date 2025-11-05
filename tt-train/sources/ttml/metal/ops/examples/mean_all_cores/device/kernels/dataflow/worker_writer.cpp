// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "tt-train/sources/ttml/metal/ops/common/dataflow_utils.hpp"

void kernel_main() {
    // Runtime arguments
    uint32_t runtime_args_counter = 0;
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t rdcn_core_x = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t rdcn_core_y = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t semaphore = get_semaphore(get_arg_val<uint32_t>(runtime_args_counter++));

    // Circular buffer indices
    constexpr uint32_t cb_interm_output = tt::CBIndex::c_2;
    constexpr uint32_t cb_transfer_output = tt::CBIndex::c_3;

    constexpr uint32_t onetile = 1U;

    // Communication config
    const uint32_t input_data_tile_size_bytes = get_tile_size(cb_interm_output);
    uint64_t sem_addr = get_noc_addr(rdcn_core_x, rdcn_core_y, semaphore);
    auto sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(semaphore);

    for (uint32_t row = 0; row < num_rows_to_process; ++row) {
        cb_wait_front(cb_interm_output, onetile);
        uint32_t l1_read_addr = get_read_ptr(cb_interm_output);

        // Wait for reduction core to be ready
        noc_semaphore_wait(sem_ptr, 1);

        // Send tile to reduction core
        const auto rdcn_core_ptr = get_write_ptr(cb_transfer_output);
        const uint64_t rdcn_core_noc_addr = get_noc_addr(rdcn_core_x, rdcn_core_y, rdcn_core_ptr);
        const auto read_data_ptr = get_read_ptr(cb_interm_output);

        noc_async_write(read_data_ptr, rdcn_core_noc_addr, input_data_tile_size_bytes);
        noc_async_write_barrier();

        // Indicate finish writing to core 1
        noc_semaphore_inc(sem_addr, 1);
        noc_async_atomic_barrier();
        noc_semaphore_set(sem_ptr, 0);  // Reset semaphore

        cb_pop_front(cb_interm_output, onetile);
    }
}
