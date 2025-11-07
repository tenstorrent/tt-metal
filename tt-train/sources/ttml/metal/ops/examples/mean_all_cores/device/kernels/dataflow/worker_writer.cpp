// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "tt-train/sources/ttml/metal/ops/common/dataflow_utils.hpp"

void kernel_main() {
    // Runtime arguments
    uint32_t runtime_args_counter = 0;
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t rdcn_core_x = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t rdcn_core_y = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t semaphore = get_semaphore(get_arg_val<uint32_t>(runtime_args_counter++));

    // Circular buffer indices
    constexpr uint32_t cb_interm_output = tt::CBIndex::c_2;
    constexpr uint32_t cb_transfer_output_01 = tt::CBIndex::c_3;
    constexpr uint32_t cb_transfer_output_02 = tt::CBIndex::c_4;

    constexpr uint32_t onetile = 1U;

    // Communication config
    const uint32_t input_data_tile_size_bytes = get_tile_size(cb_interm_output);
    uint64_t sem_addr = get_noc_addr(rdcn_core_x, rdcn_core_y, semaphore);      // reduction core semaphore address
    auto sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(semaphore);  // ptr on this core semaphore

    DPRINT << "WORKER_WRITER: num_rows_to_process: " << num_rows_to_process << " start_row: " << start_row << ENDL();
    uint32_t cb_transfer_output = (start_row == 0) ? cb_transfer_output_01 : cb_transfer_output_02;
    for (uint32_t row = 0; row < num_rows_to_process; ++row) {
        cb_wait_front(cb_interm_output, onetile);
        // print_tile(cb_interm_output, 0);

        // Wait for reduction core to be ready
        noc_semaphore_wait(sem_ptr, 1);
        DPRINT << "WORKER_WRITER: receive ready check from reduction core for row: " << row << ENDL();

        // TODO(vmelnykov): I can pass transfer buffer index as runtime arg instead of choosing here
        // choose transfer buffer
        // Send tile to reduction core
        const auto rdcn_core_ptr = get_write_ptr(cb_transfer_output);
        const uint64_t rdcn_core_noc_addr = get_noc_addr(rdcn_core_x, rdcn_core_y, rdcn_core_ptr);
        const auto read_data_ptr = get_read_ptr(cb_interm_output);

        // DPRINT << "WORKER_WRITER: Preparing to send data to reduction core at coord: (" << rdcn_core_x << ", "
        //        << rdcn_core_y << ")\n";
        // DPRINT << "WORKER_WRITER: Sending data to reduction core to CB ptr: " << rdcn_core_ptr << ENDL();
        // DPRINT << "WORKER_WRITER: Sending data from addr: " << read_data_ptr
        //        << " to reduction core at addr: " << rdcn_core_noc_addr << ENDL();

        // write 3 tile
        noc_async_write(read_data_ptr, rdcn_core_noc_addr, input_data_tile_size_bytes);
        noc_async_write_barrier();

        // Indicate finish writing to reduction core
        noc_semaphore_inc(sem_addr, 1);
        noc_async_atomic_barrier();
        noc_semaphore_set(sem_ptr, 0);  // Reset semaphore
        DPRINT << "WORKER_WRITER: Data sent to reduction core for row: " << row << ENDL();
        cb_pop_front(cb_interm_output, onetile);
    }
}
