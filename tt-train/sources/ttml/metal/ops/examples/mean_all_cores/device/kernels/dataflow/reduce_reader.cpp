// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <dataflow_api_addrgen.h>

#include <array>
#include <cstdint>
#include <cstring>

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "debug/dprint_pages.h"
#include "tt-train/sources/ttml/metal/ops/common/dataflow_utils.hpp"

void kernel_main() {
    // Runtime arguments
    uint32_t runtime_args_counter = 0U;
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);  // rows to process in this kernel
    const uint32_t worker0_core_x = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t worker0_core_y = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t worker1_core_x = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t worker1_core_y = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t semaphore = get_semaphore(get_arg_val<uint32_t>(runtime_args_counter++));

    // Circular buffer indices
    constexpr uint32_t cb_transfer_input_01 = tt::CBIndex::c_3;
    constexpr uint32_t cb_transfer_input_02 = tt::CBIndex::c_4;

    // Get tile size
    const uint32_t tile_bytes = get_tile_size(cb_transfer_input_01);
    constexpr uint32_t onetile = 1U;
    // Communication config
    const uint32_t input_data_tile_size_bytes = get_tile_size(cb_transfer_input_01);
    auto sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(semaphore);

    std::array<std::pair<uint32_t, uint32_t>, 2U> worker_cores = {
        std::pair<uint32_t, uint32_t>(worker0_core_x, worker0_core_y),
        std::pair<uint32_t, uint32_t>(worker1_core_x, worker1_core_y)};

    std::array<uint32_t, 2> transfer_buffer_selector = {cb_transfer_input_01, cb_transfer_input_02};

    DPRINT << "RDCN_READER: num_rows_to_process: " << num_rows_to_process << ENDL();
    for (uint32_t row = 0; row < num_rows_to_process; ++row) {
        // Prepare for data transfer
        DPRINT << "RDCN_READER: ROW: " << row << " Preparing for data transfer" << ENDL();
        uint32_t cb_transfer_input = transfer_buffer_selector[row];
        cb_reserve_back(cb_transfer_input, onetile);

        // Indicate readiness for data transfer to core 0
        uint64_t sem_addr = get_noc_addr(worker_cores[row].first, worker_cores[row].second, semaphore);
        noc_semaphore_inc(sem_addr, 1);
        noc_async_atomic_barrier();

        // Wait until core 0 will finish data transfer
        noc_semaphore_wait(sem_ptr, 1);  // Wait to get tile from worker core with idx: row
        noc_semaphore_set(sem_ptr, 0);   // Reset semaphore

        DPRINT << "RDCN_READER: ROW: " << row << " Data received and stored in cb_transfer_input" << ENDL();

        cb_push_back(cb_transfer_input, onetile);
        print_tile(cb_transfer_input, row);
    }
}
