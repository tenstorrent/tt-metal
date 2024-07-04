// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

constexpr uint32_t NUM_TILES_IN_TILIZED_CHUNK = 32;

void fill_zeros(uint32_t cb_id) {
    constexpr uint32_t num_zeros_reads = 2048 / MEM_ZEROS_SIZE;
    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
    uint32_t write_addr = get_write_ptr(cb_id);
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_addr);

    // Fill tile with zeros
    for (uint32_t i = 0; i < num_zeros_reads; ++i) {
        noc_async_read(zeros_noc_addr, write_addr, MEM_ZEROS_SIZE);
        write_addr += MEM_ZEROS_SIZE;
    }
    noc_async_read_barrier();
}

void kernel_main() {
    uint32_t num_tiles_per_core = get_arg_val<uint32_t>(0);
    uint32_t total_tiles_per_row = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_a_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_bx_in = get_compile_time_arg_val(1);
    constexpr uint32_t cb_h_in = get_compile_time_arg_val(2);
    constexpr uint32_t cb_zeros = get_compile_time_arg_val(3);

    const uint32_t num_chunks_per_row =
        (total_tiles_per_row + NUM_TILES_IN_TILIZED_CHUNK - 1) / NUM_TILES_IN_TILIZED_CHUNK;  // ceil(x/y)

    fill_zeros(cb_zeros);
    cb_push_back(cb_zeros, 1);

    cb_push_back(cb_a_in, num_tiles_per_core);
    cb_push_back(cb_bx_in, num_tiles_per_core);
    cb_push_back(cb_h_in, num_chunks_per_row);
}
