// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t in0_cb = get_compile_time_arg_val(0);
    constexpr uint32_t weight0_cb = get_compile_time_arg_val(1);
    constexpr uint32_t weight1_cb = get_compile_time_arg_val(2);
    constexpr uint32_t interm_cb = get_compile_time_arg_val(3);
    constexpr uint32_t interm_cb2 = get_compile_time_arg_val(4);
    constexpr uint32_t num_tiles_k = get_compile_time_arg_val(5);

    cb_reserve_back(in0_cb, num_tiles_k);
    cb_push_back(in0_cb, num_tiles_k);

    cb_reserve_back(weight0_cb, num_tiles_k);
    cb_push_back(weight0_cb, num_tiles_k);

    cb_reserve_back(weight1_cb, num_tiles_k);
    cb_push_back(weight1_cb, num_tiles_k);

    // Mcast single output tile from interm_cb to interm_cb2
    constexpr uint32_t num_output_tiles = 1;
    cb_wait_front(interm_cb, num_output_tiles);
    cb_reserve_back(interm_cb2, num_output_tiles);

    noc_async_write(get_read_ptr(interm_cb), get_noc_addr(get_write_ptr(interm_cb2)), get_tile_size(interm_cb));
    noc_async_write_barrier();

    cb_push_back(interm_cb2, num_output_tiles);
    cb_pop_front(interm_cb, num_output_tiles);
}
