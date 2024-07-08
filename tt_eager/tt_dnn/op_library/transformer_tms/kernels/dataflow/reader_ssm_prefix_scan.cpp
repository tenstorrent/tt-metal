// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

constexpr uint32_t NUM_TILES_IN_TILIZED_CHUNK = 32;

void kernel_main() {
    uint32_t num_tiles_per_core = get_arg_val<uint32_t>(0);
    uint32_t num_chunks_per_row = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_a_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_bx_in = get_compile_time_arg_val(1);
    constexpr uint32_t cb_h_in = get_compile_time_arg_val(2);

    cb_push_back(cb_a_in, num_tiles_per_core);
    cb_push_back(cb_bx_in, num_tiles_per_core);
    cb_push_back(cb_h_in, num_chunks_per_row);
}
