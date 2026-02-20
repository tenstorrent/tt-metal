// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // CB indices for sharded input tensors
    constexpr uint32_t cb_l1 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_l2 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_ms1 = get_compile_time_arg_val(2);
    constexpr uint32_t cb_ms2 = get_compile_time_arg_val(3);
    constexpr uint32_t num_l_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t num_ms_tiles = get_compile_time_arg_val(5);

    // ms1, ms2: num_ms_tiles each
    cb_reserve_back(cb_ms1, num_ms_tiles);
    cb_push_back(cb_ms1, num_ms_tiles);
    cb_reserve_back(cb_ms2, num_ms_tiles);
    cb_push_back(cb_ms2, num_ms_tiles);

    // l1 and l2: num_l_tiles each
    cb_reserve_back(cb_l1, num_l_tiles);
    cb_push_back(cb_l1, num_l_tiles);
    cb_reserve_back(cb_l2, num_l_tiles);
    cb_push_back(cb_l2, num_l_tiles);
}
