// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // CB indices for sharded output tensors
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_stats = get_compile_time_arg_val(1);
    constexpr uint32_t out_num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t stats_num_tiles = get_compile_time_arg_val(3);
    cb_wait_front(cb_out, out_num_tiles);
    cb_wait_front(cb_stats, stats_num_tiles);
}
