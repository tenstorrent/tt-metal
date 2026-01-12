// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t in0_cb = get_compile_time_arg_val(0);
    constexpr uint32_t weight0_cb = get_compile_time_arg_val(1);
    constexpr uint32_t num_tiles_k = get_compile_time_arg_val(2);

    cb_reserve_back(in0_cb, num_tiles_k);
    cb_push_back(in0_cb, num_tiles_k);

    cb_reserve_back(weight0_cb, num_tiles_k);
    cb_push_back(weight0_cb, num_tiles_k);
}
