// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t bias_cb = get_compile_time_arg_val(1);
    constexpr uint32_t indices_cb = get_compile_time_arg_val(2);

    // Signal that input buffer is ready (backed by L1 shards)
    cb_reserve_back(bias_cb, 1);
    cb_push_back(bias_cb, 1);
    cb_reserve_back(indices_cb, 1);
    cb_push_back(indices_cb, 1);
    cb_reserve_back(input_cb, 1);
    cb_push_back(input_cb, 1);
}
