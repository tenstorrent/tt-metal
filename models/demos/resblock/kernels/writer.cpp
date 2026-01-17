// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t output_cb = get_compile_time_arg_val(0);
    constexpr uint32_t bias_cb = get_compile_time_arg_val(1);
    constexpr uint32_t num_layers = get_compile_time_arg_val(2);
    constexpr uint32_t num_output_tiles = 1;

    cb_wait_front(output_cb, 1);
}
