// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t output_cb = get_compile_time_arg_val(0);
    constexpr uint32_t output_indices_cb = get_compile_time_arg_val(1);
    cb_wait_front(output_indices_cb, 1);
    cb_wait_front(output_cb, 1);
}
