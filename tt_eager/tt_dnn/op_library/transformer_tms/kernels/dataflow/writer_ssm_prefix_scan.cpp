// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t num_tiles_per_core = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    cb_wait_front(cb_out, num_tiles_per_core);
}
