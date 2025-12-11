// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    constexpr uint32_t output_cb = get_compile_time_arg_val(0);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(1);

    // Wait for all output tiles to be available in CB
    // Note: output_cb is backed by sharded tensor, data will be written directly to L1
    cb_wait_front(output_cb, num_tiles);
}
