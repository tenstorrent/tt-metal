// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // CB indices for sharded output tensors
    constexpr uint32_t cb_l_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_ms_out = get_compile_time_arg_val(1);
    constexpr uint32_t block_size = get_compile_time_arg_val(2);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(3);
    constexpr bool final_reduction = get_compile_time_arg_val(4);

    // Wait for all output tiles to be available in CBs
    // Note: output CBs are backed by sharded tensors, data will be written directly to L1
    if constexpr (!final_reduction) {
        cb_wait_front(cb_ms_out, 1);
    }
    cb_wait_front(cb_l_out, num_blocks * block_size);
}
