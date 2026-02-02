// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // CB indices for sharded input tensors
    constexpr uint32_t cb_q = get_compile_time_arg_val(0);
    constexpr uint32_t cb_k = get_compile_time_arg_val(1);
    constexpr uint32_t chunk_size = get_compile_time_arg_val(2);
    constexpr uint32_t num_tiles_k = get_compile_time_arg_val(3);

    cb_reserve_back(cb_q, chunk_size);
    cb_push_back(cb_q, chunk_size);
    cb_reserve_back(cb_k, num_tiles_k * chunk_size);
    cb_push_back(cb_k, num_tiles_k * chunk_size);
}
