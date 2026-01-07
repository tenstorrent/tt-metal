// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"

void kernel_main() {
    uint32_t arg_idx = 0;
    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t scalars_cb = get_compile_time_arg_val(1);
    constexpr uint32_t gamma_cb = get_compile_time_arg_val(2);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(3);
    constexpr bool tiny_tile = get_compile_time_arg_val(4);

    uint32_t scalar = get_arg_val<uint32_t>(arg_idx++);

    // Generate reduction scalar (1/sqrt(num_elements))
    generate_reduce_scaler<tiny_tile>(scalars_cb, scalar);

    // Signal that input and gamma buffers are ready (backed by L1 shards)
    cb_reserve_back(gamma_cb, num_tiles);
    cb_push_back(gamma_cb, num_tiles);
    cb_reserve_back(input_cb, num_tiles);
    cb_push_back(input_cb, num_tiles);
}
