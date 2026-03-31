// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for matmul_tile helper isolated tests.
// Uses the matmul_tile helper to perform tile-by-tile matrix multiplication.
// Compile-time args:
//   [0] Mt     — output rows in tiles
//   [1] Nt     — output cols in tiles
//   [2] Kt     — inner dim in tiles
//   [3] batch  — batch count

#include "api/compute/matmul.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_tile_helpers.hpp"

void kernel_main() {
    uint32_t Mt = get_compile_time_arg_val(0);
    uint32_t Nt = get_compile_time_arg_val(1);
    uint32_t Kt = get_compile_time_arg_val(2);
    uint32_t batch = get_compile_time_arg_val(3);

    constexpr uint32_t cb_in0 = 0;
    constexpr uint32_t cb_in1 = 1;
    constexpr uint32_t cb_out = 16;

    mm_init(cb_in0, cb_in1, cb_out);

    compute_kernel_lib::matmul_tile<cb_in0, cb_in1, cb_out>(Mt, Nt, Kt, batch);
}
