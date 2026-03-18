// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/matmul_1d_helpers.hpp"

using std::uint32_t;

void kernel_main() {
    uint32_t num_output_tiles = get_arg_val<uint32_t>(0);  // number of output tiles to produce
    uint32_t Kt = get_arg_val<uint32_t>(1);                // number of tiles in K dimension for dot product

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    compute_kernel_hw_startup(cb_in0, cb_in1, cb_out);
    // Each output tile is an independent dot product over Kt tiles (Nt=1, batch=1).
    compute_kernel_lib::matmul_1d<cb_in0, cb_in1, cb_out>(num_output_tiles, 1, Kt);
}
