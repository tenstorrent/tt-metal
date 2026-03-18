// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/matmul_1d_helpers.hpp"

using std::uint32_t;

void kernel_main() {
    const uint32_t Mt = get_compile_time_arg_val(0);
    const uint32_t Kt = get_compile_time_arg_val(1);
    const uint32_t Nt = get_compile_time_arg_val(2);

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    // Three-argument form required: srcA (cb_in0) and srcB (cb_in1) are different CBs.
    compute_kernel_hw_startup(cb_in0, cb_in1, cb_out);

    compute_kernel_lib::matmul_1d<cb_in0, cb_in1, cb_out>(Mt, Nt, Kt);
}
