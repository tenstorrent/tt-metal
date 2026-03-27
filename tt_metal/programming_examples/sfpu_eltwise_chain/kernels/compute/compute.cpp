// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp"

void kernel_main() {
    using namespace compute_kernel_lib;

    constexpr uint32_t src_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t ones_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t result_cb_index = get_compile_time_arg_val(2);

    init_sfpu(src_cb_index, result_cb_index);

    // softplus(x) = log(exp(x) + 1)
    auto chain = sfpu_chain(
        Load<src_cb_index, Dst::D0>{},         // D0 = x
        Load<ones_cb_index, Dst::D1>{},        // D1 = 1.0
        Exp<>{},                               // D0 = exp(x)
        SfpuAdd<Dst::D0, Dst::D1, Dst::D0>{},  // D0 = exp(x) + 1
        Log<>{});                              // D0 = log(exp(x) + 1)

    sfpu_pipeline(result_cb_index, /*pack_slot=*/0, /*num_tiles=*/1, chain);
}
