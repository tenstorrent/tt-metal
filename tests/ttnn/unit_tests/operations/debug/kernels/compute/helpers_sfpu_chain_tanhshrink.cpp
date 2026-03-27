// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Integration test: sfpu_chain tanhshrink = x - tanh(x)
// Load x into D0 and D1, tanh(D1), sub(D0, D1, D0)

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp"
using namespace compute_kernel_lib;

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    init_sfpu(cb_in, cb_out);

    auto chain = sfpu_chain(
        Load<cb_in, Dst::D0>{}, Load<cb_in, Dst::D1>{}, Tanh<Dst::D1>{}, SfpuSub<Dst::D0, Dst::D1, Dst::D0>{});

    for (uint32_t block = 0; block < per_core_block_cnt; block++) {
        sfpu_pipeline(cb_out, 0, per_core_block_dim, chain);
    }
}
