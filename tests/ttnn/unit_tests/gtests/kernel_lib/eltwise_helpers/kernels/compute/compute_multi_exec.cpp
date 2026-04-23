// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Multi-exec: calls eltwise_op num_blocks times, each for tiles_per_block tiles.
// Tests that init is correctly re-run each call and results are correct across blocks.
// Defines: CHAIN_SFPU_ONLY | CHAIN_FPU_ONLY | CHAIN_FPU_SFPU
// Runtime args: [0] num_blocks, [1] tiles_per_block

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_helpers.hpp"

void kernel_main() {
    const uint32_t num_blocks = get_arg_val<uint32_t>(0);
    const uint32_t tiles_per_block = get_arg_val<uint32_t>(1);
    if (num_blocks == 0 || tiles_per_block == 0) {
        return;
    }

    using namespace compute_kernel_lib;

#if defined(CHAIN_SFPU_ONLY)
    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;
    auto chain = sfpu_chain(Load<cb_in, Dst::D0>{}, Relu<>{});
    for (uint32_t b = 0; b < num_blocks; ++b) {
        eltwise_op<cb_out>(chain, EltwiseTileShape::flat(tiles_per_block));
    }

#elif defined(CHAIN_FPU_ONLY)
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;
    auto chain = sfpu_chain(FpuAdd<cb_in0, cb_in1, Dst::D0>{});
    for (uint32_t b = 0; b < num_blocks; ++b) {
        eltwise_op<cb_out>(chain, EltwiseTileShape::flat(tiles_per_block));
    }

#elif defined(CHAIN_FPU_SFPU)
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;
    auto chain = sfpu_chain(FpuAdd<cb_in0, cb_in1, Dst::D0>{}, Relu<>{});
    for (uint32_t b = 0; b < num_blocks; ++b) {
        eltwise_op<cb_out>(chain, EltwiseTileShape::flat(tiles_per_block));
    }

#else
#error "Define one of: CHAIN_SFPU_ONLY, CHAIN_FPU_ONLY, CHAIN_FPU_SFPU"
#endif
}
