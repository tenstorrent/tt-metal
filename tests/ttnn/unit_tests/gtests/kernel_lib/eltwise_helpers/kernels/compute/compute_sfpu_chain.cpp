// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// SFPU-only chain: Load<cb_in, D0> + SfpuOp<D0>
// Defines: SFPU_OP_EXP | SFPU_OP_RELU | SFPU_OP_SQRT
//          OUTPUT_POLICY_PER_TILE (default: Bulk)
// Runtime args: [0] n_tiles

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_helpers.hpp"

void kernel_main() {
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);
    if (n_tiles == 0) {
        return;
    }

    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;

    using namespace compute_kernel_lib;

#if defined(SFPU_OP_EXP)
    auto chain = sfpu_chain(Load<cb_in, Dst::D0>{}, Exp<>{});
#elif defined(SFPU_OP_RELU)
    auto chain = sfpu_chain(Load<cb_in, Dst::D0>{}, Relu<>{});
#elif defined(SFPU_OP_SQRT)
    auto chain = sfpu_chain(Load<cb_in, Dst::D0>{}, Sqrt<>{});
#else
#error "Define one of: SFPU_OP_EXP, SFPU_OP_RELU, SFPU_OP_SQRT"
#endif

#if defined(OUTPUT_POLICY_PER_TILE)
    eltwise_op<cb_out, Dst::D0, EltwiseOutputPolicy::PerTile>(chain, EltwiseTileShape::flat(n_tiles));
#else
    eltwise_op<cb_out>(chain, EltwiseTileShape::flat(n_tiles));
#endif
}
