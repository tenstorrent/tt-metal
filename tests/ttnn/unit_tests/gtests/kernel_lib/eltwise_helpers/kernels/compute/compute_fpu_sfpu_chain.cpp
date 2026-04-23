// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// FPU+SFPU chain: FpuOp<cb_in0, cb_in1, D0> + SfpuOp<D0>
// Defines: FPU_OP_ADD | FPU_OP_MUL  and  SFPU_OP_RELU | SFPU_OP_SQRT
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

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;

    using namespace compute_kernel_lib;

#if defined(FPU_OP_ADD) && defined(SFPU_OP_RELU)
    auto chain = sfpu_chain(FpuAdd<cb_in0, cb_in1, Dst::D0>{}, Relu<>{});
#elif defined(FPU_OP_ADD) && defined(SFPU_OP_SQRT)
    auto chain = sfpu_chain(FpuAdd<cb_in0, cb_in1, Dst::D0>{}, Sqrt<>{});
#elif defined(FPU_OP_MUL) && defined(SFPU_OP_RELU)
    auto chain = sfpu_chain(FpuMul<cb_in0, cb_in1, Dst::D0>{}, Relu<>{});
#elif defined(FPU_OP_MUL) && defined(SFPU_OP_SQRT)
    auto chain = sfpu_chain(FpuMul<cb_in0, cb_in1, Dst::D0>{}, Sqrt<>{});
#else
#error "Define FPU_OP_ADD|FPU_OP_MUL and SFPU_OP_RELU|SFPU_OP_SQRT"
#endif

    eltwise_op<cb_out>(chain, EltwiseTileShape::flat(n_tiles));
}
