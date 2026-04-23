// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// FPU broadcast + SFPU: FpuOp<BroadcastDim> + SfpuOp<D0>
// Defines: BCAST_ROW | BCAST_COL | BCAST_SCALAR
//          SFPU_OP_RELU | SFPU_OP_EXP
// Runtime args: [0] rows, [1] cols

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_helpers.hpp"

void kernel_main() {
    const uint32_t rows = get_arg_val<uint32_t>(0);
    const uint32_t cols = get_arg_val<uint32_t>(1);
    if (rows == 0 || cols == 0) {
        return;
    }

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;

    using namespace compute_kernel_lib;

    const EltwiseTileShape shape = EltwiseTileShape::of(rows, cols);

#if defined(BCAST_ROW) && defined(SFPU_OP_EXP)
    auto chain = sfpu_chain(FpuAdd<cb_in0, cb_in1, Dst::D0, BroadcastDim::ROW>{}, Exp<>{});
#elif defined(BCAST_ROW)
    auto chain = sfpu_chain(FpuAdd<cb_in0, cb_in1, Dst::D0, BroadcastDim::ROW>{}, Relu<>{});
#elif defined(BCAST_COL) && defined(SFPU_OP_EXP)
    auto chain = sfpu_chain(FpuAdd<cb_in0, cb_in1, Dst::D0, BroadcastDim::COL>{}, Exp<>{});
#elif defined(BCAST_COL)
    auto chain = sfpu_chain(FpuAdd<cb_in0, cb_in1, Dst::D0, BroadcastDim::COL>{}, Relu<>{});
#elif defined(BCAST_SCALAR) && defined(SFPU_OP_EXP)
    auto chain = sfpu_chain(FpuAdd<cb_in0, cb_in1, Dst::D0, BroadcastDim::SCALAR>{}, Exp<>{});
#elif defined(BCAST_SCALAR)
    auto chain = sfpu_chain(FpuAdd<cb_in0, cb_in1, Dst::D0, BroadcastDim::SCALAR>{}, Relu<>{});
#else
#error "Define BCAST_ROW|BCAST_COL|BCAST_SCALAR and optionally SFPU_OP_EXP"
#endif

    eltwise_op<cb_out>(chain, shape);
}
