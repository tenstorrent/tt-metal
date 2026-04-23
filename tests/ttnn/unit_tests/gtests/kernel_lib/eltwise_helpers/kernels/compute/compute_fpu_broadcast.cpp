// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// FPU broadcast: FpuOp<cb_in0, cb_in1, D0, BroadcastDim>
// Defines: BCAST_ROW | BCAST_COL | BCAST_SCALAR
//          FPU_OP_ADD | FPU_OP_MUL (default ADD)
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

#if defined(BCAST_ROW) && defined(FPU_OP_MUL)
    auto chain = sfpu_chain(FpuMul<cb_in0, cb_in1, Dst::D0, BroadcastDim::ROW>{});
#elif defined(BCAST_ROW)
    auto chain = sfpu_chain(FpuAdd<cb_in0, cb_in1, Dst::D0, BroadcastDim::ROW>{});
#elif defined(BCAST_COL) && defined(FPU_OP_MUL)
    auto chain = sfpu_chain(FpuMul<cb_in0, cb_in1, Dst::D0, BroadcastDim::COL>{});
#elif defined(BCAST_COL)
    auto chain = sfpu_chain(FpuAdd<cb_in0, cb_in1, Dst::D0, BroadcastDim::COL>{});
#elif defined(BCAST_SCALAR) && defined(FPU_OP_MUL)
    auto chain = sfpu_chain(FpuMul<cb_in0, cb_in1, Dst::D0, BroadcastDim::SCALAR>{});
#elif defined(BCAST_SCALAR)
    auto chain = sfpu_chain(FpuAdd<cb_in0, cb_in1, Dst::D0, BroadcastDim::SCALAR>{});
#else
#error "Define one of: BCAST_ROW, BCAST_COL, BCAST_SCALAR"
#endif

    eltwise_op<cb_out>(chain, shape);
}
