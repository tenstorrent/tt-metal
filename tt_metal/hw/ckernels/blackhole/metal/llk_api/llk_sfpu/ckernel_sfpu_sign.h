// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpu/ckernel_sfpu_is_fp16_zero.h"
#include "sfpi.h"
#include "llk_math_eltwise_sfpu_op.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_sign(const uint /*exponent_size_8*/) {
// All params are in FP16 format
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vFloat res = 1.0f;
        v_if(v < 0.0F) { res = -1.0f; }
        v_elseif(_sfpu_is_fp16_zero_(v)) { res = 0.0f; }
        v_endif;
        sfpi::dst_reg[0] = res;
        sfpi::dst_reg++;
    }
}

// Sign<APPROX, DST_SYNC, DST_ACCUM, ITERATIONS>: sign_tile / sign_tile_init (compute_kernel_api.h).
// init() is the shared SFPU init only.
template <bool APPROXIMATION_MODE, DstSync DST_SYNC, bool DST_ACCUM, int ITERATIONS = 8>
struct Sign : SfpuUnaryOp<Sign<APPROXIMATION_MODE, DST_SYNC, DST_ACCUM, ITERATIONS>, DST_SYNC, DST_ACCUM> {
    static void kernel(const uint32_t exponent_size_8) {
        calculate_sign<APPROXIMATION_MODE, ITERATIONS>(exponent_size_8);
    }
};

}  // namespace sfpu
}  // namespace ckernel
