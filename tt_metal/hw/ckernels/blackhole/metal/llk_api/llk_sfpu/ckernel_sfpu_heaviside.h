// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "llk_math_eltwise_sfpu_op.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_heaviside(uint value) {
    // SFPU microcode
    vFloat s = Converter::as_float(value);

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];

        v_if(v < 0.0f) { v = 0.0f; }
        v_elseif(v > 0.0f) { v = 1.0f; }
        v_else { v = s; }
        v_endif;

        dst_reg[0] = v;

        dst_reg++;
    }
}

// Heaviside<APPROX, DST_SYNC, DST_ACCUM, ITERATIONS>: heaviside_tile / heaviside_tile_init
// (compute_kernel_api.h). init() is the shared SFPU init only.
template <bool APPROXIMATION_MODE, DstSync DST_SYNC, bool DST_ACCUM, int ITERATIONS = 8>
struct Heaviside : SfpuUnaryOp<Heaviside<APPROXIMATION_MODE, DST_SYNC, DST_ACCUM, ITERATIONS>, DST_SYNC, DST_ACCUM> {
    static void kernel(uint32_t value) { calculate_heaviside<APPROXIMATION_MODE, ITERATIONS>(value); }
};

}  // namespace sfpu
}  // namespace ckernel
