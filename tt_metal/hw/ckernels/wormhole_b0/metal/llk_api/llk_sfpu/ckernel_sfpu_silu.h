// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cmath_common.h"  // math::reset_counters, p_setrwc
#include "ckernel_sfpu_sigmoid.h"
#include "ckernel_sfpu_recip.h"
#include "llk_math_eltwise_sfpu_op.h"

namespace ckernel::sfpu {

template <bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_silu() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // silu(x) = x * sigmoid(x)
        sfpi::vFloat result = x * _sfpu_sigmoid_<is_fp32_dest_acc_en>(x);

        // Round to bfloat16 if not in fp32 accumulation mode
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void silu_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    // calculate_silu always uses the non-approx sigmoid path via _sfpu_sigmoid_, so we must
    // use non-approx sigmoid_init regardless of APPROXIMATION_MODE.
    sigmoid_init<false>();
}

// Silu<APPROX, DST_SYNC, DST_ACCUM, ITERATIONS>: silu_tile / silu_tile_init and the
// silu_tile_pack / silu_tile_init_pack pack-thread variants (compute_kernel_api.h).
// APPROXIMATION_MODE only reaches silu_init; the kernel is always the accurate path.
template <bool APPROXIMATION_MODE, DstSync DST_SYNC, bool DST_ACCUM, int ITERATIONS = 8>
struct Silu : SfpuUnaryOp<Silu<APPROXIMATION_MODE, DST_SYNC, DST_ACCUM, ITERATIONS>, DST_SYNC, DST_ACCUM> {
    static void kernel() { calculate_silu<DST_ACCUM, ITERATIONS>(); }

    static void init_kernel() { silu_init<APPROXIMATION_MODE>(); }
};

}  // namespace ckernel::sfpu
