// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_ops.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_sigmoid_appx.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"

namespace ckernel {
namespace sfpu {

// Two sigmoid implementations behind three names: TTI SFPNONLINEAR path (`calculate_sigmoid` /
// `_calculate_sigmoid_` / `_calculate_sigmoid_regs_`, also what `_calculate_silu_` composes with)
// and the SFPI body `_sfpu_sigmoid_` (used by `clamped_silu_glu` / ttnn swiglu_sfpu). Not bit-equivalent.

// Calculates SIGMOID for results already in registers, used as a helper function.
// Input of sigmoid is expected to already be in src_reg, and the result will be stored in dest_reg.
// work_reg is an intermediate register used for calculations.
inline void _calculate_sigmoid_regs_(
    const std::uint32_t src_reg, const std::uint32_t work_reg, const std::uint32_t dest_reg) {
    // ALthough SFPMUL/SFPADD are 2 cycle instructions, we don't need a TTI_NOP
    // because the hardware implicitly stalls if the next instruction depends on results
    TTI_SFPMOV(src_reg, work_reg, 1);  // Copy src_reg to work_reg and invert sign bit (take negative of input)
    TTI_SFPNONLINEAR(
        work_reg,
        dest_reg,
        p_sfpnonlinear::EXP_MODE);  // Read value from work_reg, take exponential, load back into dest_reg
    TTI_SFPADD(
        p_sfpu::LCONST_1,
        dest_reg,
        p_sfpu::LCONST_1,
        work_reg,
        0);  // Add 1 to dest_reg, store in work_reg, takes 2 cycles
    TTI_SFPNONLINEAR(
        work_reg,
        dest_reg,
        p_sfpnonlinear::RECIP_MODE);  // Read value from work_reg, approximate recip, load back into dest_reg
}

// Calculates SIGMOID for number of rows of output SFPU ops (Quasar = 2 rows)
inline void _calculate_sigmoid_sfp_rows_() {
    TTI_SFPLOAD(
        p_sfpu::LREG0,
        p_sfpu::sfpmem::DEFAULT,
        ADDR_MOD_7,
        0,
        0);  // load from dest into lreg[0], uses ADDR_MOD_7 (set to all zeroes)

    _calculate_sigmoid_regs_(
        p_sfpu::LREG0,
        p_sfpu::LREG1,
        p_sfpu::LREG0);  // calculate sigmoid using lreg[0] as src and dest, and lreg[1] as work register

    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, 0, 0);  // store from lreg[0] into dest register
}

template <int ITERATIONS = SFPU_ITERATIONS>
inline void _calculate_sigmoid_() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        _calculate_sigmoid_sfp_rows_();
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>();  // does the dest_reg++ (increments by
                                                                                   // 2 rows)
    }
}

// SFPI sigmoid body used by clamped_silu_glu and ttnn's minimal_matmul swiglu_sfpu.
// See the file-top comment for how this relates to calculate_sigmoid / _calculate_sigmoid_.
template <bool is_fp32_acc_to_dest_mode = true>
sfpi_inline sfpi::vFloat _sfpu_sigmoid_(sfpi::vFloat x) {
    // Compute sigmoid as:
    // sigmoid(x) = 1 / (1 + exp(-x))

    sfpi::vFloat exp_neg_x;
    // If fp32 then use higher accuracy exp function
    // Otherwise, use exp_21f (~1 ULP on bfloat16)
    if constexpr (is_fp32_acc_to_dest_mode) {
        exp_neg_x = _sfpu_exp_accurate_<true>(-x);
    } else {
        exp_neg_x = _sfpu_exp_21f_bf16_<true>(-x);
    }

    sfpi::vFloat denominator = 1.0f + exp_neg_x;

    sfpi::vFloat result;
    if constexpr (is_fp32_acc_to_dest_mode) {
        result = _sfpu_reciprocal_<2>(denominator);
    } else {
        result = _sfpu_reciprocal_<1>(denominator);
    }

    return result;
}

template <bool APPROXIMATION_MODE>
inline void sigmoid_init() {
    math::_reset_counters_<p_setrwc::SET_ABD_F>();
    if constexpr (!APPROXIMATION_MODE) {
        _init_reciprocal_<false>();
    } else {
        sigmoid_appx_init();
    }
}

template <int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_sigmoid() {
    _calculate_sigmoid_<ITERATIONS>();
}

}  // namespace sfpu
}  // namespace ckernel
