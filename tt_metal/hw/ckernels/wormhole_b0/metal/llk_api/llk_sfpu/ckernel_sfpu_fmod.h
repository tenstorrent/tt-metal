// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_recip.h"
#include "cmath_common.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "llk_math_eltwise_sfpu_op.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE>
inline void init_fmod(const uint value, const uint recip) {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpi::vConstFloatPrgm0 = Converter::as_float(value);
    sfpi::vConstFloatPrgm1 = Converter::as_float(recip);
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_fmod() {
    // SFPU microcode
    sfpi::vFloat s = sfpi::abs(sfpi::vConstFloatPrgm0);
    sfpi::vFloat recip_val = sfpi::abs(sfpi::vConstFloatPrgm1);

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::vFloat v = sfpi::abs(val);

        vFloat quotient;
        vInt exp = sfpi::exexp(v * recip_val);
        v_if(exp < 0) { quotient = 0.0f; }
        // Since fp32 has 23 mantissa bits, the LSB represents the fractional part when exp < 23.
        // We effectively round off the fractional bits to zero by right shifting using (exp - 23) and then left
        // shifting it back using (0 - (exp - 23)).
        v_elseif(exp < 23) {
            quotient = sfpi::as<sfpi::vFloat>(
                shft((shft(sfpi::as<sfpi::vUInt>(v * recip_val), (exp - 23))), (0 - (exp - 23))));
        }
        v_else { quotient = v * recip_val; }
        v_endif

        v_if(quotient > v * recip_val) {
            quotient = quotient - 1;
        }
        v_endif;
        v = v - quotient * s;

        v = sfpi::copysgn(v, val);

        v_if(s == 0) { v = std::numeric_limits<float>::quiet_NaN(); }
        v_endif;

        constexpr auto iter = 10;
        for (int l = 0; l < iter; l++) {
            v_if(v >= s) { v = v - s; }
            v_endif;
        }
        v_if(sfpi::abs(v) - s == 0.0f) { v = 0.0f; }
        v_endif;
        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}

// ---------------------------------------------------------------------------------------------------
// Fmod<APPROX, DST_SYNC, DST_ACCUM, ITERATIONS>
//   calculate(dst_index, vector_mode) -> calculate_fmod (fmod_tile)
//   init(value, recip)                -> init_fmod      (fmod_tile_init)
// ---------------------------------------------------------------------------------------------------
template <bool APPROXIMATION_MODE, DstSync DST_SYNC, bool DST_ACCUM, int ITERATIONS = 8>
struct Fmod : SfpuUnaryOp<Fmod<APPROXIMATION_MODE, DST_SYNC, DST_ACCUM, ITERATIONS>, DST_SYNC, DST_ACCUM> {
    static void kernel() { calculate_fmod<APPROXIMATION_MODE, ITERATIONS>(); }

    static void init_kernel(uint32_t value, uint32_t recip) { init_fmod<APPROXIMATION_MODE>(value, recip); }
};
}  // namespace sfpu
}  // namespace ckernel
