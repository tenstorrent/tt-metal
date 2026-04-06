// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ckernel {
namespace sfpu {

// rpow(x, base) = base^x
//
// Implementation:
//   base^x = exp(x * ln(base))
//          = 2^(x * log2(base))
//
// We decompose log2(base) using IEEE754 exponent/mantissa, then compute
// 2^(x * log2(base)) using range reduction and a polynomial approximation.
//
// param0: bit-cast uint32_t representation of the float 'base' parameter
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_rpow(uint param0) {
    // Reconstruct the float 'base' from the bit-cast uint32_t parameter
    sfpi::vFloat base_val = Converter::as_float(param0);

    // Constants
    sfpi::vFloat ln2 = 0.6931471805599453f;
    sfpi::vFloat inv_ln2 = 1.4426950408889634f;

    // Precompute log2(base) before the tile loop since base is constant.
    // Decompose base = 2^e * m where m in [1, 2):
    //   log2(base) = e + log2(m)

    // Extract biased exponent
    sfpi::vInt e_biased = sfpi::exexp(base_val);
    sfpi::vFloat e_float = sfpi::int32_to_float(e_biased, 0);
    sfpi::vFloat e_unbiased = e_float - 127.0f;

    // Extract mantissa in [1, 2): set exponent to 127 (2^0 scaling)
    sfpi::vFloat m = sfpi::setexp(base_val, 127);
    sfpi::vFloat f = m - 1.0f;

    // Minimax polynomial: log2(1+f) for f in [0, 1)
    sfpi::vFloat log2_m = f * (1.44269504f + f * (-0.72134752f + f * 0.48089835f));

    // log2(base) = e + log2(m)
    sfpi::vFloat log2_base = e_unbiased + log2_m;

    // ln(base) = log2(base) * ln(2)
    sfpi::vFloat ln_base = log2_base * ln2;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // y = x * ln(base)
        sfpi::vFloat y = x * ln_base;

        // Compute exp(y) = 2^(y / ln(2))
        sfpi::vFloat z = y * inv_ln2;

        // Range reduction: z = n + frac where n = round(z)
        // Add-subtract trick for rounding
        sfpi::vFloat shift = 1024.0f;
        sfpi::vFloat z_shifted = z + shift;
        sfpi::vFloat n_float = z_shifted - shift;
        sfpi::vFloat frac = z - n_float;

        // 2^frac via Taylor polynomial for |frac| <= 0.5
        sfpi::vFloat exp2_frac = 1.0f + frac * (0.693147f + frac * (0.240227f + frac * 0.055505f));

        // 2^n via exponent manipulation
        sfpi::vInt n_int = sfpi::float_to_int16(n_float);
        sfpi::vInt new_exp = n_int + 127;
        sfpi::vFloat pow2_n = sfpi::setexp(1.0f, new_exp);

        // result = 2^n * 2^frac
        sfpi::vFloat result = pow2_n * exp2_frac;

        // x == 0 => base^0 = 1
        v_if(x == 0.0f) { result = 1.0f; }
        v_endif;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
