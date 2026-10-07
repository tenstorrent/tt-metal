// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// exp((qk - rowmax) * scale) for the accurate prefill softmax.
//
// Computed as 2^z with z = x * (scale * log2 e). Inputs are scores minus the row maximum
// (<= ~0) or masked -inf, so a single max(z, -126) replaces the generic overflow/underflow
// guards (-inf -> 2^-126 ~ 1e-38). n = round(z) and f = z - n in [-0.5, 0.5] are exact in
// FP32. 2^f is a degree-4 minimax polynomial with p(0) = 1, max relative error 2.86e-6.
// c1 and c2 are FP32 in programmable constant registers; c3 and c4 are FP16-exact immediates
// (one sfploadi each).
//
// It replaces the FP32 path's separate scale pass plus the full-FP32 guarded exponential:
// ~20 SFPU instructions per 32 elements instead of ~31, with no guard branches. The
// probabilities feed the PV matmul as TF32 (11-bit mantissa, 4.9e-4), so the 2.9e-6 error is
// ~170x below that truncation. Prefill attention error vs FP64 is unchanged to 4 significant digits.
#ifdef TRISC_MATH
namespace ckernel::sfpu {
inline void k2_exp_init() {
    sfpi::vConstFloatPrgm0 = 0x1.62e126p-1f;  // c1
    sfpi::vConstFloatPrgm1 = 0x1.ec038cp-3f;  // c2
}

template <int ITERATIONS>
inline void k2_exp_scaled(const uint32_t log2e_scale_bits) {
    const sfpi::vFloat c = Converter::as_float(log2e_scale_bits);
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat z = sfpi::max(sfpi::dst_reg[0] * c, -126.0f);
        sfpi::vSMag16 sm = sfpi::convert<sfpi::vSMag16>(z, sfpi::RoundMode::Nearest);
        sfpi::vFloat n = sfpi::convert<sfpi::vFloat>(sm, sfpi::RoundMode::Nearest);
        sfpi::vFloat f = z - n;
        sfpi::vFloat p = f * 0x1.3ap-7f + 0x1.cap-5f;  // c4, c3
        sfpi::vInt i = sfpi::abs(sfpi::as<sfpi::vInt>(sm));
        p = p * f + sfpi::vConstFloatPrgm1;
        i = sfpi::as<sfpi::vInt>(sfpi::copysgn(sfpi::as<sfpi::vFloat>(i), n));
        p = p * f + sfpi::vConstFloatPrgm0;
        p = p * f + 1.0f;
        sfpi::vInt e = sfpi::exexp(p, sfpi::ExponentMode::Biased) + i;
        sfpi::dst_reg[0] = sfpi::setexp(p, e);
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

// scale * log2(e) in FP32, from the FP32 bit pattern of the attention scale.
inline uint32_t k2_log2e_scale_bits(uint32_t scale_bits) {
    union {
        uint32_t u;
        float f;
    } v{scale_bits};
    v.f *= 1.4426950408889634f;
    return v.u;
}
