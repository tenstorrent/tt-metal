// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#if defined(TRISC_MATH) || defined(TRISC_PACK)
// Include after frozen compute_common.hpp and before the private streaming
// header. The selected compute_common already loads its original SFPU file;
// do not include a second frozen copy with duplicate namespace definitions.
// This file does not modify the frozen cubic implementation.

#if defined(SDPA_LOFI_EXP_DEGREE)
#if SDPA_LOFI_EXP_DEGREE != 1 && SDPA_LOFI_EXP_DEGREE != 2
#error "SDPA_LOFI_EXP_DEGREE must be 1 or 2; omit it to retain the original cubic"
#endif

namespace ckernel::sfpu {

template <int iterations>
inline void calculate_lofi_exp_refiner() {
    static_assert(iterations > 0 && iterations % 2 == 0);
    // Preserve the cubic refiner's destination addressing and the fast-grid
    // replay slots 0..7. Only the second store advances DST, by four vectors.
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_6);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 4}}.set(ADDR_MOD_7);

    // Descending relative-LS coefficients, with 2^96 folded into each value.
    // Same selection rule as calculate_sdpa_exp_stream_effective: DIAG 1/4
    // fits exp2(m-1); otherwise fit 0.995*exp2(m-1) + 1/128 before P truncation.
#if defined(SDPA_DIAG_EXP_MODE) && (SDPA_DIAG_EXP_MODE == 1 || SDPA_DIAG_EXP_MODE == 4)
#if SDPA_LOFI_EXP_DEGREE == 2
    constexpr float a = 1.78788937e28f;
    constexpr float b = -5.27842419e28f;
    constexpr float c = 1.13613290e29f;
#else
    constexpr float a = 8.42193996e26f;
    constexpr float b = 7.48452350e28f;
#endif
#else
#if SDPA_LOFI_EXP_DEGREE == 2
    constexpr float a = 1.79879414e28f;
    constexpr float b = -5.34102955e28f;
    constexpr float c = 1.14346173e29f;
#else
    constexpr float a = 5.52623980e26f;
    constexpr float b = 7.53273083e28f;
#endif
#endif

    TTI_SFPLOADI(6, 0xA, sdpa_lo16(a));
    TTI_SFPLOADI(6, 0x8, sdpa_hi16(a));
    TTI_SFPLOADI(0, 0xA, sdpa_lo16(b));
    TTI_SFPLOADI(0, 0x8, sdpa_hi16(b));
    TTI_SFPCONFIG(0, 12, 0);
#if SDPA_LOFI_EXP_DEGREE == 2
    TTI_SFPLOADI(0, 0xA, sdpa_lo16(c));
    TTI_SFPLOADI(0, 0x8, sdpa_hi16(c));
    TTI_SFPCONFIG(0, 13, 0);
    constexpr int replay_length = 12;
#else
    constexpr int replay_length = 10;
#endif

    // L0/L1 retain the two linear-grid values. L2/L3 hold their mantissas;
    // L4/L5 are independent Horner chains. Interleave the chains exactly as
    // the existing cubic does, including the final MUL-to-STORE distances.
    TTI_REPLAY(8, replay_length, 1, 1);
    TTI_SFPLOAD(0, 0, ADDR_MOD_6, 0);
    TTI_SFPLOAD(1, 0, ADDR_MOD_6, 2);
    TTI_SFPSETEXP(127, 0, 2, 1);
    TTI_SFPSETEXP(127, 1, 3, 1);
    TTI_SFPMAD(2, 6, 12, 4, 0);
    TTI_SFPMAD(3, 6, 12, 5, 0);
#if SDPA_LOFI_EXP_DEGREE == 2
    TTI_SFPMAD(2, 4, 13, 4, 0);
    TTI_SFPMAD(3, 5, 13, 5, 0);
#endif
    // Keep MUL: the quadratic's refined mantissa can cross 1 or 2, so an
    // exponent-overwrite replacement would not preserve its normalization.
    TTI_SFPMUL(0, 4, p_sfpu::LCONST_0, 0, 0);
    TTI_SFPMUL(1, 5, p_sfpu::LCONST_0, 1, 0);
    TTI_SFPSTORE(0, 0, ADDR_MOD_6, 0);
    TTI_SFPSTORE(1, 0, ADDR_MOD_7, 2);
#pragma GCC unroll 8
    for (int i = 2; i < iterations; i += 2) {
        lltt::replay(8, replay_length);
    }
}

}  // namespace ckernel::sfpu

// Deliberately installed last: only later streaming call sites are redirected.
// The fused accurate-exp and load-macro refiners are separate, unchanged APIs.
#define calculate_sdpa_exp_stream_effective calculate_lofi_exp_refiner
#endif
#endif  // TRISC_MATH || TRISC_PACK
