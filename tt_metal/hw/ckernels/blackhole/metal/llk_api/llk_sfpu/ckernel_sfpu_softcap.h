// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// softcap(x, cap) = cap * tanh(x / cap)
// tanh via piecewise degree-7 centered polynomials.
// Max error < 2 ULP in fp32 per segment.

// Helper: evaluate degree-7 centered polynomial on a segment
// tanh_pos = c0 + c1*u + c2*u^2 + ... + c7*u^7 where u = (at - center) * inv_hw
template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat eval_tanh_seg(
    sfpi::vFloat at, float center, float inv_hw,
    float c0, float c1, float c2, float c3,
    float c4, float c5, float c6, float c7) {
    sfpi::vFloat u = (at - center) * inv_hw;
    return ((((((c7 * u + c6) * u + c5) * u + c4) * u + c3) * u + c2) * u + c1) * u + c0;
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_softcap(const uint32_t param0, const uint32_t param1) {
    union { uint32_t u; float f; } u0, u1;
    u0.u = param0;
    u1.u = param1;
    sfpi::vFloat inv_cap = u0.f;
    sfpi::vFloat cap_val = u1.f;

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat t = x * inv_cap;
        sfpi::vFloat at = sfpi::abs(t);
        sfpi::vFloat at2 = at * at;

        // Seg [0, 0.5]: degree-9 odd polynomial
        sfpi::vFloat tp = at * (0.999999991f + at2 * (-0.333331294f + at2 * (0.133261660f + at2 * (-0.053086073f + at2 * 0.017356469f))));

        // Seg (0.5, 1.0]: centered at 0.75, inv_hw = 4
        v_if(at > 0.5f) { tp = eval_tanh_seg<APPROXIMATION_MODE>(at, 0.75f, 4.0f, 0.635148953f, 0.149146452f, -0.023682583f, 0.000653266f, 0.000389819f, -0.000062523f, -0.000000530f, 0.000001304f); }
        v_endif;

        // Seg (1.0, 1.5]: centered at 1.25, inv_hw = 4
        v_if(at > 1.0f) { tp = eval_tanh_seg<APPROXIMATION_MODE>(at, 1.25f, 4.0f, 0.848283640f, 0.070103716f, -0.014866962f, 0.001692358f, -0.000049155f, -0.000018754f, 0.000003682f, -0.000000272f); }
        v_endif;

        // Seg (1.5, 2.5]: centered at 2.0, inv_hw = 2
        v_if(at > 1.5f) { tp = eval_tanh_seg<APPROXIMATION_MODE>(at, 2.0f, 2.0f, 0.964027571f, 0.035325421f, -0.017026995f, 0.005263506f, -0.001120066f, 0.000149883f, 0.000000731f, -0.000005784f); }
        v_endif;

        // Seg (2.5, 3.5]: centered at 3.0, inv_hw = 2
        v_if(at > 2.5f) { tp = eval_tanh_seg<APPROXIMATION_MODE>(at, 3.0f, 2.0f, 0.995054754f, 0.004933019f, -0.002454321f, 0.000810001f, -0.000198422f, 0.000038101f, -0.000005912f, 0.000000679f); }
        v_endif;

        // Seg (3.5, 5.0]: centered at 4.25, inv_hw = 1/0.75
        v_if(at > 3.5f) { tp = eval_tanh_seg<APPROXIMATION_MODE>(at, 4.25f, 1.333333333f, 0.999593147f, 0.000610156f, -0.000457481f, 0.000228545f, -0.000085285f, 0.000025525f, -0.000006825f, 0.000001427f); }
        v_endif;

        // Seg (5.0, 9.0]: centered at 7.0, inv_hw = 0.5
        v_if(at > 5.0f) { tp = eval_tanh_seg<APPROXIMATION_MODE>(at, 7.0f, 0.5f, 0.999998358f, 0.000006609f, -0.000014055f, 0.000018364f, -0.000013757f, 0.000011834f, -0.000015874f, 0.000008557f); }
        v_endif;

        // Clamp
        v_if(tp > 1.0f) { tp = sfpi::vConst1; }
        v_endif;
        v_if(tp < 0.0f) { tp = 0.0f; }
        v_endif;
        v_if(at > 9.0f) { tp = sfpi::vConst1; }
        v_endif;

        // Apply sign
        sfpi::vFloat result = tp;
        v_if(t < 0.0f) { result = 0.0f - tp; }
        v_endif;

        sfpi::dst_reg[0] = cap_val * result;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
