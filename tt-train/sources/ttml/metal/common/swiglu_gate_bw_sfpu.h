// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// One SFPU pass for both SwiGLU-gating gradients, from three BF16 DEST tiles: gate at idst, dL/dh at
// idst + 1 and up at idst + 2. It overwrites gate with dL/d(gate) and dL/dh with dL/d(up):
//   dL/d(gate) = (dL/dh * up) * silu'(gate),  silu'(x) = s(x) * (1 + x * s(-x))
//   dL/d(up)   = dL/dh * x * s(x)
// s(x) and s(-x) come from one exp and one reciprocal, the core of ttnn's silu_bw kernel
// (ckernel_sfpu_silu_bw_bf16.h). Both gradients are evaluated in fp32 registers and rounded to BF16 once, so
// dL/d(gate) is not rounded twice.
//
// KERNEL-SIDE ONLY: include this from a compute kernel .cpp.

#pragma once

#include <cstdint>
#include <limits>

#include "api/compute/common_globals.h"

#ifdef TRISC_MATH
#include "llk_assert.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "sfpi.h"
#endif

namespace ttml::metal::sfpu {

#ifdef TRISC_MATH

template <int ITERATIONS = 32>
inline void calculate_swiglu_gate_bw() {
    using namespace sfpi;
    constexpr int kTile = 32;  // SFPU rows per 32x32 tile
    // The SFPU has few LRegs: x, dL/dh and up are read from DEST where they are used rather than held.
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
#if defined(ARCH_WORMHOLE)
        // Wormhole's multiply gives a NaN the sign of its product, which the pack stores as an infinity of that
        // sign: a NaN input becomes the positive NaN first.
        v_if(sfpi::is_nan(vFloat(dst_reg[kTile + d]))) {
            dst_reg[kTile + d] = std::numeric_limits<float>::quiet_NaN();
        }
        v_endif;
        v_if(sfpi::is_nan(vFloat(dst_reg[2 * kTile + d]))) {
            dst_reg[2 * kTile + d] = std::numeric_limits<float>::quiet_NaN();
        }
        v_endif;
#endif
        // es = exp(-|x|) * 2^64, from 2^t with t = -|x| / ln 2 split into an integer and a polynomial in the rest.
        vFloat es;
        {
            vFloat a = setsgn(vFloat(dst_reg[d]), 0);
            vFloat core_t = (-a) * 1.4426950216293335f;
            v_if(core_t < -191.0f) {
                core_t = -191.0f;
            }
            v_elseif(core_t > 127.0f) {
                core_t = 127.0f;
            }
            v_endif;
            vFloat core_f = core_t + 12582912.0f;
            vFloat core_n = core_f - 12582912.0f;
            vInt core_k = as<vInt>(core_f) - 1262485504;
            vFloat core_r = core_t - core_n;
            vFloat core = 0.00958237424492836f;
            core = core * core_r + 0.05590646713972092f;
            core = core * core_r + 0.24024108052253723f;
            core = core * core_r + 0.6931241750717163f;
            core = core * core_r + 1.0f;
            vInt core_e = exexp(core, ExponentMode::Biased) + core_k + 64;
            v_if(core_e <= 0) {
                core = 0.0f;
            }
            v_else {
                core = setexp(core, core_e);
            }
            v_endif;
            es = core;
        }
        // P = 1 / (1 + e) by Newton's method and Ns = e * P * 2^64: s(x) and s(-x) are P and N = Ns * 2^-64 in the
        // order of x's sign.
        vFloat P;
        {
            vFloat e = es * 5.421010862427522e-20f;
            vFloat denominator = e + 1.0f;
            P = denominator * -0.5f + 1.4571068286895752f;
            P = P * (-denominator * P + 2.0f);
            P = P * (-denominator * P + 2.0f);
            P = P * (-denominator * P + 2.0f);
            v_if(e < 2.9802322387695312e-08f) {
                P = 1.0f;
            }
            v_endif;
        }
        vFloat Ns = es * P;
        // For x < 0 both products are formed with s(x) at 2^64 times its value (Ns) and scaled back at the end, so
        // they stay in range when s(x) is near the bottom of fp32.
        vFloat x = dst_reg[d];
        vFloat S = P;
        vFloat T = Ns * 5.421010862427522e-20f;
        v_if(x < 0.0f) {
            S = Ns;
            T = P;
        }
        v_endif;
        vFloat dup = vFloat(dst_reg[kTile + d]) * (x * S);
        vFloat dgate = vFloat(dst_reg[kTile + d]) * (vFloat(dst_reg[2 * kTile + d]) * (S * (1.0f + (T * x))));
        x = dst_reg[d];
        v_if(x < 0.0f) {
            // For x <= -ln(2) * 100, s(x) < 2^-100: both products are rounded to BF16 at 2^64 times their value,
            // then scaled back. Above that, scaling back first is exact.
            v_if(x <= -69.31472778320312f) {
#if defined(ARCH_WORMHOLE)
                v_if(sfpi::is_nan(dgate)) {
                    dgate = std::numeric_limits<float>::quiet_NaN();
                }
                v_endif;
                v_if(sfpi::is_nan(dup)) {
                    dup = std::numeric_limits<float>::quiet_NaN();
                }
                v_endif;
#endif
                dgate = convert<vFloat16b>(dgate, RoundMode::Nearest);
                dup = convert<vFloat16b>(dup, RoundMode::Nearest);
            }
            v_endif;
            dgate = dgate * 5.421010862427522e-20f;
            dup = dup * 5.421010862427522e-20f;
        }
        v_endif;
#if defined(ARCH_WORMHOLE)
        v_if(sfpi::is_nan(dgate)) {
            dgate = std::numeric_limits<float>::quiet_NaN();
        }
        v_endif;
        v_if(sfpi::is_nan(dup)) {
            dup = std::numeric_limits<float>::quiet_NaN();
        }
        v_endif;
#endif
        dgate = convert<vFloat16b>(dgate, RoundMode::Nearest);
        dup = convert<vFloat16b>(dup, RoundMode::Nearest);
        // DEST holds x as BF16 bits: exponent 7:0, mantissa 14:8, sign 15. An infinite or NaN x gives a NaN dL/d(gate),
        // as ttnn::silu_bw does, and a NaN x a NaN dL/d(up).
        vUInt raw = dst_reg[d].mode<::sfpi::DataLayout::U16>();
        v_if((raw & 0x00ff) == 0x00ff) {
            dgate = std::numeric_limits<float>::quiet_NaN();
            v_if((raw & 0x7f00) != 0) {
                dup = std::numeric_limits<float>::quiet_NaN();
            }
            v_endif;
        }
        v_endif;
        dst_reg[d] = dgate;
        dst_reg[kTile + d] = dup;
    }
}

#endif  // TRISC_MATH

}  // namespace ttml::metal::sfpu

namespace ckernel {

#ifdef TRISC_MATH
namespace sfpu {

// SFPU_UNARY_CALL resolves its functor as ::ckernel::sfpu::FN, hence this forwarder.
template <int ITERATIONS>
inline void _calculate_swiglu_gate_bw_() {
    ttml::metal::sfpu::calculate_swiglu_gate_bw<ITERATIONS>();
}

}  // namespace sfpu
#endif  // TRISC_MATH

MATH(static_assert(!DST_ACCUM_MODE, "swiglu_gate_bw_tile evaluates BF16 DEST"));

/**
 * Initializes swiglu_gate_bw_tile. Must be called before swiglu_gate_bw_tile.
 */
ALWI void swiglu_gate_bw_tile_init() {
    MATH(SFPU_UNARY_INIT(unused));
}

/**
 * Both SwiGLU-gating gradients of one tile: DEST tiles idst, idst + 1 and idst + 2 hold gate, dL/dh and up;
 * dL/d(gate) replaces gate and dL/d(up) replaces dL/dh.
 */
ALWI void swiglu_gate_bw_tile(uint32_t idst) {
    MATH(LLK_ASSERT(
        (idst + 3 <= get_dest_max_tiles_rt<DST_SYNC_MODE, DstTileShape::Tile32x32>()),
        "swiglu_gate_bw_tile needs DEST tiles idst to idst + 2"));
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, _calculate_swiglu_gate_bw_, (32), idst, VectorMode::None));
}

}  // namespace ckernel
