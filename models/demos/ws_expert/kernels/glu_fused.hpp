// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// SwiGLU silu(g) * u as ONE SFPU pass over a (gate, up) DST tile pair (the stock path is two: silu over the gate tile,
// then a binary multiply). FAST picks the sigmoid's exp: false = the fp32-accurate exp + 2 reciprocal Newton steps
// (what silu_tile uses with fp32 DEST), true = exp_21f (~1 bf16 ULP) + 1 Newton step - h is packed to bf16 / bfp8
// anyway. Runs on the PACK thread (expert_activation-style), init: sigmoid_init<false> (Prgm0 = 2.0f, Prgm1 = 1/ln2).
#pragma once

#ifdef TRISC_PACK
#include "ckernel_sfpu_sigmoid.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"

namespace ckernel::sfpu {
// MODE 0: fp32-accurate exp + 2 Newton steps (= stock silu with fp32 DEST); 1: exp_21f + 1 Newton (constants
// reloaded per row, as _sfpu_sigmoid_ does); 2: exp_21f + 1 Newton, constants in the programmable registers;
// 3: as 2 without the Newton step (approx reciprocal, ~0.4%: speed reference only)
template <bool is_fp32_dest_acc_en, int ITERATIONS, int MODE>
inline void calculate_ws_glu(const uint gate_tile_idx, const uint up_tile_idx, const uint out_tile_idx) {
    constexpr uint dst_tile_size = 32;
    if constexpr (MODE <= 1) {
#pragma GCC unroll 4
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat g = sfpi::dst_reg[gate_tile_idx * dst_tile_size];
            sfpi::vFloat u = sfpi::dst_reg[up_tile_idx * dst_tile_size];
            sfpi::vFloat r = g * _sfpu_sigmoid_<MODE == 0>(g) * u;
            sfpi::dst_reg[out_tile_idx * dst_tile_size] = r;
            sfpi::dst_reg++;
        }
    } else {
        // exp_21f of -g with c0 / c1 / (1/ln2) in the programmable constant registers (ws_glu::init<2>), xlog2
        // clamped to [0, 254] (finite, so the plain Newton step never sees inf * 0), approx reciprocal + one Newton
        // step: same results as MODE 1, ~17% fewer SFPU cycles (no per-row SFPLOADI pairs for the constants)
#pragma GCC unroll 1
        for (int d = 0; d < ITERATIONS; d++) {
            const sfpi::vFloat g = sfpi::dst_reg[gate_tile_idx * dst_tile_size];
            const sfpi::vFloat gu = g * sfpi::dst_reg[up_tile_idx * dst_tile_size];
            sfpi::vFloat xl = -g * sfpi::vConstFloatPrgm1 + 127.0f;
            xl = sfpi::clamp(xl, 0.0f, 254.0f);
            const sfpi::vFloat z = sfpi::as<sfpi::vFloat>(_float_to_int32_for_exp_21f_(xl));
            const sfpi::vInt ep = sfpi::exexp(z, sfpi::ExponentMode::Biased);
            sfpi::vFloat f = sfpi::convert<sfpi::vFloat>(sfpi::exman(z), sfpi::RoundMode::Nearest);
            f = (f * EXP_21F_C2 + sfpi::vConstFloatPrgm2) * f + sfpi::vConstFloatPrgm0;
            const sfpi::vFloat den = sfpi::setexp(f, ep) + 1.0f;
            sfpi::vFloat y = sfpi::approx_recip(den);
            if constexpr (MODE == 2) {
                y = y * (2.0f - den * y);
            }
            sfpi::dst_reg[out_tile_idx * dst_tile_size] = gu * y;
            sfpi::dst_reg++;
        }
    }
}
inline void ws_glu_prgm_init() {
    sfpi::vConstFloatPrgm0 = EXP_21F_C0;
    sfpi::vConstFloatPrgm1 = EXP_21F_ONE_LN2;
    sfpi::vConstFloatPrgm2 = EXP_21F_C1;
}
}  // namespace ckernel::sfpu
#endif

namespace ws_glu {
template <int MODE>
ALWI void init() {
    if constexpr (MODE >= 2) {
        PACK((SFPU_BINARY_INIT_FN_NO_ARGS(unused, sfpu::ws_glu_prgm_init)));
    } else {
        PACK((SFPU_BINARY_INIT_FN_NO_ARGS(unused, sfpu::clamped_silu_glu_init)));
    }
}

template <int MODE>
ALWI void tile_pair(uint32_t gate_tile) {
    PACK((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_ws_glu,
        (DST_ACCUM_MODE, 8, MODE),
        gate_tile,
        gate_tile + 1,
        gate_tile,
        VectorMode::RC)));
}
}  // namespace ws_glu
