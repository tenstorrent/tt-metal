// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "cmath_common.h"
#include "lltt.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_load_config.h"
#include "ckernel_sfpu_sigmoid_appx.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"

namespace ckernel {
namespace sfpu {

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
        result = sfpu_reciprocal_iter<2>(denominator);
    } else {
        result = sfpu_reciprocal_iter<1>(denominator);
    }

    return result;
}

// =====================================================================================================================
// Fast bf16 sigmoid for Blackhole. Origin: llk-bench (LLM-agent-written kernel, claude-opus-5, 2026-08),
// validated exhaustively over all 65,536 bf16 inputs vs the exact golden: max 2 ULP (gate <= 2).
// Measured 447.0 cycles/tile on p150b (tt-metal v0.76.0 baseline); the manifest records no same-harness number
// for the previous kernel -- the kernel's own note below puts it at ~1234 cycles/tile.
//
// State programmed by _init_sigmoid_bf16_fast_: LREG12..14 (vConstFloatPrgm0..2), LREG11 (config path) and
// LREG6/7; replay slot 0, 25 instructions (recorded per init, like exp). No SFPLOADMACRO. Needs ADDR_MOD_7 =
// {0,0,0} (re-asserted by the init). bf16 DEST only: never use for fp32 dest or on Wormhole. Processes one face
// (8 dst vectors at ADDR_MOD_7 offsets 0/2, advancing the dst RWC by two vectors per pair with INCRWC; the
// params wrapper re-bases the RWC from CR_D between faces, so this leaves no residue).
//
// sigmoid(x) = 1 / (1 + exp(-x)), bf16 DST, single-core Blackhole SFPU.
//
// 12.5 SFPU issue slots per 32-lane vector, branch-free, no Newton iterations,
// no divides: ~447 cycles/tile vs ~1234 for the production kernel.
//
// ---------------------------------------------------------------- algorithm
//
//   u = exp(-x) = 2^(s/256 - 127),  s = clamp(256*(127 - x*log2 e), 0, 65535)
//
// s is carried as an 8-bit fixed-point exponent, so the float->int conversion
// is a single SFPSTOCHRND fp32->uint16 (which saturates at 65535 for us, all
// that is needed on the x < 0 side) plus a shift that lands the integer part
// in the exponent field:
//
//   ylin = bit_cast<float>(s_int << 15) = 2^(n-127) * (1 + f)
//
// with n = floor(s/256), f = frac(s/256): Schraudolph's piecewise-linear 2^x.
// Rather than extract f (exman9 + cast), the mantissa 1+f is recovered with a
// single SETEXP and 2^f/(1+f) is restored by a minimax cubic rho(1+f)
// (relative error 6.6e-4). The refinement multiply is folded into the "+1" of
// the denominator, so the whole exp costs one extra MAD:
//
//   d = ylin * rho(1+f) + 1,   y = 1/d   via SFPARECIP (hardware approximate
//                                        reciprocal, one instruction)
//
// Two systematic biases are cancelled for free by the single scale factor
// carried in the cubic:
//   * SFPARECIP's relative error lies in [-0.558%, +0.113%] (measured over
//     the whole bf16 range), i.e. a -0.223% bias; scaling rho down by that
//     amount re-centres it. The cancellation is exact where it matters most
//     (x < 0, where y ~ 1/u so u's scale passes straight through) and half
//     strength for x > 0, where y is in [0.5, 1] and 2 ULP is a wide target.
//   * SFPSTORE rounds fp32->bf16 to nearest, so no explicit SFP_STOCH_RND
//     conversion is needed on the way out.
//
// Exhaustively verified: max 2 ULP over all 65,536 bf16 inputs. Error budget:
// cubic 0.07% + fixed-point quantisation 0.135% + reciprocal +-0.335%.
//
// The tails fall out of the clamp alone, with no special cases:
//   s = 65535 (x <= -88) -> s_int << 15 overflows into a NaN/huge pattern ->
//                           d is huge/NaN -> SFPARECIP gives 0, and the
//                           golden there is denormal, flushed to zero;
//   s = 0     (x >=  88) -> ylin = 0 -> d = scale -> y = 1.
//
// ------------------------------------------------------------- scheduling
//
// The SFPU issues one instruction per cycle with a 2-cycle result latency
// (measured: 8 chained MADs cost 8 extra cycles, the same 8 MADs alternating
// between two independent chains cost none). The body therefore processes two
// vectors with strictly alternating chains, so every consumer sits two slots
// after its producer and nothing stalls.
//
// Holding two chains x (u, mantissa, accumulator) plus the four cubic
// coefficients needs 10 registers, and there are only 8 LREGs. The fourth
// programmable constant register (LREG11, which the LLK convention leaves at
// -1.0) takes one coefficient and LREG12..14 take the rest, freeing LREG0..5
// as the six scratch registers the schedule needs.
//
// Finally, pushing Tensix instructions from the RISC-V core costs ~1.5 cycles
// each, which would dominate a 12-instruction-per-vector kernel. The init
// therefore records the body into replay-buffer slot 0 (without executing it,
// outside the per-tile loop) and the face body is just four REPLAYs.
// =====================================================================================================================

// Re-centering factor for the SFPARECIP bias, folded into the cubic.
constexpr float SIGMOID_FAST_SCALE = 0.9977700f;
// Minimax cubic fit of 2^f / (1+f) in m = 1+f, pre-scaled by SIGMOID_FAST_SCALE.
constexpr float SIGMOID_FAST_C0 = 1.7755990f * SIGMOID_FAST_SCALE;
constexpr float SIGMOID_FAST_C1 = -1.3772819f * SIGMOID_FAST_SCALE;
constexpr float SIGMOID_FAST_C2 = 0.70746325f * SIGMOID_FAST_SCALE;
constexpr float SIGMOID_FAST_C3 = -0.10644398f * SIGMOID_FAST_SCALE;

constexpr auto sigmoid_fast_bits_ = [](float x) constexpr { return __builtin_bit_cast(std::uint32_t, x); };
constexpr auto sigmoid_fast_lo16_ = [](float x) constexpr {
    return static_cast<std::uint16_t>(sigmoid_fast_bits_(x) & 0xFFFFu);
};
constexpr auto sigmoid_fast_hi16_ = [](float x) constexpr {
    return static_cast<std::uint16_t>(sigmoid_fast_bits_(x) >> 16);
};

// Register map
//   LREG0/1  u   (chain a / chain b)      LREG11  C0     (programmable constant)
//   LREG2/3  1+f                          LREG12  -256*log2(e)
//   LREG4/5  rho accumulator              LREG13  256*127
//   LREG6    C1                           LREG14  C3
//   LREG7    C2                           LREG9/10  hardwired 0.0 / 1.0
constexpr unsigned SIGMOID_FAST_BODY_LEN = 25;

// Two vectors, chains alternating instruction by instruction.
sfpi_inline void _sigmoid_bf16_fast_body_() {
    TTI_SFPLOAD(0, 0, ADDR_MOD_7, 0);       // u = x
    TTI_SFPLOAD(1, 0, ADDR_MOD_7, 2);
    TTI_SFPMAD(0, 12, 13, 0, 0);            // s = 256*127 - x*256*log2(e)
    TTI_SFPMAD(1, 12, 13, 1, 0);
    TTI_SFPSWAP(0, 0, 9, 1);                // s = max(s, 0)
    TTI_SFPSWAP(0, 1, 9, 1);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 0, 6);    // s -> uint16, round to nearest
    TTI_SFP_STOCH_RND(0, 0, 0, 1, 1, 6);
    TTI_SFPSHFT(15, 0, 0, 5);               // ylin = bitcast(s_int << 15)
    TTI_SFPSHFT(15, 1, 1, 5);
    TTI_SFPSETEXP(127, 0, 2, 1);            // m = 1 + f
    TTI_SFPSETEXP(127, 1, 3, 1);
    TTI_SFPMAD(14, 2, 7, 4, 0);             // rho = C3*m + C2
    TTI_SFPMAD(14, 3, 7, 5, 0);
    TTI_SFPMAD(4, 2, 6, 4, 0);              // rho = rho*m + C1
    TTI_SFPMAD(5, 3, 6, 5, 0);
    TTI_SFPMAD(4, 2, 11, 4, 0);             // rho = rho*m + C0
    TTI_SFPMAD(5, 3, 11, 5, 0);
    TTI_SFPMAD(0, 4, 10, 0, 0);             // d = ylin*rho + 1
    TTI_SFPMAD(1, 5, 10, 1, 0);
    TTI_SFPARECIP(0, 0, 0, 0);              // y = 1/d
    TTI_SFPARECIP(0, 1, 1, 0);
    TTI_SFPSTORE(0, 0, ADDR_MOD_7, 0);      // rounds fp32 -> bf16 to nearest
    TTI_SFPSTORE(1, 0, ADDR_MOD_7, 2);
    TTI_INCRWC(0, 4, 0, 0);                 // advance DST by two vectors
}

inline void _init_sigmoid_bf16_fast_() {
    // Self-contained: re-assert the common SFPU state (config reg + ADDR_MOD_7 + RWC reset) as exp_init does,
    // then program this kernel's constants and record its replay body.
    _init_sfpu_config_reg();
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    math::reset_counters(p_setrwc::SET_ABD_F);

    sfpi::vConstFloatPrgm0 = -369.3299304595426f;  // LREG12 = -256*log2(e)
    sfpi::vConstFloatPrgm1 = 32512.0f;             // LREG13 = 256*127
    sfpi::vConstFloatPrgm2 = SIGMOID_FAST_C3;      // LREG14
    // LREG11 is the fourth programmable constant register (config path, staged through LREG0).
    TTI_SFPLOADI(0, 0xA, sigmoid_fast_lo16_(SIGMOID_FAST_C0));
    TTI_SFPLOADI(0, 0x8, sigmoid_fast_hi16_(SIGMOID_FAST_C0));
    TTI_SFPCONFIG(0, 11, 0);
    sfpi::l_reg[sfpi::LRegs::LReg6] = sfpi::vFloat(SIGMOID_FAST_C1);
    sfpi::l_reg[sfpi::LRegs::LReg7] = sfpi::vFloat(SIGMOID_FAST_C2);

    // Record the body once, here, so the per-face body issues one instruction
    // per two vectors instead of twenty-five.
    lltt::record<lltt::NoExec>(0, SIGMOID_FAST_BODY_LEN);
    _sigmoid_bf16_fast_body_();
}

inline void _calculate_sigmoid_bf16_fast_() {
    lltt::replay(0, SIGMOID_FAST_BODY_LEN);
    lltt::replay(0, SIGMOID_FAST_BODY_LEN);
    lltt::replay(0, SIGMOID_FAST_BODY_LEN);
    lltt::replay(0, SIGMOID_FAST_BODY_LEN);
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_sigmoid() {
    if constexpr (!APPROXIMATION_MODE && !is_fp32_dest_acc_en && ITERATIONS == 8) {
        _calculate_sigmoid_bf16_fast_();
        return;
    }
    if constexpr (!APPROXIMATION_MODE) {
        if constexpr (!is_fp32_dest_acc_en) {
            // ITERATIONS != 8: sigmoid_init<false, false> cannot see ITERATIONS and has programmed the fast
            // kernel's LREG12 over the 2.0f that sfpu_reciprocal_iter (inside _sfpu_sigmoid_) reads; re-seed it.
            sfpu_reciprocal_init<false>();
        }
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat val = sfpi::dst_reg[0];
            sfpi::vFloat result = _sfpu_sigmoid_<is_fp32_dest_acc_en>(val);
            if constexpr (!is_fp32_dest_acc_en) {
                result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
            }

            sfpi::dst_reg[0] = result;
            sfpi::dst_reg++;
        }
    } else {
        calculate_sigmoid_appx<ITERATIONS>();
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void sigmoid_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    if constexpr (!APPROXIMATION_MODE) {
        sfpu_reciprocal_init<false>();
    } else {
        sigmoid_appx_init();
    }
    // bf16 non-approx: calculate_sigmoid<false, false, 8> runs the fast kernel; program its state last so its
    // LREG11..14 values win over sfpu_reciprocal_init's vConstFloatPrgm0 (the ITERATIONS != 8 fallback re-seeds).
    if constexpr (!APPROXIMATION_MODE && !is_fp32_dest_acc_en) {
        _init_sigmoid_bf16_fast_();
    }
}

}  // namespace sfpu
}  // namespace ckernel
