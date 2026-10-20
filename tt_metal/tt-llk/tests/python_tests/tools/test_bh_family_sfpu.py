# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Blackhole SFPU family (``bh/families/sfpu.py``) on real snippets.

Every snippet below is a verbatim excerpt of the Blackhole SFPU scope on main
(8a7beeeaede), stored under its real repo-relative path: whole functions (never
edited), the constants they use, and namespace lines around them. The tree also
gets the real ``cfg_defines.h`` and ``ckernel_gpr_map.h`` of the checkout. Expected
evidence lines are computed from the snippet text, so every expectation is an exact
``path:line``.
"""

import tempfile
import unittest
from pathlib import Path

from tools.llk_state_audit.bh import effects_db, resources
from tools.llk_state_audit.bh.families import sfpu as sfpu_family

REPO_ROOT = Path(__file__).resolve().parents[5]
SFPU_API = "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/"
SFPU_LLK = "tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/"
EXP = SFPU_API + "ckernel_sfpu_exp.h"
RECIP = SFPU_API + "ckernel_sfpu_recip.h"
GELU = SFPU_API + "ckernel_sfpu_gelu.h"
SIGMOID = SFPU_API + "ckernel_sfpu_sigmoid_appx.h"
TYPECAST = SFPU_API + "ckernel_sfpu_typecast.h"
SIGNBIT = SFPU_API + "ckernel_sfpu_signbit.h"
LOAD_CONFIG = SFPU_LLK + "ckernel_sfpu_load_config.h"
TOPK = SFPU_LLK + "ckernel_sfpu_topk.h"
SINKHORN = SFPU_LLK + "experimental/ckernel_sfpu_sinkhorn.h"
TRIANGLE = SFPU_LLK + "ckernel_sfpu_triangle_solve.h"
PARAMS = "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h"
EXP_API = "tt_metal/hw/inc/api/compute/eltwise_unary/exp.h"

SNIPPETS = {
    "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h": r"""namespace ckernel {
namespace sfpu {

/*
 * Implementation of _sfpu_exp_21f_bf16_ (same algorithm) with TTI intrinsics
 * This implementation is faster, and give comparable accuracy as _sfpu_exp_21f_bf16_
 * (~< 1 ULP).
 *
 * Requires _init_exponential_tti_bf16_() to have been called to configure
 * ADDR_MOD_6 (dest auto-increment by 2 on SFPSTORE) and to load:
 *   - LREG12 = 1/ln2 (sfpi::vConstFloatPrgm0)
 *   - LREG13 = c2    (sfpi::vConstFloatPrgm1)  — poly coeff 4.791750e-15f
 */
template <bool SCALE_EN, bool is_fp32_dest_acc_en, bool CLAMP_NEGATIVE, int ITERATIONS>
inline void _sfpu_exp_21f_bf16_tti_(const std::uint16_t exp_base_scale_factor) {
    // Iteration-invariant constants. Loaded once before the loop.
    //
    //   LREG5 = 127.0f                      (bias term in z = x/ln2 + 127)
    //   LREG6 = 7.839635491371155e-08f      (poly coeff c1)
    //   LREG7 = 1.0017248f                  (poly coeff c0)
    //   LREG12 = 1/ln2                      (programmable, set in init)
    //   LREG13 = 4.791750143340323e-15f     (poly coeff c2; programmable, set in init)
    //
    // In-loop scratch:
    //   LREG0 = val → integer-part work
    //   LREG1 = 255 (loaded inside loop) → exexp result → frac (int) → frac (float) → poly result
    //   LREG2 = poly accumulator
    //   LREG3 = xlog2 (preserved through int-part work) → mask
    TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_FLOATB, 0x42fe);

    TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_UPPER, 0x33a8);
    TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_LOWER, 0x5ada);

    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_FLOATA, 0x3c02);

    // Number of instructions in one iteration of the loop body. Used by the
    // load_replay_buf record and TTI_REPLAY/replay below; MUST match exactly the
    // count of TTI_ instructions emitted by the recorded lambda, or the replay
    // buffer will misalign.
    //
    //   Base body:                                                15
    //     SFPLOAD, SFPMAD, SFPLOADI(255), SFPSWAP,
    //     SFPEXEXP, SFPEXMAN8, SFPSHFT, SFPEXMAN9, SFPCAST,
    //     SFPMAD poly1, SFPGT mask, SFPMAD poly2,
    //     SFPAND, SFPSETEXP, SFPSTORE.
    //   + SCALE_EN ? 1 : 0                  (SFPMULI scale)
    //   + is_fp32_dest_acc_en ? 0 : 1       (SFP_STOCH_RND fp32→bf16)
    constexpr int BODY_LEN = 15 + (SCALE_EN ? 1 : 0) + (is_fp32_dest_acc_en ? 0 : 1);

    // Record the loop body into replay buffer slot 0 the first time
    // through. Subsequent iterations replay the recorded sequence, which
    // shrinks the unrolled kernel binary from ~ITERATIONS*BODY_LEN
    // instructions down to BODY_LEN + (ITERATIONS - 1) replays.
    //
    // Per-element runtime is unchanged: the recorded instructions execute
    // exactly as if they had been issued inline, and dst_reg is advanced
    // by ADDR_MOD_6 inside the body so each replay walks to the next
    // element correctly.
    //
    // The accurate path (APPROXIMATION_MODE=false) does not otherwise use
    // the replay buffer, so slot 0 is free here. Callers that mix this
    // function with other replay-buffer clients should ensure they don't
    // require slot 0 to survive across the call.
    //
    // The record goes through load_replay_buf, which brackets it with
    // disable_gathering()/enable_gathering() when ENABLE_GATHERING is defined
    // and compiles to the bare record otherwise.
    load_replay_buf<Exec>(0 /*start*/, BODY_LEN, [exp_base_scale_factor] {
        // val = sfpi::dst_reg[0]
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);

        if constexpr (SCALE_EN) {
            // Multiply LREG0 by the BF16 scale immediate in-place.
            TTI_SFPMULI(exp_base_scale_factor, p_sfpu::LREG0, 0);
        }

        // xlog2 = val * (1/ln2) + 127.0f, into LREG3 (preserved past int-part work).
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG12, p_sfpu::LREG5, p_sfpu::LREG3, 0);

        // LREG1 = 255.0f. Slots into the SFPMAD's 2-cycle latency window.
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0x437f);

        // Upper clamp. SFPSWAP (mode VEC_MIN_MAX = "max into lreg_dest"):
        //   LREG1 = max(255, xlog2), LREG3 = min(255, xlog2).
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);

        // _float_to_int32_for_exp_21f_: shift mantissa left by exp-bias bits.
        // Reads xlog2 from LREG3, leaves int_part in LREG0 (LREG0 freed of val).
        TTI_SFPEXEXP(0, p_sfpu::LREG3, p_sfpu::LREG1, 0);  // LREG1 = exexp(xlog2)
        TTI_SFPEXMAN(0, p_sfpu::LREG3, p_sfpu::LREG0, 0);  // LREG0 = exman8(xlog2)
        TTI_SFPSHFT(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);   // LREG0 <<= LREG1   (int_part)

        // Extract fractional part (sfpi::exman9 with PAD9). LREG0 still holds
        // the integer-part-as-float-encoding which feeds SETEXP later.
        TTI_SFPEXMAN(0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPEXMAN_MOD1_PAD9);

        // frac = convert<vFloat>(fractional_part, RoundMode::Nearest)
        constexpr unsigned SFPCAST_MOD1_SM32_TO_FP32_RNE = 0;
        TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, SFPCAST_MOD1_SM32_TO_FP32_RNE);

        // Polynomial refinement of 2^x_f on [0, 1] in Horner form:
        //   frac = c0 + frac * (c1 + frac * c2)
        //        = 1.0017248 + frac * (7.84e-08 + frac * 4.79e-15)
        TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG13, p_sfpu::LREG6, p_sfpu::LREG2, 0);

        // Negative-input handling: instead of clamping xlog2 with a max(0, ·)
        // SFPSWAP, we build a mask with (SFPGT) and mask negative value using
        // (SFPAND) below.
        // This avoids SFPLOADI + SFPSWAP and introduced instructions
        // can be interleaved with SFPMAD to hide their latency.
        constexpr unsigned SFPGT_MOD1_SET_VD = 8;
        TTI_SFPGT(0, p_sfpu::LCONST_0, p_sfpu::LREG3, SFPGT_MOD1_SET_VD);

        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LREG7, p_sfpu::LREG1, 0);

        // Apply the mask to the integer part *before* SETEXP: for lanes with
        // xlog2 <= 0 the mask is 0, zeroing the int_part. The subsequent
        // SETEXP then produces a bf16 subnormal that flushes to 0, matching
        // the scalar max(xlog2, 0) behavior on finite inputs.
        constexpr unsigned SFPAND_MOD1_USE_VB = 1;
        TTI_SFPAND(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LREG0, SFPAND_MOD1_USE_VB);

        // y = setexp(frac, masked_int_part) — recombine 2^x_i * 2^x_f.
        constexpr unsigned SFPSETEXP_MOD1_ARG_EXPONENT = 2;
        TTI_SFPSETEXP(0, p_sfpu::LREG1, p_sfpu::LREG0, SFPSETEXP_MOD1_ARG_EXPONENT);

        if constexpr (!is_fp32_dest_acc_en) {
            // Round float32 -> bfloat16 using round-to-nearest before
            // SFPSTORE truncates. Avoids ULP loss on values like 9*9 = 80.8.
            TTI_SFP_STOCH_RND(
                sfpi::SFPSTOCHRND_RND_EVEN,
                0,
                p_sfpu::LREG0,
                p_sfpu::LREG0,
                p_sfpu::LREG0,
                sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        }

        // sfpi::dst_reg[0] = y; sfpi::dst_reg++;
        // (ADDR_MOD_6 increments dest by 2 on store.)
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 0);
    });

#pragma GCC unroll 8
    for (std::uint32_t i = 1; i < ITERATIONS; i++) {
        // Replay the recorded body for the remaining ITERATIONS - 1 elements.
        // Each replay is one REPLAY-equivalent issue; the body executes as
        // if it had been issued inline, dst_reg advancing via ADDR_MOD_6.
        TTI_REPLAY(0, BODY_LEN, 0, 0);
    }
}

template <
    bool APPROXIMATION_MODE,
    bool is_fp32_dest_acc_en,
    bool SCALE_EN = false,
    int ITERATIONS = 8,
    bool CLAMP_NEGATIVE = true>
void calculate_exponential(const std::uint32_t exp_base_scale_factor = p_sfpu::kCONST_1_FP16B) {
    if constexpr (!APPROXIMATION_MODE) {
        if constexpr (!is_fp32_dest_acc_en) {
            // bfloat16-accurate path: hand-tuned TTI exp_21f kernel.
            // CLAMP_NEGATIVE is forwarded for API symmetry with the WH kernel,
            // but on BH the negative-input handling is applied unconditionally
            // because the SFPGT mask hides in an SFPMAD latency window and is
            // effectively free (no other instruction to interleave with SFPMAD).
            _sfpu_exp_21f_bf16_tti_<SCALE_EN, is_fp32_dest_acc_en, CLAMP_NEGATIVE, ITERATIONS>(exp_base_scale_factor);
        } else {
            for (int d = 0; d < ITERATIONS; d++) {
                sfpi::vFloat val = sfpi::dst_reg[0];
                sfpi::dst_reg[0] =
                    _ckernel_sfpu_exp_accurate_<SCALE_EN, is_fp32_dest_acc_en>(val, exp_base_scale_factor);
                sfpi::dst_reg++;
            }
        }
    } else if constexpr (APPROXIMATION_MODE && CLAMP_NEGATIVE) {
#ifdef DISABLE_SFPLOADMACRO
        for (int d = 0; d < ITERATIONS; d++) {
            TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, 0);
            TTI_SFPSWAP(0, p_sfpu::LREG14, p_sfpu::LREG0, 9);
            TTI_SFPMAD(p_sfpu::LREG12, p_sfpu::LREG0, p_sfpu::LREG13, p_sfpu::LREG0, 0);
            TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPSTOCHRND_MOD1_FP32_TO_UINT16);
            TTI_SFPSHFT(15, p_sfpu::LREG0, p_sfpu::LREG0, 1);
            TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, 0);
            sfpi::dst_reg++;
        }
#else
        // Code below is hand-unrolled for 8 iterations
        // so it doesn't respect ITERATIONS. TODO: tt-llk#1486
        // static_assert(ITERATIONS == 8);

        // Sanitize the input values by loading from DEST, comparing against the value -88.5, and if the input value is
        // more negative than that, swap the input value with -88.5 and store back to DEST
        //  - in other words, after the sanitize step, the values in DEST will be in the range {-88.5 , +inf}

        // Macro Sequence Register 1 configured to read back in the original values from dest, sanitize them to a range
        // we can handle, and then store them back to dest
        //  LD     : bring in the original value from DEST (y)
        //  MAD    : unused
        //  ROUND  : unused
        //  SIMPLE : SWAP the larger value of y and -88.5 into the LREG
        //  STORE  : store the sanitized value back to dest
        TTI_SFPLOADMACRO(
            4,
            0,
            ADDR_MOD_7,
            0);      // MACRO Sequence Register 1: LD, SWAP, STORE - uses LREG[0] for loaded value - Dest offset  0 is
                     // targeting the even columns for rows   3: 0
        TTI_SFPNOP;  // NOP is necessary because the SWAP operation takes 2 cycles and unfortunately is not pipelined
        TTI_SFPLOADMACRO(
            5,
            0,
            ADDR_MOD_7,
            2);  // MACRO Sequence Register 1: LD, SWAP, STORE - uses LREG[1] for loaded value - Dest offset  2 is
                 // targeting the odd  columns for rows   3: 0
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(
            6,
            0,
            ADDR_MOD_7,
            4);  // MACRO Sequence Register 1: LD, SWAP, STORE - uses LREG[2] for loaded value - Dest offset  4 is
                 // targeting the even columns for rows   7: 4
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(
            7,
            0,
            ADDR_MOD_7,
            6);  // MACRO Sequence Register 1: LD, SWAP, STORE - uses LREG[3] for loaded value - Dest offset  6 is
                 // targeting the odd  columns for rows   7: 4
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(
            4,
            0,
            ADDR_MOD_7,
            8);  // MACRO Sequence Register 1: LD, SWAP, STORE - uses LREG[0] for loaded value - Dest offset  8 is
                 // targeting the even columns for rows  11: 8
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(
            5,
            0,
            ADDR_MOD_7,
            10);  // MACRO Sequence Register 1: LD, SWAP, STORE - uses LREG[1] for loaded value - Dest offset 10 is
                  // targeting the even columns for rows  11: 8
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(
            6,
            0,
            ADDR_MOD_7,
            12);  // MACRO Sequence Register 1: LD, SWAP, STORE - uses LREG[2] for loaded value - Dest offset 12 is
                  // targeting the odd  columns for rows  15:12
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(
            7,
            0,
            ADDR_MOD_7,
            14);  // MACRO Sequence Register 1: LD, SWAP, STORE - uses LREG[3] for loaded value - Dest offset 14 is
                  // targeting the even columns for rows  15:12
        // NOP not needed in this spot because the next LoadMacro is a computational macro which doesn't immediately use
        // the SIMPLE unit

        // Macro Sequence Register 0 configured to read back in the sanitized values and calculate the approximate
        // exponential value
        //  LD     : the sanitized value from DEST (y)
        //  MAD    : compute (A * y) + (B-C)  , where A = (2^8)/ln(2) , B = 127 * (2^8) , C = Adjustment parameter of
        //  roughly 11.2 to minimize error ROUND  : convert the MAD result from FP32 to a 16-bit unsigned integer using
        //  stochastic rounding SIMPLE : shift the 16-bit integer to the left by 15 bits to place the MSB of the
        //  computed value into the MSB of the exponent bits of the fp32 format STORE  : store the shifted value back to
        //  dest
        TTI_SFPLOADMACRO(0, 0, ADDR_MOD_7, 0);  // MACRO Sequence Register 0: LD, MAD, ROUND, SHIFT and STORE - uses
                                                // LREG[0] for loading and intermediate results
                                                // - Dest offset  0 is targeting the even columns for rows   3: 0
        TTI_SFPLOADMACRO(1, 0, ADDR_MOD_7, 2);  // MACRO Sequence Register 0: LD, MAD, ROUND, SHIFT and STORE - uses
                                                // LREG[1] for loading and intermediate results
                                                // - Dest offset  2 is targeting the odd  columns for rows   3: 0
        TTI_SFPLOADMACRO(2, 0, ADDR_MOD_7, 4);  // MACRO Sequence Register 0: LD, MAD, ROUND, SHIFT and STORE - uses
                                                // LREG[2] for loading and intermediate results
                                                // - Dest offset  4 is targeting the even columns for rows   7: 4
        TTI_SFPLOADMACRO(3, 0, ADDR_MOD_7, 6);  // MACRO Sequence Register 0: LD, MAD, ROUND, SHIFT and STORE - uses
                                                // LREG[3] for loading and intermediate results
                                                // - Dest offset  6 is targeting the odd  columns for rows   7: 4
        TTI_SFPLOADMACRO(0, 0, ADDR_MOD_7, 8);  // MACRO Sequence Register 0: LD, MAD, ROUND, SHIFT and STORE - uses
                                                // LREG[0] for loading and intermediate results
                                                // - Dest offset  8 is targeting the even columns for rows  11: 8
        TTI_SFPLOADMACRO(
            1,
            0,
            ADDR_MOD_7,
            10);  // MACRO Sequence Register 0: LD, MAD, ROUND, SHIFT and STORE - uses LREG[1] for loading and
                  // intermediate results - Dest offset 10 is targeting the even columns for rows  11: 8
        TTI_SFPLOADMACRO(
            2,
            0,
            ADDR_MOD_7,
            12);  // MACRO Sequence Register 0: LD, MAD, ROUND, SHIFT and STORE - uses LREG[2] for loading and
                  // intermediate results - Dest offset 12 is targeting the odd  columns for rows  15:12
        TTI_SFPLOADMACRO(
            3,
            0,
            ADDR_MOD_7,
            14);  // MACRO Sequence Register 0: LD, MAD, ROUND, SHIFT and STORE - uses LREG[3] for loading and
                  // intermediate results - Dest offset 14 is targeting the even columns for rows  15:12
        // NOP needed to allow time for the final Computation Loadmacro to complete before returning to the Sanitation
        // Loadmacro at the top for the next iteration
        //  - to be completely safe, use 3 NOP; in practice 1 seems to be enough, probably because the overhead of the
        //  DEST INCRW stuff introduces 2 cycles of delay
        TTI_SFPNOP;
        // TTI_SFPNOP;
        // TTI_SFPNOP;
#endif
    }
#ifdef DISABLE_SFPLOADMACRO
    else if constexpr (APPROXIMATION_MODE) {
        for (int d = 0; d < ITERATIONS; d++) {
            TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, 0);
            TTI_SFPMAD(p_sfpu::LREG12, p_sfpu::LREG0, p_sfpu::LREG13, p_sfpu::LREG0, 0);
            TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPSTOCHRND_MOD1_FP32_TO_INT16);
            TTI_SFPSHFT2(p_sfpu::LREG0, p_sfpu::LREG14, p_sfpu::LREG1, 5);  // lreg[1] = lreg[0] << 15
            TTI_SFPSETSGN(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);  // lreg[0] preserves sign, copies e/m from lreg[1]
            TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, 0);
            sfpi::dst_reg++;
        }
    }
#else
    else if constexpr (APPROXIMATION_MODE && ITERATIONS == 8) {
        // =======================================================================
        // 8-element version using replay buffer.
        // Total: ~20 cycles for 8 elements = 2.5 cycles/element
        //
        // Uses 1 replay of the 16-instruction pattern.
        // First 2 SHFT2s are dummy (timing placeholders), then 6 real SHFT2s.
        // Drain phase handles final 2 SHFT2s.
        // =======================================================================

        // Configure ADDR_MOD_7 for auto-increment (dest += 2 per LOADMACRO).
        addr_mod_t{
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 2},
        }
            .set(ADDR_MOD_7);

        // Single replay of 16 instructions = 8 LM + 8 SHFT2 (2 dummy + 6 real).
        lltt::replay(0, 16);

        // Drain: SHFT2[6-7].
        TTI_SFPNOP;
        TTI_SFPSHFT2(p_sfpu::LREG2, p_sfpu::LREG14, p_sfpu::LREG4, 5);  // SHFT2[6]
        TTI_SFPNOP;
        TTI_SFPSHFT2(p_sfpu::LREG3, p_sfpu::LREG14, p_sfpu::LREG4, 5);  // SHFT2[7]
        TTI_SFPNOP;
        TTI_SFPNOP;
    } else if constexpr (APPROXIMATION_MODE && ITERATIONS == 32) {
        // =======================================================================
        // 32-element version using replay buffer.
        // Total: ~68 cycles for 32 elements = 2.125 cycles/element
        //
        // Uses 2 replays of the 32-instruction pattern.
        // First 2 SHFT2s are dummy (timing placeholders), then 30 real SHFT2s.
        // Drain phase handles final 2 SHFT2s.
        // =======================================================================

        // Configure ADDR_MOD_7 for auto-increment (dest += 2 per LOADMACRO).
        addr_mod_t{
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 2},
        }
            .set(ADDR_MOD_7);

        // 2 replays of 32 instructions = 32 LM + 32 SHFT2 (2 dummy + 30 real).
        lltt::replay(0, 32);
        lltt::replay(0, 32);

        // Drain: SHFT2[30-31].
        TTI_SFPNOP;
        TTI_SFPSHFT2(p_sfpu::LREG2, p_sfpu::LREG14, p_sfpu::LREG4, 5);  // SHFT2[30]
        TTI_SFPNOP;
        TTI_SFPSHFT2(p_sfpu::LREG3, p_sfpu::LREG14, p_sfpu::LREG4, 5);  // SHFT2[31]
        TTI_SFPNOP;
        TTI_SFPNOP;
    } else if constexpr (APPROXIMATION_MODE) {
        static_assert(
            ITERATIONS == 8 || ITERATIONS == 32, "This version of exponential only supports 8 or 32 iterations.");
    }
#endif
}

constexpr auto bits = [](float x) constexpr { return __builtin_bit_cast(std::uint32_t, x); };
constexpr auto lo16 = [](float x) constexpr { return static_cast<std::uint16_t>(bits(x) & 0xFFFFu); };
constexpr auto hi16 = [](float x) constexpr { return static_cast<std::uint16_t>(bits(x) >> 16); };

template <
    bool APPROXIMATION_MODE,
    std::uint32_t scale,
    bool CLAMP_NEGATIVE,
    bool is_fp32_dest_acc_en>
void exp_init() {
    // Common SFPU init inlined (SFPU config register + ADDR_MOD_7 + counter reset), then the op-specific
    // exp setup below -- one self-contained init, no separate shared-common-init call. Same functionality as
    // _llk_math_eltwise_unary_sfpu_init_<exponential>() (exp uses only ADDR_MOD_7, no op-specific ADDR_MOD_6).
    sfpu::_init_sfpu_config_reg();
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    math::reset_counters(p_setrwc::SET_ABD_F);
    if constexpr (APPROXIMATION_MODE && CLAMP_NEGATIVE) {
        // Algorithm is adapted from:
        //      A Fast, Compact Approximation of the Exponential Function
        //      Nicol N. Schraudolph
        //      IDSIA, Lugano, Switzerland

        // First, set up constant values which are needed for the computation
        //      We will first sanitize the input values (y) to be in the range that won't cause underflow, which for our
        //      hardware means we need to limit negative values to be greater than or equal to -88.5 The computation
        //      that is needed is (A * y) + (B - C) , where A = (2^8)/ln(2) , B = 127 * (2^8) , C = Adjustment parameter
        //      of roughly 11.2 to minimize error
        //          - NOTE: we would like to be able to use 2^23 instead of 2^8 and compute a 32-bit quantity, but our
        //          hardware only supports rounding FP32 into a 16-bit integer, so we use 2^8 and then shift left by 15
        //          bits after rounding
        //      So we will set up the following constants:
        //          LREG[14] =       =    -88.5               = 0xc2b10000
        //          LREG[12] = A     =    369.329925537109375 = 0x43b8aa3b
        //          LREG[13] = (B-C) =  32500.818359375       = 0x46fde9a3

        constexpr float LN2_RECIP = 1.4426950408889634f;
        constexpr float A = 256.0f * LN2_RECIP;
        constexpr float B_minus_C = 32500.818359375f;
        constexpr float THRESHOLD = -88.5f;

        constexpr float scale_fp32 = __builtin_bit_cast(float, scale);

        constexpr float A_scaled = A * scale_fp32;
        constexpr float THRESHOLD_scaled = THRESHOLD / scale_fp32;

        TTI_SFPLOADI(0, 0xA, lo16(THRESHOLD_scaled));
        TTI_SFPLOADI(0, 0x8, hi16(THRESHOLD_scaled));
        TTI_SFPCONFIG(0, 14, 0);  // SFPCONFIG Dest 14 = LREG[14] =            -88.5               = 0xc2b10000

        TTI_SFPLOADI(0, 0xA, lo16(A_scaled));
        TTI_SFPLOADI(0, 0x8, hi16(A_scaled));
        TTI_SFPCONFIG(0, 12, 0);  // SFPCONFIG Dest 12 = LREG[12] = A     =    369.329925537109375 = 0x43b8aa3b

        TTI_SFPLOADI(0, 0xA, lo16(B_minus_C));
        TTI_SFPLOADI(0, 0x8, hi16(B_minus_C));
        TTI_SFPCONFIG(0, 13, 0);  // SFPCONFIG Dest 13 = LREG[13] = (B-C) =  32500.818359375       = 0x46fde9a3

#ifndef DISABLE_SFPLOADMACRO
        // Next, set up the macro instructions which will be necessary
        //  - for the sanitize function: we will need a SWAP instruction
        //  - for the main computation function: we will need MAD, ROUND, and SHIFT instructions

        // There are two ways to program the macro instruction registers, and this setup leverages both ways
        //  - we can either use the SFPCONFIG flow, by setting up the bits of the instruction into LREG[0] and then
        //  targeting the Macro instruction register
        //  - or we can use the shortcut / backdoor load method which relies on having some illegal destination register
        //  values as part of the instruction

        // Use SFPCONFIG method for the SWAP instruction, since we want the SWAP itself to use a destination register
        // which is not normally a legal value
        //      (we are cheating a bit here, since we only care about one half of the swap and we want to use a constant
        //      for the other half)
        //
        //              imm12 = 0,       lreg_src_c = 0 (will be fed by value loaded from Dest into Loadmacro
        //              lreg_dest),  lreg_dest = LREG[14] = - 88.5, instr_mod1 = 1 swap the values with the larger of
        //              the two ending up in lreg_dest -> but we will use the Loadmacro lreg_dest register as output
        // TTI_SFP_SWAP(0,               0, 14,                            1);
        TTI_SFPLOADI(0, 0xA, 0x00E1);
        TTI_SFPLOADI(0, 0x8, 0x9200);
        TTI_SFPCONFIG(
            0, 0, 0);  // SFPCONFIG Dest 0 = Programmable Macro instruction 0: TTI_SFPSWAP(0, 0, 14, 1); // compare
                       // against LREG[14] (-88.5), and put the larger value into LREG[loadmacro_lreg_dest]
        TTI_SFPNOP;

        // Backdoor load of Macro Instruction 1
        // Dummy version of MAD instruction with lreg_dest = 4'b11_01 = 13 to install into Programmable Macro
        // instruction register 1, which is Macro Instruction Register 5
        TTI_SFPMAD(12, 0, 13, 13, 0);  // MACRO Instruction 1 <--- lreg X = lreg[12] (A) * lreg[0] (y) + lreg[13] (B-C)

        // Backdoor load of Macro Instruction 2
        // ROUND instruction to convert FP32 result into an integer value (int16)
        //                Stochastic = 0,  Imm(Descale),  SrcB(unused),   SrcC(input value),  Lreg_dest = 14 to install
        //                in Programmable Macro Instruction reg 2'b10,  instr_mod1 = 14 to treat input as fp32, output
        //                as unsigned int16, use imm as descale
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 14, 14);  // Round to unsigned Int16

        // Backdoor load of Macro Instruction 3
        // If using the unsigned int rounding mode, then shift by 15; SHL to move integer bits to exponent;
        TTI_SFPSHFT(
            15,
            0,
            15,
            1);  // imm = 15 to shift left by 15 bits; lreg_c = 0 (will use macro reg); lreg_dest = 15 to install in
                 // Programmable Macro Instruction reg 2'b11, which is Macro Instruction Register 7

        // So at this point, we have the following instructions loaded into our macro registers:
        //
        // 00: (no macro instruction, just execute whatever is issued from Tensix) <-- these are fixed / not
        // programmable 01: ( Rsvd                                                            ) <-- these are fixed /
        // not programmable 02: ( NOP                                                             ) <-- these are fixed
        // / not programmable 03: ( SFPSTORE                                                        ) <-- these are
        // fixed / not programmable 04: TTI_SFPSWAP       (0, 0, 11, 1) 05: TTI_SFPMAD        (12, 0, 13, 13, 0) 06:
        // TTI_SFP_STOCH_RND (1, 0, 0, 0, 14, 14) 07: TTI_SFPSHFT       (15,0,15,1)

        // Now we want to set up our two sequences

        // Sequence 1 setup: we want to Load, SWAP, <delay>, Store
        //       Delay slot:                  0     1        2
        //                                                                                                                                                                                                 Use
        //                                                                                                                                                                                                 Loaded  Result          Macro
        //                                                                                                                                                                                                 Value   Value   Delay   Instruction
        //                                                                                                                                                                                                 SRCB    Stage   Slot    Select
        TTI_SFPLOADI(
            0,
            0xA,
            0x0004);  // slot1 : SIMPLE UNIT, want SWAP  instruction which is in macro instruction mux[4], delayed by 0
                      // ; not using staging flop as dest; not using load reg as srcb : 8'b0_______0_______000_____100
                      // = 0x04 slot2 : MAD    UNIT, unused : 8'b0_______0_______000_____000          = 0x00
        TTI_SFPLOADI(
            0, 0x8, 0x1300);  // slot3 : ROUND  UNIT, unused : 8'b0_______0_______000_____000          = 0x00 slot4 :
                              // STORE  UNIT, want STORE instruction which is in macro instruction mux[3], delayed by 2
                              // ; not using staging flop as src ; : 8'b0_______0_______010_____011          = 0x13
        TTI_SFPCONFIG(0, 5, 0);  // SFPCONFIG Dest 5 = Macro Sequence Register 1

        // Sequence 0 setup: we want to Load, MAD, <delay>, ROUND, SHIFT, Store
        //       Delay slot:                  0    1        2      3      4
        //                                                                                                                                                                                                 Use
        //                                                                                                                                                                                                 Loaded  Result          Macro
        //                                                                                                                                                                                                 Value   Value   Delay   Instruction
        //                                                                                                                                                                                                 SRCB    Stage   Slot    Select
        TTI_SFPLOADI(
            0,
            0xA,
            0x85DF);  // slot1 : SIMPLE UNIT, want SHIFT instruction which is in macro instruction mux[7], delayed by 3
                      // ;     using staging flop as dest; using load reg as srcb : 8'b1_______1_______011_____111 =
                      // 0xDF slot2 : MAD    UNIT, want MAD   instruction which is in macro instruction mux[5], delayed
                      // by 0 ; not using staging flop as dest;     using load reg as srcb :
                      // 8'b1_______0_______000_____101 = 0x85
        TTI_SFPLOADI(
            0,
            0x8,
            0x6316);  // slot3 : ROUND  UNIT, want ROUND instruction which is in macro instruction mux[6], delayed by 2
                      // ; not using staging flop as dest; using : 8'b0_______0_______010_____110          = 0x16 slot4
                      // : STORE  UNIT, want STORE instruction which is in macro instruction mux[3], delayed by 4 ;
                      // using staging flop as src ;     using                  : 8'b0_______1_______100_____011 = 0x63
        TTI_SFPCONFIG(0, 4, 0);  // Load it into macro sequence register 0 (destination = 4)

        // Reset LoadMacroConfig[Lane].Misc for all lanes, in case it has been previously set by another use of macros.
        TTI_SFPCONFIG(0, 8, 1);
#endif
    } else if constexpr (APPROXIMATION_MODE) {
        // ===================================================================
        // Based on "A Fast, Compact Approximation of the Exponential Function" by Schraudolph.
        //
        // The Schraudolph algorithm computes exp(x) by exploiting the fact that IEEE 754 floats
        // encode values as 2^(exponent) * (1 + mantissa), where the bit-pattern read as an integer
        // is linear in log2(value). This allows exp(x) to be computed as i = A*x + (B-C)
        // when reinterpreted as float.
        // This implementation adds an explicit sign-setting step (SETSGN) to ensure outputs are
        // negative for inputs below ~-88, where the algorithm would otherwise produce incorrect values.
        // To get a correct result for inputs below ~-88 the output of this function must be ReLU'd.
        // In this implementation, for inputs above 0.72 the output saturates to exp(0.72).
        // Valid input range: [-88, 0.72] with no following ReLU, [-inf, 0.72] with ReLU.
        //
        // Constants:
        //   LREG[12] = A = 256.0 * (1/ln2) = 369.329925537109375
        //   LREG[13] = B - C = 32500.818359375
        //   LREG[14] = 15 (shift amount for SFPSHFT2)
        //
        // Macro Instructions (backdoor loaded):
        //   Macro 5: MAD        - compute i = A * x + (B-C)
        //   Macro 6: STOCHRND   - convert to INT16 (sign-magnitude format)
        //   Macro 7: SETSGN     - restore sign from STOCHRND result to shifted value
        //
        // Macro Sequence Register 0
        //   Slot 1 (Simple): SETSGN @ delay 5, writes to LREG[16] (staging)
        //     - Bit 7 = 1: VB = loadmacro's VD (sign source from STOCHRND result)
        //     - Bit 6 = 1: VD = 16 (staging register, avoids write port conflict)
        //     - Bits 5:3 = 101: delay 5
        //     - Bits 2:0 = 111: macro 7 (SETSGN)
        //     - Encoding: 0b1_1_101_111 = 0xEF
        //
        //   Slot 2 (MAD): MAD @ delay 0
        //     - Bit 7 = 1: use loaded value as VB
        //     - Bit 6 = 0: write to LREG[lreg_dest]
        //     - Bits 5:3 = 000: delay 0
        //     - Bits 2:0 = 101: macro 5 (MAD)
        //     - Encoding: 0b1_0_000_101 = 0x85
        //
        //   Slot 3 (Round): STOCHRND @ delay 3
        //     - Bits 5:3 = 011: delay 3
        //     - Bits 2:0 = 110: macro 6 (STOCHRND)
        //     - Encoding: 0b0_0_011_110 = 0x1E
        //
        //   Slot 4 (Store): STORE @ delay 6, reads from LREG[16]
        //     - Bit 7 = 0: don't preserve VD from instruction
        //     - Bit 6 = 1: read from LREG[16] (staging register)
        //     - Bits 5:3 = 110: delay 6
        //     - Bits 2:0 = 011: macro 3 (STORE)
        //     - Encoding: 0b0_1_110_011 = 0x73
        //
        // =======================================================================

        constexpr float LN2_RECIP = 1.4426950408889634f;
        constexpr float A = 256.0f * LN2_RECIP;
        constexpr float B_minus_C = 32500.818359375f;

        constexpr float scale_fp32 = __builtin_bit_cast(float, scale);
        constexpr float A_scaled = A * scale_fp32;

        // Load constant A into LREG[12]
        TTI_SFPLOADI(0, 0xA, lo16(A_scaled));
        TTI_SFPLOADI(0, 0x8, hi16(A_scaled));
        TTI_SFPCONFIG(0, 12, 0);

        // Load constant (B-C) into LREG[13]
        TTI_SFPLOADI(0, 0xA, lo16(B_minus_C));
        TTI_SFPLOADI(0, 0x8, hi16(B_minus_C));
        TTI_SFPCONFIG(0, 13, 0);

        // Load shift amount (15) into LREG[14] for SFPSHFT2
        // SFPSHFT2 mode 5 reads shift amount from VC register
        TTI_SFPLOADI(0, 0xA, 15);  // Lower 16 bits = 15
        TTI_SFPLOADI(0, 0x8, 0);   // Upper 16 bits = 0
        TTI_SFPCONFIG(0, 14, 0);   // Store in LREG[14]

#ifndef DISABLE_SFPLOADMACRO
        // ===================================================================
        // Program Macro Instructions via Backdoor Load
        // ===================================================================

        // Macro Instruction 1 (slot 5): MAD
        // Computes: LREG[dest] = LREG[12] * LREG[dest] + LREG[13]
        // dest=13 triggers backdoor load to Macro Instruction Register 5
        TTI_SFPMAD(12, 0, 13, 13, 0);

        // Macro Instruction 2 (slot 6): STOCHRND with INT16 mode
        // Converts FP32 to sign-magnitude integer with max magnitude 32767
        // dest=14 triggers backdoor load to Macro Instruction Register 6
        // Mode 7 = FP32_TO_INT16 (keeps sign in bit 31, clamps magnitude to 32767)
        TTI_SFP_STOCH_RND(0, 0, 0, 0, 14, 7);

        // Macro Instruction 3 (slot 7): SETSGN
        // VC=4: reads exp/mantissa from LREG[4] (where discrete SHFT2 writes)
        // VD=15: triggers backdoor load to Macro Instruction Register 7
        // Mod1=0: sign comes from VB (which equals loadmacro's VD after override)
        //
        // When executed via LOADMACRO with sequence bits:
        //   - Bit 7=1: VB = loadmacro's VD (0-3, sign source from STOCHRND)
        //   - Bit 6=1: VD = 16 (staging register for output)
        //   - VC is preserved as 4 (exp/man source from SHFT2 result)
        TTI_SFPSETSGN(0, 4, 15, 0);

        //   Low 16 bits:  Slot2(MAD)=0x85, Slot1(Simple)=0xEF -> 0x85EF
        //   High 16 bits: Slot4(Store)=0x73, Slot3(Round)=0x1E -> 0x731E
        //   - STOCHRND delay 3: 0x1E
        //   - SETSGN delay 5: 0xEF
        //   - STORE delay 6: 0x73
        TTI_SFPLOADI(0, 0xA, 0x85EF);  // Slots 1-2: Simple=0xEF (delay 5), MAD=0x85
        TTI_SFPLOADI(0, 0x8, 0x731E);  // Slots 3-4: Round=0x1E (delay 3), Store=0x73 (delay 6)
        TTI_SFPCONFIG(0, 4, 0);        // Load into Macro Sequence Register 0 (dest=4)

        // Reset LoadMacroConfig[Lane].Misc for all lanes
        // Sets StoreMod0=0 (SRCB), UsesLoadMod0ForStore=0, UnitDelayKind=0xF
        // UnitDelayKind prevents pipeline advancement when not seeing new instructions,
        // avoiding race conditions from dest bank conflicts or other pipeline hiccups.
        TTI_SFPCONFIG(0xF00, 0x8, 0x1);

        // ===================================================================
        // Program Replay Buffer
        // ===================================================================
        // Record 32 instructions (16 LM+SHFT2 pairs) into replay buffer.
        // ADDR_MOD_7 will be configured for auto-increment at replay time.
        //
        // Pattern starts with LM(LREG0) so all LOADMACROs can be replayed.
        // The first two SHFT2s are "dummy" (operate on not-yet-loaded LREGs)
        // but provide correct pipeline timing for subsequent real SHFT2s.
        //
        // LREG pattern cycles every 4 elements:
        //   LM uses:    LREG0, LREG1, LREG2, LREG3, LREG0, ...
        //   SHFT2 uses: LREG2, LREG3, LREG0, LREG1, LREG2, ...
        // ===================================================================

        lltt::record(0, 32);

        // 16 pairs of LM + SHFT2 (LREG pattern repeats every 4 pairs)
        // Pairs 0-1: dummy SHFT2s, Pairs 2-15: real SHFT2s for elements 0-13
        TTI_SFPLOADMACRO(0, 0, ADDR_MOD_7, 0);
        TTI_SFPSHFT2(p_sfpu::LREG2, p_sfpu::LREG14, p_sfpu::LREG4, 5);
        TTI_SFPLOADMACRO(1, 0, ADDR_MOD_7, 0);
        TTI_SFPSHFT2(p_sfpu::LREG3, p_sfpu::LREG14, p_sfpu::LREG4, 5);
        TTI_SFPLOADMACRO(2, 0, ADDR_MOD_7, 0);
        TTI_SFPSHFT2(p_sfpu::LREG0, p_sfpu::LREG14, p_sfpu::LREG4, 5);
        TTI_SFPLOADMACRO(3, 0, ADDR_MOD_7, 0);
        TTI_SFPSHFT2(p_sfpu::LREG1, p_sfpu::LREG14, p_sfpu::LREG4, 5);
        TTI_SFPLOADMACRO(0, 0, ADDR_MOD_7, 0);
        TTI_SFPSHFT2(p_sfpu::LREG2, p_sfpu::LREG14, p_sfpu::LREG4, 5);
        TTI_SFPLOADMACRO(1, 0, ADDR_MOD_7, 0);
        TTI_SFPSHFT2(p_sfpu::LREG3, p_sfpu::LREG14, p_sfpu::LREG4, 5);
        TTI_SFPLOADMACRO(2, 0, ADDR_MOD_7, 0);
        TTI_SFPSHFT2(p_sfpu::LREG0, p_sfpu::LREG14, p_sfpu::LREG4, 5);
        TTI_SFPLOADMACRO(3, 0, ADDR_MOD_7, 0);
        TTI_SFPSHFT2(p_sfpu::LREG1, p_sfpu::LREG14, p_sfpu::LREG4, 5);
        TTI_SFPLOADMACRO(0, 0, ADDR_MOD_7, 0);
        TTI_SFPSHFT2(p_sfpu::LREG2, p_sfpu::LREG14, p_sfpu::LREG4, 5);
        TTI_SFPLOADMACRO(1, 0, ADDR_MOD_7, 0);
        TTI_SFPSHFT2(p_sfpu::LREG3, p_sfpu::LREG14, p_sfpu::LREG4, 5);
        TTI_SFPLOADMACRO(2, 0, ADDR_MOD_7, 0);
        TTI_SFPSHFT2(p_sfpu::LREG0, p_sfpu::LREG14, p_sfpu::LREG4, 5);
        TTI_SFPLOADMACRO(3, 0, ADDR_MOD_7, 0);
        TTI_SFPSHFT2(p_sfpu::LREG1, p_sfpu::LREG14, p_sfpu::LREG4, 5);
        TTI_SFPLOADMACRO(0, 0, ADDR_MOD_7, 0);
        TTI_SFPSHFT2(p_sfpu::LREG2, p_sfpu::LREG14, p_sfpu::LREG4, 5);
        TTI_SFPLOADMACRO(1, 0, ADDR_MOD_7, 0);
        TTI_SFPSHFT2(p_sfpu::LREG3, p_sfpu::LREG14, p_sfpu::LREG4, 5);
        TTI_SFPLOADMACRO(2, 0, ADDR_MOD_7, 0);
        TTI_SFPSHFT2(p_sfpu::LREG0, p_sfpu::LREG14, p_sfpu::LREG4, 5);
        TTI_SFPLOADMACRO(3, 0, ADDR_MOD_7, 0);
        TTI_SFPSHFT2(p_sfpu::LREG1, p_sfpu::LREG14, p_sfpu::LREG4, 5);

        TTI_SFPNOP;
#endif
    } else {
        if constexpr (!is_fp32_dest_acc_en) {
            // _calculate_exponential_tti_bf16_() path:
            // Auto-increment Dest on ADDR_MOD_6
            addr_mod_t{
                .srca = {.incr = 0},
                .srcb = {.incr = 0},
                .dest = {.incr = 2},
            }
                .set(ADDR_MOD_6);

            // LREG12 = 1/ln2
            TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, 0x3fb8);
            TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, 0xaa3b);
            TTI_SFPCONFIG(0, p_sfpu::LREG12, 0);

            // LREG13 = c2 = 4.791750143340323e-15f (0x27aca418)
            TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, 0x27ac);
            TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, 0xa418);
            TTI_SFPCONFIG(0, p_sfpu::LREG13, 0);
        } else {
            // fp32 scalar path (_sfpu_exp_fp32_accurate_) — uses the scalar
            // reciprocal LLK for negative inputs, so its constants must be
            // primed here.
            sfpu_reciprocal_init<false>();
        }
    }
}

}  // namespace sfpu
}  // namespace ckernel
""",
    "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h": r"""namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
void gelu_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    if constexpr (APPROXIMATION_MODE) {
        sfpi::vConstFloatPrgm0 = 0.5f;

        // Load the packed 6-segment LUT coefficients directly into the LRegs via sfpi.
        sfpi::l_reg[sfpi::LRegs::LReg0] = sfpi::vLut16ss(0.1928f, 0.4939f);
        sfpi::l_reg[sfpi::LRegs::LReg4] = sfpi::vLut16ii(-0.00010443f, -0.1604f);

        sfpi::l_reg[sfpi::LRegs::LReg1] = sfpi::vLut16ss(0.6188f, 0.6099f);
        sfpi::l_reg[sfpi::LRegs::LReg5] = sfpi::vLut16ii(-0.2795f, -0.2635f);

        sfpi::l_reg[sfpi::LRegs::LReg2] = sfpi::vLut16ss(0.5402f, 0.5f);
        sfpi::l_reg[sfpi::LRegs::LReg6] = sfpi::vLut16ii(-0.1194f, 0.0f);
    } else if constexpr (is_fp32_dest_acc_en) {
        // FP32 accurate mode: rational erf evaluation requires reciprocal init
        sfpu_reciprocal_init<false>();
    }
    // BF16 accurate mode: no init needed (correction polynomial has no reciprocal)
}

template <int ITERATIONS>
inline void calculate_gelu_appx() {
    sfpi::vLut16ss s01 = sfpi::l_reg[sfpi::LRegs::LReg0];
    sfpi::vLut16ss s23 = sfpi::l_reg[sfpi::LRegs::LReg1];
    sfpi::vLut16ss s45 = sfpi::l_reg[sfpi::LRegs::LReg2];
    sfpi::vLut16ii i01 = sfpi::l_reg[sfpi::LRegs::LReg4];
    sfpi::vLut16ii i23 = sfpi::l_reg[sfpi::LRegs::LReg5];
    sfpi::vLut16ii i45 = sfpi::l_reg[sfpi::LRegs::LReg6];

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat half = sfpi::vConstFloatPrgm0;
        sfpi::vFloat half_in = in * half;
        sfpi::vFloat result = sfpi::lut(in, s01, i01, s23, i23, s45, i45, sfpi::LutSign::Update);
        result = half_in + result;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }

    sfpi::l_reg[sfpi::LRegs::LReg0] = s01;
    sfpi::l_reg[sfpi::LRegs::LReg1] = s23;
    sfpi::l_reg[sfpi::LRegs::LReg2] = s45;
    sfpi::l_reg[sfpi::LRegs::LReg4] = i01;
    sfpi::l_reg[sfpi::LRegs::LReg5] = i23;
    sfpi::l_reg[sfpi::LRegs::LReg6] = i45;
}

}  // namespace sfpu
}  // namespace ckernel
""",
    "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_recip.h": r"""namespace ckernel {
namespace sfpu {

// FP32 reciprocal, with throughput of 5c/32.
inline void _calculate_reciprocal_fast_24b_5c_(const int iterations) {
#ifdef DISABLE_SFPLOADMACRO
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        TTI_SFPARECIP(0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPARECIP_MOD1_RECIP);
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG2, 1);  // SFPMAD_MOD1_NEGATE_VA
        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG2, p_sfpu::LREG2, p_sfpu::LREG3, 0);
        TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG2, p_sfpu::LREG2, p_sfpu::LREG3, 0);
        TTI_SFPSWAP(0, p_sfpu::LCONST_1, p_sfpu::LREG3, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
        TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG1, p_sfpu::LREG1, p_sfpu::LREG0, 0);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 0);
    }
#else
    // Pseudocode:
    //
    // y = arecip(x)
    // e = 1 - x*y
    // t = e * e + e
    // t2 = t * e + e    # e**3 + e**2 + e
    // t2 = min(t2, 1.0) # replace NaN with 1.0
    // y = t2 * y + y    # y = y * (e**3 + e**2 + e + 1)
    //                   # if y = ±0 or ±inf, then y = y+y
    //
    // Notation: [x] means scheduled by SFPLOADMACRO with VD=x.
    //
    //   | Load | Simple                 | MAD                     | Store   |
    // - | -----| ---------------------- | ----------------------- |-------- |
    // 0 | [y]  |                        |                         |         |
    // 1 |      | [y] = arecip(y)        |                         |         |
    // 2 | [e]  |                        |                         |         |
    // 3 |      | [e] L16 = arecip(e)    | e = mad(-e, y, 1.0)     |         |
    // 4 |      |                        |                         |         |
    // 0 |      |                        | [e] = mad(e, e, e)      | [e]     |
    // 1 | [t2] |                        |                         |         |
    // 2 |      |                        | [t2] = mad(t2, e, t2)   | [y] L16 |
    // 3 |      |                        |                         |         |
    // 4 | [z]  | [t2] = swap(t2, 1.0)   |                         |         |
    // 0 |      |                        |                         |         |
    // 1 |      |                        | [z] = mad(t2, z, z)     |         |
    // 2 |      |                        |                         |         |
    // 3 |      |                        |                         | [z]     |

    lltt::replay(0, 4);
    TTI_SFPLOAD(7, 0, ADDR_MOD_6, 0);

#pragma GCC unroll 7
    for (int d = 0; d < iterations - 1; d++) {
        lltt::replay(0, 5);
    }

    TTI_SFPNOP;
    lltt::replay(1, 1);
    TTI_SFPNOP;
    lltt::replay(3, 2);

    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
#endif
}

inline void _init_reciprocal_fast_24b_5c_() {
#ifndef DISABLE_SFPLOADMACRO
    constexpr int e = p_sfpu::LREG0;
    constexpr int t2 = p_sfpu::LREG1;
    constexpr int z = p_sfpu::LREG2;
    constexpr int y = p_sfpu::LREG3;

    // InstructionTemplate[0]
    TTI_SFPARECIP(0, 0, 12, sfpi::SFPARECIP_MOD1_RECIP);

    // InstructionTemplate[1]
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG0, 0, 13, 0);

    // InstructionTemplate[2]
    // SFPMAD(VA=t2, VB=0 or VD, VC=VD or z)
    TTI_SFPMAD(t2, p_sfpu::LREG0, z, 14, 0);

    // InstructionTemplate[3]
    TTI_SFPSWAP(0, p_sfpu::LCONST_1, 15, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);

    // Macro 0: [y]
    {
        constexpr std::uint32_t simple_bits = 0x00 | 0x00 | (0 << 3) | (4 + 0);
        constexpr std::uint32_t mad_bits = 0;
        constexpr std::uint32_t round_bits = 0;
        constexpr std::uint32_t store_bits = 0x00 | 0x40 | (6 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4, 0);
    }

    // Macro 1: [e]
    {
        constexpr std::uint32_t simple_bits = 0x00 | 0x40 | (0 << 3) | (4 + 0);
        constexpr std::uint32_t mad_bits = 0x00 | 0x00 | (2 << 3) | (4 + 1);
        constexpr std::uint32_t round_bits = 0;
        constexpr std::uint32_t store_bits = 0x00 | 0x00 | (2 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 1, 0);
    }

    // Macro 2: [t2]
    {
        constexpr std::uint32_t simple_bits = 0x80 | 0x00 | (2 << 3) | (4 + 3);
        constexpr std::uint32_t mad_bits = 0x00 | 0x00 | (0 << 3) | (4 + 2);

        TTI_SFPCONFIG((mad_bits << 8) | simple_bits, 4 + 2, 1);
    }

    // Macro 3: [z]
    {
        // Keep the corrected result in z. Macro 0 still needs L16 for its delayed
        // store of the next vector's approximate reciprocal. Even with instruction-
        // counted delays, an issued MAD completes during scalar stalls (e.g. gcov),
        // so writing its result to L16 can clobber that value before macro 0 stores it.
        // The next load of z follows this macro's store, so z has no such overlap.
        constexpr std::uint32_t simple_bits = 0;
        constexpr std::uint32_t mad_bits = 0x80 | 0x00 | (1 << 3) | (4 + 2);
        constexpr std::uint32_t round_bits = 0;
        constexpr std::uint32_t store_bits = 0x00 | 0x00 | (3 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 3, 0);
    }

    // Misc: {UsesLoadMod0ForStore=1, WaitForElapsedInstructions=1} for all macros.
    TTI_SFPCONFIG(0xff0, 8, 1);

    constexpr std::uint32_t prev_offset = -2 & 0x3ff;
    constexpr std::uint32_t offset = 0;

    load_replay_buf(0, 6, [e, t2, z, y, offset, prev_offset] {
        TTI_SFPLOADMACRO((0 << 2) | (y & 3), 0, ADDR_MOD_7, offset | (y >> 2));
        TTI_SFPLOADMACRO((2 << 2) | (t2 & 3), 0, ADDR_MOD_7, prev_offset | (t2 >> 2));
        TTI_SFPLOADMACRO((1 << 2) | (e & 3), 0, ADDR_MOD_7, offset | (e >> 2));
        TTI_SFPMAD(p_sfpu::LREG0, y, p_sfpu::LCONST_1, 0, 1);  // SFPMAD_MOD1_NEGATE_VA
        TTI_SFPLOADMACRO((3 << 2) | (z & 3), 0, ADDR_MOD_6, prev_offset | (z >> 2));
        TTI_SFPLOADMACRO((3 << 2) | (z & 3), 0, ADDR_MOD_7, prev_offset | (z >> 2));
    });
#endif
}

template <bool APPROXIMATE = false>
sfpi_inline void sfpu_reciprocal_init() {
    if constexpr (!APPROXIMATE) {
        sfpi::vConstFloatPrgm0 = 2.0f;
    }
}

}  // namespace sfpu
}  // namespace ckernel
""",
    "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_sigmoid_appx.h": r"""namespace ckernel {
namespace sfpu {

inline void sigmoid_appx_init() {
    // Load the 3 fp16b LUT coefficients into LReg0-2
    sfpi::l_reg[sfpi::LRegs::LReg0] = sfpi::vLut8si(0.22656f, 0.0f);
    sfpi::l_reg[sfpi::LRegs::LReg1] = sfpi::vLut8si(0.26562f, -0.04687f);
    sfpi::l_reg[sfpi::LRegs::LReg2] = sfpi::vLut8si(0.0f, 0.5f);
}

}  // namespace sfpu
}  // namespace ckernel
""",
    "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_signbit.h": r"""namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_signbit() {
    // This uses SFPLOADMACRO to achieve a throughput of 1 cycle per input row.
    //
    // Notation: [x] means scheduled by SFPLOADMACRO with VD=x.
    //
    // t  | Load | Simple                 | MAD | Round      | Store    |
    // -- | ---- | ---------------------- | --- | ---------- | -------- |
    //  0 | [a]  |                        |     |            |          |
    //    | ...  |                        |     | [a] >>= 31 |          |
    //    | ...  | [a] L16 = cast_fp32(a) |     |            |          |
    //    | ...  |                        |     |            | [a] L16  |

    constexpr int offset = 0;

#ifndef DISABLE_SFPLOADMACRO
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        int a = d & 1;  // alternate between p_sfpu::LREG0 and p_sfpu::LREG1
        TT_SFPLOADMACRO((0 << 2) | (a & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_6, offset | (a >> 2));
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
#else
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, offset);
        TTI_SFPSHFT(-31 & 0xfff, p_sfpu::LREG0, p_sfpu::LREG0, 1);  // SFPSHFT_MOD1_ARG_IMM
        TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 0);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_6, offset);
    }
#endif
}

}  // namespace sfpu
}  // namespace ckernel
""",
    "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_typecast.h": r"""namespace ckernel {
namespace sfpu {

// Mask that keeps the low 16 bits (the UInt16 value) of a 32-bit dest word and clears the garbage high bits.
constexpr std::uint16_t UINT16_LOW_MASK = 0xFFFF;

template <bool APPROXIMATION_MODE, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void calculate_typecast_uint16_to_uint32() {
#ifdef DISABLE_SFPLOADMACRO
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_SFPAND(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_6, 0);
    }
#else
    if constexpr (!is_fp32_dest_acc_en) {
        // 16-bit Dest: SFPLOADMACRO fast path, throughput of 1 cycle per input row. The LO16 load
        // keeps only the low 16 bits (the UInt16 value) and zero-extends them, so the INT32 store
        // matches the plain loop's INT32 load + 0xFFFF mask.
        //
        // Notation: [x] means scheduled by SFPLOADMACRO with VD=x.
        //
        // t | Load | Simple | MAD | Round | Store |
        // - | ---- | ------ | --- | ----- | ----- |
        // 0 | [v]  |        |     |       |       |
        // 0 | ...  |        |     |       | [v]   |

        constexpr int v = p_sfpu::LREG0;

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            TTI_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::LO16, ADDR_MOD_6, v >> 2);
        }
        TTI_SFPNOP;
    } else {
        // 32-bit Dest: the macro's LO16 load cannot reproduce the INT32 + 0xFFFF mask path, so
        // this case uses the plain loop instead of SFPLOADMACRO.
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
            TTI_SFPAND(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);
            TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_6, 0);
        }
    }
#endif
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint16_to_uint32() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
#ifdef DISABLE_SFPLOADMACRO
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, UINT16_LOW_MASK);
#else
    // The 32-bit Dest (is_fp32_dest_acc_en) path of calculate_typecast_uint16_to_uint32 falls
    // back to the plain loop, which masks the loaded word with LREG1. Load the mask here
    // so that path is correct; the macro-programming below only targets LREG0, so LREG1
    // survives. The 16-bit Dest macro path does not read LREG1.
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, UINT16_LOW_MASK);

    // Macro 0: store only (the LO16 load already zero-extends the UInt16 value).
    {
        constexpr std::uint32_t simple_bits = 0;
        constexpr std::uint32_t mad_bits = 0;
        constexpr std::uint32_t round_bits = 0;
        constexpr std::uint32_t store_bits = 0x00 | 0x00 | (0 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }

    // Misc: {
    //   StoreMod0: INT32,
    //   UsesLoadMod0ForStore: {0},
    //   UnitDelayKind: {1}, (WaitForElapsedInstructions=1)
    // }
    TTI_SFPCONFIG(0x100 | InstrModLoadStore::INT32, 8, 1);
#endif
}

}  // namespace sfpu
}  // namespace ckernel
""",
    "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_init.h": r"""namespace ckernel {

// Callback init entry point (SFPU_UNARY_INIT_FN / _FN_ARGS / two-arg SFPU_UNARY_INIT). The per-op common init
// (config reg + ADDR_MOD_7 + counter reset) runs first, then the op-specific init_func. Consolidated per-op
// (#50381) rather than hoisted: re-asserting the shared SFPU config/addrmod state on every init -- on both the
// MATH and PACK threads -- keeps the exp(PACK)/fp32-reciprocal(MATH) shared macro/replay programming from
// interleaving destructively, which was the #50381 fp32 SDPA accuracy regression.
template <SfpuType sfpu_op, class F, class... ARGS>
inline void llk_math_eltwise_unary_sfpu_init(F&& init_func, ARGS&&... args) {
    SAN_HOOK(unsupported());
    _llk_math_eltwise_unary_sfpu_init_<sfpu_op>();
    init_func(std::forward<ARGS>(args)...);
}

}  // namespace ckernel
""",
    "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h": r"""// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_assert.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sanitizer/api.h"

/*
 * Keep macro preconditions outside the tt-llk params wrapper. DST_SYNC is
 * explicit so tests and non-standard kernel preludes can supply their own
 * sync mode. Dest accumulation mode is read at runtime inside
 * get_dest_max_tiles_rt so it stays correct after enable/disable_fp32_dest_acc.
 * DST_ACCUM remains in the call-macro signature for compatibility.
 */

namespace ckernel {

template <DstSync DST_SYNC>
inline __attribute__((always_inline)) void _sfpu_check_(
    std::uint32_t dst_index, [[maybe_unused]] VectorMode vector_mode) {
    SAN_HOOK(unsupported());
    LLK_ASSERT(
        (dst_index < get_dest_max_tiles_rt<DST_SYNC, DstTileShape::Tile32x32>()),
        "dst_index exceeds max dest tiles");
}

}  // namespace ckernel

// Strip the parentheses around the template-argument tuple passed to SFPU_UNARY_CALL.

#define _SFPU_EXPAND(...) __VA_ARGS__

/*
 * Macro hygiene: DST_IDX and VECTOR_MODE are evaluated by both the check and
 * params call. Keep call sites to identifiers/literals, not side effects.
 */
#define SFPU_UNARY_CALL(DST_SYNC, DST_ACCUM, FN, TEMPLATES, DST_IDX, VECTOR_MODE, ...) \
    (::ckernel::_sfpu_check_<DST_SYNC>(DST_IDX, VECTOR_MODE),               \
     _llk_math_eltwise_unary_sfpu_params_(                                             \
         ::ckernel::sfpu::FN<_SFPU_EXPAND TEMPLATES>, DST_IDX, VECTOR_MODE, ##__VA_ARGS__))

// Non-templated functor in `ckernel::sfpu`.
#define SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC, DST_ACCUM, FN, DST_IDX, VECTOR_MODE, ...) \
    (::ckernel::_sfpu_check_<DST_SYNC>(DST_IDX, VECTOR_MODE),                     \
     _llk_math_eltwise_unary_sfpu_params_(::ckernel::sfpu::FN, DST_IDX, VECTOR_MODE, ##__VA_ARGS__))

/*
 * SFPU init macros (3 total)
 *
 * No dst index involved, so no bound check and no DST_SYNC argument.
 * The bare init LLK requires is_fp32_dest_acc_en; this macro supplies the
 * kernel compile-time DST_ACCUM_MODE (the same value compute APIs default to).
 * Argument order mirrors the call macros: OP first (which selects the
 * SfpuType), then the init callback (the FN-like argument), then the
 * templates that parameterise it.
 */

/*
 * Bare init: no callback.
 *   SFPU_UNARY_INIT(abs);
 */
#define SFPU_UNARY_INIT(OP) ::ckernel::llk_math_eltwise_unary_sfpu_init<::SfpuType::OP, DST_ACCUM_MODE>()

/*
 * Init with a templated callback.
 *   SFPU_UNARY_INIT_FN(erf, sfpu::erf_init, (APPROXIMATE));
 *   SFPU_UNARY_INIT_FN(log, sfpu::log_init, (APPROX, fp32, FAST));
 */
#define SFPU_UNARY_INIT_FN(OP, INIT_FN, TEMPLATES) \
    ::ckernel::llk_math_eltwise_unary_sfpu_init<::SfpuType::OP>(INIT_FN<_SFPU_EXPAND TEMPLATES>)

/*
 * Init with a templated callback and extra runtime arguments.
 *   SFPU_UNARY_INIT_FN_ARGS(exponential, sfpu::exp_init, (APPROX), scale, clamp_neg);
 */
#define SFPU_UNARY_INIT_FN_ARGS(OP, INIT_FN, TEMPLATES, ...) \
    ::ckernel::llk_math_eltwise_unary_sfpu_init<::SfpuType::OP>(INIT_FN<_SFPU_EXPAND TEMPLATES>, ##__VA_ARGS__)
""",
    "tt_metal/hw/inc/api/compute/eltwise_unary/exp.h": r"""// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#if defined(TRISC_MATH) || defined(TRISC_PACK)
#include "ckernel_sfpu_exp.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {
/**
 * Controls whether the fast approximate exponential clamps very negative inputs.
 *
 * ClampToNegative (default): Inputs below ~-88.5 are clamped to -88.5. Safer but slightly slower.
 * None: No input clamping. Faster, but inputs below ~-88.5 will produce incorrect outputs. They
 *     will be guaranteed to be negative, so consider enabling packer ReLU when using this mode.
 */
enum class InputClamping : std::uint8_t {
    ClampToNegative = 1,
    None = 0,
};

/**
 * Please refer to documentation for any_init.
 *
 * Template scale parameter is used when approx is true and exp_tile is called with scale_en set to
 * true.
 *
 */
template <
    bool approx = false,
    std::uint32_t scale = 0x3F800000,
    InputClamping input_clamping = InputClamping::ClampToNegative, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void exp_tile_init() {
    MATH(SFPU_UNARY_INIT_FN(
        exponential,
        sfpu::exp_init,
        (approx, scale, (input_clamping == InputClamping::ClampToNegative), is_fp32_dest_acc_en)));
}

// clang-format off
/**
 * Performs element-wise computation of exponential on each element of a tile
 * in the DST register. The DST register buffer must be in an
 * acquired state via an *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Template Parameter      | Description                                                    | Type     | Valid Range      | Default |
 * |-------------------------|----------------------------------------------------------------|----------|------------------|---------|
 * | approx                  | Enable approximate mode.                                       | bool     | true, false      | false   |
 * | scale_en                | Enable input scaling by a constant factor in approximate or non-approximate mode | bool     | true, false      | false   |
 * | input_clamping          | If approx, controls whether very negative inputs are clamped to prevent incorrect outputs | InputClamping | ClampToNegative, None | ClampToNegative |
 * | iterations              | Number of iterations over 32-SFPU lanes to run                 | int      | Positive integer | 8       |
 *
 * | Argument    | Description                                                                | Type     | Valid Range                                           | Required |
 * |-------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst        | The index of the tile in DST register buffer to perform the computation on | std::uint32_t | Must be less than the size of the DST register buffer | True     |
 * | vector_mode | Specifies the vector mode for computation (default: VectorMode::RC)        | VectorMode | Subject to specific hardware/kernel limits            | False    |
 * | scale       | Scale factor to apply in approximate or non-approximate mode if scale_en is true (default: 0x3F80, 1.0f in FP16b) | std::uint16_t | Valid FP16b representation                            | False    |
 */
// clang-format on
template <
    bool approx = false,
    bool scale_en = false,
    InputClamping input_clamping = InputClamping::ClampToNegative,
    int iterations = 8, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void exp_tile(std::uint32_t idst, VectorMode vector_mode = VectorMode::RC, std::uint16_t scale = p_sfpu::kCONST_1_FP16B) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_exponential,
        (approx, is_fp32_dest_acc_en, scale_en, iterations, (input_clamping == InputClamping::ClampToNegative)),
        idst,
        vector_mode,
        scale));
}

#ifndef ARCH_QUASAR

/**
 * Pack-thread variant of exp_tile_init. Runs the init on the pack thread
 * to enable FPU/SFPU overlap with math-thread matmul operations.
 */
template <
    bool approx = false,
    std::uint32_t scale = 0x3F800000,
    InputClamping input_clamping = InputClamping::ClampToNegative, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void exp_packthread_tile_init() {
    PACK(llk_math_eltwise_unary_sfpu_init<SfpuType::exponential>(
        sfpu::exp_init<approx, scale, (input_clamping == InputClamping::ClampToNegative), is_fp32_dest_acc_en>));
}

/**
 * Pack-thread variant of exp_tile. Runs the exp computation on the pack thread
 * to enable FPU/SFPU overlap with math-thread matmul operations.
 */
template <
    bool approx = false,
    bool scale_en = false,
    InputClamping input_clamping = InputClamping::ClampToNegative,
    int iterations = 8, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void exp_packthread_tile(
    std::uint32_t idst, VectorMode vector_mode = VectorMode::RC, std::uint16_t scale = p_sfpu::kCONST_1_FP16B) {
    PACK(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_exponential,
        (approx, is_fp32_dest_acc_en, scale_en, iterations, (input_clamping == InputClamping::ClampToNegative)),
        idst,
        vector_mode,
        scale));
}
#endif
}  // namespace ckernel
""",
    "tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_addrmod.h": r"""namespace ckernel
{

constexpr std::uint8_t ADDR_MOD_0 = 0;
constexpr std::uint8_t ADDR_MOD_1 = 1;
constexpr std::uint8_t ADDR_MOD_2 = 2;
constexpr std::uint8_t ADDR_MOD_3 = 3;
constexpr std::uint8_t ADDR_MOD_4 = 4;
constexpr std::uint8_t ADDR_MOD_5 = 5;
constexpr std::uint8_t ADDR_MOD_6 = 6;
constexpr std::uint8_t ADDR_MOD_7 = 7;

} // namespace ckernel
""",
    "tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_instr_params.h": r"""namespace ckernel
{

struct p_sfpu
{
    // SFPU registers
    constexpr static std::uint32_t LREG0 = 0;
    constexpr static std::uint32_t LREG1 = 1;
    constexpr static std::uint32_t LREG2 = 2;
    constexpr static std::uint32_t LREG3 = 3;
    constexpr static std::uint32_t LREG4 = 4;
    constexpr static std::uint32_t LREG5 = 5;
    constexpr static std::uint32_t LREG6 = 6;
    constexpr static std::uint32_t LREG7 = 7;

    // HW provided constants
    constexpr static std::uint32_t LCONST_0_8373 = 8;
    constexpr static std::uint32_t LCONST_0      = 9;
    constexpr static std::uint32_t LCONST_1      = 10;

    // Programmable constants
    constexpr static std::uint32_t LREG11      = 11;
    constexpr static std::uint32_t LREG12      = 12;
    constexpr static std::uint32_t LREG13      = 13;
    constexpr static std::uint32_t LREG14      = 14;
    constexpr static std::uint32_t LCONST_neg1 = 11;

    constexpr static std::uint32_t LTILEID = 15;

    constexpr static std::uint32_t kCONST_1_FP16B  = 0x3F80;
    constexpr static std::uint32_t kCONST_1_FP16A  = 0x3C00;
    constexpr static std::uint32_t kCONST_0        = 0x0000;
    constexpr static std::uint32_t kCONST_Exp_8Bit = 0;
    constexpr static std::uint32_t kCONST_Exp_5Bit = 1;
};

} // namespace ckernel
""",
    "tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_load_config.h": r"""// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

inline void _sfpu_load_imm32_(const std::uint32_t dest, const std::uint32_t val)
{
    TT_SFPLOADI(dest, 10, (val & 0xFFFF));      // insmod == 10 will write the lower bits, and not affect the upper bits;
    TT_SFPLOADI(dest, 8, (val >> 16) & 0xFFFF); // insmod == 8 will write the upper bits, and not affect the lower bits;
}

inline void _sfpu_load_imm16_(const std::uint32_t dest, const std::uint32_t val)
{
    TT_SFPLOADI(dest, 2, val & 0xFFFF); // insmod == 2 will write imm16 value treated as unsigned integer, right justified and padded with zeroes on the MSBs
}

inline void _sfpu_load_config32_(const std::uint32_t dest, const std::uint32_t upper16, const std::uint32_t lower16)
{
    // registers 11 through 14 are programmable "constants" which are shared across all 4 rows
    // They are updated only through the CONFIG path, which uses LREG[0] first and then copies it to the desired register location
    TTI_SFPLOADI(0, 10, lower16); // insmod == A will write the lower bits, and not affect the upper bits;
    TTI_SFPLOADI(0, 8, upper16);  // insmod == 8 will write the upper bits, and not affect the lower bits;
    TTI_SFPCONFIG(0, dest, 0);
}

inline void _init_sfpu_config_reg()
{
    TTI_SFPCONFIG(0, 0xF, 1);
}

// LCONST_neg1 (LREG11) is a core-wide constant other SFPU kernels read as -1.0. A kernel that reprograms LREG11
// through SFPCONFIG calls this before it returns.
inline void _restore_lconst_neg1_()
{
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATB, 0xBF80); // bf16 0xBF80 << 16 == fp32 -1.0
    TTI_SFPCONFIG(0x5555, /*LREG11=*/11, /*MOD1_IMM16_IS_LANE_MASK=*/8);
}

} // namespace sfpu
} // namespace ckernel
""",
    "tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_topk.h": r"""namespace ckernel
{
namespace sfpu
{

static std::int32_t topk_replay_init = 0;

constexpr std::uint32_t TOPK_LO16_MASK = 0x0000FFFF;

// All lanes enabled, CC result true. A macro, not a function: some sites sit inside replay-record windows.
#define TOPK_SFPENCC_ALL_LANES_ON() TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0, 0, sfpi::SFPENCC_MOD1_EI_RI)

// Rank-stamped keys carry a TAG_BITS-wide rank tag in the value word's low bits: 16 for bf16 values
// (the whole low half is free), 6..15 for fp32 keys whose low mantissa bits are known to be zero.
template <std::uint32_t TAG_BITS>
constexpr std::uint32_t topk_tag_mask()
{
    static_assert(TAG_BITS >= 6 && TAG_BITS <= 16, "rank tag field must be 6..16 bits (a 64-rank chain needs 6)");
    return (1u << TAG_BITS) - 1u;
}

// Rank-stamped init: index tracking ON (the true u32 indices ride the tracked
// swaps at DEST offset 128, as in the plain unfused modes) plus the tag
// complement constant. Written as an explicit set so a preceding fused-mode
// topk in the same kernel cannot leak tracking OFF. The merge programs the
// remaining stamp constants (LREG12/13/14) at each entry, since K and the
// rank base are runtime values there.
template <std::uint32_t TAG_BITS = 16>
inline void _init_topk_rank_stamped_()
{
    topk_replay_init = 0;
    _sfpu_load_config32_(0xF, 0x0, 0x4);              // SFPU_CONTROL_REG: ENABLE_DEST_INDEX (bit 2) = 1
    sfpi::vConstIntPrgm0 = topk_tag_mask<TAG_BITS>(); // LREG12: tag complement operand
}

// Fused-key init: index tracking stays OFF (the packed key carries the index; there is no
// L4-7 bank to mirror). Written as an explicit clear rather than a skip so a preceding
// tracked-mode topk in the same kernel cannot leak the bit in. Programs the fuse/defuse
// mask constant; the sweeps re-program it defensively at each entry as well.
inline void _init_topk_fused_()
{
    topk_replay_init = 0;
    _sfpu_load_config32_(0xF, 0x0, 0x0); // SFPU_CONTROL_REG: ENABLE_DEST_INDEX (bit 2) = 0
    sfpi::vConstIntPrgm0 = TOPK_LO16_MASK; // LREG12: #50215 mask + tie-complement operand
}

} // namespace sfpu
} // namespace ckernel
""",
    "tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_triangle_solve.h": r"""namespace ckernel
{
namespace sfpu
{

constexpr std::uint32_t TRIANGLE_SOLVE_IMM16_BITS     = 16;                                    // width of an SFPLOADI immediate
constexpr std::uint32_t TRIANGLE_SOLVE_IMM16_MASK     = (1u << TRIANGLE_SOLVE_IMM16_BITS) - 1; // a 16-bit SFPLOADI immediate
constexpr std::uint32_t TRIANGLE_SOLVE_IMM16_SIGN_BIT = 1u << (TRIANGLE_SOLVE_IMM16_BITS - 1); // its sign bit

template <std::uint32_t MOD0, std::uint32_t FLIP = 0>
inline void _triangle_solve_sfploadi_lreg7_(const std::uint32_t imm16)
{
    constexpr std::uint32_t OPCODE = TT_OP_SFPLOADI(p_sfpu::LREG7, MOD0, 0);
    static_assert(MOD0 <= 0xF, "SFPLOADI mod0 is a 4-bit field");
    static_assert((OPCODE & TRIANGLE_SOLVE_IMM16_MASK) == 0, "the immediate must fill the zero low half of the SFPLOADI word");
    static_assert(FLIP <= TRIANGLE_SOLVE_IMM16_MASK, "the flip must stay within the 16-bit immediate");
    TT_INSN((OPCODE ^ FLIP) ^ imm16);
}

/**
 * @brief TT_SFPLOADI(LREG7, MOD0, bits & TRIANGLE_SOLVE_IMM16_MASK), the word built with one Zbkb `pack` where the ISA has it.
 *
 * @tparam MOD0: SFPLOADI mode, one of sfpi::SFPLOADI_MOD0_*
 * @param bits: Raw 32-bit element; only its low half is used.
 */
template <std::uint32_t MOD0>
inline void _triangle_solve_sfploadi_lreg7_lo_pack_(const std::uint32_t bits)
{
    constexpr std::uint32_t OPCODE = TT_OP_SFPLOADI(p_sfpu::LREG7, MOD0, 0);
    static_assert(MOD0 <= 0xF, "SFPLOADI mod0 is a 4-bit field");
    static_assert((OPCODE & TRIANGLE_SOLVE_IMM16_MASK) == 0, "the immediate must fill the zero low half of the SFPLOADI word");
#if defined(__riscv_xttzbkb) || defined(__riscv_zbkb)
    const std::uint32_t opcode_hi = OPCODE >> TRIANGLE_SOLVE_IMM16_BITS;
    std::uint32_t word;
    __asm__("pack %0, %1, %2" : "=r"(word) : "r"(bits), "r"(opcode_hi));
    TT_INSN(word);
#else
    TT_INSN(OPCODE ^ (bits & TRIANGLE_SOLVE_IMM16_MASK));
#endif

} // namespace sfpu
} // namespace ckernel
""",
    "tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/experimental/ckernel_sfpu_sinkhorn.h": r"""namespace ckernel
{
namespace sfpu
{

// Program LREG11 = lane-parity mask: lane k -> (2k & 2), i.e. 2 at odd
// lanes, 0 at even. Used by the row-norm 2-lane pair fold (persistent
// across iters). Comb row-max builds a throwaway copy in a scratch LREG
// instead, so LCONST_neg1 (LREG11) stays live for its subtract.
//
// SFPCONFIG with a lane-masked LREG target copies LReg[0][lane & 7] into
// LREG[dest][lane] (per-column values, vertically broadcast across the 4
// row-groups); imm16 bit (2c) gates column c, so 0x5555 selects all columns.
//
// LCONST_neg1 aliases LREG11, so this clobbers the HW -1.0 constant. The
// row-norm body must not use LCONST_neg1 while the mask is live.
inline void _sinkhorn_program_parity_mask_()
{
    // Parity mask: LREG0 lane k -> 2k & 2 = (2 at odd lanes, 0 at even).
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 2);
    // SFPAND with Mod1=0 does VD = VD & VC; VC=15 is the read-only lane-id
    // register (LReg[15][k] = 2k).
    TTI_SFPAND(0, /*VC=*/15, p_sfpu::LREG0, 0);
    TTI_SFPCONFIG(0x5555, /*LREG11=*/11, /*MOD1_IMM16_IS_LANE_MASK=*/8);
}

} // namespace sfpu
} // namespace ckernel
""",
    "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_sfpu_common.h": r"""// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <utility>

#include "ckernel_ops.h"
#include "ckernel_sfpu.h"
#include "llk_assert.h"
#include "llk_math_common.h"
#include "llk_sfpu_types.h"

using namespace ckernel;

inline void _llk_math_eltwise_sfpu_start_(const std::uint32_t dst_index)
{
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
}

inline void _llk_math_eltwise_sfpu_done_()
{
    math::clear_dst_reg_addr();
}

inline void _llk_math_eltwise_sfpu_inc_dst_face_addr_()
{
    math::inc_dst_addr<8>();
    math::inc_dst_addr<8>();
}

inline void _llk_math_eltwise_sfpu_uninit_()
{
    // No state to restore - all states are transient or default
}

template <DstSync Dst>
inline void _llk_math_eltwise_sfpu_assert_dst_index_(std::uint32_t dst_index, [[maybe_unused]] const char* message)
{
    LLK_ASSERT((dst_index < get_dest_max_tiles_rt<Dst, DstTileShape::Tile32x32>()), message);
}

template <typename Callable, typename... Args>
inline __attribute__((always_inline)) void _llk_math_eltwise_sfpu_apply_vector_mode_(Callable&& sfpu_func, VectorMode vector_mode, Args&&... args)
{
    if (vector_mode == VectorMode::RC)
    {
        // Do all four faces, and iterate through all 4 blocks of 4 rows each
#pragma GCC unroll 0
        for (int face = 0; face < 4; face++)
        {
            sfpu_func(args...);
            _llk_math_eltwise_sfpu_inc_dst_face_addr_();
        }
    }
    else if (vector_mode == VectorMode::R)
    {
        // Do a row vector, Face0 + Face1 -- first iteration (first row)
#pragma GCC unroll 0
        for (int face = 0; face < 2; face++)
        {
            sfpu_func(args...);
            _llk_math_eltwise_sfpu_inc_dst_face_addr_();
        }
        // Skip the next 2 faces
        _llk_math_eltwise_sfpu_inc_dst_face_addr_();
        _llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
    else if (vector_mode == VectorMode::C)
    {
        // Do a column vector, Face0 + Face2 -- All iterations for full face
#pragma GCC unroll 0
        for (int face = 0; face < 2; face++)
        {
            sfpu_func(args...);
            _llk_math_eltwise_sfpu_inc_dst_face_addr_();
            _llk_math_eltwise_sfpu_inc_dst_face_addr_();
        }
    }
    else
    {
        std::forward<Callable>(sfpu_func)(std::forward<Args>(args)...);
    }
}
""",
    "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu.h": r"""// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <type_traits>

#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "ckernel_sfpu.h"
#include "cmath_common.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_sfpu_common.h"
#include "llk_sfpu_types.h"

using namespace ckernel;

// local function declarations
template <SfpuType sfpu_op>
inline void eltwise_unary_sfpu_configure_addrmod()
{
    // NOTE: this kernel is typically used in conjunction with
    //       A2D, which is using ADDR_MOD_0 and ADDR_MOD_2, so use one
    //       that doesn't conflict!

    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_7);

    if constexpr (sfpu_op == SfpuType::topk_local_sort)
    {
        addr_mod_t {
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 32},
        }
            .set(ADDR_MOD_6);
    }

    if constexpr (
        sfpu_op == SfpuType::reciprocal || sfpu_op == SfpuType::typecast || sfpu_op == SfpuType::unary_max || sfpu_op == SfpuType::unary_min ||
        sfpu_op == SfpuType::unary_max_int32 || sfpu_op == SfpuType::unary_min_int32 || sfpu_op == SfpuType::unary_max_uint32 ||
        sfpu_op == SfpuType::unary_min_uint32 || sfpu_op == SfpuType::signbit || sfpu_op == SfpuType::equal_zero || sfpu_op == SfpuType::not_equal_zero ||
        sfpu_op == SfpuType::less_than_zero || sfpu_op == SfpuType::greater_than_equal_zero || sfpu_op == SfpuType::less_than_equal_zero ||
        sfpu_op == SfpuType::greater_than_zero)
    {
        addr_mod_t {
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 2},
        }
            .set(ADDR_MOD_6);
    }
}

template <SfpuType sfpu_op>
inline void _llk_math_eltwise_unary_sfpu_init_()
{
    sfpu::_init_sfpu_config_reg();
    eltwise_unary_sfpu_configure_addrmod<sfpu_op>();
    math::reset_counters(p_setrwc::SET_ABD_F);
}

// Kernel-invariant SFPU init. Runs the parts of the per-op init that are identical for every SFPU op and
// therefore only need to run once per kernel: the SFPU config register (SFPCONFIG(0, 0xF, 1)) and the
// invariant ADDR_MOD_7 = {srca:0, srcb:0, dest:0}. Hoisted out of the per-op path so that the self-contained
// per-op init (ckernel::sfpu::_init_<op>_) does not re-run these on every op init. Metal wires this into every
// "full init" entry point (compute_kernel_hw_startup, init_sfpu, unary_op_init_common),
// and the tt-llk standalone SFPU test harness wires it into its init prelude.
inline void _llk_math_eltwise_unary_sfpu_init_once_()
{
    sfpu::_init_sfpu_config_reg();

    // NOTE: this kernel is typically used in conjunction with
    //       A2D, which is using ADDR_MOD_0 and ADDR_MOD_2, so use one
    //       that doesn't conflict!
    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_7);
}
""",
    "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h": r"""// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <utility>

#include "llk_math_eltwise_sfpu_common.h"
#include "llk_math_eltwise_unary_sfpu.h"

template <typename Callable, typename... Args>
inline void _llk_math_eltwise_unary_sfpu_params_(Callable&& sfpu_func, std::uint32_t dst_index, VectorMode vector_mode = VectorMode::RC, Args&&... args)
{
    _llk_math_eltwise_sfpu_start_(dst_index);

    _llk_math_eltwise_sfpu_apply_vector_mode_(std::forward<Callable>(sfpu_func), vector_mode, std::forward<Args>(args)...);

    _llk_math_eltwise_sfpu_done_();
}
""",
}

TYPECAST_INIT_START = next(
    number
    for number, text in enumerate(SNIPPETS[TYPECAST].split("\n"), start=1)
    if "inline void init_typecast_uint16_to_uint32()" in text
)


def _line(path: str, needle: str, occurrence: int = 1) -> int:
    """1-based line of the ``occurrence``-th line of SNIPPETS[path] containing ``needle``."""
    seen = 0
    for number, text in enumerate(SNIPPETS[path].split("\n"), start=1):
        if needle in text:
            seen += 1
            if seen == occurrence:
                return number
    raise AssertionError(f"{needle!r} not found {occurrence}x in {path}")


def _call_line(
    path: str, needle: str, macro: str = "TTI_SFPLOADI(", occurrence: int = 1
) -> int:
    """Line of the ``macro`` call (its name token) whose argument list contains ``needle``."""
    lines = SNIPPETS[path].split("\n")
    number = _line(path, needle, occurrence)
    while macro not in lines[number - 1]:
        number -= 1
    return number


def _project(effect):
    line = int(effect.evidence.rsplit(":", 1)[1])
    return (
        effect.pattern,
        effect.resource,
        effect.access,
        effect.value,
        effect.condition,
        line,
        effect.detail,
    )


PROGRAMMABLE = {f"SFPU.LREG{n}" for n in range(11, 15)}


class SfpuFamilyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        files = dict(SNIPPETS)
        for relative in (resources.CFG_DEFINES, resources.GPR_MAP):
            source = REPO_ROOT / relative
            if not source.is_file():
                raise unittest.SkipTest(
                    "needs cfg_defines.h and ckernel_gpr_map.h of the checkout"
                )
            files[relative] = source.read_text(encoding="utf-8")
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        for relative, text in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        cls.root = root
        cls.db = effects_db.build(root, families=["sfpu"])

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    # ---- helpers ------------------------------------------------------------ #
    def fn(self, name: str):
        (model,) = self.db.by_name(name)
        return model

    def direct(self, name: str, *, clobbers: bool = True) -> set:
        return {
            _project(e)
            for e in self.fn(name).effects
            if e.family == "sfpu" and (clobbers or e.pattern != "sfpu.sfpi_clobber")
        }

    def call(self, name: str, *targs: str, **kwargs) -> list:
        return [
            e
            for e in self.db.call_effects(
                name, template_args=targs, evidence="kernel.cpp:1", **kwargs
            )
            if e.family == "sfpu"
        ]

    # ---- SFPCONFIG primitives ----------------------------------------------- #
    def test_init_sfpu_config_reg_writes_sfpu_ctrl_from_imm16(self) -> None:
        line = _line(LOAD_CONFIG, "TTI_SFPCONFIG(0, 0xF, 1);")
        self.assertEqual(
            self.direct("_init_sfpu_config_reg"),
            {("sfpu.sfpconfig", "SFPU.CTRL", "write", "0x0", "", line, "src=imm16")},
        )
        (effect,) = self.fn("_init_sfpu_config_reg").effects
        self.assertEqual(
            (effect.value_kind, effect.granularity, effect.family),
            ("const", "slot", "sfpu"),
        )

    def test_load_config32_stages_the_word_in_lreg0(self) -> None:
        upper = _line(LOAD_CONFIG, "TTI_SFPLOADI(0, 8, upper16);")
        lower = _line(LOAD_CONFIG, "TTI_SFPLOADI(0, 10, lower16);")
        config = _line(LOAD_CONFIG, "TTI_SFPCONFIG(0, dest, 0);")
        value = "((upper16) << 16) | (lower16)"
        self.assertEqual(
            self.direct("_sfpu_load_config32_"),
            {
                (
                    "sfpu.lreg.live_out",
                    "SFPU.LREG0",
                    "write",
                    value,
                    "",
                    upper,
                    "live_out",
                ),
                (
                    "sfpu.sfpconfig",
                    "SFPU.?dest",
                    "write",
                    value,
                    "",
                    config,
                    f"src=LREG0;staged={lower},{upper}",
                ),
            },
        )
        effect = next(
            e
            for e in self.fn("_sfpu_load_config32_").effects
            if e.pattern == "sfpu.sfpconfig"
        )
        self.assertFalse(effect.resolved)
        self.assertEqual(effect.params, ("dest", "lower16", "upper16"))
        self.assertEqual(effect.value_kind, "expr")

    def test_load_config32_callers_resolve_dest_and_value(self) -> None:
        config = _line(LOAD_CONFIG, "TTI_SFPCONFIG(0, dest, 0);")
        fused = {
            (e.resource, e.value, e.evidence) for e in self.call("_init_topk_fused_")
        }
        self.assertIn(
            ("SFPU.CTRL", "((0x0) << 16) | (0x0)", f"{LOAD_CONFIG}:{config}"), fused
        )
        self.assertIn(
            ("SFPU.LREG12", "0xffff", f"{TOPK}:{_line(TOPK, 'TOPK_LO16_MASK; //')}"),
            fused,
        )
        stamped = {
            (e.resource, e.value)
            for e in self.call("_init_topk_rank_stamped_", "16")
            if e.access == "write"
        }
        self.assertIn(("SFPU.CTRL", "((0x0) << 16) | (0x4)"), stamped)
        self.assertIn(("SFPU.LREG12", "topk_tag_mask<16>()"), stamped)

    def test_load_imm32_merges_the_two_halves_of_a_parameter_lreg(self) -> None:
        upper = _line(LOAD_CONFIG, "TT_SFPLOADI(dest, 8, (val >> 16) & 0xFFFF);")
        self.assertEqual(
            self.direct("_sfpu_load_imm32_"),
            {
                (
                    "sfpu.lreg.live_out",
                    "SFPU.?dest",
                    "write",
                    "(((val >> 16) & 0xFFFF) << 16) | ((val & 0xFFFF))",
                    "",
                    upper,
                    "live_out",
                )
            },
        )

    # ---- LUT / mask LREGs written by inits, read by calculate_* -------------- #
    def test_sigmoid_appx_init_loads_the_lut_into_lreg0_2(self) -> None:
        rows = {
            reg: _line(SIGMOID, f"sfpi::l_reg[sfpi::LRegs::LReg{reg}] =")
            for reg in range(3)
        }
        self.assertEqual(
            self.direct("sigmoid_appx_init"),
            {
                (
                    "sfpu.lreg.live_out",
                    "SFPU.LREG0",
                    "write",
                    "sfpi::vLut8si(0.22656f, 0.0f)",
                    "",
                    rows[0],
                    "live_out",
                ),
                (
                    "sfpu.lreg.live_out",
                    "SFPU.LREG1",
                    "write",
                    "sfpi::vLut8si(0.26562f, -0.04687f)",
                    "",
                    rows[1],
                    "live_out",
                ),
                (
                    "sfpu.lreg.live_out",
                    "SFPU.LREG2",
                    "write",
                    "sfpi::vLut8si(0.0f, 0.5f)",
                    "",
                    rows[2],
                    "live_out",
                ),
            },
        )

    def test_gelu_init_lut_and_constant_depend_on_approximation_mode(self) -> None:
        const = _line(GELU, "sfpi::vConstFloatPrgm0 = 0.5f;")
        lut = {
            0: (
                "sfpi::vLut16ss(0.1928f, 0.4939f)",
                _line(GELU, "LRegs::LReg0] = sfpi::vLut16ss"),
            ),
            4: (
                "sfpi::vLut16ii(-0.00010443f, -0.1604f)",
                _line(GELU, "LRegs::LReg4] = sfpi::vLut16ii"),
            ),
            1: (
                "sfpi::vLut16ss(0.6188f, 0.6099f)",
                _line(GELU, "LRegs::LReg1] = sfpi::vLut16ss"),
            ),
            5: (
                "sfpi::vLut16ii(-0.2795f, -0.2635f)",
                _line(GELU, "LRegs::LReg5] = sfpi::vLut16ii"),
            ),
            2: (
                "sfpi::vLut16ss(0.5402f, 0.5f)",
                _line(GELU, "LRegs::LReg2] = sfpi::vLut16ss"),
            ),
            6: (
                "sfpi::vLut16ii(-0.1194f, 0.0f)",
                _line(GELU, "LRegs::LReg6] = sfpi::vLut16ii"),
            ),
        }
        expected = {
            (
                "sfpu.sfpi_constant",
                "SFPU.LREG12",
                "write",
                "0.5f",
                "APPROXIMATION_MODE",
                const,
                "vConstFloatPrgm0",
            )
        }
        expected |= {
            (
                "sfpu.lreg.live_out",
                f"SFPU.LREG{reg}",
                "write",
                value,
                "APPROXIMATION_MODE",
                line,
                "live_out",
            )
            for reg, (value, line) in lut.items()
        }
        # the constant's materialization may use any LREG the init does not rewrite afterwards
        expected |= {
            (
                "sfpu.sfpi_clobber",
                f"SFPU.LREG{reg}",
                "write",
                "?sfpi",
                "APPROXIMATION_MODE",
                const,
                "sfpi_regalloc",
            )
            for reg in (3, 7)
        }
        self.assertEqual(self.direct("gelu_init"), expected)
        self.assertFalse(
            [
                e
                for e in self.call("gelu_init", "false", "false")
                if e.resource.startswith("SFPU.LREG")
            ]
        )
        approx = {
            (e.resource, e.value)
            for e in self.call("gelu_init", "true", "false")
            if e.access == "write"
        }
        self.assertIn(("SFPU.LREG4", "sfpi::vLut16ii(-0.00010443f, -0.1604f)"), approx)

    def test_calculate_gelu_appx_reads_the_lut_it_expects_from_gelu_init(self) -> None:
        live_in = {
            (reg, _line(GELU, f"= sfpi::l_reg[sfpi::LRegs::LReg{reg}];"))
            for reg in (0, 1, 2, 4, 5, 6)
        }
        restore = {
            (reg, name, _line(GELU, f"sfpi::l_reg[sfpi::LRegs::LReg{reg}] = {name};"))
            for reg, name in (
                (0, "s01"),
                (1, "s23"),
                (2, "s45"),
                (4, "i01"),
                (5, "i23"),
                (6, "i45"),
            )
        }
        first = _line(GELU, "sfpi::vLut16ss s01")
        expected = {
            ("sfpu.lreg.live_in", f"SFPU.LREG{reg}", "read", "", "", line, "live_in")
            for reg, line in live_in
        }
        expected |= {
            (
                "sfpu.lreg.live_out",
                f"SFPU.LREG{reg}",
                "write",
                name,
                "",
                line,
                "live_out",
            )
            for reg, name, line in restore
        }
        expected |= {
            (
                "sfpu.sfpi_clobber",
                f"SFPU.LREG{reg}",
                "write",
                "?sfpi",
                "",
                first,
                "sfpi_regalloc",
            )
            for reg in (3, 7)
        }
        expected.add(
            (
                "sfpu.dst_reg.addr_mod",
                "ADDR_MOD.7",
                "read",
                "sfpi::dst_reg",
                "",
                _line(GELU, "sfpi::vFloat in = sfpi::dst_reg[0];"),
                "SFPLOAD/SFPSTORE_ADDR_MODE_NOINC",
            )
        )
        self.assertEqual(self.direct("calculate_gelu_appx"), expected)

    # ---- typecast: #ifdef DISABLE_SFPLOADMACRO and is_fp32_dest_acc_en --------- #
    def test_typecast_uint16_to_uint32_init_programs_mask_sequence_and_misc(
        self,
    ) -> None:
        start = TYPECAST_INIT_START
        mask_disabled = _line(
            TYPECAST,
            "TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, UINT16_LOW_MASK);",
            1,
        )
        mask_enabled = _line(
            TYPECAST,
            "TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, UINT16_LOW_MASK);",
            2,
        )
        lower = _line(
            TYPECAST,
            "TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);",
        )
        upper = _line(
            TYPECAST,
            "TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);",
        )
        seq = _line(TYPECAST, "TTI_SFPCONFIG(0, 4 + 0, 0);")
        misc = _line(TYPECAST, "TTI_SFPCONFIG(0x100 | InstrModLoadStore::INT32, 8, 1);")
        self.assertTrue(start < mask_disabled < mask_enabled < seq < misc)
        enabled = "!(defined(DISABLE_SFPLOADMACRO))"
        self.assertEqual(
            self.direct("init_typecast_uint16_to_uint32"),
            {
                (
                    "sfpu.lreg.live_out",
                    "SFPU.LREG1",
                    "write",
                    "0xffff",
                    "defined(DISABLE_SFPLOADMACRO)",
                    mask_disabled,
                    "live_out",
                ),
                (
                    "sfpu.lreg.live_out",
                    "SFPU.LREG1",
                    "write",
                    "0xffff",
                    enabled,
                    mask_enabled,
                    "live_out",
                ),
                (
                    "sfpu.lreg.live_out",
                    "SFPU.LREG0",
                    "write",
                    "0x3000000",
                    enabled,
                    upper,
                    "live_out",
                ),
                (
                    "sfpu.sfpconfig",
                    "SFPU.LOADMACRO.SEQ0",
                    "write",
                    "0x3000000",
                    enabled,
                    seq,
                    f"src=LREG0;staged={lower},{upper};uses=store:STORE",
                ),
                (
                    "sfpu.sfpconfig",
                    "SFPU.LOADMACRO.CTRL",
                    "write",
                    "0x104",
                    enabled,
                    misc,
                    "src=imm16",
                ),
            },
        )

    def test_typecast_uint16_to_uint32_calc_branches(self) -> None:
        effects = self.direct("calculate_typecast_uint16_to_uint32")
        macro = _line(
            TYPECAST,
            "TTI_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::LO16, ADDR_MOD_6, v >> 2);",
        )
        macro_cond = "!(defined(DISABLE_SFPLOADMACRO)) && !is_fp32_dest_acc_en"
        self.assertIn(
            (
                "sfpu.loadmacro.seq",
                "SFPU.LOADMACRO.SEQ0",
                "read",
                "",
                macro_cond,
                macro,
                "",
            ),
            effects,
        )
        self.assertIn(
            (
                "sfpu.loadmacro.ctrl",
                "SFPU.LOADMACRO.CTRL",
                "read",
                "",
                macro_cond,
                macro,
                "",
            ),
            effects,
        )
        self.assertEqual(
            {e[1] for e in effects if e[0] == "sfpu.loadmacro.instr"},
            {f"SFPU.LOADMACRO.INSTR{slot}" for slot in range(4)},
        )
        self.assertIn(
            (
                "sfpu.lreg.live_out",
                "SFPU.LREG0",
                "write",
                "?SFPLOADMACRO",
                macro_cond,
                macro,
                "live_out",
            ),
            effects,
        )
        mask_reads = {(e[4], e[5]) for e in effects if e[0] == "sfpu.lreg.live_in"}
        self.assertEqual(
            mask_reads,
            {
                (
                    "defined(DISABLE_SFPLOADMACRO)",
                    _line(
                        TYPECAST, "TTI_SFPAND(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);", 1
                    ),
                ),
                (
                    "!(defined(DISABLE_SFPLOADMACRO)) && !(!is_fp32_dest_acc_en)",
                    _line(
                        TYPECAST, "TTI_SFPAND(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);", 2
                    ),
                ),
            },
        )
        # fp32 Dest: the macro branch folds away, the LREG1 mask read stays
        fp32 = self.call("calculate_typecast_uint16_to_uint32", "false", "8", "true")
        self.assertFalse([e for e in fp32 if e.resource.startswith("SFPU.LOADMACRO")])
        self.assertIn(
            "SFPU.LREG1", {e.resource for e in fp32 if e.pattern == "sfpu.lreg.live_in"}
        )

    # ---- exp: scale-dependent constants, macro programming, replay -------------- #
    def test_exp_init_constants_follow_the_scale_template_argument(self) -> None:
        clamp = "APPROXIMATION_MODE && CLAMP_NEGATIVE"
        plain = "!(APPROXIMATION_MODE && CLAMP_NEGATIVE) && APPROXIMATION_MODE"
        a_scaled = (
            "bits((256.0f * 1.4426950408889634f) * __builtin_bit_cast(float, scale))"
        )
        writes = {
            (e[1], e[3], e[4], e[5])
            for e in self.direct("exp_init")
            if e[0] in ("sfpu.sfpconfig", "sfpu.sfpi_constant") and e[1] in PROGRAMMABLE
        }
        expected = {
            (
                "SFPU.LREG14",
                "bits(-88.5f / __builtin_bit_cast(float, scale))",
                clamp,
                _line(EXP, "TTI_SFPCONFIG(0, 14, 0);  // SFPCONFIG Dest 14"),
            ),
            (
                "SFPU.LREG12",
                a_scaled,
                clamp,
                _line(EXP, "TTI_SFPCONFIG(0, 12, 0);  // SFPCONFIG Dest 12"),
            ),
            (
                "SFPU.LREG13",
                "bits(32500.818359375f)",
                clamp,
                _line(EXP, "TTI_SFPCONFIG(0, 13, 0);  // SFPCONFIG Dest 13"),
            ),
            ("SFPU.LREG12", a_scaled, plain, _line(EXP, "TTI_SFPCONFIG(0, 12, 0);", 2)),
            (
                "SFPU.LREG13",
                "bits(32500.818359375f)",
                plain,
                _line(EXP, "TTI_SFPCONFIG(0, 13, 0);", 2),
            ),
            (
                "SFPU.LREG14",
                "0xf",
                plain,
                _line(EXP, "TTI_SFPCONFIG(0, 14, 0);   // Store in LREG[14]"),
            ),
            (
                "SFPU.LREG12",
                "0x3fb8aa3b",
                "!(APPROXIMATION_MODE && CLAMP_NEGATIVE) && !(APPROXIMATION_MODE) && !is_fp32_dest_acc_en",
                _line(EXP, "TTI_SFPCONFIG(0, p_sfpu::LREG12, 0);"),
            ),
            (
                "SFPU.LREG13",
                "0x27aca418",
                "!(APPROXIMATION_MODE && CLAMP_NEGATIVE) && !(APPROXIMATION_MODE) && !is_fp32_dest_acc_en",
                _line(EXP, "TTI_SFPCONFIG(0, p_sfpu::LREG13, 0);"),
            ),
        }
        self.assertEqual(writes, expected)
        unit = {
            (e.resource, e.value)
            for e in self.call("exp_init", "true", "0x3F800000", "true", "false")
            if e.access == "write"
        }
        scaled = {
            (e.resource, e.value)
            for e in self.call("exp_init", "true", "0x3FB8AA3B", "true", "false")
            if e.access == "write"
        }
        self.assertIn(
            (
                "SFPU.LREG12",
                "bits((256.0f * 1.4426950408889634f) * __builtin_bit_cast(float, 0x3F800000))",
            ),
            unit,
        )
        self.assertIn(
            (
                "SFPU.LREG12",
                "bits((256.0f * 1.4426950408889634f) * __builtin_bit_cast(float, 0x3FB8AA3B))",
            ),
            scaled,
        )
        self.assertIn(
            ("SFPU.LREG14", "bits(-88.5f / __builtin_bit_cast(float, 0x3FB8AA3B))"),
            scaled,
        )
        self.assertNotIn(
            ("SFPU.LREG14", "0xf"), unit
        )  # CLAMP_NEGATIVE=true folds the other branch away
        accurate = {
            (e.resource, e.value)
            for e in self.call("exp_init", "false", "0x3F800000", "true", "true")
            if e.access == "write"
        }
        self.assertEqual(
            {
                r
                for r in accurate
                if r[0] in PROGRAMMABLE or r[0].startswith("SFPU.LOADMACRO")
            },
            {
                ("SFPU.LREG12", "2.0f")
            },  # fp32 accurate path only primes the reciprocal constant
        )

    def test_exp_init_programs_templates_sequences_and_misc(self) -> None:
        effects = self.direct("exp_init")
        clamp = "!defined(DISABLE_SFPLOADMACRO) && APPROXIMATION_MODE && CLAMP_NEGATIVE"
        plain = "!defined(DISABLE_SFPLOADMACRO) && !(APPROXIMATION_MODE && CLAMP_NEGATIVE) && APPROXIMATION_MODE"
        macro = {
            (e[1], e[3], e[4], e[6])
            for e in effects
            if e[0] in ("sfpu.sfpconfig", "sfpu.backdoor") and "LOADMACRO" in e[1]
        }
        self.assertEqual(
            macro,
            {
                (
                    "SFPU.LOADMACRO.INSTR0",
                    "0x920000e1",
                    clamp,
                    f"src=LREG0;staged={_line(EXP, 'TTI_SFPLOADI(0, 0xA, 0x00E1);')},{_line(EXP, 'TTI_SFPLOADI(0, 0x8, 0x9200);')}",
                ),
                (
                    "SFPU.LOADMACRO.INSTR1",
                    "SFPMAD(12, 0, 13, 13, 0)",
                    clamp,
                    "backdoor",
                ),
                (
                    "SFPU.LOADMACRO.INSTR2",
                    "SFP_STOCH_RND(0, 0, 0, 0, 14, 14)",
                    clamp,
                    "backdoor",
                ),
                ("SFPU.LOADMACRO.INSTR3", "SFPSHFT(15, 0, 15, 1)", clamp, "backdoor"),
                (
                    "SFPU.LOADMACRO.SEQ1",
                    "0x13000004",
                    clamp,
                    f"src=LREG0;staged={_call_line(EXP, '0x0004);  // slot1')},{_call_line(EXP, '0, 0x8, 0x1300);')};uses=simple:INSTR0,store:STORE",
                ),
                (
                    "SFPU.LOADMACRO.SEQ0",
                    "0x631685df",
                    clamp,
                    f"src=LREG0;staged={_call_line(EXP, '0x85DF);  // slot1')},{_call_line(EXP, '0x6316);  // slot3')};uses=simple:INSTR3,mad:INSTR1,round:INSTR2,store:STORE",
                ),
                ("SFPU.LOADMACRO.CTRL", "0x0", clamp, "src=imm16"),
                (
                    "SFPU.LOADMACRO.INSTR1",
                    "SFPMAD(12, 0, 13, 13, 0)",
                    plain,
                    "backdoor",
                ),
                (
                    "SFPU.LOADMACRO.INSTR2",
                    "SFP_STOCH_RND(0, 0, 0, 0, 14, 7)",
                    plain,
                    "backdoor",
                ),
                ("SFPU.LOADMACRO.INSTR3", "SFPSETSGN(0, 4, 15, 0)", plain, "backdoor"),
                (
                    "SFPU.LOADMACRO.SEQ0",
                    "0x731e85ef",
                    plain,
                    f"src=LREG0;staged={_line(EXP, 'TTI_SFPLOADI(0, 0xA, 0x85EF);')},{_line(EXP, 'TTI_SFPLOADI(0, 0x8, 0x731E);')};uses=simple:INSTR3,mad:INSTR1,round:INSTR2,store:STORE",
                ),
                ("SFPU.LOADMACRO.CTRL", "0xf00", plain, "src=imm16"),
            },
        )
        # the 32 instructions after lltt::record(0, 32) only fill the replay buffer
        record = _line(EXP, "lltt::record(0, 32);")
        deferred = {(e[0], e[1], e[5]) for e in effects if e[6] == "deferred:replay"}
        self.assertIn(
            ("sfpu.loadmacro.seq", "SFPU.LOADMACRO.SEQ0", record + 4), deferred
        )
        self.assertIn(("sfpu.const_read", "SFPU.LREG14", record + 5), deferred)
        exp_lines = SNIPPETS[EXP].split("\n")
        window_end = next(
            n
            for n in range(record + 1, len(exp_lines) + 1)
            if exp_lines[n - 1].strip() == "TTI_SFPNOP;"
        )
        self.assertGreater(window_end, record + 32)
        self.assertFalse(
            [
                e
                for e in effects
                if e[0].startswith("sfpu.lreg") and record < e[5] <= window_end
            ]
        )
        self.assertEqual(
            {e[0] for e in effects if e[1] == "SFPU.CTRL"}, {"sfpu.backdoor.ctrl"}
        )

    def test_exp_tile_reaches_calculate_and_keeps_the_calling_thread(self) -> None:
        replay = _line(EXP, "lltt::replay(0, 16);")
        for name, thread in (("exp_tile", "math"), ("exp_packthread_tile", "pack")):
            effects = self.db.effects(self.fn(name).key)
            linked = [
                e
                for e in effects
                if e.pattern == "sfpu.replay.loadmacro.seq"
                and e.evidence == f"{EXP}:{replay}"
            ]
            self.assertEqual(len(linked), 1, name)
            (effect,) = linked
            self.assertEqual(
                (effect.resource, effect.thread, effect.detail),
                ("SFPU.LOADMACRO.SEQ0", thread, "via_replay=exp_init"),
            )
            self.assertEqual(
                [step[0].split("(")[0].split("<")[0] for step in effect.via],
                [
                    "_llk_math_eltwise_unary_sfpu_params_",
                    "ckernel::sfpu::calculate_exponential",
                ],
            )
            self.assertIn("approx && iterations == 8", effect.condition)
            sequences = {
                e.resource for e in effects if e.pattern == "sfpu.loadmacro.seq"
            }
            self.assertEqual(
                sequences, {"SFPU.LOADMACRO.SEQ0", "SFPU.LOADMACRO.SEQ1"}, name
            )
            self.assertEqual(
                {e.thread for e in effects if e.family == "sfpu"}, {thread}, name
            )
        wrapper = self.fn("_llk_math_eltwise_unary_sfpu_params_")
        self.assertEqual(wrapper.thread, "any")
        injected = [edge for edge in wrapper.calls if edge.name == "sfpu_func"]
        self.assertEqual([edge.resolution for edge in injected], ["param_callable"])
        self.assertEqual(injected[0].args, ("args...",))
        init = {
            (e.resource, e.value)
            for e in self.call(
                "exp_tile_init",
                "true",
                "0x3F800000",
                "InputClamping::ClampToNegative",
                "false",
            )
        }
        self.assertIn(
            (
                "SFPU.LREG12",
                "bits((256.0f * 1.4426950408889634f) * __builtin_bit_cast(float, 0x3F800000))",
            ),
            init,
        )
        self.assertIn(("SFPU.LOADMACRO.SEQ0", "0x631685df"), init)

    def test_calculate_exponential_reads_the_constants_of_the_installed_templates(
        self,
    ) -> None:
        effects = [
            e for e in self.fn("calculate_exponential").effects if e.family == "sfpu"
        ]
        seq1 = [
            e
            for e in effects
            if e.pattern == "sfpu.loadmacro.seq" and e.resource == "SFPU.LOADMACRO.SEQ1"
        ]
        self.assertEqual(len(seq1), 1)
        first_macro = seq1[0].evidence
        replay8 = f"{EXP}:{_line(EXP, 'lltt::replay(0, 16);')}"
        replay32 = f"{EXP}:{_line(EXP, 'lltt::replay(0, 32);')}"
        template = {
            (e.resource, e.evidence, e.detail, e.confidence)
            for e in effects
            if e.pattern == "sfpu.loadmacro.template_const"
        }
        expected = set()
        for reg in (
            "SFPU.LREG12",
            "SFPU.LREG13",
        ):  # SFPMAD(12, 0, 13, 13, 0) installed as INSTR1 by exp_init
            expected.add((reg, first_macro, "via_template=exp_init:INSTR1", "low"))
            for site in (replay8, replay32):
                expected.add(
                    (
                        reg,
                        site,
                        "via_replay=exp_init;via_template=exp_init:INSTR1",
                        "low",
                    )
                )
        self.assertEqual(template, expected)

    # ---- reciprocal: backdoor templates + replay of a load_replay_buf lambda ---- #
    def test_reciprocal_24b_init_backdoor_templates_and_sequences(self) -> None:
        effects = self.direct("_init_reciprocal_fast_24b_5c_")
        cond = "!defined(DISABLE_SFPLOADMACRO)"
        programmed = {
            (e[1], e[3], e[6])
            for e in effects
            if e[0] in ("sfpu.backdoor", "sfpu.sfpconfig")
        }
        lines = {
            name: _line(RECIP, f"TTI_SFPCONFIG(0, {dest}, 0);")
            for name, dest in (("SEQ0", "4"), ("SEQ1", "4 + 1"), ("SEQ3", "4 + 3"))
        }
        lower = [
            _line(
                RECIP,
                "TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);",
                n,
            )
            for n in (1, 2, 3)
        ]
        self.assertEqual(
            programmed,
            {
                (
                    "SFPU.LOADMACRO.INSTR0",
                    "SFPARECIP(0, 0, 12, sfpi::SFPARECIP_MOD1_RECIP)",
                    "backdoor",
                ),
                (
                    "SFPU.LOADMACRO.INSTR1",
                    "SFPMAD(p_sfpu::LREG0, p_sfpu::LREG0, 0, 13, 0)",
                    "backdoor",
                ),
                (
                    "SFPU.LOADMACRO.INSTR2",
                    "SFPMAD(1, p_sfpu::LREG0, 2, 14, 0)",
                    "backdoor",
                ),
                (
                    "SFPU.LOADMACRO.INSTR3",
                    "SFPSWAP(0, p_sfpu::LCONST_1, 15, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX)",
                    "backdoor",
                ),
                (
                    "SFPU.LOADMACRO.SEQ0",
                    "0x73000004",
                    f"src=LREG0;staged={lower[0]},{lower[0] + 1};uses=simple:INSTR0,store:STORE",
                ),
                (
                    "SFPU.LOADMACRO.SEQ1",
                    "0x13001544",
                    f"src=LREG0;staged={lower[1]},{lower[1] + 1};uses=simple:INSTR0,mad:INSTR1,store:STORE",
                ),
                (
                    "SFPU.LOADMACRO.SEQ2",
                    "0x697",
                    "src=imm16;uses=simple:INSTR3,mad:INSTR2",
                ),
                (
                    "SFPU.LOADMACRO.SEQ3",
                    "0x1b008e00",
                    f"src=LREG0;staged={lower[2]},{lower[2] + 1};uses=mad:INSTR2,store:STORE",
                ),
                ("SFPU.LOADMACRO.CTRL", "0xff0", "src=imm16"),
            },
        )
        self.assertEqual({e[4] for e in effects}, {cond})
        self.assertEqual(lines["SEQ0"], lower[0] + 2)
        lambda_reads = {(e[1], e[5]) for e in effects if e[0] == "sfpu.loadmacro.seq"}
        first = _line(
            RECIP,
            "TTI_SFPLOADMACRO((0 << 2) | (y & 3), 0, ADDR_MOD_7, offset | (y >> 2));",
        )
        self.assertEqual(
            lambda_reads,
            {
                ("SFPU.LOADMACRO.SEQ0", first),
                ("SFPU.LOADMACRO.SEQ2", first + 1),
                ("SFPU.LOADMACRO.SEQ1", first + 2),
                ("SFPU.LOADMACRO.SEQ3", first + 4),
            },
        )
        self.assertEqual(
            {e[6] for e in effects if e[0].startswith("sfpu.loadmacro")},
            {"deferred:replay"},
        )

    def test_reciprocal_24b_calculate_replays_the_init_macros(self) -> None:
        effects = self.direct("_calculate_reciprocal_fast_24b_5c_")
        first = _line(RECIP, "lltt::replay(0, 4);")
        loop = _line(RECIP, "lltt::replay(0, 5);")
        cond = "!(defined(DISABLE_SFPLOADMACRO))"
        via = "via_replay=_init_reciprocal_fast_24b_5c_"
        seq = {(e[1], e[5]) for e in effects if e[0] == "sfpu.replay.loadmacro.seq"}
        self.assertEqual(
            seq,
            {
                ("SFPU.LOADMACRO.SEQ0", first),
                ("SFPU.LOADMACRO.SEQ1", first),
                ("SFPU.LOADMACRO.SEQ2", first),
                ("SFPU.LOADMACRO.SEQ3", loop),
            },
        )
        self.assertEqual(
            {(e[4], e[6]) for e in effects if e[0].startswith("sfpu.replay.")},
            {(cond, via)},
        )
        self.assertIn(
            (
                "sfpu.replay.loadmacro.ctrl",
                "SFPU.LOADMACRO.CTRL",
                "read",
                "",
                cond,
                first,
                via,
            ),
            effects,
        )
        self.assertIn(
            (
                "sfpu.lreg.live_out",
                "SFPU.LREG7",
                "write",
                "?SFPLOAD",
                cond,
                _line(RECIP, "TTI_SFPLOAD(7, 0, ADDR_MOD_6, 0);"),
                "live_out",
            ),
            effects,
        )
        self.assertIn(
            (
                "sfpu.swap.ctrl",
                "SFPU.CTRL",
                "read",
                "",
                "defined(DISABLE_SFPLOADMACRO)",
                _line(RECIP, "TTI_SFPSWAP(0, p_sfpu::LCONST_1, p_sfpu::LREG3"),
                "swap_mode",
            ),
            effects,
        )

    # ---- runtime LOADMACRO operands and lane masks ---------------------------- #
    def test_signbit_loadmacro_with_runtime_vd(self) -> None:
        effects = {
            e for e in self.fn("calculate_signbit").effects if e.family == "sfpu"
        }
        macro = _line(
            SIGNBIT,
            "TT_SFPLOADMACRO((0 << 2) | (a & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_6, offset | (a >> 2));",
        )
        cond = "!defined(DISABLE_SFPLOADMACRO)"
        seq = [e for e in effects if e.pattern == "sfpu.loadmacro.seq"]
        self.assertEqual(
            [(e.resource, e.condition, e.confidence) for e in seq],
            [("SFPU.LOADMACRO.SEQ0", cond, "high")],
        )
        vd = sorted(
            (e.resource, e.detail, e.confidence)
            for e in effects
            if e.pattern == "sfpu.lreg.live_out"
            and int(e.evidence.rsplit(":", 1)[1]) == macro
        )
        self.assertEqual(
            vd,
            [
                ("SFPU.LREG0", "live_out;may", "low"),
                ("SFPU.LREG1", "live_out;may", "low"),
            ],
        )

    def test_sinkhorn_parity_mask_is_a_lane_masked_lreg11_write(self) -> None:
        line = _line(
            SINKHORN,
            "TTI_SFPCONFIG(0x5555, /*LREG11=*/11, /*MOD1_IMM16_IS_LANE_MASK=*/8);",
        )
        self.assertEqual(
            self.direct("_sinkhorn_program_parity_mask_"),
            {
                (
                    "sfpu.lreg.live_out",
                    "SFPU.LREG0",
                    "write",
                    "?SFPAND",
                    "",
                    line - 1,
                    "live_out",
                ),
                (
                    "sfpu.sfpconfig",
                    "SFPU.LREG11",
                    "write",
                    "?SFPAND",
                    "",
                    line,
                    f"src=LREG0;staged={line - 1};lane_mask=0x5555",
                ),
            },
        )

    def test_tt_insn_of_a_patched_sfploadi_word_writes_lreg7(self) -> None:
        issue = _line(TRIANGLE, "TT_INSN((OPCODE ^ FLIP) ^ imm16);")
        self.assertEqual(
            self.direct("_triangle_solve_sfploadi_lreg7_"),
            {
                (
                    "sfpu.lreg.live_out",
                    "SFPU.LREG7",
                    "write",
                    "?SFPLOADI",
                    "",
                    issue,
                    "live_out",
                )
            },
        )
        packed = self.direct("_triangle_solve_sfploadi_lreg7_lo_pack_")
        self.assertEqual(
            packed,
            {
                (
                    "sfpu.lreg.live_out",
                    "SFPU.LREG7",
                    "write",
                    "?SFPLOADI",
                    "!(defined(__riscv_xttzbkb) || defined(__riscv_zbkb))",
                    _line(
                        TRIANGLE,
                        "TT_INSN(OPCODE ^ (bits & TRIANGLE_SOLVE_IMM16_MASK));",
                    ),
                    "live_out",
                )
            },
        )

    def test_init_execute_pairs_of_the_exp_compute_api(self) -> None:
        pairs = {
            p["execute"].split("<")[0]: p
            for p in sfpu_family.init_execute_pairs(self.db)
        }
        exp = pairs["ckernel::exp_tile"]
        self.assertEqual(
            (exp["rule"], exp["self_init"], exp["path"]), ("name", False, EXP_API)
        )
        self.assertEqual(
            [k.split("<")[0] for k in exp["inits"]], ["ckernel::exp_tile_init"]
        )
        self.assertIn("SFPU.LOADMACRO.SEQ0", exp["reads"])
        self.assertIn("SFPU.LREG14", exp["reads"])
        self.assertTrue(
            {
                "SFPU.LREG12",
                "SFPU.LREG13",
                "SFPU.LREG14",
                "SFPU.LOADMACRO.SEQ0",
                "SFPU.CTRL",
            }
            <= set(exp["writes"])
        )
        pack = pairs["ckernel::exp_packthread_tile"]
        self.assertEqual(
            [k.split("<")[0] for k in pack["inits"]],
            ["ckernel::exp_packthread_tile_init"],
        )
        self.assertEqual(
            pack["reaches_init"], []
        )  # main: the execute does not run its own init

    # ---- plugin contract ------------------------------------------------------ #
    def test_core_sfpu_recognizers_are_replaced(self) -> None:
        patterns = {e.pattern for fn in self.db.functions.values() for e in fn.effects}
        for core in sfpu_family.DISABLE_CORE:
            self.assertNotIn(core, patterns)
        self.assertIn("core.sfpi_constant_read", {p.id for _, p in self.db.patterns})
        produced = {p for p in patterns if p.startswith("sfpu.")}
        self.assertTrue(
            produced <= set(sfpu_family.EXTRACT_PATTERNS),
            produced - set(sfpu_family.EXTRACT_PATTERNS),
        )

    def test_scope_and_thread_hook(self) -> None:
        self.assertEqual(sfpu_family.scope_kind(EXP), "llk_api_sfpu")
        self.assertEqual(sfpu_family.scope_kind(LOAD_CONFIG), "common_sfpu")
        self.assertEqual(sfpu_family.scope_kind(PARAMS), "llk_lib_sfpu")
        self.assertEqual(sfpu_family.scope_kind(EXP_API), "compute_api_sfpu")
        self.assertIsNone(
            sfpu_family.scope_kind(
                "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary.h"
            )
        )
        self.assertTrue(sfpu_family.in_scope(self.fn("exp_tile").definition))
        self.assertEqual(
            self.fn("_llk_math_eltwise_sfpu_apply_vector_mode_").thread, "any"
        )

    def test_output_is_deterministic(self) -> None:
        first = effects_db.dumps_effects(self.db.to_json())
        second = effects_db.dumps_effects(
            effects_db.build(self.root, families=["sfpu"]).to_json()
        )
        self.assertEqual(first, second)


class SfpuHelperTest(unittest.TestCase):
    def test_combine_halves(self) -> None:
        self.assertEqual(sfpu_family._combine("16312", "43579"), "0x3fb8aa3b")
        self.assertEqual(
            sfpu_family._combine("hi16(A * s)", "lo16(A * s)"), "bits(A * s)"
        )
        self.assertEqual(sfpu_family._combine("0", None), "((0) << 16) | (?)")

    def test_decode_sequence(self) -> None:
        self.assertEqual(
            sfpu_family._decode_sequence(0x731E85EF),
            "simple:INSTR3,mad:INSTR1,round:INSTR2,store:STORE",
        )
        self.assertEqual(
            sfpu_family._decode_sequence(0x0697, imm16=True), "simple:INSTR3,mad:INSTR2"
        )
        self.assertEqual(sfpu_family._decode_sequence(0), "none")

    def test_condition_parts(self) -> None:
        parts = sfpu_family._parts(
            "!(defined(DISABLE_SFPLOADMACRO)) && !(!is_fp32_dest_acc_en)"
        )
        self.assertEqual(
            parts, frozenset({"!defined(DISABLE_SFPLOADMACRO)", "is_fp32_dest_acc_en"})
        )
        self.assertTrue(
            sfpu_family._exclusive(
                sfpu_family._parts("defined(DISABLE_SFPLOADMACRO)"), parts
            )
        )
        self.assertEqual(sfpu_family._parts("a || b && c"), frozenset({"a || b && c"}))


if __name__ == "__main__":
    unittest.main()
