// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "cmath_common.h"
#include "lltt.h"
#include "sfpu/ckernel_sfpu_load_config.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_sqrt.h"
#include "ckernel_sfpu_sqrt_custom.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_log1p.h"
#include "sfpu/ckernel_sfpu_log.h"
#include "sfpu/ckernel_sfpu_polyval.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel::sfpu {

template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_reciprocal_gt0_(sfpi::vFloat x) {
    sfpi::vFloat y = sfpi::approx_recip(x);
    sfpi::vFloat e = -x * y + 1.0f;
    y = y * e + y;
    if constexpr (is_fp32_dest_acc_en) {
        e = -x * y + 1.0f;
        y = y * e + y;
    }

    return y;
}

sfpi_inline sfpi::vFloat _sfpu_sqrt_endpoint_(sfpi::vFloat x) {
    // SQRT_23-bits from ckernel_sfpu_sqrt.h, specialized for endpoint reduction.
    // Valid-domain inputs are non-negative; callers handle domain errors afterward.
    // Zero naturally evaluates to zero.
    sfpi::vInt i = sfpi::as<sfpi::vInt>(sfpi::as<sfpi::vUInt>(x) >> 1);
    sfpi::vFloat y = sfpi::as<sfpi::vFloat>(sfpi::vConstIntPrgm0 - i);

    sfpi::vFloat xy = x * y;
    sfpi::vFloat c = (-y) * xy;
    y = y * (sfpi::vConstFloatPrgm1 + c * (sfpi::vConstFloatPrgm2 + c));

    xy = x * y;
    sfpi::vFloat e = 1.0f + (-y) * xy;
    return e * (0.5f * xy) + xy;
}

template <bool is_fp32_dest_acc_en>
void asin_acos_init() {
    if constexpr (is_fp32_dest_acc_en) {
        sqrt_init<false, is_fp32_dest_acc_en>();
    }
}

static const float PI = 3.1415927410125732f;
static const float PI_2 = 1.5707963705062866f;
static const float PI_4 = 0.7853981852531433f;
static const float FRAC_1_PI = 0.31830987334251404f;
static const float FRAC_2_PI = 0.6366197466850281f;

template <bool is_fp32_dest_acc_en>
static sfpi::vFloat sfpu_tan(sfpi::vFloat x, sfpi::vInt i);

template <>
sfpi_inline sfpi::vFloat sfpu_tan<true>(sfpi::vFloat a, sfpi::vInt i) {
    sfpi::vFloat s = a * a;

    // tan(x) for x in [-PI/4, PI/4]
    sfpi::vFloat t = 0x1.fa9f82p-9f;
    t = t * s + 0x1.2b404p-10f;
    t = t * s + 0x1.4787dp-7f;
    t = t * s + 0x1.620abcp-6f;
    t = t * s + 0x1.ba5716p-5f;
    t = t * s + 0x1.111072p-3f;
    t = t * s + 0x1.555556p-2f;
    t = t * s;

    sfpi::vFloat r = t * a + a;

    v_if(i < 0) {
        // Compensated residual for the reciprocal-correction branch.
        // This preserves precision when tan(x) is near its poles.
        s = -1.0f * r + a;
        s = t * a + s;

        t = sfpi::approx_recip(r);

        // Newton-Raphson refinement.
        // e = 1 - r*t, then t <- t*(1 + e) = t*(2 - r*t)
        sfpi::vFloat e = -r * t + 1.0f;
        // Negate to get t = -1/r.
        t = -t * e - t;

        // Reconstruct tan from corrected reciprocal terms.
        r = r * t + 1.0f;
        r = s * t + r;
        r = r * t + t;
    }
    v_endif;

    return r;
}

template <>
sfpi_inline sfpi::vFloat sfpu_tan<false>(sfpi::vFloat a, sfpi::vInt i) {
    sfpi::vFloat s = a * a;

    // tan(x) for x in [-PI/4, PI/4]
    sfpi::vFloat t = 0x1.4f1f4ep-4f;
    t = t * s + 0x1.02b98p-3f;
    t = t * s + 0x1.55953p-2f;
    t = t * s;

    sfpi::vFloat r = t * a + a;

    v_if(i < 0) {
        t = sfpi::approx_recip(r);
        // Newton-Raphson refinement resulting in r = -1/r.
        sfpi::vFloat e = -r * t + 1.0f;
        // Negate to get t = -1/r.
        r = -t * e - t;
    }
    v_endif;

    return r;
}

// Fast bf16 tan for Blackhole. Origin: llk-bench (LLM-agent-written kernel, claude-opus-5, 2026-08),
// validated exhaustively over all 65,536 bf16 inputs vs the exact golden: max 2 ULP (gate <= 2).
// Measured 395 cycles/tile on p150b (tt-metal v0.76.0 baseline; the bench manifest records no v0.76.0
// production reference number for tan).
// Domain: the bench golden only checked |x| <= 65536 (larger |x| is don't-care there). The previous
// kernel's behaviour beyond that range is NOT preserved by construction and must be checked on hardware.
// State: programs vConstFloatPrgm0/1/2 (LREG12-14) only. No SFPLOADMACRO, no replay slots, no ADDR_MOD_6
// (sfpi dst_reg[] indexing over ADDR_MOD_7, zero increments, from the common init).
// Selected by tangent_init / calculate_tangent when !APPROXIMATION_MODE && !is_fp32_dest_acc_en
// (&& ITERATIONS == 8 in calculate); the fast init replaces the production Cody-Waite constant setup,
// which lives in the same three LREGs.
//
// tan(x) for bf16 on Blackhole SFPU.  10 SFPU ops per 32-lane vector.
//
//   1. j = round(x / pi)                       (1.5*2^23 rounding-bias trick)
//   2. a = x - j*pi                            (2-stage Cody-Waite, |a| <= pi/2)
//   3. tan(a) = a * (1 + C1 + C2 / (K - a*a)),  K = (pi/2)^2
//
// The idea that makes this cheap is keeping the pole *in* the approximation
// instead of folding the argument into [-pi/4, pi/4].  (K - a^2)*tan(a)/a is
// analytic over the whole half-period (2.4674 at a=0, 2 at a=pi/2), so a
// linear numerator already fits to 0.17%.  Consequences:
//   * no quadrant bit, no reciprocal-vs-polynomial branch, no predication --
//     the same 10 instructions run for every lane;
//   * the formula stays valid straight *through* the pole, so it does not
//     matter which side of pi/2 the rounded argument lands on;
//   * the single hardware reciprocal that the pole needs is also the whole
//     numerator evaluation, and no Newton step is required: SFPARECIP is good
//     to +-0.55%, and 2 bf16 ulp is 0.78%.  C1/C2 were fitted against the
//     *measured* SFPARECIP error over the exact set of denominators this
//     kernel produces, which leaves >= 0.3% of slack on the 2-ulp gate.
//   * C1 carries a +2^-9 bias so that the truncating SFPSTORE rounds
//     symmetrically, and the last op is a*q + a rather than a*(1+q) so tiny
//     arguments (where a*q flushes) come back bit-exact.
//
// Only 1/pi and the rounding bias need LREGs; that leaves four live values,
// i.e. two vectors in flight.  The SFPU here is latency-bound (~1.9 cycles
// per dependent op, ~1.15 with two independent chains), so the loop is
// unrolled and interleaved by two.

// -pi = P0 + P1, with P0 bf16-exact: j*P0 (|j| <= 65536/pi) and v + j*P0 are
// then both exact, so the whole reduction error is one fp32 rounding.
static constexpr float TANGENT_FAST_P0 = -3.140625f;
static constexpr float TANGENT_FAST_P1 = -0.0009676535846665502f;
static constexpr float TANGENT_FAST_K = 2.4674010276794434f;  // (pi/2)^2
static constexpr float TANGENT_FAST_INV_PI = 0.31830987334251404f;
static constexpr float TANGENT_FAST_C1 = -0.8121f;  // C1 - 1
static constexpr float TANGENT_FAST_C2 = 2.0119f;

inline void _init_tangent_bf16_fast_() {
    sfpi::vConstFloatPrgm0 = TANGENT_FAST_P0;
    sfpi::vConstFloatPrgm1 = TANGENT_FAST_K;
    sfpi::vConstFloatPrgm2 = TANGENT_FAST_P1;
}

inline void _calculate_tangent_bf16_fast_() {
    const sfpi::vFloat p0 = sfpi::vConstFloatPrgm0;
    const sfpi::vFloat k = sfpi::vConstFloatPrgm1;
    const sfpi::vFloat p1 = sfpi::vConstFloatPrgm2;
    const sfpi::vFloat bias = sfpi::sFloat16b(0x1.8p23f);
    const sfpi::vFloat inv_pi = TANGENT_FAST_INV_PI;
    const sfpi::vFloat c1 = TANGENT_FAST_C1;
    const sfpi::vFloat c2 = TANGENT_FAST_C2;

#pragma GCC unroll 4
    for (size_t i = 0; i < 8; i += 2) {
        sfpi::vFloat v0 = sfpi::dst_reg[i];
        sfpi::vFloat v1 = sfpi::dst_reg[i + 1];

        // j = round(v/pi): the bias forces rounding at the units place.
        sfpi::vFloat j0 = __builtin_rvtt_sfpmad(v0.get(), inv_pi.get(), bias.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
        sfpi::vFloat j1 = __builtin_rvtt_sfpmad(v1.get(), inv_pi.get(), bias.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
        j0 -= bias;
        j1 -= bias;

        sfpi::vFloat a0 = j0 * p0 + v0;
        sfpi::vFloat a1 = j1 * p0 + v1;
        a0 = j0 * p1 + a0;
        a1 = j1 * p1 + a1;

        sfpi::vFloat d0 = -a0 * a0 + k;
        sfpi::vFloat d1 = -a1 * a1 + k;

        sfpi::vFloat r0 = sfpi::approx_recip(d0);
        sfpi::vFloat r1 = sfpi::approx_recip(d1);

        sfpi::vFloat q0 = c2 * r0 + c1;
        sfpi::vFloat q1 = c2 * r1 + c1;

        sfpi::dst_reg[i] = a0 * q0 + a0;
        sfpi::dst_reg[i + 1] = a1 * q1 + a1;
    }
}

// P2/P3 of the four-stage Cody-Waite reduction by PI/2 plus 2/PI, read by calculate_tangent's generic body from
// vConstFloatPrgm0..2. tangent_init programs these except on the fast bf16 gate (same LREGs); the ITERATIONS != 8
// fallback in calculate_tangent re-seeds them.
inline void _init_tangent_body_constants_() {
    // P2 and P3 of four-part Cody-Waite reduction by PI/2.
    sfpi::vConstFloatPrgm0 = -0x1.51p-22f;
    sfpi::vConstFloatPrgm1 = -0x1.0b4612p-34f;

    sfpi::vConstFloatPrgm2 = FRAC_2_PI;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_tangent() {
    if constexpr (!APPROXIMATION_MODE && !is_fp32_dest_acc_en && ITERATIONS == 8) {
        _calculate_tangent_bf16_fast_();
        return;
    }
    if constexpr (
        (!APPROXIMATION_MODE && !is_fp32_dest_acc_en) &&
        !(!APPROXIMATION_MODE && !is_fp32_dest_acc_en && ITERATIONS == 8)) {
        // ITERATIONS != 8: tangent_init<false, false> cannot see ITERATIONS and has programmed the fast kernel's
        // LREG12-14 over the Cody-Waite constants this path reads from vConstFloatPrgm0..2; re-seed them.
        _init_tangent_body_constants_();
    }

    // Constants for four-stage Cody-Waite reduction with -PI/2 = P0 + P1 + P2 + P3
    const float P0 = -0x1.92p+0f;   // representable as bf16
    const float P1 = -0x1.fbp-12f;  // representable as fp16

    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vInt i;

        sfpi::vFloat inv_pio2 = sfpi::vConstFloatPrgm2;

        // j = round(v / (PI/2))
        // j = v * (2/PI) + 1.5*2**23 shifts the mantissa bits to give round-to-nearest.
        // Workaround for SFPI's insistence on generating SFPADDI+SFPMUL instead of SFPLOADI+SFPMAD here.
        sfpi::vFloat rounding_bias = sfpi::sFloat16b(0x1.8p23f);
        sfpi::vFloat j =
            __builtin_rvtt_sfpmad(v.get(), inv_pio2.get(), rounding_bias.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);

        // We need the LSB of the integer later, to determine the sign of the result.
        i = sfpi::as<sfpi::vInt>(j);

        // Shift mantissa bits back; j is now round(v / (PI/2)) in fp32.
        j += -rounding_bias;

        i <<= 31;

        // Four-stage Cody-Waite reduction; a = v - j * (PI/2).
        // P0 representable as bf16; generates a single SFPLOADI, filling NOP slot from previous SFPADDI.
        sfpi::vFloat a = v + j * P0;
        // P1 representable as fp16; generates a single SFPLOADI, filling NOP slot from previous SFPMAD.
        a = a + j * P1;
        a = a + j * sfpi::vConstFloatPrgm0;
        a = a + j * sfpi::vConstFloatPrgm1;

        a = sfpu_tan<is_fp32_dest_acc_en>(a, i);

        if constexpr (!is_fp32_dest_acc_en) {
            a = sfpi::convert<sfpi::vFloat16b>(a, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = a;
        sfpi::dst_reg++;
    }
}

#ifndef DISABLE_SFPLOADMACRO
// Fast bf16 sin for Blackhole. Origin: llk-bench (LLM-agent-written kernel, claude-fable-5, 2026-08),
// validated exhaustively over all 65,536 bf16 inputs vs the exact golden: max 1 ULP (gate <= 2).
// Measured 370 cycles/tile vs 935 for the previous kernel on p150b (tt-metal v0.76.0 baseline).
// Domain: the bench golden only checked |x| <= 65536 (larger |x| is don't-care there). The previous
// kernel's behaviour beyond that range is NOT preserved by construction and must be checked on hardware.
// State: programs LREG11-14 via SFPCONFIG (LREG11 = rounding bias, LREG12 = 1/pi, LREG13 = -P1,
// LREG14 = C0), SFPLOADMACRO instruction templates 0..3, macro sequences 0..3 (SFPCONFIG dest 4..7) and
// the macro Misc register (dest 8). No replay slots, no ADDR_MOD_6 (raw TTI_* with ADDR_MOD_7).
// Selected by sine_init / calculate_sine when !APPROXIMATION_MODE && !is_fp32_dest_acc_en
// (&& ITERATIONS == 8 in calculate); the fast init replaces the production Cody-Waite constant setup
// (vConstFloatPrgm0/1/2 = LREG12-14, the same LREGs the fast kernel programs).
// Integration note: LREG11 is the architectural -1.0f constant (sfpi vConstNeg1) and nothing in the
// per-op common init (SFPCONFIG(0, 0xF, 1) only writes LaneConfig) restores it, so the per-face body
// below re-loads the bias into LREG11 on entry and restores -1.0f on exit; the bench kernel left the
// bias in LREG11 permanently (it ran alone). Those two config writes are the only additions.
//
// Algorithm (validated bit-exactly offline against the Blackhole FMA model):
//   j  = round(x / pi)            (bias trick: x*(1/pi) + 1.5*2^23)
//   a  = x - j*P0 - j*P1          (2-stage Cody-Waite; P0 = 3.140625 has an
//                                  exact 28-bit product with any |j| <= 2^15,
//                                  P1 = fp32(pi - P0) rides the partially
//                                  fused MAD's 28-bit product precision)
//   sin(x) = (-1)^j * (a + a^3*(C0 + C1*a^2))   (degree-5 odd minimax)
// Sign fold: parity bit of the biased j, shifted to bit 31, XORed into the
// reduced argument. Plain truncating SFPSTORE to bf16 (whole pipeline stays
// within 1 ULP of the correctly rounded golden over all 65536 bf16 inputs).
//
// Performance: hand-scheduled with SFPLOADMACRO so three ops per vector ride
// the loads for free (jb-MAD + parity-shift on the first load, stage-1
// reduction MAD on the second load). Two dst vectors are processed in
// lockstep so every issued instruction is independent of its predecessor:
// 21 issued instructions per vector pair, ~1 instruction/cycle.
//
// Register map (per pair; A = even vector, B = odd vector):
//   L0: xA -> jbA -> qA -> sA        L4: -P0 (pinned per face)
//   L1: xB -> jbB -> qB -> sB        L5: jA -> pA -> vA
//   L2: xA' -> a1A -> a2A -> aA -> rB L6: jB -> pB -> vB
//   L3: xB' -> a1B -> a2B -> aB      L7: C1 -> rA
// Constant regs: L11 = bias, L12 = 1/pi, L13 = -P1, L14 = C0.
//
// Macros (configured in _init_sine_bf16_fast_):
//   M0 (first load of a vector, VD=L0/L1):
//     MAD slot, delay 0:   VD = 1/pi * VD + bias          (template T0)
//     Simple slot, delay 3: VD <<= 31 (parity -> sign bit) (template T1)
//   M1 (second load of vector A, VD=L2):
//     MAD slot, delay 2:   VD = jA * (-P0) + VD            (template T2)
//   M2 (second load of vector B, VD=L3):
//     MAD slot, delay 0:   VD = jB * (-P0) + VD            (template T3)
// Delays are instruction-counted (UnitDelayKind = all ones), so the schedule
// is robust against any issue-side stalls.
inline void _init_sine_bf16_fast_() {
    // Programmable constant LREGs. LREG11 normally holds -1.0f; nothing in
    // this kernel uses -1.0f, so it is repurposed for the rounding bias
    // (re-loaded per face and restored to -1.0f afterwards, see
    // _calculate_sine_bf16_fast_).
    _sfpu_load_config32_(11, 0x4B40, 0x0000);  // bias = 1.5*2^23
    _sfpu_load_config32_(12, 0x3EA2, 0xF983);  // 1/pi
    _sfpu_load_config32_(13, 0xBA7D, 0xAA22);  // -P1 = -(pi - 3.140625)
    _sfpu_load_config32_(14, 0xBE2A, 0x1BF3);  // C0 (refit for fp16 C1)

    // Instruction templates for the load macros.
    _sfpu_load_imm32_(0, TT_OP_SFPMAD(12, 0, 11, 0, 0));  // T0: jb = 1/pi*VB + bias
    TTI_SFPCONFIG(0, 0, 0);
    _sfpu_load_imm32_(0, TT_OP_SFPSHFT(31, 0, 0, 1));     // T1: VD <<= 31
    TTI_SFPCONFIG(0, 1, 0);
    _sfpu_load_imm32_(0, TT_OP_SFPMAD(5, 4, 0, 0, 0));    // T2: a1A = jA*(-P0) + VC
    TTI_SFPCONFIG(0, 2, 0);
    _sfpu_load_imm32_(0, TT_OP_SFPMAD(6, 4, 0, 0, 0));    // T3: a1B = jB*(-P0) + VC
    TTI_SFPCONFIG(0, 3, 0);

    // Macro sequences: per sub-unit byte = [7]=VB-not-VC override,
    // [6]=VD->LREG16, [5:3]=delay, [2:0]=template select (4..7 = T0..T3).
    // Word layout: Simple | MAD<<8 | Round<<16 | Store<<24.
    TTI_SFPCONFIG(0x849D, 4, 1);  // M0: MAD=T0 d0 VB<-load; Simple=T1 d3 VB<-load
    TTI_SFPCONFIG(0x1600, 5, 1);  // M1: MAD=T2 d2 VC<-load
    TTI_SFPCONFIG(0x0700, 6, 1);  // M2: MAD=T3 d0 VC<-load
    TTI_SFPCONFIG(0x0000, 7, 1);  // M3: unused
    // Misc: UnitDelayKind = 0xF (instruction-counted delays on all sub-units).
    TTI_SFPCONFIG(0x0F00, 8, 1);
    TTI_SFPNOP;
}

template <int A>
inline __attribute__((always_inline)) void _sine_bf16_fast_pair_() {
    constexpr int B = A + 2;
    TTI_SFPLOADMACRO((0 << 2) | 0, 0, 7, A);  // M0: L0 = xA; jbA; qA
    TTI_SFPLOADMACRO((0 << 2) | 1, 0, 7, B);  // M0: L1 = xB; jbB; qB
    TTI_SFPLOADMACRO((1 << 2) | 2, 0, 7, A);  // M1: L2 = xA; a1A (after jA)
    TTI_SFPMAD(0, 10, 11, 5, 2);              // jA = jbA - bias        -> L5
    TTI_SFPMAD(1, 10, 11, 6, 2);              // jB = jbB - bias        -> L6
    TTI_SFPLOADMACRO((2 << 2) | 3, 0, 7, B);  // M2: L3 = xB; a1B
    TTI_SFPLOADI(7, 1, 0x1FD6);               // C1 (fp16)              -> L7
    TTI_SFPMAD(5, 13, 2, 2, 0);               // a2A = jA*(-P1) + a1A   -> L2
    TTI_SFPMAD(6, 13, 3, 3, 0);               // a2B = jB*(-P1) + a1B   -> L3
    TTI_SFPXOR(0, 0, 2, 0);                   // aA ^= qA
    TTI_SFPXOR(0, 1, 3, 0);                   // aB ^= qB
    TTI_SFPMAD(2, 2, 9, 0, 0);                // sA = aA*aA             -> L0
    TTI_SFPMAD(3, 3, 9, 1, 0);                // sB = aB*aB             -> L1
    TTI_SFPMAD(0, 7, 14, 5, 0);               // pA = sA*C1 + C0        -> L5
    TTI_SFPMAD(1, 7, 14, 6, 0);               // pB = sB*C1 + C0        -> L6
    TTI_SFPMAD(5, 0, 10, 5, 0);               // vA = pA*sA + 1         -> L5
    TTI_SFPMAD(6, 1, 10, 6, 0);               // vB = pB*sB + 1         -> L6
    TTI_SFPMAD(5, 2, 9, 7, 0);                // rA = vA*aA             -> L7
    TTI_SFPMAD(6, 3, 9, 2, 0);                // rB = vB*aB             -> L2
    TTI_SFPSTORE(7, 0, 7, A);                 // dst[A] = rA (truncate to bf16)
    TTI_SFPSTORE(2, 0, 7, B);                 // dst[B] = rB
}

inline void _calculate_sine_bf16_fast_() {
    // LREG11 holds the rounding bias for this op (programmed by _init_sine_bf16_fast_). The next SFPU op's init
    // re-establishes the architectural -1.0f through _init_sfpu_config_reg(), so no per-face restore is needed.
    TTI_SFPLOADI(4, 0, 0xC049);  // -P0 = -3.140625 (bf16) -> L4, per face
    _sine_bf16_fast_pair_<0>();
    _sine_bf16_fast_pair_<4>();
    _sine_bf16_fast_pair_<8>();
    _sine_bf16_fast_pair_<12>();
}
#endif  // DISABLE_SFPLOADMACRO

// P2/P3 of the four-stage Cody-Waite reduction by PI plus 1/PI, read by calculate_sine's generic body from
// vConstFloatPrgm0..2. sine_init programs these except on the fast bf16 gate (same LREGs); the ITERATIONS != 8
// fallback in calculate_sine re-seeds them.
inline void _init_sine_body_constants_() {
    // P2 and P3 of four-part Cody-Waite reduction by PI.
    sfpi::vConstFloatPrgm0 = -0x1.51p-21f;
    sfpi::vConstFloatPrgm1 = -0x1.0b4612p-33f;

    sfpi::vConstFloatPrgm2 = FRAC_1_PI;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_sine() {
#ifndef DISABLE_SFPLOADMACRO
    if constexpr (!APPROXIMATION_MODE && !is_fp32_dest_acc_en && ITERATIONS == 8) {
        _calculate_sine_bf16_fast_();
        return;
    }
    if constexpr (
        (!APPROXIMATION_MODE && !is_fp32_dest_acc_en) &&
        !(!APPROXIMATION_MODE && !is_fp32_dest_acc_en && ITERATIONS == 8)) {
        // ITERATIONS != 8: sine_init<false, false> cannot see ITERATIONS and has programmed the fast kernel's
        // LREG11-14 over the Cody-Waite constants this path reads from vConstFloatPrgm0..2 and over the
        // architectural -1.0f in LREG11 that sfpi-compiled code assumes; re-seed both (SFPCONFIG imm mode
        // writes the default, as in _init_sfpu_config_reg).
        _init_sine_body_constants_();
        TTI_SFPCONFIG(0, 11, 1);
    }
#endif

    // 1. Reduce argument using a four-stage Cody-Waite reduction to the interval [-PI/2, PI/2].
    // 2. Use odd symmetry (sin(-x) = -sin(x)) via quadrant/sign tracking.
    // 3. Evaluate sin(a) = a + a^3 (C0 + a^2 (C1 + a^2 (C2 + a^2 C3))) on [0, PI/2].

    // Constants for four-stage Cody-Waite reduction with -PI = P0 + P1 + vConstFloatPrgm0 + vConstFloatPrgm1
    const float P0 = -0x1.92p+1f;   // representable as bf16
    const float P1 = -0x1.fbp-11f;  // representable as fp16

    sfpi::vFloat C3, C2, C1, C0;

    // Coefficients are chosen per destination precision target for sin(a) on [0, PI/2].
    if (is_fp32_dest_acc_en) {
        C3 = 0x1.5dc908p-19f;
        C2 = -0x1.9f70fp-13f;
        C1 = 0x1.110edap-7f;
        C0 = -0x1.55554cp-3f;
    } else {
        C2 = -0x1.8b10a4p-13f;
        C1 = 0x1.10c2a2p-7f;
        C0 = -0x1.5554a4p-3f;
    }

    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];

        // Workaround for SFPI's insistence on generating SFPADDI+SFPMUL instead of SFPLOADI+SFPMAD here.
        sfpi::vFloat rounding_bias = sfpi::sFloat16b(0x1.8p23f);
        sfpi::vFloat inv_pi = sfpi::vConstFloatPrgm2;

        // Compute j = round(v / PI).
        // First, j = v * (1 / PI) + 1.5*2^23 shifts the mantissa bits to give round-to-nearest.
        sfpi::vFloat j =
            __builtin_rvtt_sfpmad(v.get(), inv_pi.get(), rounding_bias.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);

        // At this point, the mantissa bits of j contain the integer.
        // Store for later; the LSB determines the sign of the result.
        sfpi::vInt q = sfpi::as<sfpi::vInt>(j);
        // Shift mantissa bits back; j is now round(v / PI) in fp32.
        j = j - rounding_bias;

        // Four-stage Cody-Waite reduction; a = v + j * -PI.
        // P0 representable as bf16; generates a single SFPLOADI, filling NOP slot from previous SFPADDI.
        sfpi::vFloat a = v + j * P0;
        // P1 representable as fp16; generates a single SFPLOADI, filling NOP slot from previous SFPMAD.
        a = a + j * P1;
        a = a + j * sfpi::vConstFloatPrgm0;
        a = a + j * sfpi::vConstFloatPrgm1;

        q <<= 31;
        sfpi::vFloat s = a * a;
        a = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(a) ^ q);

        sfpi::vFloat r;
        if (is_fp32_dest_acc_en) {
            r = C3 * s + C2;
            r = r * s + C1;
            sfpi::vFloat c = a * s;
            r = r * s + C0;
            r = r * c + a;
        } else {
            r = C2 * s + C1;
            sfpi::vFloat c = a * s;
            r = r * s + C0;
            r = r * c + a;
            r = sfpi::convert<sfpi::vFloat16b>(r, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = r;
        sfpi::dst_reg++;
    }
}

// Fast bf16 cos for Blackhole. Origin: llk-bench (LLM-agent-written kernel, claude-fable-5, 2026-08),
// validated exhaustively over all 65,536 bf16 inputs vs the exact golden: max 1 ULP (gate <= 2).
// Measured 375 cycles/tile vs 1095 for the previous kernel on p150b (tt-metal v0.76.0 baseline).
// Domain: the bench golden only checked |x| <= 65536 (larger |x| is don't-care there). The previous
// kernel's behaviour beyond that range is NOT preserved by construction and must be checked on hardware.
// State: programs vConstFloatPrgm0/1/2 (LREG12-14) plus loop constants in L4-L7 at init time (L4-L7 must
// survive between tiles, i.e. no other SFPU op may run between cosine_init and the cos_tile calls it
// serves -- the same init-immediately-before-use contract the Prgm constants already impose). Uses
// replay slots 0..17 (two 9-instruction cores recorded per calculate call, lltt::record<lltt::Exec>).
// No SFPLOADMACRO, no ADDR_MOD_6 (raw TTI_* with ADDR_MOD_7).
// Face handling: the whole 32x32 tile (dst offsets 0..62) is processed on the FIRST of the four per-face
// calls made by the VectorMode::RC loop in _llk_math_eltwise_sfpu_apply_vector_mode_; the other three
// calls return immediately, tracked by a function-local static call counter. calculate_cosine must
// therefore only be reached through that 4-call RC loop (cos_tile does); VectorMode::R/C would
// desynchronise the counter.
// Selected by cosine_init / calculate_cosine when !APPROXIMATION_MODE && !is_fp32_dest_acc_en
// (&& ITERATIONS == 8 in calculate); the fast init replaces the production Cody-Waite constant setup,
// which lives in the same three LREGs.
//
// Algorithm (validated bit-exactly against the documented Blackhole FMA model
// over all 65536 bf16 inputs; max observed error 1 ULP):
//   j = round(x/(2pi))  via the 1.5*2^23 bias trick
//   a = x - j*2pi       via 2-stage Cody-Waite: M0 = -6.28125 (8-bit mantissa,
//                       j*M0 exact in the 27-bit partial product),
//                       M1 = fp32(6.28125 - 2pi)
//   cos(x) = E0 + s*(E1 + s*(E2 + s*E3)),  s = a^2,  a in [-pi, pi]
// The full-period reduction makes cos's sign fall out of the polynomial
// itself (root aligned with pi/2), so there is no quadrant/parity logic at
// all: no shift, no xor, no sign register.
//
// Implementation: hand-scheduled TTI software pipeline, two vectors in
// flight, 12 issue slots per vector with zero idle slots. The whole
// 32-vector tile is processed on the first of the four per-face wrapper
// calls (VectorMode::RC), so the pipeline fills/drains once per tile.
// The 9-instruction loop-invariant core is issued from the replay buffer
// (recorded while executing bodies 1 and 2).
//
// Correctness is interlock-safe: every producer/consumer pair is either
// spaced >= 2 cycles by construction or covered by the hardware's automatic
// one-cycle stall (none of the buggy-stall instructions are used).
//
// Register map:
//   L0/L1 = alternating v -> a1 -> a -> s chain (body k works vector k in V[k%2])
//   L2    = u -> t -> j        L3 = r chain
//   L4 = E3   L5 = M0 = -6.28125   L6 = E0   L7 = E2
//   L12 (Prgm0) = 1/(2pi)   L13 (Prgm1) = M1   L14 (Prgm2) = E1
//
// Steady-state body for vector i (12 slots; tail of vector i-1 interleaved):
//   c0  MAD  Vb = T*M1 + Vb        a2 of i-1 (T = j of i-1 until c1)
//   c1  MUL  T  = Va*(1/2pi)       u of i
//   c2  MUL  Vb = Vb*Vb            s of i-1
//   c3  ADDI T += 12582912         t of i
//   c4  MAD  R  = E3*Vb + E2       r1 of i-1
//   c5  ADDI T -= 12582912         j of i
//   c6  MAD  R  = R*Vb + E1        r2 of i-1
//   c7  MAD  Va = T*M0 + Va        a1 of i (j ready exactly now)
//   c8  MAD  R  = R*Vb + E0        r3 of i-1 (last read of Vb)
//   c9  LOAD Vb <- dst[2(i+1)]     v of i+1 (inline)
//   c10 STORE dst[2(i-1)] <- R     (inline; bf16 truncating store --
//        coefficients are refit with a +2^-9 relative bias to recenter
//        the truncation error, so no explicit rounding op is needed)
inline void _init_cosine_bf16_fast_() {
    sfpi::vConstFloatPrgm0 = 0x1.45f306p-3f;    // L12 = 1/(2pi)
    sfpi::vConstFloatPrgm1 = -0x1.fb5444p-10f;  // L13 = M1 = fp32(6.28125 - 2pi)
    sfpi::vConstFloatPrgm2 = -0x1.fc7f18p-2f;   // L14 = E1 (bias-refit)
    // Loop constants. Nothing else touches the SFPU LRegs between tiles of one op,
    // so these survive across _calculate_cosine_bf16_fast_ calls.
    TTI_SFPLOADI(4, 8, 0xBA81);   // L4 = E3 hi
    TTI_SFPLOADI(4, 10, 0xDC27);  // L4 = E3 lo
    TTI_SFPLOADI(5, 0, 0xC0C9);   // L5 = M0 = -6.28125 (bf16)
    TTI_SFPLOADI(6, 8, 0x3F7F);   // L6 = E0 hi
    TTI_SFPLOADI(6, 10, 0xE0DA);  // L6 = E0 lo
    TTI_SFPLOADI(7, 8, 0x3D21);   // L7 = E2 hi
    TTI_SFPLOADI(7, 10, 0xE284);  // L7 = E2 lo
}

// Loop-invariant 9-instruction core (slots c0..c8). VA/VB: (1,0) for odd
// bodies, (0,1) for even bodies.
#define COSINE_FAST_CORE(VA, VB)                  \
    TTI_SFPMAD(2, 13, VB, VB, 0);   /* c0 a2'  */ \
    TTI_SFPMUL(VA, 12, 9, 2, 0);    /* c1 u    */ \
    TTI_SFPMUL(VB, VB, 9, VB, 0);   /* c2 s'   */ \
    TTI_SFPADDI(0x4B40, 2, 0);      /* c3 t    */ \
    TTI_SFPMAD(4, VB, 7, 3, 0);     /* c4 r1'  */ \
    TTI_SFPADDI(0xCB40, 2, 0);      /* c5 j    */ \
    TTI_SFPMAD(3, VB, 14, 3, 0);    /* c6 r2'  */ \
    TTI_SFPMAD(2, 5, VA, VA, 0);    /* c7 a1   */ \
    TTI_SFPMAD(3, VB, 6, 3, 0)      /* c8 r3'  */

inline void _calculate_cosine_bf16_fast_() {
    // The RC face loop invokes this once per face; the first call handles the
    // whole tile with immediate dst offsets 0..62.
    static unsigned cosine_fast_call_idx = 0;
    if ((cosine_fast_call_idx++ & 3u) != 0) {
        return;
    }

    // Prologue: front of vector 0 (into L0), preload vector 1 (into L1).
    // Producer/consumer gaps here rely on the hardware interlock.
    TTI_SFPLOAD(0, 0, ADDR_MOD_7, 0);  // L0 = v0
    TTI_SFPLOAD(1, 0, ADDR_MOD_7, 2);  // L1 = v1
    TTI_SFPMUL(0, 12, 9, 2, 0);        // u0
    TTI_SFPADDI(0x4B40, 2, 0);         // t0
    TTI_SFPADDI(0xCB40, 2, 0);         // j0
    TTI_SFPMAD(2, 5, 0, 0, 0);         // a1 of v0

    // Body 1 (odd): execute + record the core into replay slots 0..8.
    lltt::record<lltt::Exec>(0, 9);
    COSINE_FAST_CORE(1, 0);
    TTI_SFPLOAD(0, 0, ADDR_MOD_7, 4);   // v2
    TTI_SFPSTORE(3, 0, ADDR_MOD_7, 0);  // result 0

    // Body 2 (even): execute + record into replay slots 9..17.
    lltt::record<lltt::Exec>(9, 9);
    COSINE_FAST_CORE(0, 1);
    TTI_SFPLOAD(1, 0, ADDR_MOD_7, 6);   // v3
    TTI_SFPSTORE(3, 0, ADDR_MOD_7, 2);  // result 1

    // Bodies 3..31: replay the recorded cores. Body 31 skips the prefetch
    // load (there is no vector 32).
#define COSINE_FAST_BODY(K)                                 \
    lltt::replay(((K) & 1) ? 0 : 9, 9);                     \
    TTI_SFPLOAD(((K) + 1) & 1, 0, ADDR_MOD_7, 2 * (K) + 2); \
    TTI_SFPSTORE(3, 0, ADDR_MOD_7, 2 * (K) - 2)
    COSINE_FAST_BODY(3);
    COSINE_FAST_BODY(4);
    COSINE_FAST_BODY(5);
    COSINE_FAST_BODY(6);
    COSINE_FAST_BODY(7);
    COSINE_FAST_BODY(8);
    COSINE_FAST_BODY(9);
    COSINE_FAST_BODY(10);
    COSINE_FAST_BODY(11);
    COSINE_FAST_BODY(12);
    COSINE_FAST_BODY(13);
    COSINE_FAST_BODY(14);
    COSINE_FAST_BODY(15);
    COSINE_FAST_BODY(16);
    COSINE_FAST_BODY(17);
    COSINE_FAST_BODY(18);
    COSINE_FAST_BODY(19);
    COSINE_FAST_BODY(20);
    COSINE_FAST_BODY(21);
    COSINE_FAST_BODY(22);
    COSINE_FAST_BODY(23);
    COSINE_FAST_BODY(24);
    COSINE_FAST_BODY(25);
    COSINE_FAST_BODY(26);
    COSINE_FAST_BODY(27);
    COSINE_FAST_BODY(28);
    COSINE_FAST_BODY(29);
    COSINE_FAST_BODY(30);
#undef COSINE_FAST_BODY
    lltt::replay(0, 9);                  // body 31 (odd)
    TTI_SFPMAD(2, 13, 1, 1, 0);          // a2 of 31 (in the prefetch slot)
    TTI_SFPSTORE(3, 0, ADDR_MOD_7, 60);  // result 30

    // Epilogue: tail of vector 31 (odd -> chain lives in L1, j in L2).
    TTI_SFPMUL(1, 1, 9, 1, 0);           // s
    TTI_SFPMAD(4, 1, 7, 3, 0);           // r1
    TTI_SFPMAD(3, 1, 14, 3, 0);          // r2
    TTI_SFPMAD(3, 1, 6, 3, 0);           // r3
    TTI_SFPSTORE(3, 0, ADDR_MOD_7, 62);  // result 31
}

#undef COSINE_FAST_CORE

// P2/P3 of the four-stage Cody-Waite reduction by PI/2 plus 1/PI, read by calculate_cosine's generic body from
// vConstFloatPrgm0..2. cosine_init programs these except on the fast bf16 gate (same LREGs); the ITERATIONS != 8
// fallback in calculate_cosine re-seeds them.
inline void _init_cosine_body_constants_() {
    // P2 and P3 of four-part Cody-Waite reduction by PI/2.
    sfpi::vConstFloatPrgm0 = -0x1.51p-22f;
    sfpi::vConstFloatPrgm1 = -0x1.0b4612p-34f;

    sfpi::vConstFloatPrgm2 = FRAC_1_PI;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_cosine() {
    if constexpr (!APPROXIMATION_MODE && !is_fp32_dest_acc_en && ITERATIONS == 8) {
        _calculate_cosine_bf16_fast_();
        return;
    }
    if constexpr (
        (!APPROXIMATION_MODE && !is_fp32_dest_acc_en) &&
        !(!APPROXIMATION_MODE && !is_fp32_dest_acc_en && ITERATIONS == 8)) {
        // ITERATIONS != 8: cosine_init<false, false> cannot see ITERATIONS and has programmed the fast kernel's
        // LREG12-14 over the Cody-Waite constants this path reads from vConstFloatPrgm0..2; re-seed them. (Its
        // L4-L7 loop constants and LREG11 = -1.0f need nothing: sfpi allocates L0-L7 itself and the fast init
        // leaves LREG11 alone.)
        _init_cosine_body_constants_();
    }

    // 1. Build an odd quadrant index j for PI/2-based reduction.
    // 2. Reduce to a in [-PI/2, PI/2] and fold sign from the quadrant parity.
    // 3. Evaluate sin(a) polynomial and use identity cos(x) = sin(x + PI/2).

    // Constants for four-stage Cody-Waite reduction with -PI/2 = P0 + P1 + vConstFloatPrgm0 + vConstFloatPrgm1
    const float P0 = -0x1.92p+0f;   // representable as bf16
    const float P1 = -0x1.fbp-12f;  // representable as fp16

    sfpi::vFloat C3, C2, C1, C0;

    if constexpr (is_fp32_dest_acc_en) {
        // Constants for sin(a) = a + a^3 (C0 + a^2 (C1 + a^2 (C2 + a^2 C3))) on [0, PI/2].
        C3 = 0x1.5dc908p-19f;
        C2 = -0x1.9f70fp-13f;
        C1 = 0x1.110edap-7f;
        C0 = -0x1.55554cp-3f;
    } else {
        C2 = -0x1.8b10a4p-13f;
        C1 = 0x1.10c2a2p-7f;
        C0 = -0x1.5554a4p-3f;
    }

    const float ROUNDING_BIAS = 12582912.0f;
    const float NEG_ROUNDING_BIAS = -12582912.0f;

    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];

        // Force v * (1/PI) + 0.5 to compile as a single SFPMAD sequence for consistent instruction scheduling.
        sfpi::vFloat half = sfpi::sFloat16b(0.5f);
        sfpi::vFloat inv_pi = sfpi::vConstFloatPrgm2;
        sfpi::vFloat neg_one = -1.0f;

        // Start from j = v * (1 / PI) + 0.5; after bias-round and 2*j - 1, j is an odd quadrant index.
        // ROUNDING_BIAS shifts mantissa bits to perform round-to-nearest.
        sfpi::vFloat j = __builtin_rvtt_sfpmad(v.get(), inv_pi.get(), half.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);

        j = j + ROUNDING_BIAS;

        // At this point, the mantissa bits of j contain the rounded integer.
        // Store for later; the LSB tracks quadrant parity for sign selection.
        sfpi::vInt q = sfpi::as<sfpi::vInt>(j);

        j = j + NEG_ROUNDING_BIAS;

        sfpi::vFloat two = sfpi::sFloat16b(2.0f);
        j = __builtin_rvtt_sfpmad(j.get(), two.get(), neg_one.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);

        // Four-stage Cody-Waite reduction; a = v + j * -PI / 2.
        // P0 representable as bf16; generates a single SFPLOADI, filling NOP slot from previous SFPADDI.
        sfpi::vFloat a = v + j * P0;
        // P1 representable as fp16; generates a single SFPLOADI, filling NOP slot from previous SFPMAD.
        a = a + j * P1;
        a = a + j * sfpi::vConstFloatPrgm0;
        a = a + j * sfpi::vConstFloatPrgm1;

        q <<= 31;
        sfpi::vFloat s = a * a;
        a = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(a) ^ q);

        sfpi::vFloat r;
        if constexpr (is_fp32_dest_acc_en) {
            r = C3 * s + C2;
            r = r * s + C1;
            sfpi::vFloat c = a * s;
            r = r * s + C0;
            r = r * c + a;
        } else {
            r = C2 * s + C1;
            sfpi::vFloat c = a * s;
            r = r * s + C0;
            r = r * c + a;
            r = sfpi::convert<sfpi::vFloat16b>(r, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = r;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat sfpu_atan_bf16(sfpi::vFloat val) {
    sfpi::vFloat t0 = sfpi::abs(val);
    sfpi::vFloat result = 0.0f;

    // If input is NaN then output must be NaN as well
    v_if(sfpi::is_nan(val)) { result = std::numeric_limits<float>::quiet_NaN(); }
    v_else {
        sfpi::vFloat absval_minus_1 = t0 - 1.0f;

        v_if(absval_minus_1 >= 0.0f) { t0 = sfpu_reciprocal<false>(t0); }
        v_endif;

        sfpi::vFloat t1 = t0 * t0;

        // Low-degree minimax polynomial (Sollya) for reduced-precision destination path.
        // > fpminimax(atan(x), [|1,3,5,7|], [|single...|], [2^(-40); 1], relative);
        t1 = PolynomialEvaluator::eval(
            t1,
            0.999787867069244384765625f,
            -0.325808584690093994140625f,
            0.1555790007114410400390625f,
            -4.4326744973659515380859375e-2f);

        t1 = t1 * t0;

        v_if(absval_minus_1 >= 0.0f) { t1 = PI_2 - t1; }
        v_endif;

        result = sfpi::copysgn(t1, val);
    }
    v_endif;

    return result;
}

sfpi_inline sfpi::vFloat sfpu_atan_fp32(sfpi::vFloat x) {
    sfpi::vFloat r;
    sfpi::vFloat p;
    sfpi::vFloat s;
    sfpi::vFloat a;
    sfpi::vFloat ax;
    sfpi::vFloat pio2;

    ax = sfpi::setsgn(x, 0);
    a = ax;
    sfpi::vInt e = sfpi::exexp(a);

    v_if(e >= 0) {
        // Use a = 0 for the pi/2 asymptote; NaNs remain NaN.
        a = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(a) - 1) * 0.0f;

        // atan(|x|) rounds to pi/2 for |x| >= 2^26, so skip the reciprocal there.
        // This also avoids the inf * 0 residual produced when approx_recip
        // underflows to zero for very large finite values or infinity.
        v_if(e < 26) { a = _sfpu_reciprocal_gt0_<true>(ax); }
        v_endif;
    }
    v_endif;

    // Minimax approximation of atan(a) on [0, 1].
    {
        p = 0x1.01cp-8f;
        s = a * a;
        sfpi::vFloat c6 = -0x1.4bcp-6f;
        p = __builtin_rvtt_sfpmad(p.get(), s.get(), c6.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
        sfpi::vFloat c5 = 0x1.93p-5f;
        p = __builtin_rvtt_sfpmad(p.get(), s.get(), c5.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
        sfpi::vFloat c4 = -0x1.48cp-4f;
        p = __builtin_rvtt_sfpmad(p.get(), s.get(), c4.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
        sfpi::vFloat c3 = 0x1.bd4p-4f;
        p = __builtin_rvtt_sfpmad(p.get(), s.get(), c3.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
        sfpi::vFloat c2 = -0x1.24p-3f;
        p = __builtin_rvtt_sfpmad(p.get(), s.get(), c2.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
        sfpi::vFloat a3 = s * a;
        p = p * s + sfpi::vConstFloatPrgm1;
        p = p * s + sfpi::vConstFloatPrgm2;
        pio2 = PI_2;
        r = p * a3 + a;
    }

    // atan(|x|) = pi/2 - atan(1/|x|) for |x| >= 1.
    v_if(e >= 0) { r = pio2 - r; }
    v_endif;

    r = sfpi::copysgn(r, x);

    return r;
}

// Fast bf16 atan for Blackhole. Origin: llk-bench (LLM-agent-written kernel, claude-fable-5, 2026-08),
// validated exhaustively over all 65,536 bf16 inputs vs the exact golden: max 2 ULP (gate <= 2).
// Measured 443 cycles/tile vs 1704 for the previous kernel on p150b (tt-metal v0.76.0 baseline).
// Domain: validated on the full bf16 domain (NaN inputs were don't-care in the bench because the unpacker
// flushes them to +/-inf; unlike the previous kernel there is no explicit NaN pass-through -- check).
// State: programs vConstFloatPrgm0/1/2 (LREG12-14) only. No SFPLOADMACRO, no replay slots, no ADDR_MOD_6
// (sfpi dst_reg[] indexing over ADDR_MOD_7, zero increments, from the common init).
// Selected by atan_init / calculate_atan when !APPROXIMATION_MODE && !is_fp32_dest_acc_en
// (&& ITERATIONS == 8 in calculate); the fast init replaces sfpu_reciprocal_init<false>() (which
// only sets vConstFloatPrgm0 = 2.0f for the production bf16 path).
//
// Algorithm:
//   t = min(|x|, approx_recip(|x|))    -- SFPARECIP hardware reciprocal
//   p = t + t^2*(C1 + C2*t)            -- 3-op tuned correction (bit-exact
//                                         search against the full bf16 sweep)
//   r = (|x| >= 1) ? PIO2C - p : p     -- PIO2C tuned above pi/2 to center
//                                         the truncating bf16 store
//   result = setsgn(r, x)              -- store truncates fp32->bf16
//
// approx_recip(|x|) returns 0 for |x| = inf or >= 2^126, so atan(+/-inf)
// lands on PIO2C (rounds to bf16(pi/2)). The correction polynomial has
// P(0) == 1 exactly, so p == t for tiny t (atan(x) == x at bf16 precision
// there; keeps the smallest normal 2^-126 from underflowing to zero).
// NaN inputs are don't-care (the unpacker flushes them to +/-inf anyway).
inline void _init_atan_bf16_fast_() {
    sfpi::vConstFloatPrgm0 = 1.5746880519769257f;    // pi/2 + truncation bias
    sfpi::vConstFloatPrgm1 = -0.04294930753096212f;  // C1
    sfpi::vConstFloatPrgm2 = -0.1777010704459626f;   // C2
}

inline void _calculate_atan_bf16_fast_() {
#pragma GCC unroll 4
    for (int i = 0; i < 8; i += 2) {
        sfpi::vFloat x0 = sfpi::dst_reg[i];
        sfpi::vFloat x1 = sfpi::dst_reg[i + 1];
        sfpi::vFloat ax0 = sfpi::abs(x0);
        sfpi::vFloat u0 = sfpi::approx_recip(ax0);
        sfpi::vFloat ax1 = sfpi::abs(x1);
        sfpi::vFloat t0 = sfpi::min(ax0, u0);
        sfpi::vFloat u1 = sfpi::approx_recip(ax1);
        sfpi::vFloat t1 = sfpi::min(ax1, u1);
        sfpi::vFloat q0 = t0 * sfpi::vConstFloatPrgm2 + sfpi::vConstFloatPrgm1;
        sfpi::vFloat q1 = t1 * sfpi::vConstFloatPrgm2 + sfpi::vConstFloatPrgm1;
        sfpi::vFloat s0 = t0 * t0;
        sfpi::vFloat s1 = t1 * t1;
        sfpi::vFloat p0 = q0 * s0 + t0;
        sfpi::vFloat p1 = q1 * s1 + t1;
        v_if(sfpi::exexp(x0) >= 0) {  // |x| >= 1 (denormals/zero give exp < 0)
            p0 = sfpi::vConstFloatPrgm0 - p0;
        }
        v_endif;
        v_if(sfpi::exexp(x1) >= 0) {
            p1 = sfpi::vConstFloatPrgm0 - p1;
        }
        v_endif;
        p0 = sfpi::copysgn(p0, x0);  // sfpi >= 7.80: setsgn(vFloat, vFloat) removed, copysgn is the replacement
        p1 = sfpi::copysgn(p1, x1);
        sfpi::dst_reg[i] = p0;
        sfpi::dst_reg[i + 1] = p1;
    }
}

// Constants read by calculate_atan's generic body: the fp32 path's polynomial tail in vConstFloatPrgm1/2, the
// bf16 path's sfpu_reciprocal<false> Newton constant (vConstFloatPrgm0 = 2.0f). atan_init programs these except
// on the fast bf16 gate (same LREGs); the ITERATIONS != 8 fallback in calculate_atan re-seeds them.
template <bool is_fp32_dest_acc_en>
inline void _init_atan_body_constants_() {
    if constexpr (is_fp32_dest_acc_en) {
        sfpi::vConstFloatPrgm1 = 0x1.999384p-3f;
        sfpi::vConstFloatPrgm2 = -0x1.555552p-2f;
    } else {
        // sfpu_atan_bf16 uses sfpu_reciprocal<false>.
        sfpu_reciprocal_init<false>();
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_atan() {
    if constexpr (!APPROXIMATION_MODE && !is_fp32_dest_acc_en && ITERATIONS == 8) {
        _calculate_atan_bf16_fast_();
        return;
    }
    if constexpr (
        (!APPROXIMATION_MODE && !is_fp32_dest_acc_en) &&
        !(!APPROXIMATION_MODE && !is_fp32_dest_acc_en && ITERATIONS == 8)) {
        // ITERATIONS != 8: atan_init<false, false> cannot see ITERATIONS and has programmed the fast kernel's
        // LREG12-14 over sfpu_reciprocal_init's vConstFloatPrgm0 = 2.0f that sfpu_reciprocal<false> (inside
        // sfpu_atan_bf16) reads; re-seed it.
        _init_atan_body_constants_<is_fp32_dest_acc_en>();
    }

    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat result;

        if constexpr (is_fp32_dest_acc_en) {
            result = sfpu_atan_fp32(in);
        } else {
            result = sfpu_atan_bf16<APPROXIMATION_MODE>(in);
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat sfpu_asin_poly_bf16(sfpi::vFloat val) {
    sfpi::lreg_pressure _;

    // asin(z) = z*P(z^2) for |z| <= 5/8.
    sfpi::vFloat z2 = val * val;
    // Single-precision fit to asin(sqrt(u))/sqrt(u). Regenerate with:
    // > fpminimax(asin(sqrt(x))/sqrt(x), [|0,1,2,3|], [|single...|], [2^(-40); (5/8)^2], relative);
    sfpi::vFloat ratio = PolynomialEvaluator::eval(
        z2,
        0.999978601932525634765625f,
        0.16771225631237030029296875f,
        0.06381262838840484619140625f,
        0.083148844540119171142578125f);
    return val * ratio;
}

template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat sfpu_asin_range_reduced_bf16(sfpi::vFloat val) {
    sfpi::lreg_pressure _;

    // Range reduction near the endpoints:
    // asin(x) = sign(x) * [pi/2 - 2*asin(sqrt((1-|x|)/2))].
    sfpi::vFloat abs_v = sfpi::abs(val);
    sfpi::vFloat endpoint = abs_v - 0.625f;
    sfpi::vFloat z = sfpu_sqrt_custom<APPROXIMATION_MODE>((1.0f - abs_v) * 0.5f);

    v_if(endpoint < 0.0f) { z = abs_v; }
    v_endif;

    sfpi::vFloat asin_abs = sfpu_asin_poly_bf16<APPROXIMATION_MODE>(z);

    v_if(endpoint >= 0.0f) { asin_abs = PI_2 - 2.0f * asin_abs; }
    v_endif;

    return sfpi::copysgn(asin_abs, val);
}

template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat sfpu_asin_bf16(sfpi::vFloat val) {
    sfpi::vFloat result = std::numeric_limits<float>::quiet_NaN();
    v_if(sfpi::abs(val) <= 1.0f) { result = sfpu_asin_range_reduced_bf16<APPROXIMATION_MODE>(val); }
    v_endif;
    return result;
}

template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat sfpu_acos_bf16(sfpi::vFloat val) {
    sfpi::vFloat result = std::numeric_limits<float>::quiet_NaN();
    v_if(sfpi::abs(val) <= 1.0f) { result = PI_2 - sfpu_asin_range_reduced_bf16<APPROXIMATION_MODE>(val); }
    v_endif;
    return result;
}

sfpi_inline sfpi::vFloat sfpu_asin_fp32(sfpi::vFloat x) {
    sfpi::lreg_pressure _;

    sfpi::vFloat r;
    sfpi::vFloat ax = sfpi::abs(x);
    sfpi::vFloat d = 1.0f - ax;
    sfpi::vFloat cutoff = 0.5625f;
    sfpi::vFloat half_d = d * 0.5f;
    sfpi::vFloat t = ax - cutoff;

    // Reduce the endpoint region using asin(|x|) = pi/2 - 2*asin(sqrt((1 - |x|)/2)).
    sfpi::vFloat z = _sfpu_sqrt_endpoint_(half_d);

    v_if(t < 0.0f) { z = ax; }
    v_endif;

    // Minimax approximation of asin(z) on [0, 0.5625].
    sfpi::vFloat s = z * z;
    sfpi::vFloat p = 0x1.9e0000p-5f;
    sfpi::vFloat c = 0x1.364p-6f;
    p = __builtin_rvtt_sfpmad(p.get(), s.get(), c.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
    c = 0x1.7dcp-5f;
    p = __builtin_rvtt_sfpmad(p.get(), s.get(), c.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
    c = 0x1.329a74p-4f;
    p = __builtin_rvtt_sfpmad(p.get(), s.get(), c.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
    c = 0x1.55578cp-3f;
    p = __builtin_rvtt_sfpmad(p.get(), s.get(), c.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
    sfpi::vFloat neg_two = -2.0f;
    p *= s;
    sfpi::vFloat pio2 = PI_2;
    r = __builtin_rvtt_sfpmad(p.get(), z.get(), z.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);

    // Undo the endpoint reduction and restore the input sign.
    v_if(t >= 0.0f) { r = pio2 + neg_two * r; }
    v_endif;

    r = sfpi::copysgn(r, x);

    // Domain error for |x| > 1.
    v_if(half_d < 0.0f) { r = std::numeric_limits<float>::quiet_NaN(); }
    v_endif;

    return r;
}

sfpi_inline sfpi::vFloat sfpu_acos_fp32(sfpi::vFloat x) {
    sfpi::vFloat r;
    sfpi::vFloat ax = sfpi::abs(x);
    sfpi::vFloat d = 1.0f - ax;
    sfpi::vFloat cutoff = 0.5625f;
    sfpi::vFloat half_d = d * 0.5f;
    sfpi::vFloat t = ax - cutoff;

    // Reduce the endpoint region with sqrt((1 - |x|)/2); the sign mapping below
    // reconstructs acos(x) for both signs of x.
    sfpi::vFloat z = _sfpu_sqrt_endpoint_(half_d);

    v_if(t < 0.0f) { z = ax; }
    v_endif;

    // Minimax approximation of asin(z) on [0, 0.5625].
    sfpi::vFloat s = z * z;
    sfpi::vFloat p = 0x1.830p-5f;
    sfpi::vFloat c = 0x1.ca0000p-8f;
    p = __builtin_rvtt_sfpmad(p.get(), s.get(), c.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
    c = 0x1.158p-5f;
    p = __builtin_rvtt_sfpmad(p.get(), s.get(), c.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
    c = 0x1.6acp-5f;
    p = __builtin_rvtt_sfpmad(p.get(), s.get(), c.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
    c = 0x1.33411ep-4f;
    p = __builtin_rvtt_sfpmad(p.get(), s.get(), c.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
    c = 0x1.555552p-3f;
    p = __builtin_rvtt_sfpmad(p.get(), s.get(), c.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
    sfpi::vUInt x_bits = sfpi::as<sfpi::vUInt>(x);
    sfpi::vUInt t_bits = sfpi::as<sfpi::vUInt>(t);
    x_bits ^= t_bits;
    p *= s;
    // copysgn observes the XORed sign bit: sign(z) = sign(x) XOR sign(t).
    // This selects the central pi/2 +/- asin(ax) mapping and the endpoint sign.
    z = sfpi::copysgn(z, sfpi::as<sfpi::vFloat>(x_bits));
    r = __builtin_rvtt_sfpmad(p.get(), z.get(), z.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);

    // Add pi/2 in the central interval and at the negative endpoint.
    sfpi::vUInt pio2_bits = sfpi::as<sfpi::vUInt>(z);
    pio2_bits |= t_bits;
    sfpi::vFloat pio2_pred = sfpi::as<sfpi::vFloat>(pio2_bits);
    v_if(pio2_pred < 0.0f) { r += PI_2; }
    v_endif;

    // Endpoint reconstruction.
    v_if(t >= 0.0f) { r += r; }
    v_endif;

    // Domain error for |x| > 1.
    v_if(half_d < 0.0f) { r = std::numeric_limits<float>::quiet_NaN(); }
    v_endif;

    return r;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_asin() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat result;

        if constexpr (is_fp32_dest_acc_en) {
            result = sfpu_asin_fp32(in);
        } else {
            result = sfpu_asin_bf16<APPROXIMATION_MODE>(in);
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

// fp32-dest asin/acos route through the endpoint sqrt (_sfpu_sqrt_endpoint_), which reads the sqrt
// seed/refinement constants from vConstIntPrgm0/1/2. Prime them via asin_acos_init (a no-op for bf16
// dest, which uses the self-contained sfpu_sqrt_custom). Templated on the dest-acc flag so the bare
// init path picks the right variant; the counter reset preserves the previous bare-init behavior.
template <bool is_fp32_dest_acc_en>
inline void asin_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    asin_acos_init<is_fp32_dest_acc_en>();
}

template <bool is_fp32_dest_acc_en>
inline void acos_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    asin_acos_init<is_fp32_dest_acc_en>();
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_acos() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat result;

        if constexpr (is_fp32_dest_acc_en) {
            result = sfpu_acos_fp32(in);
        } else {
            result = sfpu_acos_bf16<APPROXIMATION_MODE>(in);
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

// computes exp(abs(x))/4 without overflow
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_quarter_exp_abs_(sfpi::vFloat x) {
    // j = x * log2(e); i = round(abs(j)); j = (float)i;
    sfpi::vFloat j = x * sfpi::vConstFloatPrgm0;
    sfpi::vFloat a = sfpi::setsgn(x, 0);
    // Rounds the absolute value of j, clamped to [0, 255].
    sfpi::vMag m = sfpi::convert<sfpi::vUInt8>(j, sfpi::RoundMode::Nearest);
    j = sfpi::convert<sfpi::vFloat>(m, sfpi::RoundMode::Nearest);
    sfpi::vInt i = m;

    sfpi::vFloat r, f;

    if constexpr (!is_fp32_dest_acc_en) {
        f = j * sfpi::vConstFloatPrgm1 + a;  // f = a - j * ln(2)

        r = 0.038877178f;
        r = r * f + 0.168174848f;
        i += 125;
        r = r * f + sfpi::vConstFloatPrgm2;
        r = r * f + 0.999963462f;

    } else {
        f = j * sfpi::vConstFloatPrgm1 + a;  // f = a - j * ln(2)_hi
        f = j * -1.42860677e-6f + f;         // f = f - j * ln(2)_lo

        r = 1.37805939e-3f;
        r = r * f + 8.37312452e-3f;
        r = r * f + 4.16695364e-2f;
        r = r * f + 1.66664720e-1f;
        r = r * f + sfpi::vConstFloatPrgm2;
        i += 125;
        r = r * f + 1.0f;
    }

    // Handle a * log2(e) >= 130, while propagating NaN.
    sfpi::vFloat y = a * std::numeric_limits<float>::infinity();
    r = r * f + 1.0f;

    v_if(i < 255) {
        // Keep reconstruction quarter-scaled: scale is 0.25 * 2**i. Avoids
        // materialising 2**i directly near overflow boundary.
        y = r * sfpi::as<sfpi::vFloat>(i << 23);
    }
    v_endif;

    return y;
}

// t = exp(a); cosh(a) = 0.5 * (t + 1/t)
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_cosh() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat a = sfpi::setsgn(x, 0);
        sfpi::vFloat q = _sfpu_quarter_exp_abs_<is_fp32_dest_acc_en>(a);
        sfpi::vFloat r = _sfpu_reciprocal_gt0_<is_fp32_dest_acc_en>(q);
        sfpi::vFloat y = q + q;
        r *= 0.125f;
        sfpi::vInt q_exp = sfpi::exexp(q);
        v_if(q_exp < 24) { y += r; }
        v_endif;

        if constexpr (!is_fp32_dest_acc_en) {
            y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
        }

        sfpi::dst_reg[0] = y;
        sfpi::dst_reg++;
    }
}

// computes expm1(abs(x))/4 without overflow
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_quarter_expm1_abs_(sfpi::vFloat x) {
    sfpi::vFloat j = x * sfpi::vConstFloatPrgm0;  // j = x * log2(e)
    sfpi::vFloat a = sfpi::setsgn(x, 0);
    // Rounds the absolute value of j, clamped to [0, 255].
    sfpi::vMag m = sfpi::convert<sfpi::vUInt8>(j, sfpi::RoundMode::Nearest);
    j = sfpi::convert<sfpi::vFloat>(m, sfpi::RoundMode::Nearest);
    sfpi::vInt i = m;

    sfpi::vFloat r, s, f, w, y, scale, bias, c0;

    if constexpr (!is_fp32_dest_acc_en) {
        f = j * sfpi::vConstFloatPrgm1 + a;  // f = a - j * ln(2)

        r = 8.361816406e-03f;
        r = r * f + 4.177856445e-02f;
        s = f * f;  // hide SFPMAD latency
        r = r * f + sfpi::vConstFloatPrgm2;
        c0 = 0.5f;
        r = __builtin_rvtt_sfpmad(r.get(), f.get(), c0.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);

    } else {
        f = j * sfpi::vConstFloatPrgm1 + a;  // f = a - j * ln(2)_hi
        f = j * -1.42860677e-6f + f;         // f = f - j * ln(2)_lo

        r = 1.974105835e-04f;
        r = r * f + 1.393107930e-3f;
        r = r * f + 8.333439939e-3f;
        r = r * f + 4.166680202e-2f;
        s = f * f;  // hide SFPMAD latency
        r = r * f + sfpi::vConstFloatPrgm2;
        r = r * f + 4.999999702e-1f;
    }

    w = 0.25f;
    r = r * s + f;

    // Keep reconstruction quarter-scaled: scale is 0.25 * 2**i. Avoids
    // materialising 2**i directly near overflow boundary.
    scale = sfpi::as<sfpi::vFloat>((i << 23) + sfpi::as<sfpi::vInt>(w));
    bias = scale - w;
    // Handle a * log2(e) >= 130, while propagating NaN.
    y = a * std::numeric_limits<float>::infinity();

    v_if(i < 130) { y = r * scale + bias; }
    v_endif;

    return y;
}

// a = abs(x); t = expm1(a); sinh(a) = 0.5 * (t + t / (t + 1))
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_sinh_(sfpi::vFloat x) {
    sfpi::vFloat q = _sfpu_quarter_expm1_abs_<is_fp32_dest_acc_en>(x);
    sfpi::vFloat e = 4.0f * q + 1.0f;

    sfpi::vFloat r = _sfpu_reciprocal_gt0_<is_fp32_dest_acc_en>(e);

    // t < 2^-25: t + 1 rounds to 1, so sinh(x) rounds to x. Since q = t / 4, this is q < 2^-27.
    sfpi::vFloat y = x;
    sfpi::vInt q_exp = sfpi::exexp(q);
    v_if(q_exp >= -27) {
        // t >= 2^25: t + 1 rounds to t, so sinh(abs(x)) = expm1(abs(x)) / 2 = 2q.
        y = q + q;
        v_if(q_exp < 23) {
            // Middle range: sinh(abs(x)) = 0.5t + 0.5t/(t+1), with t = 4q.
            y = y * r + y;
        }
        v_endif;
    }
    v_endif;
    return sfpi::copysgn(y, x);
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_sinh() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat y = _sfpu_sinh_<is_fp32_dest_acc_en>(sfpi::dst_reg[0]);

        if constexpr (!is_fp32_dest_acc_en) {
            y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
        }

        sfpi::dst_reg[0] = y;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
void sine_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
#ifndef DISABLE_SFPLOADMACRO
    if constexpr (!APPROXIMATION_MODE && !is_fp32_dest_acc_en) {
        // Fast bf16 kernel: programs LREG11-14 + load macros itself and does not use the Cody-Waite
        // constants below (same LREGs), so they are skipped rather than overwritten.
        _init_sine_bf16_fast_();
        return;
    }
#endif
    _init_sine_body_constants_();
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
void cosine_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    if constexpr (!APPROXIMATION_MODE && !is_fp32_dest_acc_en) {
        // Fast bf16 kernel: programs vConstFloatPrgm0/1/2 + L4-L7 itself and does not use the
        // Cody-Waite constants below (same LREGs), so they are skipped rather than overwritten.
        _init_cosine_bf16_fast_();
        return;
    }
    _init_cosine_body_constants_();
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
void tangent_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    if constexpr (!APPROXIMATION_MODE && !is_fp32_dest_acc_en) {
        // Fast bf16 kernel: programs vConstFloatPrgm0/1/2 itself and does not use the Cody-Waite
        // constants below (same LREGs), so they are skipped rather than overwritten.
        _init_tangent_bf16_fast_();
        return;
    }
    _init_tangent_body_constants_();
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
void cosh_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpi::vConstFloatPrgm0 = 1.442695f;  // log2(e) == 1 / ln(2)
    if constexpr (is_fp32_dest_acc_en) {
        sfpi::vConstFloatPrgm1 = -0.693145752f;   // -ln(2)_hi
        sfpi::vConstFloatPrgm2 = 4.99999851e-1f;  // c2
    } else {
        sfpi::vConstFloatPrgm1 = -0.6931471805599453f;  // -ln(2)
        sfpi::vConstFloatPrgm2 = 0.500122011f;          // c2
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
void sinh_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpi::vConstFloatPrgm0 = 1.442695f;  // log2(e) == 1 / ln(2)
    if constexpr (is_fp32_dest_acc_en) {
        sfpi::vConstFloatPrgm1 = -0.693145752f;    // -ln(2)_hi
        sfpi::vConstFloatPrgm2 = 1.666667163e-1f;  // c1
    } else {
        sfpi::vConstFloatPrgm1 = -0.6931471805599453f;  // -ln(2)
        sfpi::vConstFloatPrgm2 = 1.666259766e-01f;      // c1
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
void atan_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    if constexpr (!APPROXIMATION_MODE && !is_fp32_dest_acc_en) {
        // Fast bf16 kernel: programs vConstFloatPrgm0/1/2 itself; sfpu_reciprocal_init<false>() below
        // (vConstFloatPrgm0 = 2.0f) is only needed by the production bf16 path, so it is skipped.
        _init_atan_bf16_fast_();
        return;
    }
    _init_atan_body_constants_<is_fp32_dest_acc_en>();
}

template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat _sfpu_sine_maclaurin_series_(sfpi::vFloat val) {
    // Good for [-pi:pi]
    // Maclaurin series = x - x^3/3! + x^5/5! - x^7/7! + x^9/9! - x^11/11!
    sfpi::vFloat tmp = val;
    // x
    sfpi::vFloat output = tmp;
    // x^3/3!
    tmp = tmp * val * val;
    output += -0.166666666 * tmp;
    // x^5/5!
    tmp = tmp * val * val;
    output += 0.0083333333 * tmp;
    // x^7/7!
    tmp = tmp * val * val;
    output += -0.0001984126 * tmp;
    if constexpr (not APPROXIMATION_MODE) {
        // x^9/9!
        tmp = tmp * val * val;
        output += 0.0000027557 * tmp;
        // x^11/11!
        tmp = tmp * val * val;
        output += -0.00000002505 * tmp;
    }

    // Write out output
    return output;
}

template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat _sfpu_cosine_maclaurin_series_(sfpi::vFloat val) {
    // Good for [-pi:pi]
    // Maclaurin series = 1 - x^2/2! + x^4/4! - x^6/6! + x^8/8! - x^10/10! + x^12/12!
    // 1
    sfpi::vFloat output = 1.0f;
    // x^2/2!
    sfpi::vFloat tmp = val * val;
    output += -0.5 * tmp;
    // x^4/4!
    tmp = tmp * val * val;
    output += 0.0416666666 * tmp;
    // x^6/6!
    tmp = tmp * val * val;
    output += -0.0013888888 * tmp;
    if constexpr (not APPROXIMATION_MODE) {
        // x^8/8!
        tmp = tmp * val * val;
        output += 0.0000248015 * tmp;
        // x^10/10!
        tmp = tmp * val * val;
        output += -0.0000002755 * tmp;
    }

    // Write out output
    return output;
}

// Legacy implementation
// Candidate for removal in future versions. See https://github.com/tenstorrent/tt-llk/issues/225 for more details.
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_sine_(const int iterations) {
    // SFPU microcode
    for (int d = 0; d < iterations; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        v = 0.318309886183791f * v;  // *1/pi to get number of pi rads.
        auto whole_v = sfpi::convert<sfpi::vSMag16>(v, sfpi::RoundMode::Nearest);
        auto whole_v_float = sfpi::convert<sfpi::vFloat>(whole_v, sfpi::RoundMode::Nearest);
        v = v - whole_v_float;
        v *= 3.141592653589793f;  // fractional * pi to get it in [-pi:pi]
        v = _sfpu_sine_maclaurin_series_<APPROXIMATION_MODE>(v);
        v_if((whole_v & 1) != 0) {
            // odd so flip the sign
            v *= -1;
        }
        v_endif;
        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}

// Legacy implementation, replaced by newer void _calculate_sine_() which produces more accurate results
// Candidate for removal in future versions. See https://github.com/tenstorrent/tt-llk/issues/225 for more details.
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_cosine_(const int iterations) {
    // SFPU microcode
    for (int d = 0; d < iterations; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        v = 0.318309886183791f * v;  // *1/pi to get number of pi rads.
        auto whole_v = sfpi::convert<sfpi::vSMag16>(v, sfpi::RoundMode::Nearest);
        auto whole_v_float = sfpi::convert<sfpi::vFloat>(whole_v, sfpi::RoundMode::Nearest);
        v = v - whole_v_float;
        v *= 3.141592653589793f;  // fractional * pi to get it in [-pi:pi]
        v = _sfpu_cosine_maclaurin_series_<APPROXIMATION_MODE>(v);
        v_if((whole_v & 1) != 0) {
            // odd so flip the sign
            v *= -1;
        }
        v_endif;
        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}

// Self-contained square root for the inverse-hyperbolic kernels.
//
// The shared _calculate_sqrt_body_ stores its magic seed and Newton refinement
// constants in vConstIntPrgm0 / vConstFloatPrgm1 / vConstFloatPrgm2. Those same
// program registers are owned by the log1p polynomial (vConstFloatPrgm0/1/2)
// that asinh/acosh now route through, so the two cannot coexist in one pass.
// This helper bakes the magic seed and refinement constants in as immediates so
// it leaves the program registers untouched for log1p. Input is assumed >= 0
// (always true here: x*x + 1 for asinh, (x-1)*(x+1) for acosh with x >= 1).
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_sqrt_ge0_(sfpi::vFloat x) {
    // Fast inverse-square-root seed (same constant the shared sqrt kernel uses
    // for the high-precision path) followed by Newton-Raphson refinement of
    // y ~= 1 / sqrt(x): y <- y * (1.5 - 0.5 * x * y * y).
    sfpi::vFloat half_x = sfpi::addexp(x, -1);  // 0.5 * x
    sfpi::vInt i = sfpi::as<sfpi::vInt>(sfpi::as<sfpi::vUInt>(x) >> 1);
    sfpi::vFloat y = sfpi::as<sfpi::vFloat>(0x5f1110a0 - i);

    y = y * (1.5f - half_x * y * y);
    y = y * (1.5f - half_x * y * y);
    if constexpr (is_fp32_dest_acc_en) {
        y = y * (1.5f - half_x * y * y);
    }

    // sqrt(x) = x / sqrt(x) = x * (1 / sqrt(x)). One Newton step on the product
    // form (a = x * y; a <- 0.5 * (a + x / a)) is folded as a <- a + 0.5 * (x - a*a) * y.
    sfpi::vFloat a = x * y;
    a = a + 0.5f * (x - a * a) * y;

    // sqrt(0) must be exactly 0; the reciprocal seed produces inf*0 = NaN there.
    v_if(x == 0.0f) { a = 0.0f; }
    v_endif;
    return a;
}

// acosh(x) = log(x + sqrt(x^2 - 1)), reformulated through log1p to remove the
// absorption error at x -> 1+ and the x^2 overflow at large x. Three regions:
//   x < 1            -> NaN
//   x == 1           -> +0
//   1 < x < 1.5      -> log1p((x - 1) + sqrt((x - 1) * (x + 1)))  (small-(x-1) stable)
//   1.5 <= x < 2^28  -> log1p((x + sqrt(x^2 - 1)) - 1)            (safe reconstruction)
//   x >= 2^28        -> ln(2x) = log1p(2x - 1)                    (avoids x^2 overflow)
// LOG1P_LARGE is the threshold past which x^2 - 1 == x^2 to working precision and
// acosh(x) == ln(2x) to <1 ulp; using log1p(2x - 1) also dodges the x^2 overflow
// that makes the classic form return +inf for x >= ~1.84e19.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_acosh() {
    constexpr float LOG1P_LARGE = 268435456.0f;  // 2^28
    constexpr float LN2 = 0.6931471805599453f;
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat inp = sfpi::dst_reg[0];

        // Build the log1p argument per region, clamping the out-of-domain lanes
        // (x < 1) to a safe value so the shared log1p runs over the whole vector.
        // The argument is materialised to DST before log1p; round-tripping through
        // DST severs the sqrt/reciprocal expression from the log1p polynomial so
        // the SFPU register allocator stays within its reload budget. The x <= 1
        // lanes are overwritten with their exact results afterwards.
        //
        // arg = x - 1 is the common term in every region, so it is computed once
        // and the per-region sqrt term is accumulated onto it:
        //   x >= 2^28        -> arg = x - 1                 (large; acosh ~= LN2 +
        //                                                    log1p(x-1), +LN2 added later)
        //   1 < x < 2^28     -> arg += sqrt((x-1)(x+1))     (sqrt(x^2-1) without the
        //                                                    x^2-1 cancellation)
        // The large region falls through the predicated block and keeps arg = x-1.
        sfpi::vFloat arg = inp - 1.0f;
        v_if(inp < LOG1P_LARGE) { arg = arg + _sfpu_sqrt_ge0_<is_fp32_dest_acc_en>((inp + 1.0f) * arg); }
        v_endif;
        sfpi::dst_reg[0] = arg;

        sfpi::vFloat res = calculate_log1p_fp32<is_fp32_dest_acc_en>(sfpi::dst_reg[0]);
        // Large region carries the extra ln(2) from acosh(x) ~= LN2 + ln(x).
        v_if(inp >= LOG1P_LARGE) { res = res + LN2; }
        v_endif;

        // Domain fix-ups: x == 1 -> +0, x < 1 -> NaN.
        v_if(inp == 1.0f) { res = 0.0f; }
        v_elseif(inp < 1.0f) { res = std::numeric_limits<float>::quiet_NaN(); }
        v_endif;

        if constexpr (!is_fp32_dest_acc_en) {
            res = sfpi::convert<sfpi::vFloat16b>(res, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = res;
        sfpi::dst_reg++;
    }
}

// asinh(x) = sign(x) * log(|x| + sqrt(x^2 + 1)), reformulated to remove the
// cancellation at x -> 0 and the x^2 overflow at large |x|. Regions in a = |x|:
//   a < 0.75          -> a * P(a^2), degree-6 minimax polynomial (<=1 ulp)
//   0.75 <= a < 2^28  -> log1p(a + a*a / (1 + sqrt(1 + a*a)))  (cancellation-free)
//   a >= 2^28         -> ln(2a) = LN2 + log1p(a - 1)           (avoids x^2 overflow)
// Sign is restored at the end. The small region is a plain polynomial (no
// sqrt/reciprocal/log1p), which keeps SFPU register pressure within the reload
// budget. The mid region uses a + sqrt(1+a^2) - 1 = a + a^2 / (1 + sqrt(1+a^2))
// so the log1p argument never loses precision through a subtract-1 cancellation;
// the large region exits the x^2 regime entirely so |x| up to fp32 max no longer
// overflows (the old x^2 + 1 produced +inf at ~1.84e19).
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_asinh() {
    constexpr float LOG1P_LARGE = 268435456.0f;  // 2^28
    constexpr float LN2 = 0.6931471805599453f;
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        // Keep only the original input live across the body (matching calculate_acosh,
        // which compiles within the SFPU reload budget). a = |x| and x2 = x*x are
        // recomputed inline rather than held in their own long-lived registers:
        // hoisting them into vFloats pushes the SFPU register allocator past its
        // reload budget and the kernel fails to compile (internal compiler error:
        // maximum number of generated reload insns), so the recompute is deliberate.
        // Build the per-region log1p argument over |x|, clamp |x| < 0.75 lanes to a
        // safe value, materialise to DST, run the shared log1p, then overwrite the
        // |x| < 0.75 lanes with the direct polynomial. Sign is restored from inp.
        sfpi::vFloat inp = sfpi::dst_reg[0];

        // Mid/large region (|x| >= 0.75): asinh(|x|) = log1p(arg). The large
        // sub-region drops the x^2 term (LN2 + log1p(|x| - 1)) to dodge fp32
        // overflow. For the safe sub-region use the cancellation-free identity
        //   |x| + sqrt(1+x^2) - 1 = |x| + x^2 / (1 + sqrt(1+x^2))
        // which avoids the subtract-1 cancellation that otherwise costs ~3-4 ulp
        // near the crossover. Lanes below 0.75 are clamped here and overwritten by
        // the polynomial after log1p.
        sfpi::vFloat arg = 0.0f;
        v_if(sfpi::abs(inp) >= LOG1P_LARGE) { arg = sfpi::abs(inp) - 1.0f; }
        v_elseif(sfpi::abs(inp) >= 0.75f) {
            sfpi::vFloat root = _sfpu_sqrt_ge0_<is_fp32_dest_acc_en>(inp * inp + 1.0f);
            arg = sfpi::abs(inp) + (inp * inp) * _sfpu_reciprocal_gt0_<is_fp32_dest_acc_en>(1.0f + root);
        }
        v_endif;
        sfpi::dst_reg[0] = arg;

        sfpi::vFloat res = calculate_log1p_fp32<is_fp32_dest_acc_en>(sfpi::dst_reg[0]);
        v_if(sfpi::abs(inp) >= LOG1P_LARGE) { res = res + LN2; }
        v_endif;

        // Small region (|x| < 0.75): asinh(|x|) = |x| * P(x^2), a degree-6 (in x^2)
        // minimax fit (<=1 ulp on [0, 0.75]). No sqrt/reciprocal/log1p here.
        v_if(sfpi::abs(inp) < 0.75f) {
            sfpi::vFloat s = inp * inp;
            sfpi::vFloat p = 4.375355784e-03f;
            p = p * s + -1.484858524e-02f;
            p = p * s + 2.785361186e-02f;
            p = p * s + -4.417778924e-02f;
            p = p * s + 7.495806366e-02f;
            p = p * s + -1.666652262e-01f;
            p = p * s + 1.000000000e+00f;
            res = sfpi::abs(inp) * p;
        }
        v_endif;

        // res is asinh(|x|) >= 0; restore the original sign.
        res = sfpi::copysgn(res, inp);

        if constexpr (!is_fp32_dest_acc_en) {
            res = sfpi::convert<sfpi::vFloat16b>(res, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = res;
        sfpi::dst_reg++;
    }
}

// atanh(x) = 0.5 * log((1 + x) / (1 - x)), reformulated as
// 0.5 * log1p(2 * x / (1 - x)) to remove the cancellation at x -> 0 and the
// (1 + x)/(1 - x) ratio that loses precision there.
//   |x| > 1   -> NaN
//   |x| == 1  -> copysgn(+inf, x)
//   else      -> sign(x) * 0.5 * log1p(2 * a / (1 - a)),  a = |x|
// Working with a = |x| keeps 1 - a positive (away from 0 except at the |x| == 1
// boundary, handled separately), and the sign is restored at the end.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_atanh() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat inp = sfpi::dst_reg[0];
        sfpi::vFloat a = sfpi::abs(inp);

        // Clamp |x| >= 1 lanes to 0 so the interior formula stays finite there;
        // those lanes are overwritten by the boundary fix-up below.
        v_if(a >= 1.0f) { a = 0.0f; }
        v_endif;

        // Build the log1p argument, then materialise it to DST before the log1p
        // polynomial. Round-tripping through DST cuts the reciprocal->log1p
        // expression so the SFPU register allocator does not exceed its reload
        // budget (the fused form overflows it). The boundary lanes are restored
        // from `inp` afterwards, so clobbering DST here is safe.
        sfpi::vFloat den = 1.0f - a;
        sfpi::dst_reg[0] = (a + a) * _sfpu_reciprocal_gt0_<is_fp32_dest_acc_en>(den);

        sfpi::vFloat res = calculate_log1p_fp32<is_fp32_dest_acc_en>(sfpi::dst_reg[0]);
        res = sfpi::copysgn(0.5f * res, inp);

        // Boundary fix-ups: |x| == 1 -> +/-inf, |x| > 1 -> NaN. abs(inp) is
        // recomputed inline here rather than cached in a register; a cached
        // |x| - 1 variant pushed the allocator past the reload budget.
        v_if(sfpi::abs(inp) > 1.0f) { res = std::numeric_limits<float>::quiet_NaN(); }
        v_elseif(sfpi::abs(inp) == 1.0f) {
            sfpi::vFloat inf = std::numeric_limits<float>::infinity();
            res = sfpi::copysgn(inf, inp);
        }
        v_endif;

        if constexpr (!is_fp32_dest_acc_en) {
            res = sfpi::convert<sfpi::vFloat16b>(res, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = res;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
void init_inverse_hyperbolic() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    // asinh/acosh route through calculate_log1p_fp32, which expects the log1p
    // polynomial constants in vConstFloatPrgm0/1/2. The sqrt used internally is
    // self-contained (_sfpu_sqrt_ge0_) and does not touch the program registers.
    // Production constants, not log1p_init: that would program the bf16 fast-kernel state on bf16 dest, which
    // calculate_log1p_fp32 does not read.
    _init_log1p_body_constants_<is_fp32_dest_acc_en>();
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
void init_atanh() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    // atanh routes through calculate_log1p_fp32; the reciprocal it uses is the
    // self-contained _sfpu_reciprocal_gt0_, so log1p owns the program registers.
    // Production constants, not log1p_init: that would program the bf16 fast-kernel state on bf16 dest, which
    // calculate_log1p_fp32 does not read.
    _init_log1p_body_constants_<is_fp32_dest_acc_en>();
}

}  // namespace ckernel::sfpu
