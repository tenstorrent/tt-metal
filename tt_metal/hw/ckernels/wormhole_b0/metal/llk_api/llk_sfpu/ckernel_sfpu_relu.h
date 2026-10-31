// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "sfpu/ckernel_sfpu_load_config.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <typename T>
constexpr bool is_supported_relu_type_v = std::is_same_v<T, float> || std::is_same_v<T, std::uint32_t>;

template <bool APPROXIMATION_MODE>
inline void _calculate_lrelu_(const int iterations, std::uint32_t slope) {
    const sfpi::vFloat slope_v = Converter::as_float(slope);
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        v_if(v < 0.0f) { v = v * slope_v; }
        v_endif;
        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}

// relu_max(x) = max(min(x, threshold), 0): two SFPSWAP min/max folds in place of two predicated
// v_if blocks. On Wormhole each block was an SFPMAD subtract (+SFPNOP) and SFPSETCC, or a bare
// SFPSETCC, then SFPMOV/SFPENCC; the replayed row goes from 11 to 5 instructions (load, two
// SFPSWAPs, store, dst increment) when `threshold` is a literal or a constant register. A threshold
// in a general LReg costs one more SFPMOV per row: SFPSWAP writes both operands, so the compiler
// copies the threshold first.
//
// The order is the one the predicated chain had -- threshold first, then the relu clamp -- and
// it is not interchangeable with min(max(x, 0), threshold): for threshold < 0 this form returns
// +0.0 on every lane (min(x, t) <= t < 0, then the clamp), the other would return the threshold.
// SFPSWAP orders FP32 by sign-magnitude, -NaN < -Inf < ... < -0 < +0 < ... < +Inf < +NaN, so
//   +NaN -> threshold (outranks every value, the min lands on the threshold)
//   -NaN -> +0.0      (ranks below every value, survives the min, the clamp takes it)
//   -0.0 -> +0.0      (ranks below +0.0)
// which is what the predicated chain returned as well. On Wormhole silicon the output is
// bit-identical to the previous v_if implementation for every bf16 input pattern (±0, ±Inf, NaNs,
// subnormals) and for random fp32 inputs, at thresholds 6, 0.5 and -0.5; see also
// test_eltwise_unary_sfpu_relu_max_threshold.
sfpi_inline sfpi::vFloat _relu_max_body_(sfpi::vFloat val, sfpi::vFloat threshold) {
    return sfpi::max(sfpi::min(val, threshold), 0.0f);
}

// Float relu_max with a per-call scalar threshold. The threshold is parked in vConstFloatPrgm1
// (LREG13) for the duration of the call: SFPSWAP writes both of its operands, so a threshold held
// in a general LReg costs an SFPMOV copy per row, while a write to a constant register is dropped.
// One SFPCONFIG (+2 SFPLOADI) per call buys one instruction per row: 5 instead of 6, 11 before.
//
// Prgm1 is call-scoped here. relu_max_init programs nothing, and every production caller runs its
// init next to its call (ttnn unary chains emit `<op>_init(); <op>(idst);` per op per tile), so no
// kernel relies on Prgm1 surviving across a relu_max call. A fused kernel that programs Prgm1 in
// its own init and calls relu_max_tile before its own compute would have to re-init.
//
// sfpi 7.83.0 compiles exactly this shape -- max(min(v, Prgm), literal) -- to SFPSWAP against
// LREG13 and LREG9; the bare min/max/clamp(v, vFloat(vConstFloatPrgmN)) forms hit an internal
// compiler error (lra-constraints), so do not "simplify" this into _relu_max_body_.
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _relu_max_float_impl_(const int iterations, const float threshold) {
    sfpi::vConstFloatPrgm1 = threshold;
    for (int d = 0; d < iterations; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = sfpi::max(sfpi::min(v, sfpi::vFloat(sfpi::vConstFloatPrgm1)), 0.0f);
        sfpi::dst_reg++;
    }
}

// Integer relu_max. No production caller (relu_max_tile_int32 uses the metal relu_clamp_int);
// kept as it was.
template <typename VecType, bool APPROXIMATION_MODE, int ITERATIONS>
inline void _relu_max_impl_(const int iterations, VecType threshold) {
    for (int d = 0; d < iterations; d++) {
        VecType result = sfpi::dst_reg[0];
        v_if(result > threshold) { result = threshold; }
        v_endif;
        v_if(result < 0) { result = 0; }
        v_endif;
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

// Wrappers
template <typename VectorType, bool APPROXIMATION_MODE, int ITERATIONS, typename T>
inline void _relu_max_(T threshold) {
    static_assert(
        std::is_same_v<VectorType, sfpi::vFloat> || std::is_same_v<VectorType, sfpi::vInt>,
        "VectorType must be sfpi::vFloat or sfpi::vInt");

    static_assert(
        std::is_same_v<T, float> || std::is_same_v<T, std::uint32_t>, "Threshold type must be float or uint32_t");

    if constexpr (std::is_same_v<VectorType, sfpi::vFloat>) {
        float f;
        if constexpr (std::is_same_v<T, float>) {
            f = threshold;
        } else {
            f = Converter::as_float(threshold);
        }
        _relu_max_float_impl_<APPROXIMATION_MODE, ITERATIONS>(ITERATIONS, f);
    } else {
        static_assert(
            std::is_same_v<T, std::uint32_t>,
            "A float threshold requires VectorType == sfpi::vFloat: sfpi::vInt has no float constructor, so the "
            "conversion below would otherwise be ambiguous");
        const sfpi::vInt v_threshold = int(Converter::as_float(threshold));
        _relu_max_impl_<VectorType, APPROXIMATION_MODE, ITERATIONS>(ITERATIONS, v_threshold);
    }
}

// Wormhole keeps its own _relu_min_ rather than Blackhole's sfpi form: on Wormhole silicon the
// raw SFPSWAP body below is faster for a non-zero float threshold (204 vs 247 cycles/tile) and
// for int32 (204 vs 212). Everything else in this file is shared with Blackhole.
//
// Contract: the threshold is an implicit input in LREG2, not a parameter -- every caller must
// load it with _sfpu_load_imm32_ first, which the raw-TTI body cannot express in its signature.
// SFPLOAD_INSTR_MOD has to stay a template parameter because it feeds an immediate asm operand:
// as a runtime argument it compiles only while the optimiser folds it, and fails at -O0.
template <bool APPROXIMATION_MODE, int ITERATIONS, InstrModLoadStore SFPLOAD_INSTR_MOD>
inline void _relu_min_impl_(const int iterations) {
    static_assert(
        SFPLOAD_INSTR_MOD == InstrModLoadStore::DEFAULT || SFPLOAD_INSTR_MOD == InstrModLoadStore::INT32_2S_COMP,
        "SFPLOAD_INSTR_MOD must be DEFAULT (fp32 datapath) or INT32_2S_COMP (integer datapath)");

    for (int d = 0; d < iterations; d++) {
        // Load input tensor to lreg0
        TTI_SFPLOAD(p_sfpu::LREG0, SFPLOAD_INSTR_MOD, ADDR_MOD_3, 0);
        // Copy value param from lreg2 to lreg1
        TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);
        // Swap and store maximum in lreg1, minimum in lreg0 (sign + magnitude format)
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, 1);
        // Store the result
        TTI_SFPSTORE(p_sfpu::LREG1, SFPLOAD_INSTR_MOD, ADDR_MOD_3, 0);
        sfpi::dst_reg++;
    }
}

// Wrappers
template <typename VectorType, bool APPROXIMATION_MODE, int ITERATIONS, typename T>
inline void _relu_min_(T threshold) {
    static_assert(
        std::is_same_v<VectorType, sfpi::vFloat> || std::is_same_v<VectorType, sfpi::vInt>,
        "VectorType must be sfpi::vFloat or sfpi::vInt");

    // The load/store mode the branches below encode the threshold for. Only the integer
    // datapath needs a non-default mode, and which branch runs is fixed by <T, VectorType>,
    // so this is a compile-time constant rather than a variable the branches assign -- see
    // the note on _relu_min_impl_'s SFPLOAD_INSTR_MOD parameter.
    constexpr InstrModLoadStore SFPLOAD_INSTR_MOD =
        (std::is_same_v<T, std::uint32_t> && std::is_same_v<VectorType, sfpi::vInt>) ? InstrModLoadStore::INT32_2S_COMP
                                                                                     : InstrModLoadStore::DEFAULT;

    // Invariant every branch below must uphold: leave the threshold in LREG2, in the encoding
    // SFPLOAD_INSTR_MOD selects. A branch that sets only a local vector compiles clean and then
    // reads whatever the previously executed SFPU kernel left in LREG2, so the load is not
    // optional on any path.
    if constexpr (std::is_same_v<T, float>) {
        static_assert(
            std::is_same_v<VectorType, sfpi::vFloat>,
            "A float threshold requires VectorType == sfpi::vFloat: the LREG2 load below bit-casts the float, which is "
            "meaningless for an integer datapath");
        _sfpu_load_imm32_(p_sfpu::LREG2, __builtin_bit_cast(std::uint32_t, threshold));
    } else if constexpr (std::is_same_v<T, std::uint32_t>) {
        if constexpr (std::is_same_v<VectorType, sfpi::vInt>) {
            // SFPSWAP orders its operands as sign+magnitude, and INT32_2S_COMP is the load
            // mode that converts a two's-complement DEST word into that form. The threshold
            // does not arrive through a load, so it is re-encoded by hand here to match --
            // scoped to this branch, since on a float the same bits mean something else.
            constexpr std::uint32_t SIGN_MAG_SIGN_BIT = 0x80000000u;
            constexpr std::uint32_t SIGN_MAG_MAX_MAGNITUDE = 0x7FFFFFFFu;

            const int scalar = static_cast<int>(threshold);
            std::uint32_t sign_mag = static_cast<std::uint32_t>(scalar);
            if (scalar < 0) {
                // Negate in unsigned: -INT_MIN is signed overflow. INT_MIN has no
                // sign+magnitude form, so it saturates to -(2^31 - 1); the clamp is what
                // saturates it, where a mask would encode negative zero and clamp at 0.
                const std::uint32_t magnitude = -static_cast<std::uint32_t>(scalar);
                sign_mag =
                    SIGN_MAG_SIGN_BIT | (magnitude > SIGN_MAG_MAX_MAGNITUDE ? SIGN_MAG_MAX_MAGNITUDE : magnitude);
            }
            _sfpu_load_imm32_(p_sfpu::LREG2, sign_mag);
        } else {
            _sfpu_load_imm32_(p_sfpu::LREG2, threshold);
        }
    } else {
        static_assert(
            std::is_same_v<T, float> || std::is_same_v<T, std::uint32_t>, "Threshold type must be float or uint32_t");
    }

    _relu_min_impl_<APPROXIMATION_MODE, ITERATIONS, SFPLOAD_INSTR_MOD>(ITERATIONS);
}

// One Dest row clamped against the threshold parked in LREG12 (vConstIntPrgm0), shared by the
// full-range integer clamps below: load, SFPSWAP against the constant register, store. SFPSWAP
// writes both operands, but a write to a constant register is dropped, so parking the threshold
// there removes the per-row SFPMOV copy the compiler otherwise emits; the store advances the
// dest counter through ADDR_MOD_2 (physical slot 6, programmed to +2 by relu_min_init /
// relu_max_init), so the row needs no separate increment. With INVERT the row is complemented
// before and after the swap (unsigned order mapped onto SFPSWAP's sign-magnitude order).
//
// Raw rather than sfpi: sfpi 7.90.0 lowers every max() to SFPSWAP's "VD gets min" mode with the
// larger operand returned in VC, which needs the constant as VD -- illegal for LREG12 (an
// lra-constraints internal error for `max(x, vSMag(vConstIntPrgm0))`, an SFPMOV copy per row for
// `max(vSMag(vConstIntPrgm0), x)`). The min direction compiles to exactly this stream; one raw
// helper keeps both directions the same shape.
template <InstrModLoadStore LOAD_STORE_MOD, bool WANT_MAX, bool INVERT>
sfpi_inline void _clamp_row_against_lreg12_() {
    TTI_SFPLOAD(p_sfpu::LREG0, LOAD_STORE_MOD, ADDR_MOD_3, 0);
    if constexpr (INVERT) {
        TTI_SFPNOT(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);
    }
    // mod1 = 9: VD = max, VC = min; mod1 = 1: VD = min, VC = max. VC = LREG12 is never written.
    TTI_SFPSWAP(0, p_sfpu::LREG12, p_sfpu::LREG0, WANT_MAX ? 9 : sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
    if constexpr (INVERT) {
        TTI_SFPNOT(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);
    }
    TTI_SFPSTORE(p_sfpu::LREG0, LOAD_STORE_MOD, ADDR_MOD_2, 0);
}

// Full-range unsigned clamp against a threshold, shared by relu_min and relu_max/relu6.
// IS_LOWER_BOUND selects the path:
//   IS_LOWER_BOUND == true -> relu_min: result = max(x, threshold)
//   IS_LOWER_BOUND == false -> relu_max/relu6:  result = min(x, threshold)
//
// For U32 we can't do a plain vUInt compare (it subtracts and tests the sign, which overflows for
// operands >= 2^31 apart). Instead we use the hardware sign-magnitude min/max (SFPSWAP).
// When threshold MSB = 0, invert operands (~x, ~t) so that unsigned order maps to sign magnitude
// order, run min/max, then invert the result back.
// When threshold MSB = 1, raw bits already sort correctly under sign-magnitude with min/max roles swapped.
// The threshold (inverted or not) sits in LREG12 for the call; see _clamp_row_against_lreg12_.
// Rows are 5 (MSB = 0) / 3 (MSB = 1) instructions instead of 7 / 5.
template <bool APPROXIMATION_MODE, bool IS_LOWER_BOUND, DataFormat FORMAT, int ITERATIONS = 8>
inline void relu_clamp_uint(uint threshold) {
    static_assert(
        FORMAT == DataFormat::UInt32 || FORMAT == DataFormat::UInt16,
        "relu_clamp_uint requires DataFormat::UInt32 or UInt16 (uint8 dispatches through UInt32)");
    constexpr sfpi::DataLayout LAYOUT = (FORMAT == DataFormat::UInt16) ? sfpi::DataLayout::U16 : sfpi::DataLayout::U32;
    if constexpr (LAYOUT == sfpi::DataLayout::U32) {
        // relu_min needs max(x, t) and relu_max/relu6 needs min(x, t)
        constexpr bool WANT_MAX = IS_LOWER_BOUND;
        sfpi::l_reg[sfpi::LRegs::LReg0].in_use();
        if (static_cast<int>(threshold) >= 0) {
            // threshold MSB = 0: invert operands
            sfpi::vConstIntPrgm0 = ~threshold;
#pragma GCC unroll 8
            for (int d = 0; d < ITERATIONS; d++) {
                _clamp_row_against_lreg12_<InstrModLoadStore::INT32, WANT_MAX, true /* INVERT */>();
            }
        } else {
            // threshold MSB = 1: SM min/max applies directly, with the roles swapped
            sfpi::vConstIntPrgm0 = threshold;
#pragma GCC unroll 8
            for (int d = 0; d < ITERATIONS; d++) {
                _clamp_row_against_lreg12_<InstrModLoadStore::INT32, !WANT_MAX, false /* INVERT */>();
            }
        }
    } else {
        // U16: operands are < 2^16, so the direct vUInt compare cannot overflow.
        const vUInt t = static_cast<unsigned>(threshold);
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            vUInt x = dst_reg[0].mode<LAYOUT>();
            if constexpr (IS_LOWER_BOUND) {
                v_if(x < t) { x = t; }
                v_endif;
            } else {
                v_if(x > t) { x = t; }
                v_endif;
            }
            dst_reg[0].mode<LAYOUT>() = x;
            dst_reg++;
        }
    }
}

// Signed int32 clamp against a threshold, shared by relu_min and relu_max/relu6.
// IS_LOWER_BOUND selects the direction using the user-facing op's semantics:
//   IS_LOWER_BOUND == true  -> relu_min:       result = max(x, threshold)
//   IS_LOWER_BOUND == false -> relu_max/relu6: result = max(0, min(x, threshold))  (relu built in)
//
// The Dest word is a two's-complement int32. For threshold >= 0 (every production relu_max/relu6
// call, and relu_min with a non-negative bound) the row goes through SFPSWAP: the INT32_2S_COMP
// load (mod 12) maps the two's-complement word onto its sign-magnitude image, which SFPSWAP
// orders numerically, and the mod-12 store maps the result back. The one non-bijective point is
// INT_MIN, which the load turns into 0 (31-bit magnitude, sign dropped): harmless here, since
// max(0, t) = t = max(INT_MIN, t) and max(0, min(0, t)) = 0 = max(0, min(INT_MIN, t)) for every
// t >= 0, and a threshold >= 0 is its own sign-magnitude image. Verified bit-identical with the
// predicated compare it replaces on Wormhole silicon over INT_MIN, INT_MIN + 1, -1, 0, 1, INT_MAX,
// the powers of two of both signs with their neighbours and random words, at thresholds 0, 1, 5,
// 6, 1000, 2^30, INT_MAX - 1 and INT_MAX. Rows are 3 (relu_min) / 4 (relu_max) instructions
// instead of 10.
//
// A negative relu_min threshold keeps the plain int32 compare: there max(x, t) must return t for
// x = INT_MIN, which the sign-magnitude image (0) would rank above t.
template <bool APPROXIMATION_MODE, bool IS_LOWER_BOUND, int ITERATIONS = 8>
inline void relu_clamp_int(uint threshold) {
    const vInt t = static_cast<int>(threshold);
    const bool is_pos_threshold = static_cast<int>(threshold) >= 0;
    if constexpr (IS_LOWER_BOUND) {
        // relu_min: max(x, threshold)
        if (is_pos_threshold) {
            sfpi::l_reg[sfpi::LRegs::LReg0].in_use();
            sfpi::vConstIntPrgm0 = static_cast<int>(threshold);
#pragma GCC unroll 8
            for (int d = 0; d < ITERATIONS; d++) {
                _clamp_row_against_lreg12_<InstrModLoadStore::INT32_2S_COMP, true /* WANT_MAX */, false>();
            }
        } else {
#pragma GCC unroll 8
            for (int d = 0; d < ITERATIONS; d++) {
                vInt x = dst_reg[0].mode<sfpi::DataLayout::I32>();
                v_if(x < 0) {
                    v_if(x < t) { x = t; }
                    v_endif;
                }
                v_endif;  // x >= 0 > t, so max is x (keep)
                dst_reg[0].mode<sfpi::DataLayout::I32>() = x;
                dst_reg++;
            }
        }
    } else {
        // relu_max/relu6: max(0, min(x, threshold))
        if (is_pos_threshold) {
            sfpi::l_reg[sfpi::LRegs::LReg0].in_use();
            sfpi::vConstIntPrgm0 = static_cast<int>(threshold);
#pragma GCC unroll 8
            for (int d = 0; d < ITERATIONS; d++) {
                TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32_2S_COMP, ADDR_MOD_3, 0);
                // LREG0 = min(x, threshold), then LREG0 = max(LREG0, 0) against the constant zero
                // in LREG9 (mod1 = 9: VD = max; the constant VC is never written).
                TTI_SFPSWAP(0, p_sfpu::LREG12, p_sfpu::LREG0, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
                TTI_SFPSWAP(0, sfpi::CREG_IDX_0, p_sfpu::LREG0, 9);
                TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32_2S_COMP, ADDR_MOD_2, 0);
            }
        } else {
            // threshold < 0: max(0, min(x, threshold)) == 0 for every lane
            const vInt zero = 0;
#pragma GCC unroll 8
            for (int d = 0; d < ITERATIONS; d++) {
                dst_reg[0].mode<sfpi::DataLayout::I32>() = zero;
                dst_reg++;
            }
        }
    }
}

// The integer clamps above store through ADDR_MOD_2, which the unary SFPU addr-mod base maps to
// physical slot 6: program it to advance the dest counter by one sfpi row (two counter steps).
// The float paths load and store with ADDR_MOD_3 and dst_reg++, so this is inert for them.
inline void _relu_clamp_addr_mod_init_() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
}

inline void relu_min_init() {
    _relu_clamp_addr_mod_init_();
    math::reset_counters(p_setrwc::SET_ABD_F);
}

template <bool APPROXIMATION_MODE>
inline void relu_min(uint uint_threshold) {
    vFloat threshold = Converter::as_float(uint_threshold);
    for (int d = 0; d < 8; d++) {
        vFloat a = dst_reg[0];
        v_if(a < threshold) { a = threshold; }
        v_endif;
        dst_reg[0] = a;
        dst_reg++;
    }
}

inline void relu_max_init() {
    _relu_clamp_addr_mod_init_();
    math::reset_counters(p_setrwc::SET_ABD_F);
}

template <bool APPROXIMATION_MODE>
inline void relu_max(uint uint_threshold) {
    vFloat threshold = Converter::as_float(uint_threshold);
    for (int d = 0; d < 8; d++) {
        vFloat a = dst_reg[0];
        v_if(a > threshold) { a = threshold; }
        v_endif;
        v_if(a < 0.0f) { a = 0.0f; }
        v_endif;
        dst_reg[0] = a;
        dst_reg++;
    }
}

inline void lrelu_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_lrelu(const uint slope) {
    _calculate_lrelu_<APPROXIMATION_MODE>(ITERATIONS, slope);
}

}  // namespace sfpu
}  // namespace ckernel
