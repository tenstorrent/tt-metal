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
// blocks (SFPGT/SFPMOV/SFPENCC each), 9 -> 5 instructions per row when `threshold` is a literal or
// a constant register, 6 when it lives in a general LReg (SFPSWAP writes both operands, so the
// compiler copies the threshold first).
//
// The order is the one the predicated chain had -- threshold first, then the relu clamp -- and
// it is not interchangeable with min(max(x, 0), threshold): for threshold < 0 this form returns
// +0.0 on every lane (min(x, t) <= t < 0, then the clamp), the other would return the threshold.
// SFPSWAP orders FP32 by sign-magnitude, -NaN < -Inf < ... < -0 < +0 < ... < +Inf < +NaN, so
//   +NaN -> threshold (outranks every value, the min lands on the threshold)
//   -NaN -> +0.0      (ranks below every value, survives the min, the clamp takes it)
//   -0.0 -> +0.0      (ranks below +0.0)
// which is what the SFPGT/SFPSETCC chain returned as well; measured on Blackhole silicon, see
// test_eltwise_unary_sfpu_relu_max_threshold.
sfpi_inline sfpi::vFloat _relu_max_body_(sfpi::vFloat val, sfpi::vFloat threshold) {
    return sfpi::max(sfpi::min(val, threshold), 0.0f);
}

// Float relu_max with a per-call scalar threshold. The threshold is parked in vConstFloatPrgm1
// (LREG13) for the duration of the call: SFPSWAP writes both of its operands, so a threshold held
// in a general LReg costs an SFPMOV copy per row, while a write to a constant register is dropped.
// One SFPCONFIG (+2 SFPLOADI) per call buys one instruction per row: 5 instead of 6, 9 before.
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

// Integer relu_max. No production caller on Blackhole (relu_max_tile_int32 uses the metal
// relu_clamp_int); kept as it was.
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

// Dst holds int32 in whatever encoding the caller put there, and ttnn's is two's complement,
// which is what the compares below assume. Named rather than left to sfpi's default, because
// that default is per-arch -- I32 here, SM32 (converting) on Wormhole.
inline constexpr sfpi::DataLayout INT_DEST_LAYOUT = sfpi::DataLayout::I32;

// Sign of an integer threshold. It is uniform across lanes, so it picks which of the integer
// forms below runs from outside the loop.
enum class ThresholdSign { Negative, Zero, Positive };

inline constexpr ThresholdSign threshold_sign_of(const int scalar) {
    return scalar < 0 ? ThresholdSign::Negative : scalar == 0 ? ThresholdSign::Zero : ThresholdSign::Positive;
}

template <typename VecType, bool APPROXIMATION_MODE, int ITERATIONS>
inline void _relu_min_impl_(const int iterations, VecType threshold, const ThresholdSign threshold_sign) {
    if constexpr (std::is_same_v<VecType, sfpi::vInt>) {
        // A plain `a < threshold` is not a safe compare over the full int32 range: the signed
        // compare subtracts, so it inverts its answer once the operands are 2^31 or more apart.
        // Splitting on the sign of the threshold leaves only same-sign operands to it, whose
        // difference cannot overflow. Wormhole compares differently and needs no split.
        if (threshold_sign == ThresholdSign::Negative) {
            for (int d = 0; d < iterations; d++) {
                sfpi::vInt a = sfpi::dst_reg[0].mode<INT_DEST_LAYOUT>();
                // a >= 0 > threshold keeps a, so only the negative lanes can lose, and there
                // both operands are negative.
                v_if(a < 0) {
                    v_if(a < threshold) { a = threshold; }
                    v_endif;
                }
                v_endif;
                sfpi::dst_reg[0].mode<INT_DEST_LAYOUT>() = a;
                sfpi::dst_reg++;
            }
        } else if (threshold_sign == ThresholdSign::Zero) {
            // The sign test is the whole compare, which is the form relu_tile_int32 takes.
            for (int d = 0; d < iterations; d++) {
                sfpi::vInt a = sfpi::dst_reg[0].mode<INT_DEST_LAYOUT>();
                v_if(a < 0) { a = threshold; }
                v_endif;
                sfpi::dst_reg[0].mode<INT_DEST_LAYOUT>() = a;
                sfpi::dst_reg++;
            }
        } else {
            for (int d = 0; d < iterations; d++) {
                sfpi::vInt a = sfpi::dst_reg[0].mode<INT_DEST_LAYOUT>();
                // Lifting the negative lanes to a positive threshold first is the whole answer
                // for them, and it leaves only non-negative operands to the compare.
                v_if(a < 0) { a = threshold; }
                v_endif;
                v_if(a < threshold) { a = threshold; }
                v_endif;
                sfpi::dst_reg[0].mode<INT_DEST_LAYOUT>() = a;
                sfpi::dst_reg++;
            }
        }
    } else {
        for (int d = 0; d < iterations; d++) {
            VecType a = sfpi::dst_reg[0];
            v_if(a < threshold) { sfpi::dst_reg[0] = threshold; }
            v_endif;
            sfpi::dst_reg++;
        }
    }
}

// Wrappers
template <typename VectorType, bool APPROXIMATION_MODE, int ITERATIONS, typename T>
inline void _relu_min_(T threshold) {
    static_assert(
        std::is_same_v<VectorType, sfpi::vFloat> || std::is_same_v<VectorType, sfpi::vInt>,
        "VectorType must be sfpi::vFloat or sfpi::vInt");

    VectorType v_threshold;
    // The integer path alone reads this; its compare is the one that splits on the sign.
    ThresholdSign threshold_sign = ThresholdSign::Zero;
    if constexpr (std::is_same_v<T, float>) {
        static_assert(
            std::is_same_v<VectorType, sfpi::vFloat>,
            "A float threshold requires VectorType == sfpi::vFloat: sfpi::vInt has no float constructor, so the "
            "assignment below would otherwise fail as an "
            "ambiguous conversion");
        v_threshold = threshold;
    } else if constexpr (std::is_same_v<T, std::uint32_t>) {
        if constexpr (std::is_same_v<VectorType, sfpi::vInt>) {
            const int scalar = static_cast<int>(threshold);
            v_threshold = scalar;
            threshold_sign = threshold_sign_of(scalar);
        } else {
            v_threshold = Converter::as_float(threshold);
        }
    } else {
        static_assert(
            std::is_same_v<T, float> || std::is_same_v<T, std::uint32_t>, "Threshold type must be float or uint32_t");
    }

    _relu_min_impl_<VectorType, APPROXIMATION_MODE, ITERATIONS>(ITERATIONS, v_threshold, threshold_sign);
}

// Full-range unsigned clamp against a threshold, shared by relu_min and relu_max/relu6.
// IS_LOWER_BOUND selects the path:
//   IS_LOWER_BOUND == true -> relu_min: result = max(x, threshold)
//   IS_LOWER_BOUND == false -> relu_max/relu6: result = min(x, threshold)
//
// For U32 we can't do a plain vUInt compare (it subtracts and tests the sign, which overflows for
// operands >= 2^31 apart). Instead we use the hardware sign-magnitude min/max (SFPSWAP).
// When threshold MSB = 0, invert operands (~x, ~t) so that unsigned order maps to sign magnitude
// order, run min/max, then invert the result back.
// When threshold MSB = 1, raw bits already sort correctly under sign-magnitude with min/max roles swapped.
template <bool APPROXIMATION_MODE, bool IS_LOWER_BOUND, DataFormat FORMAT, int ITERATIONS = 8>
inline void relu_clamp_uint(uint threshold) {
    static_assert(
        FORMAT == DataFormat::UInt32 || FORMAT == DataFormat::UInt16,
        "relu_clamp_uint requires DataFormat::UInt32 or UInt16 (uint8 dispatches through UInt32)");
    constexpr sfpi::DataLayout LAYOUT = (FORMAT == DataFormat::UInt16) ? sfpi::DataLayout::U16 : sfpi::DataLayout::U32;
    if constexpr (LAYOUT == sfpi::DataLayout::U32) {
        // relu_min needs max(x, t) and relu_max/relu6 needs min(x, t)
        constexpr bool WANT_MAX = IS_LOWER_BOUND;
        const vUInt t = static_cast<unsigned>(threshold);
        if (static_cast<int>(threshold) >= 0) {
            // threshold MSB = 0: invert operands
            const vSMag nt = sfpi::as<vSMag>(~t);
#pragma GCC unroll 8
            for (int d = 0; d < ITERATIONS; d++) {
                vUInt raw = dst_reg[0].mode<LAYOUT>();
                vSMag x = sfpi::as<vSMag>(~raw);
                vUInt r = sfpi::as<vUInt>(WANT_MAX ? sfpi::max(x, nt) : sfpi::min(x, nt));
                dst_reg[0].mode<LAYOUT>() = ~r;  // invert the result back
                dst_reg++;
            }
        } else {
            // threshold MSB = 1: SM min/max applies directly
            const vSMag ts = sfpi::as<vSMag>(t);
#pragma GCC unroll 8
            for (int d = 0; d < ITERATIONS; d++) {
                vUInt raw = dst_reg[0].mode<LAYOUT>();
                vSMag x = sfpi::as<vSMag>(raw);
                dst_reg[0].mode<LAYOUT>() = sfpi::as<vUInt>(WANT_MAX ? sfpi::min(x, ts) : sfpi::max(x, ts));
                dst_reg++;
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
// IS_LOWER_BOUND selects the path:
//   IS_LOWER_BOUND == true -> relu_min: result = max(x, threshold)
//   IS_LOWER_BOUND == false -> relu_max/relu6: result = max(0, min(x, threshold))
//
// Load/store as signed 2's complement (DataLayout::I32). Since the sign-magnitude SFPSWAP path cannot
// represent -2^31 (0x80000000 is -0), we use the plain int32 compare.
template <bool APPROXIMATION_MODE, bool IS_LOWER_BOUND, int ITERATIONS = 8>
inline void relu_clamp_int(uint threshold) {
    const vInt t = static_cast<int>(threshold);
    const bool is_pos_threshold = static_cast<int>(threshold) >= 0;
    if constexpr (IS_LOWER_BOUND) {
        // relu_min: max(x, threshold)
        if (is_pos_threshold) {
#pragma GCC unroll 8
            for (int d = 0; d < ITERATIONS; d++) {
                vInt x = dst_reg[0].mode<sfpi::DataLayout::I32>();
                // v_if(x < 0) is NOT redundant: the SFPU signed compare subtracts and tests the sign,
                // which overflows when x and t are >= 2^31 apart and wrongly gives x>=t. Lifting negatives
                // up to t first leaves only non-negative lanes for the x < t compare, which then cannot overflow.
                v_if(x < 0) { x = t; }
                v_endif;
                v_if(x < t) { x = t; }
                v_endif;
                dst_reg[0].mode<sfpi::DataLayout::I32>() = x;
                dst_reg++;
            }
        } else {
#pragma GCC unroll 8
            for (int d = 0; d < ITERATIONS; d++) {
                vInt x = dst_reg[0].mode<sfpi::DataLayout::I32>();
                v_if(x < 0) {
                    v_if(x < t) { x = t; }
                    v_endif;
                }
                v_endif;
                dst_reg[0].mode<sfpi::DataLayout::I32>() = x;
                dst_reg++;
            }
        }
    } else {
        // relu_max/relu6: max(0, min(x, threshold))
        if (is_pos_threshold) {
#pragma GCC unroll 8
            for (int d = 0; d < ITERATIONS; d++) {
                vInt x = dst_reg[0].mode<sfpi::DataLayout::I32>();
                v_if(x < 0) { x = 0; }
                v_endif;
                v_if(x > t) { x = t; }
                v_endif;
                dst_reg[0].mode<sfpi::DataLayout::I32>() = x;
                dst_reg++;
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
inline void relu_min_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

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

inline void relu_max_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

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

bool bf16_dest_leaky_relu();
template <int ITERATIONS>
void calculate_leaky_relu_bf16();
void init_leaky_relu_bf16();
// The slope the generated leaky_relu kernel is fitted for, as FP32 bits; any other slope runs the stock kernel.
inline constexpr std::uint32_t leaky_relu_bf16_slope_bits = 0x3c23d70au;
// Whether BF16 DEST runs the generated leaky_relu kernel as one call over the whole tile.
inline constexpr bool leaky_relu_bf16_whole_tile = true;
// Sets up the generated BF16 leaky_relu kernel for the instance it serves.
template <bool bf16_kernel>
inline void leaky_relu_bf16_tile_init() {
    if constexpr (bf16_kernel) {
        init_leaky_relu_bf16();
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool is_fp32_dest_acc_en = true>
inline void calculate_lrelu(uint slope) {
    if constexpr (!is_fp32_dest_acc_en && ITERATIONS == 32) {
        if (bf16_dest_leaky_relu() && slope == ckernel::sfpu::leaky_relu_bf16_slope_bits) {
            calculate_leaky_relu_bf16<ITERATIONS>();
            return;
        }
    }
    _calculate_lrelu_<APPROXIMATION_MODE>(ITERATIONS, slope);
}

}  // namespace sfpu
}  // namespace ckernel

#include "ckernel_sfpu_leaky_relu_bf16.h"
