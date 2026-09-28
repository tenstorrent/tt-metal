// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Generated architecture configuration; shared primitives live in common/.
#pragma once
#include <cstdint>
#include <array>
namespace ttpoly_generated {
struct SoftplusBf16Config {
    static constexpr float kNumerator[] = {
        __builtin_bit_cast(float, 0x3f800000u),
        __builtin_bit_cast(float, 0xbefab96eu),
        __builtin_bit_cast(float, 0x3e85799eu),
        __builtin_bit_cast(float, 0xbda01eecu)};
    static constexpr float kDenominator[] = {__builtin_bit_cast(float, 0x3f800000u)};
    static constexpr float kExpCoefficients[] = {
        __builtin_bit_cast(float, 0x3f803884u),
        __builtin_bit_cast(float, 0x3f285adeu),
        __builtin_bit_cast(float, 0x3eaca410u)};
    static constexpr uint32_t kExpDegree = 2;
    static constexpr float kMultiplier = __builtin_bit_cast(float, 0xbfb8aa3bu);
    static constexpr uint32_t kBoundBits = 0x42af0000u;
    static constexpr bool kHasBound = true;
    static constexpr bool kPolynomial = true;
    static constexpr bool kSquareDecay = false;
    static constexpr bool kResidualFold = true;
    static constexpr bool kSkipInputAbs = true;
    static constexpr bool kSquareStore = false;
    static constexpr bool kCorrectionStore = true;
    static constexpr bool kPositivePart = true;
    static constexpr bool kRawNanAffinePart = false;
    static constexpr bool kResidualTti = true;
    static constexpr float kPositiveNonfiniteConstant = __builtin_bit_cast(float, 0x00000000u);

    // Typed domain actions.
    struct TtDomainActionRecord {
        float bound;
        uint8_t direction;
        uint8_t inclusive;
        uint8_t action_kind;   // 0=constant, 1=identity, 2=affine, 3=class, 4=signed-inf
        uint8_t return_class;  // 0=NaN, 1=+Inf, 2=-Inf, 3=+0, 4=-0
        float value;
        float scale;
        float bias;
    };
    static constexpr uint32_t kDomainActionCount = 2;
    static constexpr std::array<TtDomainActionRecord, kDomainActionCount> kDomainActions = {
        {{-8.7500000000000000e+01f,
          0,
          1,
          0,
          0,
          0.0000000000000000e+00f,
          0.0000000000000000e+00f,
          0.0000000000000000e+00f},
         {4.1250000000000000e+00f,
          1,
          0,
          1,
          0,
          0.0000000000000000e+00f,
          0.0000000000000000e+00f,
          0.0000000000000000e+00f}}};
};
}  // namespace ttpoly_generated
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_abs_exp_correction.h"
#ifndef TT_POLY_LLK_ABS_EXP_CORRECTION_SELECTED_V1
#error "typed selected abs-exp correction runtime required"
#endif
#define TT_POLY_SOFTPLUS_BF16_AVAILABLE 1
