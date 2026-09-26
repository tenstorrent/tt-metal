// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Generated architecture configuration; shared primitives live in common/.
#pragma once
#include <cstdint>
#include <array>
namespace ttpoly_generated {
struct ErfcBf16Config {
    static constexpr float kNumerator[] = {
        __builtin_bit_cast(float, 0x3f1cb49cu), __builtin_bit_cast(float, 0x3ea79f6du)};
    static constexpr float kDenominator[] = {
        __builtin_bit_cast(float, 0x3f1ceb0bu),
        __builtin_bit_cast(float, 0x3f800000u),
        __builtin_bit_cast(float, 0x3f1651e0u)};
    static constexpr float kExpCoefficients[] = {
        __builtin_bit_cast(float, 0x3f800000u),
        __builtin_bit_cast(float, 0x3f32107du),
        __builtin_bit_cast(float, 0x3e6798ecu),
        __builtin_bit_cast(float, 0x3da00879u)};
    static constexpr uint32_t kExpDegree = 3;
    static constexpr float kMultiplier = __builtin_bit_cast(float, 0x3fb8aa3bu);
    static constexpr uint32_t kBoundBits = 0x41140000u;
    static constexpr bool kHasBound = true;
    static constexpr bool kPolynomial = false;
    static constexpr bool kSquareDecay = true;
    static constexpr bool kResidualFold = false;
    static constexpr bool kSkipInputAbs = false;
    static constexpr bool kSquareStore = true;
    static constexpr bool kCorrectionStore = false;
    static constexpr bool kPositivePart = false;
    static constexpr int kNegativeInfinityClass = 2;
    static constexpr float kPositiveNonfiniteConstant = __builtin_bit_cast(float, 0x2c4e0000u);

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
        {{-5.0000000000000000e+00f,
          0,
          0,
          0,
          0,
          2.0000000000000000e+00f,
          0.0000000000000000e+00f,
          0.0000000000000000e+00f},
         {9.5000000000000000e+00f,
          1,
          1,
          0,
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
