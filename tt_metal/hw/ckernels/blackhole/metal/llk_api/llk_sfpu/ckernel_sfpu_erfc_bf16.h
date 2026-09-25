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
    static constexpr bool kSquareStore = false;
    static constexpr bool kCorrectionStore = false;
    static constexpr bool kPositivePart = false;
    static constexpr int kNegativeInfinityClass = 2;
    static constexpr float kPositiveNonfiniteConstant = __builtin_bit_cast(float, 0x2c4f0000u);
    static constexpr uint32_t kDomainActionCount = 0;
};
}  // namespace ttpoly_generated
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_abs_exp_correction.h"
#ifndef TT_POLY_LLK_ABS_EXP_CORRECTION_SELECTED_V1
#error "typed selected abs-exp correction runtime required"
#endif
