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
    static constexpr bool kResidualFold = false;
    static constexpr bool kSkipInputAbs = false;
    static constexpr bool kSquareStore = false;
    static constexpr bool kCorrectionStore = false;
    static constexpr bool kPositivePart = true;
    static constexpr bool kRawNanAffinePart = true;
    static constexpr bool kResidualTti = true;
    static constexpr float kPositiveNonfiniteConstant = __builtin_bit_cast(float, 0x00000000u);
    static constexpr uint32_t kDomainActionCount = 0;
};
}  // namespace ttpoly_generated
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_abs_exp_correction.h"
#ifndef TT_POLY_LLK_ABS_EXP_CORRECTION_SELECTED_V1
#error "typed selected abs-exp correction runtime required"
#endif
#define TT_POLY_SOFTPLUS_BF16_AVAILABLE 1
