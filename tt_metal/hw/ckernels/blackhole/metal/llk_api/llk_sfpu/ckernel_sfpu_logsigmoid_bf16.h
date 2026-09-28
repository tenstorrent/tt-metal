// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace ckernel::sfpu {}

#if !defined(TT_POLY_LLK_DISABLE)
#include <cstdint>
#include <array>
namespace ttpoly_generated {
struct LogsigmoidBf16Config {
    static constexpr float kNumerator[] = {
        __builtin_bit_cast(float, 0x3f7fe63eu),
        __builtin_bit_cast(float, 0xbef90b38u),
        __builtin_bit_cast(float, 0x3e820f76u),
        __builtin_bit_cast(float, 0xbd9840f8u)};
    static constexpr float kDenominator[] = {__builtin_bit_cast(float, 0x3f800000u)};
    static constexpr float kExpCoefficients[] = {
        __builtin_bit_cast(float, 0x3f803884u),
        __builtin_bit_cast(float, 0x3f285adeu),
        __builtin_bit_cast(float, 0x3eaca410u)};
    static constexpr uint32_t kExpDegree = 2;
    static constexpr float kMultiplier = __builtin_bit_cast(float, 0xbfb8aa3bu);
    static constexpr uint32_t kBoundBits = 0x42b00000u;
    static constexpr bool kHasBound = true;
    static constexpr bool kPolynomial = true;
    static constexpr bool kSquareDecay = false;
    static constexpr bool kResidualFold = false;
    static constexpr bool kSkipInputAbs = false;
    static constexpr bool kSquareStore = false;
    static constexpr bool kCorrectionStore = false;
    static constexpr bool kPositivePart = false;
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
#define TT_POLY_LOGSIGMOID_BF16_AVAILABLE 1
#endif

namespace ckernel::sfpu {

#if !defined(TT_POLY_LLK_DISABLE)
template <int ITERATIONS = 8>
inline void calculate_logsigmoid_tt_poly_bf16() {
    ckernel::sfpu::ttpoly::calculate_abs_exp_correction<ttpoly_generated::LogsigmoidBf16Config, ITERATIONS>();
}
#endif

}  // namespace ckernel::sfpu
