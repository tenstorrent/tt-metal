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
    static constexpr bool kResidualFold = true;
    static constexpr bool kSkipInputAbs = true;
    static constexpr bool kSquareStore = false;
    static constexpr bool kCorrectionStore = true;
    static constexpr bool kPositivePart = false;
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
    static constexpr uint32_t kDomainActionCount = 3;
    static constexpr std::array<TtDomainActionRecord, kDomainActionCount> kDomainActions = {
        {{3.3895313892515355e+38f,
          1,
          0,
          3,
          3,
          0.0000000000000000e+00f,
          0.0000000000000000e+00f,
          0.0000000000000000e+00f},
         {-1.0000000000000000e+01f,
          0,
          0,
          1,
          0,
          0.0000000000000000e+00f,
          0.0000000000000000e+00f,
          0.0000000000000000e+00f},
         {9.3000000000000000e+01f,
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
