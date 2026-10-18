// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <array>
namespace ckernel::sfpu {
struct LogsigmoidBf16Config {
    static constexpr float kNumerator[] = {
        __builtin_bit_cast(float, 0x3f7fe63eu),
        __builtin_bit_cast(float, 0xbef90b38u),
        __builtin_bit_cast(float, 0x3e820f76u),
        __builtin_bit_cast(float, 0xbd9840f8u)};
    static constexpr float kExpCoefficients[] = {
        __builtin_bit_cast(float, 0x3f803884u),
        __builtin_bit_cast(float, 0x3f285adeu),
        __builtin_bit_cast(float, 0x3eaca410u)};
    static constexpr uint32_t kExpDegree = 2;
    static constexpr float kMultiplier = __builtin_bit_cast(float, 0xbfb8aa3bu);
    static constexpr uint32_t kBoundBits = 0x42b00000u;
    static constexpr bool kResidualFold = true;

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
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_logsigmoid_bf16_abs_exp_correction.h"

namespace ckernel::sfpu {

template <int ITERATIONS = 32>
inline void calculate_logsigmoid_bf16() {
    ckernel::sfpu::bf16::logsigmoid::calculate_abs_exp_correction<ckernel::sfpu::LogsigmoidBf16Config, ITERATIONS>();
}
inline void init_logsigmoid_bf16() {
    ckernel::sfpu::bf16::logsigmoid::init_abs_exp_correction<ckernel::sfpu::LogsigmoidBf16Config>();
}

}  // namespace ckernel::sfpu
