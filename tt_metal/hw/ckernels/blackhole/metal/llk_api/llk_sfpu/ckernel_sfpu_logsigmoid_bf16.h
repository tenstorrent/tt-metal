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
    static constexpr bool kSquareDecay = false;
};
}  // namespace ckernel::sfpu
#include "sfpu/ckernel_sfpu_bf16_abs_exp_correction.h"

namespace ckernel::sfpu {

template <int ITERATIONS = 8>
inline void calculate_logsigmoid_bf16() {
    ckernel::sfpu::bf16::calculate_abs_exp_correction<ckernel::sfpu::LogsigmoidBf16Config, ITERATIONS>();
}

}  // namespace ckernel::sfpu
