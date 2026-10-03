// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <array>
#include <cstdint>
namespace ckernel::sfpu {
struct LogitBf16Config {
    static constexpr unsigned kLogDegree = 2u;
    static constexpr unsigned kCorrectionDegree = 1u;
    static constexpr unsigned kIntervalCount = 2u;
    static constexpr unsigned kPositiveFirst = 32385u;
    static constexpr unsigned kPositiveEnd = 32640u;
    static constexpr unsigned kNegativeFirst = 65153u;
    static constexpr unsigned kNegativeEnd = 65408u;
    static constexpr unsigned kPatternCount = 510u;
    static constexpr bool kBf16 = true;
    static constexpr std::array<float, 7> kLut = {
        __builtin_bit_cast(float, 0x00000000u),
        __builtin_bit_cast(float, 0x3f800000u),
        __builtin_bit_cast(float, 0xbfecb5ccu),
        __builtin_bit_cast(float, 0xbf785579u),
        __builtin_bit_cast(float, 0x3f75c8a6u),
        __builtin_bit_cast(float, 0xbec21922u),
        __builtin_bit_cast(float, 0x3de5a23cu)};
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_bf16_normalized_log_odds.h"

namespace ckernel::sfpu {

template <int ITERATIONS = 8>
inline void calculate_logit_bf16() {
    ckernel::sfpu::bf16::calculate_normalized_log_odds<ckernel::sfpu::LogitBf16Config, ITERATIONS>();
}

}  // namespace ckernel::sfpu
