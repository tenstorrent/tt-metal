// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace ckernel::sfpu {}

#if !defined(TT_POLY_LLK_DISABLE)
#include <array>
#include <cstdint>
namespace ttpoly_generated {
struct LogitBf16Config {
    static constexpr unsigned kLogDegree = 2u;
    static constexpr unsigned kCorrectionDegree = 1u;
    static constexpr unsigned kIntervalCount = 2u;
    static constexpr unsigned kPositiveFirst = 32384u;
    static constexpr unsigned kPositiveEnd = 32640u;
    static constexpr unsigned kNegativeFirst = 65152u;
    static constexpr unsigned kNegativeEnd = 65408u;
    static constexpr unsigned kPatternCount = 512u;
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
}  // namespace ttpoly_generated
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_normalized_log_odds.h"
#ifndef TT_POLY_LLK_NORMALIZED_LOG_ODDS_SELECTED_V1
#error "typed selected normalized log-odds runtime required"
#endif
#define TT_POLY_LOGIT_BF16_AVAILABLE 1
#endif

namespace ckernel::sfpu {

#if !defined(TT_POLY_LLK_DISABLE)
template <int ITERATIONS = 8>
inline void calculate_logit_tt_poly_bf16() {
    ckernel::sfpu::ttpoly::calculate_normalized_log_odds<ttpoly_generated::LogitBf16Config, ITERATIONS>();
}
#endif

}  // namespace ckernel::sfpu
