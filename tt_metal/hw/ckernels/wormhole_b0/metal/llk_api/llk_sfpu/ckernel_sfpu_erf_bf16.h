// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Generated architecture configuration; shared primitives live in common/.
#pragma once
#include <cstdint>

namespace ttpoly_generated {
struct ErfBf16Config {
    static constexpr const char kTerminalAbi[] = "decoded_bf16_paired_saturating_signed_abs_v2";
    static constexpr uint32_t kExhaustivePopulation = 65536u;
    static constexpr uint32_t kDegree = 7u;
    static constexpr uint32_t kCoefficientBits[] = {
        0x00000000u, 0x3f904bb0u, 0x3ceceac4u, 0xbf007bbau, 0x3e40e7f0u, 0x3c90cff8u, 0xbca6e156u, 0x3b39f45cu};
    static constexpr uint32_t kClampMaxBits = 0x3f800000u;
    static constexpr bool kIntrinsicExceptional = false;
    // Indexed by: +0, -0, +subnormal, -subnormal, finite, +Inf, -Inf, +NaN, -NaN.
    static constexpr uint8_t kRawClassOutput[] = {3u, 3u, 3u, 3u, 5u, 5u, 5u, 5u, 5u};
    static constexpr uint8_t kRawToEffectiveClass[] = {3u, 3u, 3u, 3u, 5u, 1u, 2u, 1u, 2u};
    static constexpr bool kInputDaz = true;
    static constexpr bool kOutputFtz = true;
};
}  // namespace ttpoly_generated

#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_cascade_signed_abs.h"

#if !defined(TT_POLY_LLK_DECODED_BF16_PAIRED_SATURATING_SIGNED_ABS_V2)
#error "typed signed-abs plan requires decoded BF16 saturation support"
#endif
