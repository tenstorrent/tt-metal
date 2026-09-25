// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Generated architecture configuration; shared primitives live in common/.
#pragma once
#include <cstdint>
namespace ttpoly_generated {
struct AtanhBf16Config {
    static constexpr uint32_t kNumDegree = 11u;
    static constexpr uint32_t kDenDegree = 4u;
    static constexpr float kNumerator[] = {
        __builtin_bit_cast(float, 0x00000000u),
        __builtin_bit_cast(float, 0x3f800000u),
        __builtin_bit_cast(float, 0x00000000u),
        __builtin_bit_cast(float, 0xbfcc2d0bu),
        __builtin_bit_cast(float, 0x00000000u),
        __builtin_bit_cast(float, 0x3eebcd4fu),
        __builtin_bit_cast(float, 0x00000000u),
        __builtin_bit_cast(float, 0x3e42f154u),
        __builtin_bit_cast(float, 0x00000000u),
        __builtin_bit_cast(float, 0xbe3e8d7cu),
        __builtin_bit_cast(float, 0x00000000u),
        __builtin_bit_cast(float, 0x3e068b65u)};
    static constexpr float kDenominator[] = {
        __builtin_bit_cast(float, 0x3f800000u),
        __builtin_bit_cast(float, 0x00000000u),
        __builtin_bit_cast(float, 0xbff6f85fu),
        __builtin_bit_cast(float, 0x00000000u),
        __builtin_bit_cast(float, 0x3f6e010bu)};
    static constexpr uint32_t kBoundBits = 0x3f800000u;
    static constexpr bool kPinTop = true;
    static constexpr uint32_t kRawClass = 2u;
    static constexpr bool kRawEarly = false;
};
}  // namespace ttpoly_generated
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_rational_parity.h"
#ifndef TT_POLY_LLK_MIRRORED_PARITY_RATIONAL_V1
#error "typed mirrored parity rational runtime required"
#endif
