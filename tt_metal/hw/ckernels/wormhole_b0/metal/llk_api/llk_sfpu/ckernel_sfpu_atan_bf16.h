// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Generated architecture configuration; shared primitives live in common/.
#pragma once
#include <cstdint>
namespace ttpoly_generated {
struct AtanBf16Config {
    static constexpr unsigned kDegree = 0x00000003u;
    static constexpr unsigned kCoreBoundBits = 0x3f800000u;
    static constexpr unsigned kComplementBits = 0x3fc90fdbu;
    static constexpr unsigned kBodySlots = 0x00000000u;
    static constexpr unsigned kMacroSequenceBits = 0x00000000u;
    static constexpr unsigned kShadowBase = 0x00000000u;
    static constexpr unsigned kShadowRows = 0x00000000u;
    static constexpr unsigned kCoefficientBits[] = {0x3f800000u, 0xbea7be2cu, 0x3e232346u, 0xbd3e731cu};
    static constexpr float kCoefficients[] = {
        __builtin_bit_cast(float, 0x3f800000u),
        __builtin_bit_cast(float, 0xbea7be2cu),
        __builtin_bit_cast(float, 0x3e232346u),
        __builtin_bit_cast(float, 0xbd3e731cu)};
};
}  // namespace ttpoly_generated
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_reciprocal_complement.h"
#ifndef TT_POLY_LLK_RECIPROCAL_COMPLEMENT_SELECTED_V1
#error "typed selected reciprocal-complement runtime required"
#endif
#define TT_POLY_ATAN_BF16_AVAILABLE 1
