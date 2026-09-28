// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Generated architecture configuration; shared primitives live in common/.
#pragma once
#include <cstdint>
namespace ttpoly_generated {
struct AtanBf16Config {
    static constexpr unsigned kDegree = 0x00000008u;
    static constexpr unsigned kCoreBoundBits = 0x3f800000u;
    static constexpr unsigned kComplementBits = 0x3fc90fdbu;
    static constexpr unsigned kBodySlots = 0x00000020u;
    static constexpr unsigned kMacroSequenceBits = 0x63550087u;
    static constexpr unsigned kShadowBase = 0x00000060u;
    static constexpr unsigned kShadowRows = 0x00000020u;
    static constexpr unsigned kCoefficientBits[] = {
        0x00000000u,
        0x3f800022u,
        0xb9e1a4f4u,
        0xbea6c2e2u,
        0xbd53ff36u,
        0x3ebe7e06u,
        0xbe9a38c2u,
        0x3ddc317eu,
        0xbc76ece6u};
    static constexpr float kCoefficients[] = {
        __builtin_bit_cast(float, 0x00000000u),
        __builtin_bit_cast(float, 0x3f800022u),
        __builtin_bit_cast(float, 0xb9e1a4f4u),
        __builtin_bit_cast(float, 0xbea6c2e2u),
        __builtin_bit_cast(float, 0xbd53ff36u),
        __builtin_bit_cast(float, 0x3ebe7e06u),
        __builtin_bit_cast(float, 0xbe9a38c2u),
        __builtin_bit_cast(float, 0x3ddc317eu),
        __builtin_bit_cast(float, 0xbc76ece6u)};
};
}  // namespace ttpoly_generated
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_reciprocal_complement.h"
#ifndef TT_POLY_LLK_RECIPROCAL_COMPLEMENT_SELECTED_V1
#error "typed selected reciprocal-complement runtime required"
#endif
#define TT_POLY_ATAN_BF16_AVAILABLE 1
