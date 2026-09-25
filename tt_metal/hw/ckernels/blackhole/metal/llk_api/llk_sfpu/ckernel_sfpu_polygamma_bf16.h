// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Generated architecture configuration; shared primitives live in common/.
#pragma once
#include <cstdint>
namespace ttpoly_generated {
struct PolygammaBf16Config {
    static constexpr float kP0 = __builtin_bit_cast(float, 0x3e210000u);
    static constexpr float kFiniteThreshold = __builtin_bit_cast(float, 0xc0d00000u);
    static constexpr float kZeroTransitionPreviousMagnitude = __builtin_bit_cast(float, 0x7e7f0000u);
    static constexpr unsigned kFiniteRepresentative = 48661u;
    static constexpr unsigned kZeroRepresentative = 0u;
    static constexpr unsigned kPositiveZeroFirstRaw = 32384u;
    static constexpr unsigned kNegativeInfinityWord = 0u;
    static constexpr float kP2[] = {
        __builtin_bit_cast(float, 0x40530000u),
        __builtin_bit_cast(float, 0x40b40000u),
        __builtin_bit_cast(float, 0x41930000u)};
    static constexpr bool kRepair = true;
    static constexpr bool kWhRepair = false;
    static constexpr bool kDirectTail = false;
    static constexpr bool kFiniteReferencePrecedence = true;
};
}  // namespace ttpoly_generated
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_inverse_square.h"
#ifndef TT_POLY_LLK_INVERSE_SQUARE_SELECTED_V1
#error "selected inverse square runtime required"
#endif
