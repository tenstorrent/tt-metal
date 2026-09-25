// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Generated architecture configuration; shared primitives live in common/.
#pragma once
#include <cstdint>
namespace ttpoly_generated {
struct SigmoidBf16Config {
    static constexpr uint32_t kDegree = 2u;
    static constexpr uint32_t kCoefficientBits[] = {0x3f803884u, 0x3f285adeu, 0x3eaca410u};
    static constexpr uint32_t kMultiplierBits = 0xbfb8aa3bu;
    static constexpr uint32_t kBias = 127u;
    static constexpr uint32_t kLowerClampBits = 0x00000000u;
    static constexpr uint32_t kUpperClampBits = 0x437f0000u;
    static constexpr uint32_t kFinitePopulation = 65280u;
    static constexpr uint32_t kRawClassOutput[] = {5u, 5u, 5u, 5u, 5u, 5u, 3u, 5u, 3u};
    static constexpr bool kPositiveNanZero = false;
    static constexpr bool kInputDaz = true;
    static constexpr bool kOutputFtz = true;
};
}  // namespace ttpoly_generated
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_exp2_reciprocal.h"
#ifndef TT_POLY_LLK_EXP2_RECIPROCAL_BF16_V2
#error "typed exponent-ALU reciprocal runtime required"
#endif
