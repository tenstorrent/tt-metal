// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Generated architecture configuration; shared primitives live in common/.
#pragma once
#include <cstdint>
namespace ttpoly_generated {
struct Exp2Bf16Config {
    static constexpr uint32_t kDegree = 2u;
    static constexpr uint32_t kScaledCoefficientBits[] = {0x3f803884u, 0x33a85adeu, 0x27aca410u};
    static constexpr bool kRawNegativeNanInfinity = true;
    static constexpr uint32_t kMultiplierBits = 0x3f800000u;
    static constexpr uint32_t kBodySlots = 28u;
};
}  // namespace ttpoly_generated
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_exp2.h"
#ifndef TT_POLY_LLK_EXP2_REPLAY_V1
#error "typed paired exponential replay runtime required"
#endif
