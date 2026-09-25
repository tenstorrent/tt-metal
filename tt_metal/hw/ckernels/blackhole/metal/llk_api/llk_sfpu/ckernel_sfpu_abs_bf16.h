// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Generated architecture configuration; shared primitives live in common/.
#pragma once
#include <cstdint>
namespace ttpoly_generated {
struct AbsBf16Config {
    static constexpr uint32_t kEqualMask = 0x80ffu;
    static constexpr uint32_t kEqualValue = 0x80ffu;
    static constexpr uint32_t kNonzeroMask = 0x7f00u;
    static constexpr uint32_t kOutputBf16 = 0xff80u;
    static constexpr uint32_t kTruePatterns = 127u;
    static constexpr uint32_t kSelectedBodySlots = 23u;
    static constexpr uint32_t kSelectedPeakLregs = 8u;
    static constexpr uint32_t kLoadFormat = 0u;
    static constexpr uint32_t kAbsMode = 1u;
    static constexpr uint32_t kStoreFormat = 0u;
    static constexpr uint32_t kBodySlots = 6u;
    static constexpr uint32_t kPeakLregs = 2u;
    static constexpr uint32_t kProvedPatterns = 65536u;
};
}  // namespace ttpoly_generated
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_abs_value.h"
#ifndef TT_POLY_LLK_ABS_VALUE_NATIVE_TERMINAL_V2
#error "typed absolute-value raw terminal runtime required"
#endif
