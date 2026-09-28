// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Typed Config only. The canonical shared helper owns selected arithmetic.
#pragma once
#include <cstdint>
namespace ttpoly_generated {
struct TanhshrinkBf16Config {
    static constexpr uint32_t kDegree = 8u;
    static constexpr uint32_t kCoefficientBits[] = {
        0x00000000u,
        0x00000000u,
        0x00000000u,
        0x3eaa738eu,
        0x3c52ec0cu,
        0xbe4a3e90u,
        0x3dec2feeu,
        0xbce1764au,
        0x3b20a66au};
    static constexpr uint32_t kBoundBits = 0x40400000u;
    static constexpr uint32_t kScaleBits = 0x3f800000u;
    static constexpr uint32_t kBiasBits = 0xbf800000u;
    static constexpr uint32_t kLeftBiasBits = 0x3f800000u;
    static constexpr uint32_t kBodySlots = 32u;
    static constexpr uint32_t kRawEqualValue = 0x80ffu;
    static constexpr uint32_t kRepairedPatterns = 127u;
    static constexpr uint32_t kMacroSequenceBits = 0x13850000u;
};
}  // namespace ttpoly_generated
#include "ckernel_sfpu_tt_poly_cascade_signed_abs_affine.h"
#if !defined(TT_POLY_LLK_SIGNED_ABS_AFFINE_SELECTED_V1)
#error "typed affine continuation requires the canonical selected backend"
#endif
