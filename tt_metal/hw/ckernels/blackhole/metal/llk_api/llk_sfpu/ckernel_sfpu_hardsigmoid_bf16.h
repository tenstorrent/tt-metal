// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Generated architecture configuration; shared primitives live in common/.
#pragma once
#include <cstdint>
namespace ttpoly_generated {
struct HardsigmoidBf16Config {
    static constexpr uint32_t kRoute = 0x00000001u;
    static constexpr uint32_t kBodySlots = 0x0000001bu;
    static constexpr uint32_t kSlopeBits = 0x3e2aaaabu;
    static constexpr uint32_t kInterceptBits = 0x3f000000u;
    static constexpr uint32_t kHasLower = 0x00000001u;
    static constexpr uint32_t kHasUpper = 0x00000001u;
    static constexpr uint32_t kLowerBits = 0x00000000u;
    static constexpr uint32_t kUpperBits = 0x3f800000u;
    static constexpr uint32_t kHasTerminal = 0x00000001u;
    static constexpr uint32_t kTerminalEqual = 0x000080ffu;
    static constexpr uint32_t kTerminalMask = 0x000080ffu;
    static constexpr uint32_t kTerminalNonzero = 0x00007f00u;
    static constexpr uint32_t kTerminalOutput = 0x00003f80u;
    static constexpr uint32_t kPosNanConstant = 0x00000001u;
    static constexpr uint32_t kNegNanConstant = 0x00000001u;
    static constexpr uint32_t kPosNanBits = 0x3f800000u;
    static constexpr uint32_t kNegNanBits = 0x3f800000u;
    static constexpr uint32_t kPackReluThresholdBits = 0x00000000u;
};
}  // namespace ttpoly_generated
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_clamped_affine.h"
#ifndef TT_POLY_LLK_CLAMPED_AFFINE_V2
#error "typed affine clamp runtime required"
#endif
