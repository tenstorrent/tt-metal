// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Generated architecture configuration; shared primitives live in common/.
#pragma once
#include <cstdint>
namespace ttpoly_generated {
struct CeluBf16Config {
    static constexpr uint32_t kSegments = 2u;
    static constexpr uint32_t kCoefficientOffset = kSegments + 1;
    static constexpr uint32_t kCoefficientsPerSegment = 15u;
    static constexpr uint32_t kDegree[] = {14, 1};
    static constexpr uint32_t kLutBits[] = {
        0xc1200000u, 0x00000000u, 0x41200000u, 0x00000000u, 0x3f800000u, 0x3effffeau, 0x3e2aa982u,
        0x3d2a9f4au, 0x3c084e54u, 0x3ab4a13au, 0x394a3902u, 0x37bf4b2eu, 0x36161ca6u, 0x343c9f4au,
        0x32342bb8u, 0x2ff2d2c2u, 0x2d4bc41cu, 0x2a1f125cu, 0x00000000u, 0x3f800000u, 0x00000000u,
        0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
        0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u};
    static constexpr int kParkRows[] = {-1, -1, -1, -1, -1, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                        12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    static constexpr uint32_t kRowBase = 32u;
    static constexpr uint32_t kRowLimit = 128u;
    static constexpr uint32_t kParkedCount = 13u;
    static constexpr uint32_t kLowerBits = 0xc0c80000u;
    static constexpr uint32_t kTerminalBits = 0xbf800000u;
    static constexpr bool kOrdered = false;
    static constexpr bool kRawIngress = true;
    static constexpr bool kRawTerminal = false;
    static constexpr uint32_t degree(uint32_t segment) { return kDegree[segment]; }
    static constexpr int park_row(uint32_t index) { return kParkRows[index]; }
    static constexpr float lut(uint32_t index) { return __builtin_bit_cast(float, kLutBits[index]); }
};
}  // namespace ttpoly_generated
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_cascade_dense.h"
#ifndef TT_POLY_LLK_DENSE_POLYNOMIAL_LOWER_CLAMP_V1
#error "typed whole-tile dense polynomial runtime required"
#endif
