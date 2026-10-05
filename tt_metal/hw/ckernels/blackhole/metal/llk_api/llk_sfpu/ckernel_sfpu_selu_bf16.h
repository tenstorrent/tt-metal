// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct SeluBf16Config {
    static constexpr uint32_t kSegments = 2u;
    static constexpr uint32_t kCoefficientOffset = kSegments + 1;
    static constexpr uint32_t kCoefficientsPerSegment = 9u;
    // Fit: degree-8 polynomial on [-6.0625, 10], 2 segments, max pure (continuous) ULP 0.5.
    static constexpr uint32_t kDegree[] = {8, 1};
    static constexpr uint32_t kLutBits[] = {
        0xc0c20000u, 0x00000000u, 0x41200000u, 0x00000000u, 0x3fe10966u, 0x3f6118f6u, 0x3e95c291u,
        0x3d927a57u, 0x3c54e4eau, 0x3ad80000u, 0x3905727fu, 0x36940000u, 0x00000000u, 0x3f867d5eu,
        0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u};
    static constexpr int kParkRows[] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    static constexpr uint32_t kRowBase = 32u;
    static constexpr uint32_t kRowLimit = 128u;
    static constexpr uint32_t kParkedCount = 0u;
    static constexpr uint32_t kLowerBits = 0xc0c20000u;
    static constexpr uint32_t kTerminalBits = 0xbfe10000u;
    static constexpr bool kOrdered = false;
    static constexpr bool kRawIngress = false;
    static constexpr bool kRawTerminal = false;
    static constexpr uint32_t degree(uint32_t segment) { return kDegree[segment]; }
    static constexpr int park_row(uint32_t index) { return kParkRows[index]; }
    static constexpr float lut(uint32_t index) { return __builtin_bit_cast(float, kLutBits[index]); }
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_bf16_cascade_dense.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format.
inline bool bf16_dest_selu() {
#if defined(TRISC_MATH) || defined(LLK_TRISC_MATH)
    return ckernel::math::src_zero_flag_srcb_fmt == static_cast<std::uint32_t>(DataFormat::Float16_b);
#else
    return false;
#endif
}
template <int ITERATIONS = 8>
inline void calculate_selu_bf16() {
    ckernel::sfpu::bf16::calculate_dense_polynomial<ckernel::sfpu::SeluBf16Config, ITERATIONS>();
}

}  // namespace ckernel::sfpu
