// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct EluBf16Config {
    static constexpr uint32_t kSegments = 2u;
    static constexpr uint32_t kCoefficientOffset = kSegments + 1;
    static constexpr uint32_t kCoefficientsPerSegment = 9u;
    // Fit: degree-8 polynomial on [-6.25, 10], 2 segments, max pure (continuous) ULP 0.5.
    static constexpr uint32_t kDegree[] = {8, 1};
    static constexpr uint32_t kLutBits[] = {
        0xc0c80000u, 0x00000000u, 0x41200000u, 0x00000000u, 0x3f800000u, 0x3effdb9eu, 0x3e295070u,
        0x3d22c6dbu, 0x3be47be7u, 0x3a5c41d7u, 0x38800000u, 0x36050000u, 0x00000000u, 0x3f800000u,
        0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u};
    static constexpr int kParkRows[] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    static constexpr uint32_t kRowBase = 32u;
    static constexpr uint32_t kRowLimit = 128u;
    static constexpr uint32_t kParkedCount = 0u;
    static constexpr uint32_t kLowerBits = 0xc0c80000u;
    static constexpr uint32_t kTerminalBits = 0xbf800000u;
    static constexpr bool kOrdered = false;
    static constexpr bool kRawIngress = true;
    static constexpr bool kRawTerminal = false;
    static constexpr uint32_t degree(uint32_t segment) { return kDegree[segment]; }
    static constexpr int park_row(uint32_t index) { return kParkRows[index]; }
    static constexpr float lut(uint32_t index) { return __builtin_bit_cast(float, kLutBits[index]); }
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_bf16_cascade_dense.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format.
inline bool bf16_dest_elu() {
#if defined(TRISC_MATH) || defined(LLK_TRISC_MATH)
    return ckernel::math::src_zero_flag_srcb_fmt == static_cast<std::uint32_t>(DataFormat::Float16_b);
#else
    return false;
#endif
}
template <int ITERATIONS = 8>
inline void calculate_elu_bf16() {
    ckernel::sfpu::bf16::calculate_dense_polynomial<ckernel::sfpu::EluBf16Config, ITERATIONS>();
}

}  // namespace ckernel::sfpu
