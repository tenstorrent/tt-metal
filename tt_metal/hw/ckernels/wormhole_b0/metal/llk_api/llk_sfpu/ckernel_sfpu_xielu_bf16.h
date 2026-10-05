// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct XieluBf16Config {
    static constexpr uint32_t kSegments = 4u;
    static constexpr uint32_t kCoefficientOffset = kSegments + 1;
    static constexpr uint32_t kCoefficientsPerSegment = 10u;
    // Fit: degree-9 polynomial on [-7.0001, 2.06158e+19], 3 segments, max pure (continuous) ULP 0.5.
    static constexpr uint32_t kDegree[] = {1, 9, 2, 2};
    static constexpr uint32_t kLutBits[] = {
        0xff7fffffu, 0xc0e00000u, 0xb58637bdu, 0x00800000u, 0x5f8f0d18u, 0xbf4ccccdu, 0xbe99999au, 0x00000000u,
        0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0xb0ffd6d4u,
        0x3f000000u, 0x3eccd1ccu, 0x3e089a94u, 0x3d086efdu, 0x3bd662c9u, 0x3a83559fu, 0x38e90000u, 0x37010000u,
        0x34820000u, 0xb556bf8eu, 0xbe99999au, 0x3b0fa562u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
        0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x3f000000u, 0x3f4ccccdu, 0x00000000u, 0x00000000u,
        0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u};
    static constexpr int kParkRows[] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    static constexpr uint32_t kRowBase = 64u;
    static constexpr uint32_t kRowLimit = 108u;
    static constexpr uint32_t kParkedCount = 0u;
    static constexpr uint32_t kLowerBits = 0xff800000u;
    static constexpr uint32_t kTerminalBits = 0xff800000u;
    static constexpr bool kOrdered = true;
    static constexpr uint32_t degree(uint32_t segment) { return kDegree[segment]; }
    static constexpr int park_row(uint32_t index) { return kParkRows[index]; }
    static constexpr float lut(uint32_t index) { return __builtin_bit_cast(float, kLutBits[index]); }
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_bf16_cascade_dense.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format.
inline bool bf16_dest_xielu() {
#if defined(TRISC_MATH) || defined(LLK_TRISC_MATH)
    return ckernel::math::src_zero_flag_srcb_fmt == static_cast<std::uint32_t>(DataFormat::Float16_b);
#else
    return false;
#endif
}
template <int ITERATIONS = 8>
inline void calculate_xielu_bf16() {
    ckernel::sfpu::bf16::calculate_dense_polynomial<ckernel::sfpu::XieluBf16Config, ITERATIONS>();
}

}  // namespace ckernel::sfpu
