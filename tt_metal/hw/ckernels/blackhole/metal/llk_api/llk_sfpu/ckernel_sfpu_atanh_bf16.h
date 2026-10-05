// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct AtanhBf16Config {
    static constexpr uint32_t kNumDegree = 11u;
    static constexpr uint32_t kDenDegree = 4u;
    // Fit: degree-11/4 rational on [-0.999, 0.999], 1 segment, max pure (continuous) ULP 0.723.
    static constexpr float kNumerator[] = {
        __builtin_bit_cast(float, 0x00000000u),
        __builtin_bit_cast(float, 0x3f800000u),
        __builtin_bit_cast(float, 0x00000000u),
        __builtin_bit_cast(float, 0xbfcc2d0bu),
        __builtin_bit_cast(float, 0x00000000u),
        __builtin_bit_cast(float, 0x3eebcd4fu),
        __builtin_bit_cast(float, 0x00000000u),
        __builtin_bit_cast(float, 0x3e42f154u),
        __builtin_bit_cast(float, 0x00000000u),
        __builtin_bit_cast(float, 0xbe3e8d7cu),
        __builtin_bit_cast(float, 0x00000000u),
        __builtin_bit_cast(float, 0x3e068b65u)};
    static constexpr float kDenominator[] = {
        __builtin_bit_cast(float, 0x3f800000u),
        __builtin_bit_cast(float, 0x00000000u),
        __builtin_bit_cast(float, 0xbff6f85fu),
        __builtin_bit_cast(float, 0x00000000u),
        __builtin_bit_cast(float, 0x3f6e010bu)};
    static constexpr uint32_t kBoundBits = 0x3f800000u;
    static constexpr bool kPinTop = false;
    static constexpr int kRawClass = -1;
    static constexpr bool kRawEarly = true;
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_bf16_rational_parity.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format.
inline bool bf16_dest_atanh() {
#if defined(TRISC_MATH) || defined(LLK_TRISC_MATH)
    return ckernel::math::src_zero_flag_srcb_fmt == static_cast<std::uint32_t>(DataFormat::Float16_b);
#else
    return false;
#endif
}
template <int ITERATIONS = 8>
inline void calculate_atanh_bf16() {
    ckernel::sfpu::bf16::calculate_rational_parity<ckernel::sfpu::AtanhBf16Config, ITERATIONS>();
}

}  // namespace ckernel::sfpu
