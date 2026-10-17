// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct Expm1Bf16Config {
    static constexpr unsigned kLowerBf16 = 0x0000c0c8u;
    static constexpr unsigned kUpperBf16 = 0x000042b2u;
    static constexpr unsigned kRoundingBiasBits = 0x4b400000u;
    static constexpr unsigned kInverseScaleBits = 0x3fb8aa3bu;
    static constexpr unsigned kResidualScaleBits = 0xbf317218u;
    static constexpr unsigned kHalfBits = 0x3f000000u;
    static constexpr float kCoefficients[] = {
        __builtin_bit_cast(float, 0x3f800000u),
        __builtin_bit_cast(float, 0x3f000000u),
        __builtin_bit_cast(float, 0x3e2aaa56u),
        __builtin_bit_cast(float, 0x3d2b440cu),
        __builtin_bit_cast(float, 0x3c0914c2u)};
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_bf16_factored_cw_expm1.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format.
inline bool bf16_dest_expm1() {
#if defined(TRISC_MATH) || defined(LLK_TRISC_MATH)
    return ckernel::math::src_zero_flag_srcb_fmt == static_cast<std::uint32_t>(DataFormat::Float16_b);
#else
    return false;
#endif
}
template <int ITERATIONS = 8>
inline void calculate_expm1_bf16() {
    ckernel::sfpu::bf16::calculate_factored_cw_expm1<ckernel::sfpu::Expm1Bf16Config, ITERATIONS>();
}
inline void init_expm1_bf16() {
    if (bf16_dest_expm1()) {
        ckernel::sfpu::bf16::init_factored_cw_expm1<ckernel::sfpu::Expm1Bf16Config>();
    }
}

}  // namespace ckernel::sfpu
