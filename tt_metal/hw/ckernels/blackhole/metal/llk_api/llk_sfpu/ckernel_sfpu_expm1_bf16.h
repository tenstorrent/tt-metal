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
#include "ckernel_sfpu_expm1_bf16_factored_cw_expm1.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format, as BF16 for
// Float16_b and the block formats sharing its exponent.
inline bool bf16_dest_expm1() {
#if defined(TRISC_MATH) || defined(LLK_TRISC_MATH)
    const std::uint32_t format = ckernel::math::src_zero_flag_srcb_fmt;
    return format == static_cast<std::uint32_t>(DataFormat::Float16_b) ||
           format == static_cast<std::uint32_t>(DataFormat::Bfp8_b) ||
           format == static_cast<std::uint32_t>(DataFormat::Bfp4_b) ||
           format == static_cast<std::uint32_t>(DataFormat::Bfp2_b);
#else
    return false;
#endif
}
template <int ITERATIONS = 32>
inline void calculate_expm1_bf16_body() {
    ckernel::sfpu::bf16::expm1::calculate_factored_cw_expm1<ckernel::sfpu::Expm1Bf16Config, ITERATIONS>();
}
// The constants stock's init also sets hold the kernel's own values for its call only.
// Each call records replay slots 0-15, below replay_buf_offset = 16, where matmul and
// reduce record.
template <int ITERATIONS = 32>
__attribute__((noinline)) inline void calculate_expm1_bf16() {
    sfpi::vConstFloatPrgm2 = __builtin_bit_cast(float, 0x3f000000u);
    calculate_expm1_bf16_body<ITERATIONS>();
    sfpi::vConstFloatPrgm2 = __builtin_bit_cast(float, 0x3e2aa000u);
}
inline void init_expm1_bf16() {
    ckernel::sfpu::bf16::expm1::init_factored_cw_expm1<ckernel::sfpu::Expm1Bf16Config>();
    sfpi::vConstFloatPrgm2 = __builtin_bit_cast(float, 0x3e2aa000u);
}

}  // namespace ckernel::sfpu
