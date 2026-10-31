// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct PolygammaBf16Config {
    static constexpr float kP0 = __builtin_bit_cast(float, 0x3e210000u);
    static constexpr unsigned kNanResultWord = 32704u;
    static constexpr unsigned kPositiveInfinityWord = 0u;
    static constexpr unsigned kNegativeInfinityWord = 32704u;
    static constexpr float kP2[] = {
        __builtin_bit_cast(float, 0x40530000u),
        __builtin_bit_cast(float, 0x40b40000u),
        __builtin_bit_cast(float, 0x41930000u)};
    static constexpr bool kWhRepair = true;
    static constexpr bool kDirectTail = false;
    static constexpr bool kFiniteReferencePrecedence = true;
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_polygamma_bf16_inverse_square.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format, as BF16 for
// Float16_b and the block formats sharing its exponent.
inline bool bf16_dest_polygamma() {
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
// Each call records replay slots 0-3, below replay_buf_offset = 16, where matmul and
// reduce record.
template <int ITERATIONS = 32>
inline void calculate_polygamma_bf16() {
    ckernel::sfpu::bf16::polygamma::calculate_inverse_square<ckernel::sfpu::PolygammaBf16Config, ITERATIONS>();
}
inline void init_polygamma_bf16() {
    ckernel::sfpu::bf16::polygamma::init_inverse_square<ckernel::sfpu::PolygammaBf16Config>();
}

}  // namespace ckernel::sfpu
