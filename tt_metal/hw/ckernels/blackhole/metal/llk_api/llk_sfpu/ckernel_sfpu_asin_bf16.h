// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct AsinBf16Config {
    static constexpr float kCoordinateScale = __builtin_bit_cast(float, 0x3f000000u);
    static constexpr float kSplit = __builtin_bit_cast(float, 0x3f100000u);
    static constexpr float kResultScale = __builtin_bit_cast(float, 0xc0000000u);
    static constexpr float kResultBias = __builtin_bit_cast(float, 0x3fc90fdbu);
    static constexpr float kLimit = __builtin_bit_cast(float, 0x3f800000u);
    static constexpr float kOutputScale = __builtin_bit_cast(float, 0x3f800000u);
    static constexpr float kOutputBias = __builtin_bit_cast(float, 0x00000000u);
    static constexpr float kCoefficients[] = {
        __builtin_bit_cast(float, 0x3f800000u),
        __builtin_bit_cast(float, 0x3e2afc6eu),
        __builtin_bit_cast(float, 0x3d8e2c75u),
        __builtin_bit_cast(float, 0x3d9296fdu)};
    static constexpr bool kSharedPool = false;
    static constexpr bool kPostcompose = false, kReconstructFromValue = true;
    static constexpr unsigned kMagic = 0x5f370000u;
    static constexpr unsigned kRootIterations = 1u;
    static constexpr unsigned kInvalidClass = 1u, kRawClass = 1u;
    static constexpr unsigned kRawMask = 0x80ffu;
    static constexpr unsigned kRawValue = 0x80ffu;
    static constexpr unsigned kRawExcluded = 0x80ffu;
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_asin_bf16_half_angle_ratio.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format, as BF16 for
// Float16_b and the block formats sharing its exponent.
inline bool bf16_dest_asin() {
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
__attribute__((noinline)) inline void calculate_asin_bf16() {
    ckernel::sfpu::bf16::asin::calculate_half_angle_ratio<ckernel::sfpu::AsinBf16Config, ITERATIONS>();
}
inline void init_asin_bf16() { ckernel::sfpu::bf16::asin::init_half_angle_ratio<ckernel::sfpu::AsinBf16Config>(); }

}  // namespace ckernel::sfpu
