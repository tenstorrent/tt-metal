// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct AsinhBf16Config {
    // Fit: degree-8 polynomial on [-10, 10], 2 segments, max pure (continuous) ULP 0.886.
    static constexpr float kNegative[] = {
        __builtin_bit_cast(float, 0x3f80336au),
        __builtin_bit_cast(float, 0x3b8c467cu),
        __builtin_bit_cast(float, 0xbe4f8e3bu),
        __builtin_bit_cast(float, 0xbdec9997u),
        __builtin_bit_cast(float, 0xbd08126cu),
        __builtin_bit_cast(float, 0xbbb63430u),
        __builtin_bit_cast(float, 0xba0eeaafu),
        __builtin_bit_cast(float, 0xb7f3512du),
        __builtin_bit_cast(float, 0xb52d7ae6u)};
    static constexpr float kPositive[] = {
        __builtin_bit_cast(float, 0x3f80336au),
        __builtin_bit_cast(float, 0xbb8c467cu),
        __builtin_bit_cast(float, 0xbe4f8e3bu),
        __builtin_bit_cast(float, 0x3dec9997u),
        __builtin_bit_cast(float, 0xbd08126cu),
        __builtin_bit_cast(float, 0x3bb63430u),
        __builtin_bit_cast(float, 0xba0eeaafu),
        __builtin_bit_cast(float, 0x37f3512du),
        __builtin_bit_cast(float, 0xb52d7ae6u)};
    static constexpr float kLog[] = {
        __builtin_bit_cast(float, 0xbefff282u),
        __builtin_bit_cast(float, 0x3eac415eu),
        __builtin_bit_cast(float, 0xbe84fd79u),
        __builtin_bit_cast(float, 0x3e1960f0u)};
    static constexpr uint32_t kLowerBits = 0xc1200000u;
    static constexpr uint32_t kUpperBits = 0x41200000u;
    static constexpr uint32_t kAddendBits = 0x3f317218u;
    static constexpr bool kMirrorFold = true;
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_asinh_bf16_symmetric_factored_log.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format, as BF16 for
// Float16_b and the block formats sharing its exponent.
inline bool bf16_dest_asinh() {
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
// Each call records replay slots 0-8, below replay_buf_offset = 16, where matmul and
// reduce record.
template <int ITERATIONS = 32>
__attribute__((noinline)) inline void calculate_asinh_bf16() {
    ckernel::sfpu::bf16::asinh::calculate_symmetric_factored_log<ckernel::sfpu::AsinhBf16Config, ITERATIONS>();
}

}  // namespace ckernel::sfpu
