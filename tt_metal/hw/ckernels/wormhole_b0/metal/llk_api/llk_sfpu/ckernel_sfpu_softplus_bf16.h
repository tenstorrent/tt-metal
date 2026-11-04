// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <array>
namespace ckernel::sfpu {
struct SoftplusBf16Config {
    static constexpr float kNumerator[] = {
        __builtin_bit_cast(float, 0x3f800000u),
        __builtin_bit_cast(float, 0xbefab96eu),
        __builtin_bit_cast(float, 0x3e85799eu),
        __builtin_bit_cast(float, 0xbda01eecu)};
    static constexpr float kExpCoefficients[] = {
        __builtin_bit_cast(float, 0x3f803884u),
        __builtin_bit_cast(float, 0x3f285adeu),
        __builtin_bit_cast(float, 0x3eaca410u)};
    static constexpr uint32_t kExpDegree = 2;
    static constexpr float kMultiplier = __builtin_bit_cast(float, 0xbfb8aa3bu);
    static constexpr uint32_t kBoundBits = 0x42af0000u;
    static constexpr bool kResidualFold = true;
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_softplus_bf16_abs_exp_correction.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format, as BF16 for
// Float16_b and the block formats sharing its exponent.
inline bool bf16_dest_softplus() {
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
// Each call records replay slots 0-31, overwriting what an init recorded there (matmul and reduce
// record from replay_buf_offset = 16).
template <int ITERATIONS = 32>
__attribute__((noinline)) inline void calculate_softplus_bf16() {
    ckernel::sfpu::bf16::softplus::calculate_abs_exp_correction<ckernel::sfpu::SoftplusBf16Config, ITERATIONS>();
}
inline void init_softplus_bf16() {
    ckernel::sfpu::bf16::softplus::init_abs_exp_correction<ckernel::sfpu::SoftplusBf16Config>();
}

}  // namespace ckernel::sfpu
