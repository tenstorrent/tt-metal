// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct CoshBf16Config {
    static constexpr uint32_t kDegree = 4u;
    static constexpr bool kOdd = false;
    static constexpr bool kBare = true;
    static constexpr uint32_t kScaledCoefficientBits[] = {
        0x3f800016u, 0x33b168acu, 0x27773d30u, 0x1ad514a6u, 0x0e5dc2acu};
    static constexpr uint32_t kMultiplierBits = 0x3fb8aa3bu;
    static constexpr uint32_t kComposeScaleBits = 0x3e800100u;
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_cosh_bf16_hyperbolic_exp.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format, as BF16 for
// Float16_b and the block formats sharing its exponent.
inline bool bf16_dest_cosh() {
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
inline void calculate_cosh_bf16_body() {
    ckernel::sfpu::bf16::cosh::calculate_hyperbolic_exp<ckernel::sfpu::CoshBf16Config, ITERATIONS>();
}
// The constants stock's init also sets hold the kernel's own values for its call only.
template <int ITERATIONS = 32>
inline void calculate_cosh_bf16() {
    sfpi::vConstFloatPrgm0 = __builtin_bit_cast(float, 0x40000000u);
    sfpi::vConstFloatPrgm1 = __builtin_bit_cast(float, 0x0e5dc2acu);
    sfpi::vConstFloatPrgm2 = __builtin_bit_cast(float, 0x1ad514a6u);
    calculate_cosh_bf16_body<ITERATIONS>();
    sfpi::vConstFloatPrgm0 = __builtin_bit_cast(float, 0x3fb8aa3bu);
    sfpi::vConstFloatPrgm1 = __builtin_bit_cast(float, 0xbf317218u);
    sfpi::vConstFloatPrgm2 = __builtin_bit_cast(float, 0x3f0007ffu);
}
inline void init_cosh_bf16() {
    ckernel::sfpu::bf16::cosh::init_hyperbolic_exp<ckernel::sfpu::CoshBf16Config>();
    sfpi::vConstFloatPrgm0 = __builtin_bit_cast(float, 0x3fb8aa3bu);
    sfpi::vConstFloatPrgm1 = __builtin_bit_cast(float, 0xbf317218u);
    sfpi::vConstFloatPrgm2 = __builtin_bit_cast(float, 0x3f0007ffu);
}

}  // namespace ckernel::sfpu
