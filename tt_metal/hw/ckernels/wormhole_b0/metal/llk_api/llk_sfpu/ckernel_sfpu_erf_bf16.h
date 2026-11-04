// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

namespace ckernel::sfpu {
struct ErfBf16Config {
    static constexpr uint32_t kDegree = 7u;
    // Fit: minimax degree-7 polynomial on [0, 10], 1 segment, max pure (continuous) ULP 0.73.
    static constexpr uint32_t kCoefficientBits[] = {
        0x00000000u, 0x3f904bb0u, 0x3ceceac4u, 0xbf007bbau, 0x3e40e7f0u, 0x3c90cff8u, 0xbca6e156u, 0x3b39f45cu};
    static constexpr uint32_t kClampMaxBits = 0x3f800000u;
    static constexpr bool kIntrinsicExceptional = false;
};
}  // namespace ckernel::sfpu

#include "ckernel_sfpu_erf_bf16_cascade_signed_abs.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format, as BF16 for
// Float16_b and the block formats sharing its exponent.
inline bool bf16_dest_erf() {
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
inline void calculate_erf_bf16_body() {
    ckernel::sfpu::bf16::erf::calculate_signed_abs<ckernel::sfpu::ErfBf16Config, ITERATIONS>();
}
// The constants stock's init also sets hold the kernel's own values for its call only.
// Each call records replay slots 0-31, overwriting what an init recorded there (matmul and reduce
// record from replay_buf_offset = 16).
template <int ITERATIONS = 32>
__attribute__((noinline)) inline void calculate_erf_bf16() {
    sfpi::vConstFloatPrgm0 = __builtin_bit_cast(float, 0x3b39f45cu);
    sfpi::vConstFloatPrgm1 = __builtin_bit_cast(float, 0xbca6e156u);
    sfpi::vConstFloatPrgm2 = __builtin_bit_cast(float, 0x3c90cff8u);
    calculate_erf_bf16_body<ITERATIONS>();
    sfpi::vConstFloatPrgm0 = __builtin_bit_cast(float, 0x3ea57ebbu);
    sfpi::vConstFloatPrgm1 = __builtin_bit_cast(float, 0x3fba2e90u);
    sfpi::vConstFloatPrgm2 = __builtin_bit_cast(float, 0x4007c1f2u);
}
// The call sets the constants it holds and puts stock's back, so the init leaves them to stock's.
inline void init_erf_bf16() {}

}  // namespace ckernel::sfpu
