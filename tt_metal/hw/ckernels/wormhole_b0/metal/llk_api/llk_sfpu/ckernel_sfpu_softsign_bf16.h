// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct SoftsignBf16Config {
    static constexpr uint32_t kBodySlots = 14u;
    static constexpr uint32_t kBoundExponent = 136u;
    static constexpr uint32_t kLutSlopes = 0xb549b9a8u;
    static constexpr uint32_t kLutIntercepts = 0x3c9f3ed4u;
    static constexpr bool kLateRound = true;
    static constexpr bool kRawNegativeInfinity = false;
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_softsign_bf16_abs_denominator.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format, as BF16 for
// Float16_b and the block formats sharing its exponent.
inline bool bf16_dest_softsign() {
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
inline void calculate_softsign_bf16_body() {
    ckernel::sfpu::bf16::softsign::calculate_abs_denominator<ckernel::sfpu::SoftsignBf16Config, ITERATIONS>();
}
// The constants stock's init also sets hold the kernel's own values for its call only.
// Each call records replay slots 0-19, overwriting what an init recorded there (matmul and reduce
// record from replay_buf_offset = 16).
template <int ITERATIONS = 32>
inline void calculate_softsign_bf16() {
    sfpi::vConstFloatPrgm0 = __builtin_bit_cast(float, 0x000080ffu);
    calculate_softsign_bf16_body<ITERATIONS>();
    sfpi::vConstFloatPrgm0 = __builtin_bit_cast(float, 0x3ea57ebbu);
}
inline void init_softsign_bf16() {
    ckernel::sfpu::bf16::softsign::init_abs_denominator<ckernel::sfpu::SoftsignBf16Config>();
    sfpi::vConstFloatPrgm0 = __builtin_bit_cast(float, 0x3ea57ebbu);
}

}  // namespace ckernel::sfpu
