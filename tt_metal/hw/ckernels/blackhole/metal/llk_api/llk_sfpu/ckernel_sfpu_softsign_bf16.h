// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct SoftsignBf16Config {
    static constexpr uint32_t kBodySlots = 19u;
    static constexpr uint32_t kBoundBits = 0x44000000u;
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
// Each call records replay slots 0-14, below replay_buf_offset = 16, where matmul and
// reduce record.
template <int ITERATIONS = 32>
__attribute__((noinline)) inline void calculate_softsign_bf16() {
    ckernel::sfpu::bf16::softsign::calculate_abs_denominator<ckernel::sfpu::SoftsignBf16Config, ITERATIONS>();
}
inline void init_softsign_bf16() {
    ckernel::sfpu::bf16::softsign::init_abs_denominator<ckernel::sfpu::SoftsignBf16Config>();
}

}  // namespace ckernel::sfpu
