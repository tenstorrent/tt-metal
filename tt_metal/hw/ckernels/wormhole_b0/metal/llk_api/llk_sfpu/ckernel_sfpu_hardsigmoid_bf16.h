// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct HardsigmoidBf16Config {
    static constexpr uint32_t kBodySlots = 0x0000000eu;
    static constexpr uint32_t kSlopeBits = 0x3e2aaaabu;
    static constexpr uint32_t kInterceptBits = 0x3f000000u;
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_hardsigmoid_bf16_clamped_affine.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format, as BF16 for
// Float16_b and the block formats sharing its exponent.
inline bool bf16_dest_hardsigmoid() {
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
// Each call records replay slots 0-13, below replay_buf_offset = 16, where matmul and
// reduce record.
template <int ITERATIONS = 32>
__attribute__((noinline)) inline void calculate_hardsigmoid_bf16() {
    ckernel::sfpu::bf16::hardsigmoid::calculate_clamped_affine<ckernel::sfpu::HardsigmoidBf16Config, ITERATIONS>();
}
inline void init_hardsigmoid_bf16() {
    ckernel::sfpu::bf16::hardsigmoid::init_clamped_affine<ckernel::sfpu::HardsigmoidBf16Config>();
}

}  // namespace ckernel::sfpu
