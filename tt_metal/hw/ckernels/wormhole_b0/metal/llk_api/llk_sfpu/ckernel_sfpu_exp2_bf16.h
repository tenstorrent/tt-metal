// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct Exp2Bf16Config {
    static constexpr uint32_t kDegree = 2u;
    // Fit: minimax degree-2 polynomial on [0, 1], 1 segment, max pure (continuous) ULP 0.898.
    static constexpr uint32_t kScaledCoefficientBits[] = {0x3f803884u, 0x33a85adeu, 0x27aca410u};
    static constexpr uint32_t kMultiplierBits = 0x3f800000u;
    static constexpr uint32_t kBodySlots = 28u;
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_exp2_bf16_exp2_held.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format, as BF16 for
// Float16_b and the block formats sharing its exponent.
inline bool bf16_dest_exp2() {
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
// Each call records replay slots 0-15, below replay_buf_offset = 16, where matmul and
// reduce record.
template <int ITERATIONS = 32>
__attribute__((noinline)) inline void calculate_exp2_bf16() {
    ckernel::sfpu::bf16::exp2::calculate_exp2_held<ckernel::sfpu::Exp2Bf16Config, ITERATIONS>();
}
inline void init_exp2_bf16() { ckernel::sfpu::bf16::exp2::init_exp2_held<ckernel::sfpu::Exp2Bf16Config>(); }

}  // namespace ckernel::sfpu
