// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct Log1PBf16Config {
    // Fit: minimax degree-2 polynomial on [-0.25, 0.5], 1 segment, max pure (continuous) ULP 0.594.
    static constexpr uint32_t kCoefficientBits[] = {0xbf006fd0u, 0x3eb06794u, 0xbe4debdau};
    static constexpr uint32_t kBodySlots = 31u;
    static constexpr uint32_t kLowerBits = 0x3f400000u;
    static constexpr uint32_t kQuotientBits = 0x7f800000u;
    static constexpr uint32_t kRawEqual = 0x00ffu;
    static constexpr uint32_t kSharedPrgm1Bits = 0x3eba2000u;
    static constexpr uint32_t kSharedPrgm2Bits = 0xbf004000u;
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_log1p_bf16_log1p.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format, as BF16 for
// Float16_b and the block formats sharing its exponent.
inline bool bf16_dest_log1p() {
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
__attribute__((noinline)) inline void calculate_log1p_bf16() {
    ckernel::sfpu::bf16::log1p::calculate_log1p<ckernel::sfpu::Log1PBf16Config, ITERATIONS>();
}
inline void init_log1p_bf16() { ckernel::sfpu::bf16::log1p::init_log1p<ckernel::sfpu::Log1PBf16Config>(); }

}  // namespace ckernel::sfpu
