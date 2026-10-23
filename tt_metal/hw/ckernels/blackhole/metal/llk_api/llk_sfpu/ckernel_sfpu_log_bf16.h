// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct LogBf16Config {
    static constexpr uint32_t kDegree = 4u;
    // Fit: degree-4 polynomial on [1, 2], 1 segment, max pure (continuous) ULP 0.64.
    static constexpr uint32_t kCoefficientBits[] = {0x00000000u, 0x3f7fbd33u, 0xbef3a895u, 0x3e6a802fu, 0xbd7180f6u};
    static constexpr uint32_t kScaleBits = 0x3f317218u;
    static constexpr uint32_t kBodySlots = 23u;
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_log_bf16_log_fused.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format, as BF16 for
// Float16_b and the block formats sharing its exponent.
inline bool bf16_dest_log() {
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
__attribute__((noinline)) inline void calculate_log_bf16() {
    ckernel::sfpu::bf16::log::calculate_log_fused<ckernel::sfpu::LogBf16Config, ITERATIONS>();
}
inline void init_log_bf16() { ckernel::sfpu::bf16::log::init_log_fused<ckernel::sfpu::LogBf16Config>(); }

}  // namespace ckernel::sfpu
