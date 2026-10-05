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
    static constexpr uint32_t kBodySlots = 22u;
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_bf16_log_fused.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format.
inline bool bf16_dest_log() {
#if defined(TRISC_MATH) || defined(LLK_TRISC_MATH)
    return ckernel::math::src_zero_flag_srcb_fmt == static_cast<std::uint32_t>(DataFormat::Float16_b);
#else
    return false;
#endif
}
template <int ITERATIONS = 8>
inline void calculate_log_bf16() {
    ckernel::sfpu::bf16::calculate_log_fused<ckernel::sfpu::LogBf16Config, ITERATIONS>();
}
inline void init_log_bf16() {
    if (bf16_dest_log()) {
        ckernel::sfpu::bf16::init_log_fused<ckernel::sfpu::LogBf16Config>();
    }
}

}  // namespace ckernel::sfpu
