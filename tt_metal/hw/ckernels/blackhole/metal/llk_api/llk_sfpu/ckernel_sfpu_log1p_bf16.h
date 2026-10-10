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
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_bf16_log1p.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format.
inline bool bf16_dest_log1p() {
#if defined(TRISC_MATH) || defined(LLK_TRISC_MATH)
    return ckernel::math::src_zero_flag_srcb_fmt == static_cast<std::uint32_t>(DataFormat::Float16_b);
#else
    return false;
#endif
}
template <int ITERATIONS = 8>
inline void calculate_log1p_bf16() {
    ckernel::sfpu::bf16::calculate_log1p<ckernel::sfpu::Log1PBf16Config, ITERATIONS>();
}
inline void init_log1p_bf16() {
    if (bf16_dest_log1p()) {
        ckernel::sfpu::bf16::init_log1p<ckernel::sfpu::Log1PBf16Config>();
    }
}

}  // namespace ckernel::sfpu
