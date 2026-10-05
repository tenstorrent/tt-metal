// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct Log2Bf16Config {
    static constexpr uint32_t kDegree = 4u;
    // Fit: degree-4 polynomial on [1, 2], 1 segment, max pure (continuous) ULP 0.621.
    static constexpr uint32_t kCoefficientBits[] = {0x00000000u, 0x3fb86abcu, 0xbf2f3310u, 0x3ea7754au, 0xbdaaf354u};
    static constexpr uint32_t kScaleBits = 0x3f800000u;
    static constexpr uint32_t kBodySlots = 22u;
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_bf16_log_fused.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format.
inline bool bf16_dest_log2() {
#if defined(TRISC_MATH) || defined(LLK_TRISC_MATH)
    return ckernel::math::src_zero_flag_srcb_fmt == static_cast<std::uint32_t>(DataFormat::Float16_b);
#else
    return false;
#endif
}
template <int ITERATIONS = 8>
inline void calculate_log2_bf16() {
    ckernel::sfpu::bf16::calculate_log_fused<ckernel::sfpu::Log2Bf16Config, ITERATIONS>();
}
inline void init_log2_bf16() {
    if (bf16_dest_log2()) {
        ckernel::sfpu::bf16::init_log_fused<ckernel::sfpu::Log2Bf16Config>();
    }
}

}  // namespace ckernel::sfpu
