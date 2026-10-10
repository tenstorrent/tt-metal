// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct Log10Bf16Config {
    static constexpr uint32_t kDegree = 4u;
    // Fit: degree-4 polynomial on [1, 2], 1 segment, max pure (continuous) ULP 0.639.
    static constexpr uint32_t kCoefficientBits[] = {0x00000000u, 0x3ede1a72u, 0xbe531575u, 0x3dc9950au, 0xbccd5243u};
    static constexpr uint32_t kScaleBits = 0x3e9a209bu;
    static constexpr uint32_t kBodySlots = 22u;
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_bf16_log_fused.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format.
inline bool bf16_dest_log10() {
#if defined(TRISC_MATH) || defined(LLK_TRISC_MATH)
    return ckernel::math::src_zero_flag_srcb_fmt == static_cast<std::uint32_t>(DataFormat::Float16_b);
#else
    return false;
#endif
}
template <int ITERATIONS = 8>
inline void calculate_log10_bf16() {
    ckernel::sfpu::bf16::calculate_log_fused<ckernel::sfpu::Log10Bf16Config, ITERATIONS>();
}
inline void init_log10_bf16() {
    if (bf16_dest_log10()) {
        ckernel::sfpu::bf16::init_log_fused<ckernel::sfpu::Log10Bf16Config>();
    }
}

}  // namespace ckernel::sfpu
