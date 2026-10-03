// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct Log2Bf16Config {
    static constexpr uint32_t kDegree = 5u;
    // Fit: minimax degree-5 polynomial on [1, 2], 1 segment, max pure (continuous) ULP 0.536.
    static constexpr uint32_t kCoefficientBits[] = {
        0x00000000u, 0x3fb8aa3bu, 0xbf36c4adu, 0x3ed98b83u, 0xbe4ce63fu, 0x3d3e4052u};
    static constexpr uint32_t kScaleBits = 0x3f800000u;
    static constexpr uint32_t kBodySlots = 29u;
    static constexpr uint32_t kTailSlots = 0u;
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_bf16_log2.h"

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
    ckernel::sfpu::bf16::calculate_log2<ckernel::sfpu::Log2Bf16Config, ITERATIONS>();
}
inline void init_log2_bf16() {
    if (bf16_dest_log2()) {
        ckernel::sfpu::bf16::init_log2<ckernel::sfpu::Log2Bf16Config>();
    }
}

}  // namespace ckernel::sfpu
