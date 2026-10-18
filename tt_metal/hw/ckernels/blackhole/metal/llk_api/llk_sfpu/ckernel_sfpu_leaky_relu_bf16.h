// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct LeakyReluBf16Config {
    static constexpr uint32_t kSlopeBits = 0x3c23d70au;
    static constexpr uint32_t kBodySlots = 12u;
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_leaky_relu_bf16_slope_max.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format.
inline bool bf16_dest_leaky_relu() {
#if defined(TRISC_MATH) || defined(LLK_TRISC_MATH)
    return ckernel::math::src_zero_flag_srcb_fmt == static_cast<std::uint32_t>(DataFormat::Float16_b);
#else
    return false;
#endif
}
template <int ITERATIONS = 32>
inline void calculate_leaky_relu_bf16() {
    ckernel::sfpu::bf16::leaky_relu::calculate_slope_max<ckernel::sfpu::LeakyReluBf16Config, ITERATIONS>();
}
inline void init_leaky_relu_bf16() {
    if (bf16_dest_leaky_relu()) {
        ckernel::sfpu::bf16::leaky_relu::init_slope_max<ckernel::sfpu::LeakyReluBf16Config>();
    }
}

}  // namespace ckernel::sfpu
