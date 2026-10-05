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
#include "ckernel_sfpu_bf16_clamped_affine.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format.
inline bool bf16_dest_hardsigmoid() {
#if defined(TRISC_MATH) || defined(LLK_TRISC_MATH)
    return ckernel::math::src_zero_flag_srcb_fmt == static_cast<std::uint32_t>(DataFormat::Float16_b);
#else
    return false;
#endif
}
template <int ITERATIONS = 8>
inline void calculate_hardsigmoid_bf16() {
    ckernel::sfpu::bf16::calculate_clamped_affine<ckernel::sfpu::HardsigmoidBf16Config, ITERATIONS>();
}
inline void init_hardsigmoid_bf16() {
    if (bf16_dest_hardsigmoid()) {
        ckernel::sfpu::bf16::init_clamped_affine<ckernel::sfpu::HardsigmoidBf16Config>();
    }
}

}  // namespace ckernel::sfpu
