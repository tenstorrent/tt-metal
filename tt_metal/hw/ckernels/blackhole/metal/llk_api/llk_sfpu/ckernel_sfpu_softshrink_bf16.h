// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct SoftshrinkBf16Config {
    static constexpr uint32_t kKind = 0x00000002u;
    static constexpr uint32_t kThresholdBits = 0x3f000000u;
    static constexpr uint32_t kBodySlots = 0x00000000u;
    static constexpr uint32_t kRowsPerReplay = 0x00000000u;
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_bf16_simple_forward.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format.
inline bool bf16_dest_softshrink() {
#if defined(TRISC_MATH) || defined(LLK_TRISC_MATH)
    return ckernel::math::src_zero_flag_srcb_fmt == static_cast<std::uint32_t>(DataFormat::Float16_b);
#else
    return false;
#endif
}
template <int ITERATIONS = 8>
inline void calculate_softshrink_bf16() {
    ckernel::sfpu::bf16::calculate_simple_forward<ckernel::sfpu::SoftshrinkBf16Config, ITERATIONS>();
}
inline void init_softshrink_bf16() {
    if (bf16_dest_softshrink()) {
        ckernel::sfpu::bf16::init_simple_forward<ckernel::sfpu::SoftshrinkBf16Config>();
    }
}

}  // namespace ckernel::sfpu
