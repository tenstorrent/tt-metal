// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct SigmoidBf16Config {
    // Fit: minimax degree-2 polynomial on [0, 1], 1 segment, max pure (continuous) ULP 0.788.
    static constexpr uint32_t kCoefficientBits[] = {0x3f803884u, 0x3f285adeu, 0x3eaca410u};
    static constexpr uint32_t kMultiplierBits = 0xbfb8aa3bu;
    static constexpr uint32_t kBias = 127u;
    static constexpr uint32_t kLowerClampBits = 0x00000000u;
    static constexpr uint32_t kUpperClampBits = 0x437f0000u;
    static constexpr bool kNanTerminal = true;
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_bf16_exp2_reciprocal.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format.
inline bool bf16_dest_sigmoid() {
#if defined(TRISC_MATH) || defined(LLK_TRISC_MATH)
    return ckernel::math::src_zero_flag_srcb_fmt == static_cast<std::uint32_t>(DataFormat::Float16_b);
#else
    return false;
#endif
}
template <int ITERATIONS = 8>
inline void calculate_sigmoid_bf16() {
    ckernel::sfpu::bf16::calculate_exp2_reciprocal<ckernel::sfpu::SigmoidBf16Config, ITERATIONS>();
}
inline void init_sigmoid_bf16() {
    if (bf16_dest_sigmoid()) {
        ckernel::sfpu::bf16::init_exp2_reciprocal<ckernel::sfpu::SigmoidBf16Config>();
    }
}

}  // namespace ckernel::sfpu
