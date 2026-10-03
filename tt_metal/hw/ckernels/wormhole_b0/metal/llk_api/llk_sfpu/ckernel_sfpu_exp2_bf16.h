// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct Exp2Bf16Config {
    static constexpr uint32_t kDegree = 2u;
    // Fit: minimax degree-2 polynomial on [0, 1], 1 segment, max pure (continuous) ULP 0.898.
    static constexpr uint32_t kScaledCoefficientBits[] = {0x3f803884u, 0x33a85adeu, 0x27aca410u};
    static constexpr uint32_t kMultiplierBits = 0x3f800000u;
    static constexpr uint32_t kBodySlots = 28u;
    static constexpr bool kNegativeNanTerminal = false;
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_bf16_exp2.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format.
inline bool bf16_dest_exp2() {
#if defined(TRISC_MATH) || defined(LLK_TRISC_MATH)
    return ckernel::math::src_zero_flag_srcb_fmt == static_cast<std::uint32_t>(DataFormat::Float16_b);
#else
    return false;
#endif
}
template <int ITERATIONS = 8>
inline void calculate_exp2_bf16() {
    ckernel::sfpu::bf16::calculate_exp2<ckernel::sfpu::Exp2Bf16Config, ITERATIONS>();
}
inline void init_exp2_bf16() {
    if (bf16_dest_exp2()) {
        ckernel::sfpu::bf16::init_exp2<ckernel::sfpu::Exp2Bf16Config>();
    }
}

}  // namespace ckernel::sfpu
