// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct SoftsignBf16Config {
    static constexpr uint32_t kBodySlots = 14u;
    static constexpr uint32_t kBoundExponent = 136u;
    static constexpr uint32_t kLutSlopes = 0xb549b9a8u;
    static constexpr uint32_t kLutIntercepts = 0x3c9f3ed4u;
    static constexpr bool kLateRound = true;
    static constexpr bool kRawNegativeInfinity = false;
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_bf16_abs_denominator.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format.
inline bool bf16_dest_softsign() {
#if defined(TRISC_MATH) || defined(LLK_TRISC_MATH)
    return ckernel::math::src_zero_flag_srcb_fmt == static_cast<std::uint32_t>(DataFormat::Float16_b);
#else
    return false;
#endif
}
template <int ITERATIONS = 8>
inline void calculate_softsign_bf16() {
    ckernel::sfpu::bf16::calculate_abs_denominator<ckernel::sfpu::SoftsignBf16Config, ITERATIONS>();
}
inline void init_softsign_bf16() {
    if (bf16_dest_softsign()) {
        ckernel::sfpu::bf16::init_abs_denominator<ckernel::sfpu::SoftsignBf16Config>();
    }
}

}  // namespace ckernel::sfpu
