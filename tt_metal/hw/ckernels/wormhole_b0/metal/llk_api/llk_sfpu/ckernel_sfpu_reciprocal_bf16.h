// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct ReciprocalBf16Config {
    static constexpr uint32_t kMagic = 0x7f800000u;
    static constexpr uint32_t kC0Bits = 0x3ea57ebbu;
    static constexpr uint32_t kC1Bits = 0x3fba2e90u;
    static constexpr uint32_t kC2Bits = 0x4007c1f2u;
    static constexpr uint32_t kBodySlots = 26u;
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_bf16_newton_reciprocal.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format.
inline bool bf16_dest_reciprocal() {
#if defined(TRISC_MATH) || defined(LLK_TRISC_MATH)
    return ckernel::math::src_zero_flag_srcb_fmt == static_cast<std::uint32_t>(DataFormat::Float16_b);
#else
    return false;
#endif
}
template <int ITERATIONS = 8>
inline void calculate_reciprocal_bf16() {
    ckernel::sfpu::bf16::calculate_newton_reciprocal<ckernel::sfpu::ReciprocalBf16Config, ITERATIONS>();
}
inline void init_reciprocal_bf16() {
    if (bf16_dest_reciprocal()) {
        ckernel::sfpu::bf16::init_newton_reciprocal<ckernel::sfpu::ReciprocalBf16Config>();
    }
}

}  // namespace ckernel::sfpu
