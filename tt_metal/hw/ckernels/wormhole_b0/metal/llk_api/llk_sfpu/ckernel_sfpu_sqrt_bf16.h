// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct SqrtBf16Config {
    static constexpr uint32_t kMagic = 0x5f1110a0u;
    static constexpr uint32_t kC1Bits = 0x401214c9u;
    static constexpr uint32_t kC2Bits = 0x40103626u;
    static constexpr uint32_t kBodySlots = 29u;
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_bf16_newton_root.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format.
inline bool bf16_dest_sqrt() {
#if defined(TRISC_MATH) || defined(LLK_TRISC_MATH)
    return ckernel::math::src_zero_flag_srcb_fmt == static_cast<std::uint32_t>(DataFormat::Float16_b);
#else
    return false;
#endif
}
template <int ITERATIONS = 8>
inline void calculate_sqrt_bf16() {
    ckernel::sfpu::bf16::calculate_newton_root<ckernel::sfpu::SqrtBf16Config, ITERATIONS>();
}
inline void init_sqrt_bf16() {
    if (bf16_dest_sqrt()) {
        ckernel::sfpu::bf16::init_newton_root<ckernel::sfpu::SqrtBf16Config>();
    }
}

}  // namespace ckernel::sfpu
