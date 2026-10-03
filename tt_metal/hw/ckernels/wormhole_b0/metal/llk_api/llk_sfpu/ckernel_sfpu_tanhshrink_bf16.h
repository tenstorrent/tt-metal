// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct TanhshrinkBf16Config {
    static constexpr uint32_t kDegree = 8u;
    // Fit: minimax degree-8 polynomial on [0, 3], 1 segment, max pure (continuous) ULP 0.781.
    static constexpr uint32_t kCoefficientBits[] = {
        0x00000000u,
        0x00000000u,
        0x00000000u,
        0x3eaa738eu,
        0x3c52ec0cu,
        0xbe4a3e90u,
        0x3dec2feeu,
        0xbce1764au,
        0x3b20a66au};
    static constexpr uint32_t kBoundBits = 0x40400000u;
    static constexpr uint32_t kScaleBits = 0x3f800000u;
    static constexpr uint32_t kBiasBits = 0xbf800000u;
    static constexpr bool kNegativeNanTerminal = false;
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_bf16_cascade_signed_abs_affine.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format.
inline bool bf16_dest_tanhshrink() {
#if defined(TRISC_MATH) || defined(LLK_TRISC_MATH)
    return ckernel::math::src_zero_flag_srcb_fmt == static_cast<std::uint32_t>(DataFormat::Float16_b);
#else
    return false;
#endif
}
template <int ITERATIONS = 8>
inline void calculate_tanhshrink_bf16() {
    ckernel::sfpu::bf16::calculate_signed_abs_affine<ckernel::sfpu::TanhshrinkBf16Config, ITERATIONS>();
}
inline void init_tanhshrink_bf16() {
    if (bf16_dest_tanhshrink()) {
        ckernel::sfpu::bf16::init_signed_abs_affine<ckernel::sfpu::TanhshrinkBf16Config>();
    }
}

}  // namespace ckernel::sfpu
