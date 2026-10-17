// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

namespace ckernel::sfpu {
struct ErfBf16Config {
    static constexpr uint32_t kDegree = 7u;
    // Fit: minimax degree-7 polynomial on [0, 10], 1 segment, max pure (continuous) ULP 0.73.
    static constexpr uint32_t kCoefficientBits[] = {
        0x00000000u, 0x3f904bb0u, 0x3ceceac4u, 0xbf007bbau, 0x3e40e7f0u, 0x3c90cff8u, 0xbca6e156u, 0x3b39f45cu};
    static constexpr uint32_t kClampMaxBits = 0x3f800000u;
};
}  // namespace ckernel::sfpu

#include "ckernel_sfpu_bf16_cascade_signed_abs.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format.
inline bool bf16_dest_erf() {
#if defined(TRISC_MATH) || defined(LLK_TRISC_MATH)
    return ckernel::math::src_zero_flag_srcb_fmt == static_cast<std::uint32_t>(DataFormat::Float16_b);
#else
    return false;
#endif
}
template <int ITERATIONS = 8>
inline void calculate_erf_bf16() {
    ckernel::sfpu::bf16::calculate_signed_abs_nan<ckernel::sfpu::ErfBf16Config, ITERATIONS>();
}
inline void init_erf_bf16() {
    if (bf16_dest_erf()) {
        ckernel::sfpu::bf16::init_signed_abs_nan<ckernel::sfpu::ErfBf16Config>();
    }
}

}  // namespace ckernel::sfpu
