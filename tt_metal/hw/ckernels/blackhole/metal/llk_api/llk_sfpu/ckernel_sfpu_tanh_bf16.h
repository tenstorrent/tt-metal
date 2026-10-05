// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

namespace ckernel::sfpu {
struct TanhBf16Config {
    static constexpr uint32_t kDegree = 6u;
    // Fit: degree-6 polynomial on [0, 5], 1 segment, max pure (continuous) ULP 0.839.
    static constexpr uint32_t kCoefficientBits[] = {
        0x00000000u, 0x3f800000u, 0x3cbfe000u, 0xbef28000u, 0x3e8a1bd5u, 0xbd8035c9u, 0x3bb0e52du};
    static constexpr uint32_t kClampMaxBits = 0x3f800000u;
    static constexpr bool kIntrinsicExceptional = true;
    static constexpr bool kIntrinsicTerminal = true;
    static constexpr uint32_t kTerminalBoundBits = 0x40a00000u;
};
}  // namespace ckernel::sfpu

#include "ckernel_sfpu_bf16_cascade_signed_abs.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format.
inline bool bf16_dest_tanh() {
#if defined(TRISC_MATH) || defined(LLK_TRISC_MATH)
    return ckernel::math::src_zero_flag_srcb_fmt == static_cast<std::uint32_t>(DataFormat::Float16_b);
#else
    return false;
#endif
}
template <int ITERATIONS = 8>
inline void calculate_tanh_bf16() {
    ckernel::sfpu::bf16::calculate_signed_abs_terminal<ckernel::sfpu::TanhBf16Config, ITERATIONS>();
}
inline void init_tanh_bf16() {
    if (bf16_dest_tanh()) {
        ckernel::sfpu::bf16::init_signed_abs<ckernel::sfpu::TanhBf16Config>();
    }
}

}  // namespace ckernel::sfpu
