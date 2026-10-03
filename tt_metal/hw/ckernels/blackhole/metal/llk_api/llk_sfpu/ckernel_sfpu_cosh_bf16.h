// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct CoshBf16Config {
    static constexpr uint32_t kDegree = 4u;
    static constexpr bool kOdd = false;
    static constexpr bool kBare = true;
    static constexpr uint32_t kScaledCoefficientBits[] = {
        0x3f800016u, 0x33b168acu, 0x27773d30u, 0x1ad514a6u, 0x0e5dc2acu};
    static constexpr uint32_t kMultiplierBits = 0x3fb8aa3bu;
    static constexpr uint32_t kComposeScaleBits = 0x3e800100u;
    static constexpr uint32_t kOriginCubicBits = 0x00000000u;
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_bf16_hyperbolic_exp.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format.
inline bool bf16_dest_cosh() {
#if defined(TRISC_MATH) || defined(LLK_TRISC_MATH)
    return ckernel::math::src_zero_flag_srcb_fmt == static_cast<std::uint32_t>(DataFormat::Float16_b);
#else
    return false;
#endif
}
template <int ITERATIONS = 8>
inline void calculate_cosh_bf16() {
    ckernel::sfpu::bf16::calculate_hyperbolic_exp<ckernel::sfpu::CoshBf16Config, ITERATIONS>();
}
inline void init_cosh_bf16() {
    if (bf16_dest_cosh()) {
        ckernel::sfpu::bf16::init_hyperbolic_exp<ckernel::sfpu::CoshBf16Config>();
    }
}

}  // namespace ckernel::sfpu
