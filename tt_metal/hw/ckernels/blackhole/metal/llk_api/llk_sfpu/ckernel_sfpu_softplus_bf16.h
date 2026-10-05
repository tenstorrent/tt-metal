// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <array>
namespace ckernel::sfpu {
struct SoftplusBf16Config {
    static constexpr float kNumerator[] = {
        __builtin_bit_cast(float, 0x3f800000u),
        __builtin_bit_cast(float, 0xbefab96eu),
        __builtin_bit_cast(float, 0x3e85799eu),
        __builtin_bit_cast(float, 0xbda01eecu)};
    static constexpr float kExpCoefficients[] = {
        __builtin_bit_cast(float, 0x3f803884u),
        __builtin_bit_cast(float, 0x3f285adeu),
        __builtin_bit_cast(float, 0x3eaca410u)};
    static constexpr uint32_t kExpDegree = 2;
    static constexpr float kMultiplier = __builtin_bit_cast(float, 0xbfb8aa3bu);
    static constexpr uint32_t kBoundBits = 0x42af0000u;
    static constexpr bool kHasBound = true;
    static constexpr bool kPolynomial = true;
    static constexpr bool kSquareDecay = false;
    static constexpr bool kResidualFold = false;
    static constexpr bool kSkipInputAbs = false;
    static constexpr bool kSquareStore = false;
    static constexpr bool kCorrectionStore = false;
    static constexpr bool kPositivePart = true;
    static constexpr bool kRawNanAffinePart = false;
    static constexpr bool kRawNanClass = false;
    static constexpr bool kRawNanTerminal = false;
    static constexpr bool kResidualTti = true;
    static constexpr uint32_t kDomainActionCount = 0;
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_bf16_abs_exp_correction.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format.
inline bool bf16_dest_softplus() {
#if defined(TRISC_MATH) || defined(LLK_TRISC_MATH)
    return ckernel::math::src_zero_flag_srcb_fmt == static_cast<std::uint32_t>(DataFormat::Float16_b);
#else
    return false;
#endif
}
template <int ITERATIONS = 8>
inline void calculate_softplus_bf16() {
    ckernel::sfpu::bf16::calculate_abs_exp_correction<ckernel::sfpu::SoftplusBf16Config, ITERATIONS>();
}

}  // namespace ckernel::sfpu
