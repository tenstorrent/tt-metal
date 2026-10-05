// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ckernel_sfpu_bf16_sfpi_isa.h"
#include <cstdint>
#include <array>
namespace ckernel::sfpu {
struct ErfcBf16Config {
    static constexpr float kNumerator[] = {
        __builtin_bit_cast(float, 0x3f1cb49cu), __builtin_bit_cast(float, 0x3ea79f6du)};
    static constexpr float kDenominator[] = {
        __builtin_bit_cast(float, 0x3f1ceb0bu),
        __builtin_bit_cast(float, 0x3f800000u),
        __builtin_bit_cast(float, 0x3f1651e0u)};
    static constexpr float kExpCoefficients[] = {
        __builtin_bit_cast(float, 0x3f800000u),
        __builtin_bit_cast(float, 0x3f32107du),
        __builtin_bit_cast(float, 0x3e6798ecu),
        __builtin_bit_cast(float, 0x3da00879u)};
    static constexpr uint32_t kExpDegree = 3;
    static constexpr float kMultiplier = __builtin_bit_cast(float, 0x3fb8aa3bu);
    static constexpr uint32_t kBoundBits = 0x41140000u;
    static constexpr bool kSquareDecay = true;
    static constexpr float kPositiveNonfiniteConstant = __builtin_bit_cast(float, 0x2c4f0000u);
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_bf16_abs_exp_correction.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format.
inline bool bf16_dest_erfc() {
#if defined(TRISC_MATH) || defined(LLK_TRISC_MATH)
    return ckernel::math::src_zero_flag_srcb_fmt == static_cast<std::uint32_t>(DataFormat::Float16_b);
#else
    return false;
#endif
}
template <int ITERATIONS = 8>
inline void calculate_erfc_bf16() {
    ckernel::sfpu::bf16::calculate_abs_exp_correction<ckernel::sfpu::ErfcBf16Config, ITERATIONS>();
}
inline void init_erfc_bf16() {
    if (bf16_dest_erfc()) {
        ckernel::sfpu::bf16::init_abs_exp_correction<ckernel::sfpu::ErfcBf16Config>();
    }
}

}  // namespace ckernel::sfpu
