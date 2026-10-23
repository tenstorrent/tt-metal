// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
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
    static constexpr float kPositiveNonfiniteConstant = __builtin_bit_cast(float, 0x2c4e0000u);
    static constexpr float kPositiveInfinityConstant = __builtin_bit_cast(float, 0x00000000u);

    // Typed domain actions.
    struct TtDomainActionRecord {
        float bound;
        uint8_t direction;
        uint8_t inclusive;
        uint8_t action_kind;   // 0=constant, 1=identity, 2=affine, 3=class, 4=signed-inf
        uint8_t return_class;  // 0=NaN, 1=+Inf, 2=-Inf, 3=+0, 4=-0
        float value;
        float scale;
        float bias;
    };
    static constexpr uint32_t kDomainActionCount = 2;
    static constexpr std::array<TtDomainActionRecord, kDomainActionCount> kDomainActions = {
        {{-5.0000000000000000e+00f,
          0,
          0,
          0,
          0,
          2.0000000000000000e+00f,
          0.0000000000000000e+00f,
          0.0000000000000000e+00f},
         {9.5000000000000000e+00f,
          1,
          1,
          0,
          0,
          0.0000000000000000e+00f,
          0.0000000000000000e+00f,
          0.0000000000000000e+00f}}};
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_erfc_bf16_abs_exp_correction.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format, as BF16 for
// Float16_b and the block formats sharing its exponent.
inline bool bf16_dest_erfc() {
#if defined(TRISC_MATH) || defined(LLK_TRISC_MATH)
    const std::uint32_t format = ckernel::math::src_zero_flag_srcb_fmt;
    return format == static_cast<std::uint32_t>(DataFormat::Float16_b) ||
           format == static_cast<std::uint32_t>(DataFormat::Bfp8_b) ||
           format == static_cast<std::uint32_t>(DataFormat::Bfp4_b) ||
           format == static_cast<std::uint32_t>(DataFormat::Bfp2_b);
#else
    return false;
#endif
}
// Each call records replay slots 0-31, overwriting what an init recorded there (matmul and reduce
// record from replay_buf_offset = 16).
template <int ITERATIONS = 32>
__attribute__((noinline)) inline void calculate_erfc_bf16() {
    ckernel::sfpu::bf16::erfc::calculate_abs_exp_correction<ckernel::sfpu::ErfcBf16Config, ITERATIONS>();
}
inline void init_erfc_bf16() { ckernel::sfpu::bf16::erfc::init_abs_exp_correction<ckernel::sfpu::ErfcBf16Config>(); }

}  // namespace ckernel::sfpu
