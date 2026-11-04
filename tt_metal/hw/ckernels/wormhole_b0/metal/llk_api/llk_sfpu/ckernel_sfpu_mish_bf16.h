// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <array>
namespace ckernel::sfpu {
struct MishBf16Config {
    static constexpr float kMultiplier = __builtin_bit_cast(float, 0x3fb8aa3bu);
    static constexpr float kBias = __builtin_bit_cast(float, 0x43060000u);
    static constexpr float kScale = __builtin_bit_cast(float, 0x43000000u);
    static constexpr float kNegativeSlope = __builtin_bit_cast(float, 0x3f804189u);
    static constexpr float kNegativeBias = __builtin_bit_cast(float, 0x43800000u);
    static constexpr float kPositiveScale = __builtin_bit_cast(float, 0x46ffbe77u);
    static constexpr float kLowerBound = __builtin_bit_cast(float, 0xc2b70000u);
    static constexpr float kUpperBound = __builtin_bit_cast(float, 0x41200000u);
    static constexpr float kCoefficients[] = {
        __builtin_bit_cast(float, 0x3f803884u),
        __builtin_bit_cast(float, 0x3f285adeu),
        __builtin_bit_cast(float, 0x3eaca410u)};
    static constexpr uint32_t kOriginEncoded = 54529u;

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
    static constexpr uint32_t kDomainActionCount = 3;
    static constexpr std::array<TtDomainActionRecord, kDomainActionCount> kDomainActions = {
        {{-3.3895313892515355e+38f,
          0,
          0,
          3,
          2,
          0.0000000000000000e+00f,
          0.0000000000000000e+00f,
          0.0000000000000000e+00f},
         {-9.2000000000000000e+01f,
          0,
          1,
          0,
          0,
          0.0000000000000000e+00f,
          0.0000000000000000e+00f,
          0.0000000000000000e+00f},
         {1.0000000000000000e+01f,
          1,
          0,
          1,
          0,
          0.0000000000000000e+00f,
          0.0000000000000000e+00f,
          0.0000000000000000e+00f}}};
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_mish_bf16_exponent_alu_product.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format, as BF16 for
// Float16_b and the block formats sharing its exponent.
inline bool bf16_dest_mish() {
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
__attribute__((noinline)) inline void calculate_mish_bf16() {
    ckernel::sfpu::bf16::mish::calculate_exponent_alu_product<ckernel::sfpu::MishBf16Config, ITERATIONS>();
}
inline void init_mish_bf16() { ckernel::sfpu::bf16::mish::init_exponent_alu_product<ckernel::sfpu::MishBf16Config>(); }

}  // namespace ckernel::sfpu
