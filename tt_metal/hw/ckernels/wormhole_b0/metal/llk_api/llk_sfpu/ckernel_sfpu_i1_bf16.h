// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <array>
namespace ckernel::sfpu {
struct I1Bf16Config {
    static constexpr float kExpCoefficients[] = {
        __builtin_bit_cast(float, 0x41aea390u),
        __builtin_bit_cast(float, 0x4165bb4cu),
        __builtin_bit_cast(float, 0x40e99381u)};
    static constexpr unsigned kExpDegree = 2u;
    static constexpr float kCoreCoefficients[] = {
        __builtin_bit_cast(float, 0x3f000d27u),
        __builtin_bit_cast(float, 0x3d7df36du),
        __builtin_bit_cast(float, 0x3b35759au),
        __builtin_bit_cast(float, 0x381e2bd7u),
        __builtin_bit_cast(float, 0x35aaf404u)};
    static constexpr unsigned kCoreDegree = 4u;
    static constexpr float kCorrectionCoefficients[] = {
        __builtin_bit_cast(float, 0x3f800000u), __builtin_bit_cast(float, 0xbeca5b42u)};
    static constexpr unsigned kCorrectionDegree = 1u;
    static constexpr float kCoreBoundary = __builtin_bit_cast(float, 0x40c00000u);
    static constexpr float kFiniteTerminal = __builtin_bit_cast(float, 0x42b80000u);
    static constexpr float kExponentShift = __builtin_bit_cast(float, 0x40800000u);
    static constexpr float kRootC1 = __builtin_bit_cast(float, 0x401214c9u);
    static constexpr float kRootC2 = __builtin_bit_cast(float, 0x40103626u);
    static constexpr unsigned kRootMagic = 1594953888u;
    static constexpr unsigned kOriginShift = 126u;
    static constexpr unsigned kOriginScaledBits = 1073676288u;
    static constexpr unsigned kLateNegativeNanClass = 1u;
    static constexpr bool kOdd = true;
    static constexpr bool kOriginRepair = true;
    static constexpr bool kWormhole = true;
    static constexpr bool kLateNegativeInf = true;

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
        {{-9.2000000000000000e+01f,
          0,
          1,
          3,
          2,
          0.0000000000000000e+00f,
          0.0000000000000000e+00f,
          0.0000000000000000e+00f},
         {9.2000000000000000e+01f,
          1,
          1,
          3,
          1,
          0.0000000000000000e+00f,
          0.0000000000000000e+00f,
          0.0000000000000000e+00f}}};
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_bf16_exp_root.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format.
inline bool bf16_dest_i1() {
#if defined(TRISC_MATH) || defined(LLK_TRISC_MATH)
    return ckernel::math::src_zero_flag_srcb_fmt == static_cast<std::uint32_t>(DataFormat::Float16_b);
#else
    return false;
#endif
}
template <int ITERATIONS = 8>
inline void calculate_i1_bf16() {
    ckernel::sfpu::bf16::calculate_exp_root<ckernel::sfpu::I1Bf16Config, ITERATIONS>();
}

}  // namespace ckernel::sfpu
