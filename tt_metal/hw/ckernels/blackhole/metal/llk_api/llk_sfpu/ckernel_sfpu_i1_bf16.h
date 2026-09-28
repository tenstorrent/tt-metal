// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Generated architecture configuration; shared primitives live in common/.
#pragma once
#include <cstdint>
#include <array>
namespace ttpoly_generated {
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
    static constexpr bool kWormhole = false;

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
        {{-8.8500000000000000e+01f,
          0,
          1,
          0,
          0,
          -1.1547668213381457e+37f,
          0.0000000000000000e+00f,
          0.0000000000000000e+00f},
         {8.8500000000000000e+01f,
          1,
          1,
          0,
          0,
          1.1547668213381457e+37f,
          0.0000000000000000e+00f,
          0.0000000000000000e+00f}}};
};
}  // namespace ttpoly_generated
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_exp_root.h"
#ifndef TT_POLY_LLK_EXP_ROOT_SELECTED_V1
#error "typed selected exp-root runtime required"
#endif
#define TT_POLY_I1_BF16_AVAILABLE 1
