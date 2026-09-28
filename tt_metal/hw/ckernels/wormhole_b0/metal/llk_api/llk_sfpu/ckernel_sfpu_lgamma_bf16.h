// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace ckernel::sfpu {}

#if !defined(TT_POLY_LLK_DISABLE)
#include <cstdint>
#include <array>
namespace ttpoly_generated {
struct LgammaBf16Config {
    static constexpr float kCoefficients[] = {
        __builtin_bit_cast(float, 0xbf7cbb8eu),
        __builtin_bit_cast(float, 0x40940bbdu),
        __builtin_bit_cast(float, 0xc1db4d3bu),
        __builtin_bit_cast(float, 0x43010ad2u),
        __builtin_bit_cast(float, 0xc3b1a657u),
        __builtin_bit_cast(float, 0x43fe3705u),
        __builtin_bit_cast(float, 0xc391d2a9u)};
    static constexpr float kLog[] = {
        __builtin_bit_cast(float, 0xc0054684u),
        __builtin_bit_cast(float, 0x407e2f9eu),
        __builtin_bit_cast(float, 0xc03adf8du),
        __builtin_bit_cast(float, 0x3fa0ac83u),
        __builtin_bit_cast(float, 0xbe65fb41u)};
    static constexpr float kUnit[] = {
        __builtin_bit_cast(float, 0x3f7f2e19u),
        __builtin_bit_cast(float, 0xbf261ef5u),
        __builtin_bit_cast(float, 0x3e8d21c3u),
        __builtin_bit_cast(float, 0xbd426304u)};
    static constexpr float kSinc[] = {
        __builtin_bit_cast(float, 0x3b0988d8u),
        __builtin_bit_cast(float, 0xbc92cbc3u),
        __builtin_bit_cast(float, 0xbfd4bb28u),
        __builtin_bit_cast(float, 0x3e9e8b1cu),
        __builtin_bit_cast(float, 0xbf8c2451u)};
    static constexpr unsigned kDegree = 6u;
    static constexpr unsigned kLogDegree = 4u;
    static constexpr unsigned kUnitDegree = 3u;
    static constexpr unsigned kSincDegree = 4u;
    static constexpr unsigned kLogBase = 64u;
    static constexpr unsigned kUnitBase = 69u;
    static constexpr unsigned kSincBase = 73u;
    static constexpr unsigned kCoreBase = 78u;
    static constexpr unsigned kRawShadowBase = 0u;
    static constexpr float kRoot = __builtin_bit_cast(float, 0x40000000u);
    static constexpr unsigned kRawNegativeNanClass = 2u;
    static constexpr bool kStore = true;
    static constexpr bool kRawShadow = false;
    static constexpr bool kSourceTerminal = true;
    static constexpr bool kBf16 = true;

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
        {{4.0915299245254442e+36f,
          1,
          1,
          3,
          1,
          0.0000000000000000e+00f,
          0.0000000000000000e+00f,
          0.0000000000000000e+00f},
         {-4.0915299245254442e+36f,
          0,
          1,
          3,
          2,
          0.0000000000000000e+00f,
          0.0000000000000000e+00f,
          0.0000000000000000e+00f}}};
};
}  // namespace ttpoly_generated
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_root_native_log.h"
#ifndef TT_POLY_LLK_ROOT_NATIVE_LOG_SELECTED_V1
#error "selected root/native log runtime required"
#endif
#define TT_POLY_LGAMMA_BF16_AVAILABLE 1
#endif

namespace ckernel::sfpu {

#if !defined(TT_POLY_LLK_DISABLE)
template <int ITERATIONS = 8>
inline void calculate_lgamma_tt_poly_bf16() {
    ckernel::sfpu::ttpoly::calculate_root_native_log<ttpoly_generated::LgammaBf16Config, ITERATIONS>();
}
#endif

}  // namespace ckernel::sfpu
