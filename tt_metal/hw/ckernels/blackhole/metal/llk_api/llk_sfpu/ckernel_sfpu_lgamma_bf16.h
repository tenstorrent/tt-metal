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
        __builtin_bit_cast(float, 0xbf7cbe07u),
        __builtin_bit_cast(float, 0x40942851u),
        __builtin_bit_cast(float, 0xc1dbd0bbu),
        __builtin_bit_cast(float, 0x43017f69u),
        __builtin_bit_cast(float, 0xc3b26d69u),
        __builtin_bit_cast(float, 0x43ff7f4du),
        __builtin_bit_cast(float, 0xc392a3f4u)};
    static constexpr float kLog[] = {
        __builtin_bit_cast(float, 0xc0058486u),
        __builtin_bit_cast(float, 0x407f0d69u),
        __builtin_bit_cast(float, 0xc03c034au),
        __builtin_bit_cast(float, 0x3fa1fb01u),
        __builtin_bit_cast(float, 0xbe68300bu)};
    static constexpr float kUnit[] = {
        __builtin_bit_cast(float, 0x3f7f201fu),
        __builtin_bit_cast(float, 0xbf25fed2u),
        __builtin_bit_cast(float, 0x3e8cf30du),
        __builtin_bit_cast(float, 0xbd420c52u)};
    static constexpr float kSinc[] = {
        __builtin_bit_cast(float, 0x3b0df0cbu),
        __builtin_bit_cast(float, 0xbc8ec15eu),
        __builtin_bit_cast(float, 0xbfd54454u),
        __builtin_bit_cast(float, 0x3ea2c565u),
        __builtin_bit_cast(float, 0xbf8cc273u)};
    static constexpr unsigned kDegree = 6u;
    static constexpr unsigned kLogDegree = 4u;
    static constexpr unsigned kUnitDegree = 3u;
    static constexpr unsigned kSincDegree = 4u;
    static constexpr unsigned kLogBase = 0u;
    static constexpr unsigned kUnitBase = 0u;
    static constexpr unsigned kSincBase = 0u;
    static constexpr unsigned kCoreBase = 0u;
    static constexpr unsigned kRawShadowBase = 64u;
    static constexpr float kRoot = __builtin_bit_cast(float, 0x40000000u);
    static constexpr unsigned kRawNegativeNanClass = 2u;
    static constexpr bool kStore = false;
    static constexpr bool kRawShadow = true;
    static constexpr bool kSourceTerminal = false;
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
    static constexpr uint32_t kDomainActionCount = 1;
    static constexpr std::array<TtDomainActionRecord, kDomainActionCount> kDomainActions = {
        {{4.0915299245254442e+36f,
          1,
          1,
          3,
          1,
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
