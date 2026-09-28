// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Generated architecture configuration; shared primitives live in common/.
#pragma once
#include <cstdint>
namespace ttpoly_generated {
struct DigammaBf16Config {
    static constexpr float TT_EXPONENT_BUCKET_COEFFS[] = {
        __builtin_bit_cast(float, 0xbfbddb9cu),
        __builtin_bit_cast(float, 0x4006731au),
        __builtin_bit_cast(float, 0xbf39dd50u),
        __builtin_bit_cast(float, 0x3de17ec8u),
        __builtin_bit_cast(float, 0x3f31697cu),
        __builtin_bit_cast(float, 0xbf004201u),
        __builtin_bit_cast(float, 0xbda33ff3u)};
    static constexpr float TT_EXPONENT_BUCKET_REFLECTION_COEFFS[] = {
        __builtin_bit_cast(float, 0x3f800000u),
        __builtin_bit_cast(float, 0x00000000u),
        __builtin_bit_cast(float, 0xc0527a38u),
        __builtin_bit_cast(float, 0x00000000u),
        __builtin_bit_cast(float, 0xc00d43cbu),
        __builtin_bit_cast(float, 0x00000000u),
        __builtin_bit_cast(float, 0xbfc7707du),
        __builtin_bit_cast(float, 0x00000000u),
        __builtin_bit_cast(float, 0xc07e28c5u)};
    static constexpr unsigned TT_EXPONENT_BUCKET_COUNT = 0u;
    static constexpr unsigned TT_EXPONENT_BUCKET_NATIVE_LOG = 0u;
    static constexpr unsigned TT_EXPONENT_BUCKET_RECIPROCAL_DEGREE = 2u;
    static constexpr unsigned TT_EXPONENT_BUCKET_DIVISOR_RECIPROCAL_ITERATIONS = 1u;
    static constexpr unsigned TT_EXPONENT_BUCKET_REFLECTION_DEGREE = 8u;
    static constexpr unsigned TT_TARGET_CLASS_TTI_RAW_SHADOW_ROW_BASE = 0u;
    static constexpr unsigned TT_TARGET_CLASS_TTI_BODY_SLOTS = 32u;
    static constexpr unsigned TT_TARGET_CLASS_TTI_MAN_MASK = 8388607u;
    static constexpr unsigned TT_TARGET_CLASS_TTI_POS_INF_RAW = 2139095040u;
    static constexpr unsigned TT_TARGET_CLASS_TTI_FINITE_INTEGER = 49475u;
    static constexpr unsigned TT_TARGET_CLASS_TTI_ZERO_SUBNORMAL = 51364u;
    static constexpr unsigned TT_TARGET_CLASS_TTI_NAN = 17073u;
    static constexpr unsigned TT_TARGET_CLASS_TTI_NEG_NAN = 32640u;
    static constexpr unsigned TT_TARGET_CLASS_TTI_NEG_INF_FIRST = 51827u;
    static constexpr unsigned TT_TARGET_CLASS_TTI_POS_INF_FIRST = 52108u;
    static constexpr unsigned TT_TARGET_CLASS_TTI_NEG_INF = 65408u;
    static constexpr bool kStore = false;
    static constexpr bool kFuse = false;
};
}  // namespace ttpoly_generated
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_exponent_bucket.h"
#ifndef TT_POLY_LLK_EXPONENT_BUCKET_SELECTED_V1
#error "selected exponent bucket runtime required"
#endif
#define TT_POLY_DIGAMMA_BF16_AVAILABLE 1
