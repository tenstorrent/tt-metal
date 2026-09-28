// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Generated architecture configuration; shared primitives live in common/.
#pragma once
#include <cstdint>
namespace ttpoly_generated {
struct DigammaBf16Config {
    static constexpr float TT_EXPONENT_BUCKET_COEFFS[] = {
        __builtin_bit_cast(float, 0xbfbdb274u),
        __builtin_bit_cast(float, 0x40064940u),
        __builtin_bit_cast(float, 0xbf396838u),
        __builtin_bit_cast(float, 0x3de0a961u),
        __builtin_bit_cast(float, 0x3f31697au),
        __builtin_bit_cast(float, 0xbf005814u),
        __builtin_bit_cast(float, 0xbda28be5u)};
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
    static constexpr unsigned TT_TARGET_CLASS_FIRST[] = {49024u, 51827u};
    static constexpr unsigned TT_TARGET_CLASS_OUTPUT[] = {49475u, 65408u};
    static constexpr unsigned TT_EXPONENT_BUCKET_COUNT = 0u;
    static constexpr unsigned TT_EXPONENT_BUCKET_NATIVE_LOG = 0u;
    static constexpr unsigned TT_EXPONENT_BUCKET_RECIPROCAL_DEGREE = 2u;
    static constexpr unsigned TT_EXPONENT_BUCKET_DIVISOR_RECIPROCAL_ITERATIONS = 2u;
    static constexpr unsigned TT_EXPONENT_BUCKET_REFLECTION_DEGREE = 8u;
    static constexpr unsigned TT_TARGET_CLASS_ZERO = 51364u;
    static constexpr unsigned TT_TARGET_CLASS_POS_NAN = 17073u;
    static constexpr unsigned TT_TARGET_CLASS_POS_INF = 32640u;
    static constexpr unsigned TT_TARGET_CLASS_NEG_INF = 65408u;
    static constexpr unsigned TT_TARGET_CLASS_NEG_NAN = 65408u;
    static constexpr unsigned TT_TARGET_CLASS_TERMINAL_PEAK_LIVE = 6u;
    static constexpr unsigned TT_TARGET_CLASS_PARTITIONS = 2u;
    static constexpr bool kStore = true;
    static constexpr bool kFuse = true;
};
}  // namespace ttpoly_generated
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_exponent_bucket.h"
#ifndef TT_POLY_LLK_EXPONENT_BUCKET_SELECTED_V1
#error "selected exponent bucket runtime required"
#endif
#define TT_POLY_DIGAMMA_BF16_AVAILABLE 1
