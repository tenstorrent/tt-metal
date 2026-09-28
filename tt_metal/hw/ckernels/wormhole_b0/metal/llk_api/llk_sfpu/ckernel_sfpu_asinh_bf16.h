// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Generated architecture configuration; shared primitives live in common/.
#pragma once
#include <cstdint>
namespace ttpoly_generated {
struct AsinhBf16Config {
    static constexpr float kNegative[] = {
        __builtin_bit_cast(float, 0x3f80336au),
        __builtin_bit_cast(float, 0x3b8c467cu),
        __builtin_bit_cast(float, 0xbe4f8e3bu),
        __builtin_bit_cast(float, 0xbdec9997u),
        __builtin_bit_cast(float, 0xbd08126cu),
        __builtin_bit_cast(float, 0xbbb63430u),
        __builtin_bit_cast(float, 0xba0eeaafu),
        __builtin_bit_cast(float, 0xb7f3512du),
        __builtin_bit_cast(float, 0xb52d7ae6u)};
    static constexpr float kPositive[] = {
        __builtin_bit_cast(float, 0x3f80336au),
        __builtin_bit_cast(float, 0xbb8c467cu),
        __builtin_bit_cast(float, 0xbe4f8e3bu),
        __builtin_bit_cast(float, 0x3dec9997u),
        __builtin_bit_cast(float, 0xbd08126cu),
        __builtin_bit_cast(float, 0x3bb63430u),
        __builtin_bit_cast(float, 0xba0eeaafu),
        __builtin_bit_cast(float, 0x37f3512du),
        __builtin_bit_cast(float, 0xb52d7ae6u)};
    static constexpr float kLog[] = {
        __builtin_bit_cast(float, 0xbefff282u),
        __builtin_bit_cast(float, 0x3eac415eu),
        __builtin_bit_cast(float, 0xbe84fd79u),
        __builtin_bit_cast(float, 0x3e1960f0u)};
    static constexpr uint32_t kLowerBits = 0xc1200000u;
    static constexpr uint32_t kUpperBits = 0x41200000u;
    static constexpr uint32_t kAddendBits = 0x3f317218u;
    static constexpr bool kMirrorFold = true;
    static constexpr bool kSignedNanFinalizer = false;
};
}  // namespace ttpoly_generated
#include "../../../../common/llk_sfpu/ckernel_sfpu_tt_poly_symmetric_factored_log.h"
#ifndef TT_POLY_LLK_SYMMETRIC_FACTORED_LOG_V1
#error "typed symmetric factored-log runtime required"
#endif
#define TT_POLY_ASINH_BF16_AVAILABLE 1
