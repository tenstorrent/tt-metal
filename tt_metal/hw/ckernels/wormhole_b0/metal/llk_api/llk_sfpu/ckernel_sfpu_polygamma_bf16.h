// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct PolygammaBf16Config {
    static constexpr float kP0 = __builtin_bit_cast(float, 0x3e210000u);
    static constexpr unsigned kNanResultWord = 32704u;
    static constexpr unsigned kPositiveInfinityWord = 0u;
    static constexpr unsigned kNegativeInfinityWord = 32704u;
    static constexpr float kP2[] = {
        __builtin_bit_cast(float, 0x40530000u),
        __builtin_bit_cast(float, 0x40b40000u),
        __builtin_bit_cast(float, 0x41930000u)};
    static constexpr bool kRepair = false;
    static constexpr bool kWhRepair = true;
    static constexpr bool kDirectTail = false;
    static constexpr bool kFiniteReferencePrecedence = true;
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_bf16_inverse_square.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format.
inline bool bf16_dest_polygamma() {
#if defined(TRISC_MATH) || defined(LLK_TRISC_MATH)
    return ckernel::math::src_zero_flag_srcb_fmt == static_cast<std::uint32_t>(DataFormat::Float16_b);
#else
    return false;
#endif
}
template <int ITERATIONS = 8>
inline void calculate_polygamma_bf16() {
    ckernel::sfpu::bf16::calculate_inverse_square<ckernel::sfpu::PolygammaBf16Config, ITERATIONS>();
}

}  // namespace ckernel::sfpu
