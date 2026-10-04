// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct GeluBf16Config {
    static constexpr float kExpCoefficients[] = {
        __builtin_bit_cast(float, 0x3f800000u),
        __builtin_bit_cast(float, 0x3f27dc51u),
        __builtin_bit_cast(float, 0x3eaa0299u)};
    static constexpr unsigned kExpDegree = 2u;
    static constexpr float kCoreCoefficients[] = {
        __builtin_bit_cast(float, 0x3ecb4b84u),
        __builtin_bit_cast(float, 0xbd8262c6u),
        __builtin_bit_cast(float, 0x3c095ac4u),
        __builtin_bit_cast(float, 0xba3e17aau),
        __builtin_bit_cast(float, 0x3818921cu),
        __builtin_bit_cast(float, 0xb5556e76u)};
    static constexpr unsigned kCoreDegree = 5u;
    static constexpr float kDecayCoefficients[] = {
        __builtin_bit_cast(float, 0xbe9f9763u),
        __builtin_bit_cast(float, 0x3cd3682eu),
        __builtin_bit_cast(float, 0x3b2ecf9fu),
        __builtin_bit_cast(float, 0x38c1d21bu)};
    static constexpr unsigned kDecayDegree = 3u;
    static constexpr float kNegativeTerminal = __builtin_bit_cast(float, 0xc1530000u);
    static constexpr float kDecayCoreBoundary = __builtin_bit_cast(float, 0xc0480000u);
    static constexpr float kPositiveIdentity = __builtin_bit_cast(float, 0x40310000u);
    static constexpr float kOriginSlope = __builtin_bit_cast(float, 0x3f000000u);
    static constexpr unsigned kBodySlots = 25u;
    static constexpr unsigned kReplayCapacity = 32u;
    static constexpr unsigned kPeakLive = 8u;
    static constexpr unsigned kOriginMask = 32767u;
    static constexpr unsigned kOriginValue = 32513u;
    static constexpr unsigned kOriginMantissa = 0u;
    static constexpr unsigned kOriginPatterns = 2u;
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_bf16_zone_affine_even_decay.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format.
inline bool bf16_dest_gelu() {
#if defined(TRISC_MATH) || defined(LLK_TRISC_MATH)
    return ckernel::math::src_zero_flag_srcb_fmt == static_cast<std::uint32_t>(DataFormat::Float16_b);
#else
    return false;
#endif
}
template <int ITERATIONS = 8>
inline void calculate_gelu_bf16() {
    ckernel::sfpu::bf16::calculate_zone_affine_even_decay<ckernel::sfpu::GeluBf16Config, ITERATIONS>();
}

}  // namespace ckernel::sfpu
