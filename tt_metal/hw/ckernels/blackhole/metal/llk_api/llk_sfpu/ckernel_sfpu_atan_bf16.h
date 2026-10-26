// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ckernel_sfpu_bf16_sfpi_isa.h"
#include <cstdint>
namespace ckernel::sfpu {
struct AtanBf16Config {
    static constexpr unsigned kDegree = 0x00000004u;
    static constexpr unsigned kComplementBits = 0x3fc90fdbu;
    static constexpr unsigned kBodySlots = 0x00000016u;
    static constexpr unsigned kMacroSequenceBits = 0x63550087u;
    static constexpr unsigned kCoefficientBits[] = {0x00000000u, 0x3f800000u, 0x3aa8eb14u, 0xbebd4004u, 0x3e1daa80u};
    // Fit: minimax degree-4 polynomial on [0, 1], 1 segment, max pure (continuous) ULP 0.55.
    static constexpr float kCoefficients[] = {
        __builtin_bit_cast(float, 0x00000000u),
        __builtin_bit_cast(float, 0x3f800000u),
        __builtin_bit_cast(float, 0x3aa8eb14u),
        __builtin_bit_cast(float, 0xbebd4004u),
        __builtin_bit_cast(float, 0x3e1daa80u)};
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_bf16_reciprocal_complement.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format.
inline bool bf16_dest_atan() {
#if defined(TRISC_MATH) || defined(LLK_TRISC_MATH)
    return ckernel::math::src_zero_flag_srcb_fmt == static_cast<std::uint32_t>(DataFormat::Float16_b);
#else
    return false;
#endif
}
template <int ITERATIONS = 8>
inline void calculate_atan_bf16() {
    ckernel::sfpu::bf16::calculate_reciprocal_complement<ckernel::sfpu::AtanBf16Config, ITERATIONS>();
}
inline void init_atan_bf16() {
    if (bf16_dest_atan()) {
        ckernel::sfpu::bf16::init_reciprocal_complement<ckernel::sfpu::AtanBf16Config>();
    }
}

}  // namespace ckernel::sfpu
