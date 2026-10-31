// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "sfpi.h"
namespace ckernel::sfpu {
struct AcosBf16Config {
    static constexpr uint32_t kDegree = 5u;
    static constexpr bool kSafeInput = true;
    // Fit: minimax degree-5 polynomial on [0, 1], 1 segment, max pure (continuous) ULP 0.5.
    static constexpr uint32_t kCoefficientBits[] = {
        0x3fc90fd2u, 0xbe5ba9b0u, 0x3db4017au, 0xbd386186u, 0x3c9f23c6u, 0xbb8f523eu};
    inline sfpi::vFloat operator[](uint32_t index) const {
        return sfpi::as<sfpi::vFloat>(sfpi::vUInt(kCoefficientBits[index]));
    }
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_acos_bf16_sqrt_factored.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format, as BF16 for
// Float16_b and the block formats sharing its exponent.
inline bool bf16_dest_acos() {
#if defined(TRISC_MATH) || defined(LLK_TRISC_MATH)
    const std::uint32_t format = ckernel::math::src_zero_flag_srcb_fmt;
    return format == static_cast<std::uint32_t>(DataFormat::Float16_b) ||
           format == static_cast<std::uint32_t>(DataFormat::Bfp8_b) ||
           format == static_cast<std::uint32_t>(DataFormat::Bfp4_b) ||
           format == static_cast<std::uint32_t>(DataFormat::Bfp2_b);
#else
    return false;
#endif
}
template <int ITERATIONS = 32>
inline void calculate_acos_bf16() {
    ckernel::sfpu::bf16::acos::calculate_sqrt_factored<ckernel::sfpu::AcosBf16Config, ITERATIONS>();
}

}  // namespace ckernel::sfpu
