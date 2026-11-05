// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

namespace ckernel::sfpu {
struct TanhBf16Config {
    static constexpr uint32_t kDegree = 6u;
    // Fit: degree-6 polynomial on [0, 5], 1 segment, max pure (continuous) ULP 0.839.
    static constexpr uint32_t kCoefficientBits[] = {
        0x00000000u, 0x3f800000u, 0x3cbfe000u, 0xbef28000u, 0x3e8a1bd5u, 0xbd8035c9u, 0x3bb0e52du};
    static constexpr uint32_t kClampMaxBits = 0x3f800000u;
    static constexpr bool kIntrinsicExceptional = true;
    static constexpr bool kIntrinsicTerminal = true;
    static constexpr uint32_t kTerminalBoundBits = 0x40a00000u;
};
}  // namespace ckernel::sfpu

#include "ckernel_sfpu_tanh_bf16_cascade_signed_abs.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format, as BF16 for
// Float16_b and the block formats sharing its exponent.
inline bool bf16_dest_tanh() {
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
inline void calculate_tanh_bf16_body() {
    ckernel::sfpu::bf16::tanh::calculate_signed_abs_terminal<ckernel::sfpu::TanhBf16Config, ITERATIONS>();
}
// The constants stock's init also sets hold the kernel's own values for its call only.
// Each call records replay slots 0-25, overwriting what an init recorded there (matmul and reduce
// record from replay_buf_offset = 16).
template <int ITERATIONS = 32>
__attribute__((noinline)) inline void calculate_tanh_bf16() {
    sfpi::vConstFloatPrgm0 = __builtin_bit_cast(float, 0x3bb0e52du);
    sfpi::vConstFloatPrgm1 = __builtin_bit_cast(float, 0xbd8035c9u);
    sfpi::vConstFloatPrgm2 = __builtin_bit_cast(float, 0x3e8a1bd5u);
    calculate_tanh_bf16_body<ITERATIONS>();
    sfpi::vConstFloatPrgm0 = __builtin_bit_cast(float, 0x3bc0919eu);
    sfpi::vConstFloatPrgm1 = __builtin_bit_cast(float, 0xbd887f48u);
    sfpi::vConstFloatPrgm2 = __builtin_bit_cast(float, 0x3e905782u);
}
// The call sets the constants it holds and puts stock's back, so the init leaves them to stock's.
inline void init_tanh_bf16() {}

}  // namespace ckernel::sfpu
