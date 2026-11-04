// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct HardmishBf16Config {
    static constexpr uint32_t kSlopeBits = 0x3f000000u;
    static constexpr uint32_t kInterceptBits = 0x3f800000u;
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_hardmish_bf16_simple_forward.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format, as BF16 for
// Float16_b and the block formats sharing its exponent.
inline bool bf16_dest_hardmish() {
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
// Each call records replay slots 0-21, overwriting what an init recorded there (matmul and reduce
// record from replay_buf_offset = 16).
template <int ITERATIONS = 32>
__attribute__((noinline)) inline void calculate_hardmish_bf16() {
    ckernel::sfpu::bf16::hardmish::calculate_simple_forward<ckernel::sfpu::HardmishBf16Config, ITERATIONS>();
}
inline void init_hardmish_bf16() {
    ckernel::sfpu::bf16::hardmish::init_simple_forward<ckernel::sfpu::HardmishBf16Config>();
}

}  // namespace ckernel::sfpu
