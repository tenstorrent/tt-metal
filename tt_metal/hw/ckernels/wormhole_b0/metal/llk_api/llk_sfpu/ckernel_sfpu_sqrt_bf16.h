// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct SqrtBf16Config {
    static constexpr uint32_t kKind = 0u;
    static constexpr uint32_t kMagic = 0x5f1110a0u;
    static constexpr uint32_t kC1Bits = 0x401214c9u;
    static constexpr uint32_t kC2Bits = 0x40103626u;
    static constexpr uint32_t kBodySlots = 26u;
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_sqrt_bf16_newton_root.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format, as BF16 for
// Float16_b and the block formats sharing its exponent.
inline bool bf16_dest_sqrt() {
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
// Each call records replay slots 0-15, below replay_buf_offset = 16, where matmul and
// reduce record.
template <int ITERATIONS = 32>
__attribute__((noinline)) inline void calculate_sqrt_bf16() {
    ckernel::sfpu::bf16::sqrt::calculate_newton_root<ckernel::sfpu::SqrtBf16Config, ITERATIONS>();
}
inline void init_sqrt_bf16() { ckernel::sfpu::bf16::sqrt::init_newton_root<ckernel::sfpu::SqrtBf16Config>(); }

}  // namespace ckernel::sfpu
