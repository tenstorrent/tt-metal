// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct AbsBf16Config {
    static constexpr uint32_t kLoadFormat = 0u;
    static constexpr uint32_t kAbsMode = 1u;
    static constexpr uint32_t kStoreFormat = 0u;
    static constexpr uint32_t kBodySlots = 6u;
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_abs_bf16_abs_value.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format, as BF16 for
// Float16_b and the block formats sharing its exponent.
inline bool bf16_dest_abs() {
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
// Each call records replay slots 0-5, below replay_buf_offset = 16, where matmul and
// reduce record.
template <int ITERATIONS = 32>
__attribute__((noinline)) inline void calculate_abs_bf16() {
    ckernel::sfpu::bf16::abs::calculate_abs_value<ckernel::sfpu::AbsBf16Config, ITERATIONS>();
}
inline void init_abs_bf16() { ckernel::sfpu::bf16::abs::init_abs_value<ckernel::sfpu::AbsBf16Config>(); }

}  // namespace ckernel::sfpu
