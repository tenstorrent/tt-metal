// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

namespace ckernel::sfpu {
struct ErfinvBf16Config {
    static constexpr uint32_t kOuterDegree = 3u;
    static constexpr uint32_t kLogCoefficientBits[] = {0xbf00677fu, 0x3eb0bfb5u, 0xbe507697u};
    static constexpr uint32_t kOuterCoefficientBits[] = {0x3f62dfc5u, 0x3e734869u, 0x3bc58be5u, 0xba96dc26u};

    struct TerminalAction {
        uint8_t direction;
        uint32_t bound_bits;
        uint8_t inclusive;
        uint8_t return_class;
    };
};
}  // namespace ckernel::sfpu

#include "ckernel_sfpu_erfinv_bf16_log_square_factorized_odd.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format, as BF16 for
// Float16_b and the block formats sharing its exponent.
inline bool bf16_dest_erfinv() {
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
// Each call records replay slots 0-31, overwriting what an init recorded there (matmul and reduce
// record from replay_buf_offset = 16).
template <int ITERATIONS = 8>
__attribute__((noinline)) inline void calculate_erfinv_bf16() {
    ckernel::sfpu::bf16::erfinv::calculate<ckernel::sfpu::ErfinvBf16Config, ITERATIONS>();
}

}  // namespace ckernel::sfpu
