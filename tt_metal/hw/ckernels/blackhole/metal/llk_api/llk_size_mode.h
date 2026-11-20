// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

// Programs with this many circular buffers sit near the kernel config buffer limit (hybrid_routed_expert clears it by a few
// hundred bytes on main), so they take main's code size. Counted from unpack_src_format, which every TRISC sees.
namespace ckernel {

inline constexpr std::uint32_t llk_size_mode_min_cbs = 36;  // hybrid_routed_expert has 37-39, no other kernel more than 33

template <std::size_t n, typename T>
constexpr std::uint32_t llk_configured_cbs(const T (&formats)[n]) {
    std::uint32_t count = 0;
    for (std::size_t i = 0; i < n; i++) {
        count += formats[i] != 255;
    }
    return count;
}

// Whether any CB of the kernel holds Int32 or UInt32 data (formats 8 and 24)
template <std::size_t n, typename T>
constexpr bool llk_int32_cbs(const T (&formats)[n]) {
    for (std::size_t i = 0; i < n; i++) {
        if (formats[i] == 8 || formats[i] == 24) {
            return true;
        }
    }
    return false;
}

}  // namespace ckernel

// Use after the JIT descriptors (chlkc_descriptors.h), where unpack_src_format is defined.
#define LLK_SIZE_MODE (::ckernel::llk_configured_cbs(unpack_src_format) >= ::ckernel::llk_size_mode_min_cbs)

// Unpack to dest takes main's per-tile handshake (the math RISC takes its own post back) at 16-bit DEST and in kernels with an
// Int32 or UInt32 CB (pow_int 32x32 +0.6 %, fp32 sort with its UInt32 index +0.15 % on the other one); Float32-only kernels keep
// the handshake where unpack takes the math post (fp32 fft, welford layernorm, cumprod +1 to +2 % without it)
#define LLK_U2D_INT_KERNEL (::ckernel::llk_int32_cbs(unpack_src_format))
#define LLK_U2D_MAIN_HANDSHAKE (!DST_ACCUM_MODE || LLK_U2D_INT_KERNEL)
