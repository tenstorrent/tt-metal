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

}  // namespace ckernel

// Use after the JIT descriptors (chlkc_descriptors.h), where unpack_src_format is defined. The ring SDPA kernels
// (ring_joint_sdpa, sliding_ring_sdpa: those given SLIDING_HALO_SLOT_COUNT) also sit 1 KB under the limit on main.
#if defined(SLIDING_HALO_SLOT_COUNT)
#define LLK_SIZE_MODE true
#else
#define LLK_SIZE_MODE (::ckernel::llk_configured_cbs(unpack_src_format) >= ::ckernel::llk_size_mode_min_cbs)
#endif
