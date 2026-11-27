// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

#include "llk_defs.h"

// Programs with this many circular buffers sit near the kernel config buffer limit (hybrid_routed_expert clears it by a few
// hundred bytes on main), so they take main's code size. Counted from unpack_src_format, which every TRISC sees.
namespace ckernel {

inline constexpr std::uint32_t llk_size_mode_min_cbs = 36;  // hybrid_routed_expert has 37-39, no other kernel more than 33
// The chunk gated delta rule kernels (chunk_gdn_prep, chunk_gated_delta_rule) map CBs 0-31 at HiFi4 with a 32-bit DEST;
// with the device profiler on, chunk_gdn_prep clears the limit by 1 KB on main
inline constexpr std::uint32_t llk_size_mode_gdn_cbs = 32;

template <std::size_t n, typename T>
constexpr std::uint32_t llk_configured_cbs(const T (&formats)[n]) {
    std::uint32_t count = 0;
    for (std::size_t i = 0; i < n; i++) {
        count += formats[i] != 255;
    }
    return count;
}

template <std::size_t n, typename T>
constexpr bool llk_gdn_kernel(const T (&formats)[n], const MathFidelity fidelity, const bool fp32_dest) {
    return llk_configured_cbs(formats) == llk_size_mode_gdn_cbs && fidelity == MathFidelity::HiFi4 && fp32_dest;
}

}  // namespace ckernel

// Use after the JIT descriptors (chlkc_descriptors.h), where unpack_src_format is defined. The ring SDPA kernels
// (ring_joint_sdpa, sliding_ring_sdpa: those given SLIDING_HALO_SLOT_COUNT) also sit 1 KB under the limit on main.
#define LLK_SIZE_MODE_GDN (::ckernel::llk_gdn_kernel(unpack_src_format, MATH_FIDELITY, DST_ACCUM_MODE))
#if defined(SLIDING_HALO_SLOT_COUNT)
#define LLK_SIZE_MODE true
#else
#define LLK_SIZE_MODE \
    (::ckernel::llk_configured_cbs(unpack_src_format) >= ::ckernel::llk_size_mode_min_cbs || LLK_SIZE_MODE_GDN)
#endif
