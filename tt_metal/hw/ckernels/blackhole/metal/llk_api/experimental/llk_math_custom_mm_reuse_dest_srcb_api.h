// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "llk_math_common_api.h"
#include "experimental/llk_math_custom_mm_reuse_dest_srcb.h"
#include "sanitizer/api.h"

/*************************************************************************
 * LLK CUSTOM_MM_REUSE_DEST_SRCB
 *
 * Second matmul of a fused chain: SrcB is moved out of DEST (where the
 * preceding custom_mm left its output) instead of being unpacked from L1.
 * Only SrcA (the weights) is unpacked.
 *
 * Uses llk_math_custom_mm_reuse_dest_srcb.h as the low-level implementation.
 *************************************************************************/

inline void llk_math_custom_mm_reuse_dest_srcb_replay_init() {
    SAN_HOOK(unsupported());
    _llk_math_custom_mm_reuse_dest_srcb_replay_init_();
}

template <bool load_replay = true>
inline void llk_math_custom_mm_reuse_dest_srcb_init() {
    SAN_HOOK(unsupported());
    _llk_math_custom_mm_reuse_dest_srcb_init_<load_replay>();
}

template <std::uint32_t in0_tile_r_dim>
inline void llk_math_custom_mm_reuse_dest_srcb(
    const std::uint32_t src_index,
    const std::uint32_t dst_index,
    const std::uint32_t kt_dim,
    const std::uint32_t nt_dim,
    const std::uint32_t src_tile_stride) {
    SAN_HOOK(unsupported());
    _llk_math_custom_mm_reuse_dest_srcb_<in0_tile_r_dim>(src_index, dst_index, kt_dim, nt_dim, src_tile_stride);
}
