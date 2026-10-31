// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_sfpu_binary_max_min.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"

namespace ckernel {

/*
 * Binary max/min dispatch.
 *
 * The SFPLOADMACRO kernels end every call with a drain (float: three SFPNOPs
 * for the scheduled swap/store; int32: a replay-buffer record plus a two-row
 * tail), so calling them once per face pays that four times per tile. For a
 * full-tile call (VectorMode::RC) the kernel is invoked once for all 32 rows
 * under VectorMode::RC_custom: the store-side ADDR_MOD dest increment carries
 * the dest RWC across the face boundaries exactly where the per-face
 * SETRWC(CR_D += 8) x2 would have put it, so every load/store address is
 * unchanged and the drain is issued once per tile. The other vector modes
 * keep the per-face path.
 */
template <DstSync DST_SYNC, bool DST_ACCUM, bool IS_MAX_OP>
inline void llk_math_eltwise_binary_sfpu_max_min(
    std::uint32_t dst_index_in0, std::uint32_t dst_index_in1, std::uint32_t dst_index_out, VectorMode vector_mode) {
    if (vector_mode == VectorMode::RC) {
        SFPU_BINARY_CALL(
            DST_SYNC,
            DST_ACCUM,
            calculate_binary_max_min,
            (IS_MAX_OP, 32 /* ITERATIONS: whole tile */),
            dst_index_in0,
            dst_index_in1,
            dst_index_out,
            VectorMode::RC_custom);
    } else {
        SFPU_BINARY_CALL(
            DST_SYNC,
            DST_ACCUM,
            calculate_binary_max_min,
            (IS_MAX_OP, 8 /* ITERATIONS: one face */),
            dst_index_in0,
            dst_index_in1,
            dst_index_out,
            vector_mode);
    }
}

template <DstSync DST_SYNC, bool DST_ACCUM, bool IS_MAX_OP, bool IS_UNSIGNED>
inline void llk_math_eltwise_binary_sfpu_max_min_int32(
    std::uint32_t dst_index_in0, std::uint32_t dst_index_in1, std::uint32_t dst_index_out) {
    SFPU_BINARY_CALL(
        DST_SYNC,
        DST_ACCUM,
        calculate_binary_max_min_int32,
        (IS_MAX_OP, IS_UNSIGNED, 32 /* ITERATIONS: whole tile */),
        dst_index_in0,
        dst_index_in1,
        dst_index_out,
        VectorMode::RC_custom);
}

}  // namespace ckernel
