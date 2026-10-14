
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// reconfig_tile_shape is a genuine split of the unpack data-format reconfig: the tile-size + face
// geometry (x-dim, num_faces) writes factored out so a caller can reprogram tile shape without the
// format writes. This test asserts the split is behavior-identical:
//   run_idx 0: hw_configure(prev geometry), then a single FACE_ROW_MAJOR data-format reconfig to
//              (next format, next geometry) -- format + shape together.
//   run_idx 1: hw_configure(prev geometry), then a format-only (IGNORE) reconfig to next format
//              (shape stays prev), then _llk_unpack_reconfig_tile_shape_* to next shape.
// Both paths must leave the same unpack state. Shape changes between prev/next in BOTH face_r_dim
// (16 -> 8, so the observable tile-descriptor X-dim changes) and num_faces (4 -> 2, the Z-dim), so a
// regression in either geometry write is caught by the state compare. FormatConfig carries the prev
// formats in the unpack_A slots and the next formats in the pack slots.

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const std::uint32_t prev_src = (std::uint32_t)params.formats.unpack_A_src;
    const std::uint32_t prev_dst = (std::uint32_t)params.formats.unpack_A_dst;
    const std::uint32_t next_src = (std::uint32_t)params.formats.pack_src;
    const std::uint32_t next_dst = (std::uint32_t)params.formats.pack_dst;

    // Distinct prev/next tile sizes; both paths need to hit NEXT_SIZE.
    constexpr std::uint32_t PREV_SIZE      = 16 * 16 * 2;
    constexpr std::uint32_t NEXT_SIZE      = 16 * 16 * 4;
    // Shape changes between prev and next in BOTH dims so the X-dim and Z-dim writes are exercised.
    constexpr std::uint32_t PREV_FACE_R    = FACE_R_DIM; // 16
    constexpr std::uint32_t NEXT_FACE_R    = 8;          // -> observable tile-descriptor X-dim change
    constexpr std::uint32_t PREV_NUM_FACES = 4;
    constexpr std::uint32_t NEXT_NUM_FACES = 2;

    // TO_FROM_INT8 lands in the impl's third template param, which is skip_int8 (inverted polarity: the
    // int8 derivation runs only when skip_int8 is false). Both runs use the same value, so it does not
    // affect the equivalence being asserted here.
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        prev_src, prev_src, prev_dst, prev_dst, PREV_FACE_R, PREV_FACE_R, PREV_NUM_FACES, PREV_NUM_FACES, PREV_SIZE, PREV_SIZE);

    if (params.CONFIGURE_TEST_RUN_IDX == 0)
    {
        // Format + shape in one reconfig.
        _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, p_dim_stride_target::FACE_ROW_MAJOR, /*skip_int8=*/TO_FROM_INT8>(
            next_src, next_dst, NEXT_SIZE, NEXT_FACE_R, NEXT_NUM_FACES);
        _llk_unpack_reconfig_data_format_srcb_impl_<is_fp32_dest_acc_en, p_dim_stride_target::FACE_ROW_MAJOR, /*skip_int8=*/TO_FROM_INT8>(
            next_src, next_dst, NEXT_SIZE, NEXT_FACE_R, NEXT_NUM_FACES);
    }
    else
    {
        // Format only (shape left at prev), then shape-only via reconfig_tile_shape.
        _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, p_dim_stride_target::IGNORE, /*skip_int8=*/TO_FROM_INT8>(
            next_src, next_dst, NEXT_SIZE, NEXT_FACE_R, NEXT_NUM_FACES);
        _llk_unpack_reconfig_data_format_srcb_impl_<is_fp32_dest_acc_en, p_dim_stride_target::IGNORE, /*skip_int8=*/TO_FROM_INT8>(
            next_src, next_dst, NEXT_SIZE, NEXT_FACE_R, NEXT_NUM_FACES);

        _llk_unpack_reconfig_tile_shape_srca_(NEXT_SIZE, NEXT_FACE_R, NEXT_NUM_FACES);
        _llk_unpack_reconfig_tile_shape_srcb_(NEXT_SIZE, NEXT_FACE_R, NEXT_NUM_FACES);
    }

    ckernel::unpacker::are_unpackers_AB_configured_correctly(
        next_src, next_dst, next_src, next_dst, NEXT_FACE_R, NEXT_FACE_R, NEXT_NUM_FACES, NEXT_NUM_FACES);
}

#endif

#ifdef LLK_TRISC_MATH

void run_kernel(RUNTIME_PARAMETERS params)
{
}

#endif

#ifdef LLK_TRISC_PACK

void run_kernel(RUNTIME_PARAMETERS params)
{
}

#endif
