
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// reconfig_tile_shape is a genuine split of the unpack data-format reconfig: the FACE_ROW_MAJOR
// dim/stride block factored out so a caller can reprogram tile/face geometry without the format
// writes. This test asserts the split is behavior-identical:
//   run_idx 0: hw_configure(prev geometry), then a single FACE_ROW_MAJOR data-format reconfig to
//              (next format, next geometry) -- format + geometry together.
//   run_idx 1: hw_configure(prev geometry), then a format-only (IGNORE) reconfig to next format
//              (geometry stays prev), then _llk_unpack_reconfig_tile_shape_* to next geometry.
// Both paths must leave the same unpack state. Geometry changes between prev/next (num_faces 4 -> 2)
// so the X-dim / Z-dim(num_faces) / Z-stride writes are actually exercised. FormatConfig carries the
// prev formats in the unpack_A slots and the next formats in the pack slots.

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
    constexpr std::uint32_t PREV_SIZE       = 16 * 16 * 2;
    constexpr std::uint32_t NEXT_SIZE       = 16 * 16 * 4;
    // Geometry changes between prev and next so the tile-shape writes are exercised.
    constexpr std::uint32_t PREV_NUM_FACES  = 4;
    constexpr std::uint32_t NEXT_NUM_FACES  = 2;

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        prev_src, prev_src, prev_dst, prev_dst, FACE_R_DIM, FACE_R_DIM, PREV_NUM_FACES, PREV_NUM_FACES, PREV_SIZE, PREV_SIZE);

    if (params.CONFIGURE_TEST_RUN_IDX == 0)
    {
        // Format + geometry in one reconfig.
        _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, p_dim_stride_target::FACE_ROW_MAJOR, TO_FROM_INT8>(
            next_src, next_dst, NEXT_SIZE, FACE_R_DIM, NEXT_NUM_FACES);
        _llk_unpack_reconfig_data_format_srcb_impl_<is_fp32_dest_acc_en, p_dim_stride_target::FACE_ROW_MAJOR, TO_FROM_INT8>(
            next_src, next_dst, NEXT_SIZE, FACE_R_DIM, NEXT_NUM_FACES);
    }
    else
    {
        // Format only (geometry left at prev), then geometry-only via reconfig_tile_shape.
        _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, p_dim_stride_target::IGNORE, TO_FROM_INT8>(
            next_src, next_dst, NEXT_SIZE, FACE_R_DIM, NEXT_NUM_FACES);
        _llk_unpack_reconfig_data_format_srcb_impl_<is_fp32_dest_acc_en, p_dim_stride_target::IGNORE, TO_FROM_INT8>(
            next_src, next_dst, NEXT_SIZE, FACE_R_DIM, NEXT_NUM_FACES);

        _llk_unpack_reconfig_tile_shape_srca_(next_dst, FACE_R_DIM, NEXT_NUM_FACES);
        _llk_unpack_reconfig_tile_shape_srcb_(next_dst, FACE_R_DIM, NEXT_NUM_FACES);
    }

    ckernel::unpacker::are_unpackers_AB_configured_correctly(
        next_src, next_dst, next_src, next_dst, FACE_R_DIM, FACE_R_DIM, NEXT_NUM_FACES, NEXT_NUM_FACES);
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
