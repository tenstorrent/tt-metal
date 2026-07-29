
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Configuring a format directly (run_idx 0) must leave the same unpack state as
// reconfiguring srcA/srcB to it from another format (run_idx 1). FormatConfig carries
// the prev formats in the unpack_A slots and the next formats in the pack slots.

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
    constexpr std::uint32_t PREV_SIZE = 16 * 16 * 2;
    constexpr std::uint32_t NEXT_SIZE = 16 * 16 * 4;
    constexpr std::uint32_t num_faces = 4;

    if (params.CONFIGURE_TEST_RUN_IDX == 0)
    {
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            next_src, next_src, next_dst, next_dst, FACE_R_DIM, FACE_R_DIM, num_faces, num_faces, NEXT_SIZE, NEXT_SIZE);
    }
    else
    {
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            prev_src, prev_src, prev_dst, prev_dst, FACE_R_DIM, FACE_R_DIM, num_faces, num_faces, PREV_SIZE, PREV_SIZE);

        _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, p_dim_stride_target::IGNORE, TO_FROM_INT8>(
            next_src, next_dst, NEXT_SIZE, FACE_R_DIM, num_faces);
        _llk_unpack_reconfig_data_format_srcb_impl_<is_fp32_dest_acc_en, p_dim_stride_target::IGNORE, TO_FROM_INT8>(
            next_src, next_dst, NEXT_SIZE, FACE_R_DIM, num_faces);
    }

    ckernel::unpacker::are_unpackers_AB_configured_correctly(next_src, next_dst, next_src, next_dst, FACE_R_DIM, FACE_R_DIM, num_faces, num_faces);
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
