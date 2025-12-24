// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"

// Globals
uint32_t unp_cfg_context          = 0;
uint32_t pack_sync_tile_dst_ptr   = 0;
uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel()
{
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_src, formats.unpack_src, formats.unpack_dst, formats.unpack_dst, FACE_R_DIM, FACE_R_DIM, 4 /* num_faces */, 4 /* num_faces */);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0, 0, FACE_R_DIM, 4, formats.unpack_src, formats.unpack_dst);
    for (int i = 0; i < TILE_CNT; ++i)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(buffer_A[i]), formats.unpack_src, formats.unpack_dst);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "params.h"

using namespace ckernel;

void run_kernel()
{
    const bool is_int_fpu_en = false;

// copy srca to dest
#ifdef ARCH_BLACKHOLE
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, is_int_fpu_en>(4, formats.math);
#else
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, is_int_fpu_en>(4, formats.math);
#endif
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_(formats.math, formats.math);
#ifdef ARCH_BLACKHOLE
    _llk_math_reconfig_remap_(true);
#endif
    _llk_math_wait_for_dest_available_<DST_SYNC>();
    for (int i = 0; i < TILE_CNT; ++i)
    {
        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(i, formats.math, formats.math);
    }
    _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel()
{
    const bool UNTILIZE = true;

#ifdef ARCH_BLACKHOLE
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, UNTILIZE, false>(formats.pack_src, formats.pack_dst, 16 * 16 * 4 /* tile_size */);
    _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en>();
    _llk_pack_untilize_init_<BLOCK_CT_DIM, FULL_CT_DIM>(formats.pack_src, formats.pack_dst, FACE_R_DIM, 4 /* num_faces */);
#else
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, UNTILIZE>(formats.pack_src, formats.pack_dst, 16 * 16 * 4 /* tile_size */);
    _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en, UNTILIZE>();
    _llk_pack_untilize_init_<BLOCK_CT_DIM, FULL_CT_DIM>(formats.pack_dst, FACE_R_DIM, 4 /* num_faces */);
#endif

    _llk_packer_wait_for_math_done_();

    // Loop over all tiles vertically.
    for (uint32_t rt = 0; rt < FULL_RT_DIM; rt++)
    {
        _llk_pack_untilize_<BLOCK_CT_DIM, FULL_CT_DIM>(
            L1_ADDRESS(buffer_Res[rt * BLOCK_CT_DIM]),
            formats.pack_dst,
            FACE_R_DIM,
            4,
            rt * FULL_CT_DIM // tile_dst_rt_offset - offset by full row width
        );
    }
    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}

#endif
