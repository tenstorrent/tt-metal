// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_AB_matmul.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(const volatile struct RuntimeParams *params)
{
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src,
        formats.unpack_B_src,
        formats.unpack_A_dst,
        formats.unpack_B_dst,
        FACE_R_DIM,
        FACE_R_DIM,
        params->num_faces_A,
        params->num_faces_B,
        TILE_SIZE_UNPACK_A,
        TILE_SIZE_UNPACK_B);
    _llk_unpack_AB_matmul_init_<>(
        0 /* transpose */,
        params->CT_DIM,
        params->RT_DIM,
        params->KT_DIM,
        FACE_R_DIM,
        FACE_R_DIM,
        4 /* unpA_num_faces */,
        4 /* unpB_num_faces */,
        false /* unpA_partial_face */,
        false /* unpB_partial_face */);
    for (std::uint32_t j = 0; j < params->KT_DIM; j++)
    {
        _llk_unpack_AB_matmul_<>(
            L1_ADDRESS(params->buffer_A[0]),
            L1_ADDRESS(params->buffer_B[0]),
            j,
            j * params->CT_DIM,
            TILE_SIZE_UNPACK_A,
            TILE_SIZE_UNPACK_B,
            false /* unpA_partial_face */,
            false /* unpB_partial_face */,
            params->CT_DIM,
            params->RT_DIM,
            params->KT_DIM);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_matmul_custom_no_mop.h"
#include "llk_math_common.h"
#include "params.h"

void run_kernel(const volatile struct RuntimeParams *params)
{
    _llk_math_matmul_init_no_mop_<MATH_FIDELITY>(
        TILE_R_DIM, TILE_C_DIM, TILE_R_DIM, TILE_C_DIM, false /* partial_face */, 0 /* transpose */, params->CT_DIM, params->RT_DIM);
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    for (std::uint32_t j = 0; j < params->KT_DIM; j++)
    {
        _llk_math_matmul_no_mop_<MATH_FIDELITY>(0 /* dst_index */, params->CT_DIM, params->RT_DIM);
    }
    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(const volatile struct RuntimeParams *params)
{
#ifdef ARCH_BLACKHOLE
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, false /* untilize */, false /* tilize */>(formats.pack_src, formats.pack_dst, TILE_SIZE_PACK);
    _llk_pack_init_<false /* untilize */, false /* zero_output */, false /* tilize */>(formats.pack_dst);
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
#else
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, false /* untilize */>(formats.pack_src, formats.pack_dst, TILE_SIZE_PACK);
    _llk_pack_init_<false /* untilize */, false /* zero_output */>(formats.pack_dst);
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en, false /* untilize */>();
#endif
    _llk_packer_wait_for_math_done_();
    for (int i = 0; i < params->TILE_CNT; i++)
    {
        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, false /* untilize */>(i, L1_ADDRESS(params->buffer_Res[i]));
    }
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif
