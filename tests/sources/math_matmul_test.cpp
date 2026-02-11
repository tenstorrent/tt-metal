// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"

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
        formats.unpack_src,
        formats.unpack_src,
        formats.unpack_dst,
        formats.unpack_dst,
        FACE_R_DIM,
        FACE_R_DIM,
        params->num_faces_A,
        params->num_faces_B,
        TILE_SIZE_UNPACK_A,
        TILE_SIZE_UNPACK_B);
    _llk_unpack_AB_matmul_init_<>(
        params->UNPACK_TRANSPOSE_FACES,
        params->CT_DIM,
        params->RT_DIM,
        params->KT_DIM,
        FACE_R_DIM,
        FACE_R_DIM,
        params->num_faces_A,
        params->num_faces_B,
        params->PARTIAL_FACE_A,
        params->PARTIAL_FACE_B);
    for (std::uint32_t j = 0; j < params->KT_DIM; j++)
    {
        _llk_unpack_AB_matmul_<>(
            L1_ADDRESS(buffer_A[0]),
            L1_ADDRESS(buffer_B[0]),
            j,
            j * params->CT_DIM,
            TILE_SIZE_UNPACK_A,
            TILE_SIZE_UNPACK_B,
            params->PARTIAL_FACE_A,
            params->PARTIAL_FACE_B,
            params->CT_DIM,
            params->RT_DIM,
            params->KT_DIM);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_matmul.h"
#include "params.h"

void run_kernel(const volatile struct RuntimeParams *params)
{
    _llk_math_matmul_init_<MATH_FIDELITY, THROTTLE_LEVEL>(
        params->in0_tile_r_dim,
        params->in0_tile_c_dim,
        params->in1_tile_r_dim,
        params->in1_tile_c_dim,
        params->PARTIAL_FACE_MATH,
        params->UNPACK_TRANSPOSE_FACES,
        params->CT_DIM,
        params->RT_DIM);
    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_wait_for_dest_available_<dest_sync>();
    LLK_ASSERT(
        (get_dest_max_matmul_tiles(params->DST_INDEX, params->CT_DIM, params->RT_DIM) <
         get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
        "Block tile index exceeds maximum destination tiles for matmul");
    for (std::uint32_t j = 0; j < params->KT_DIM; j++)
    {
        _llk_math_matmul_<MATH_FIDELITY, THROTTLE_LEVEL>(params->DST_INDEX, params->CT_DIM, params->RT_DIM);
    }

    _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(const volatile struct RuntimeParams *params)
{
#ifdef ARCH_BLACKHOLE
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, false, false>(
        formats.pack_src, formats.pack_dst, TILE_SIZE_PACK, FACE_R_DIM, TILE_C_DIM, params->num_faces, params->PARTIAL_FACE_PACK);
    _llk_pack_init_<false, false, false>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, params->num_faces, false /* partial_face parameter is unused on BH */);
    _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();
#else
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, false>(
        formats.pack_src, formats.pack_dst, TILE_SIZE_PACK, FACE_R_DIM, params->num_faces, params->PARTIAL_FACE_PACK);
    _llk_pack_init_<false, false>(formats.pack_dst, FACE_R_DIM, params->num_faces, params->PARTIAL_FACE_PACK);
    _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en, false>();
#endif
    _llk_packer_wait_for_math_done_();
    for (int i = 0; i < params->TILE_CNT; i++)
    {
        const std::uint32_t tile_index = params->DST_INDEX + i;
        LLK_ASSERT((tile_index < get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()), "tile_index exceeds max dest tiles");
        _llk_pack_<dest_sync, is_fp32_dest_acc_en, false>(tile_index, L1_ADDRESS(buffer_Res[i]));
    }
    _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
}

#endif
