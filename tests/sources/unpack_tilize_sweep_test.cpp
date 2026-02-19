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

#include "ckernel_template.h"
#include "llk_unpack_common.h"
#include "llk_unpack_tilize.h"
#include "params.h"

void run_kernel(const volatile struct RuntimeParams *params)
{
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_src, formats.unpack_src, formats.unpack_dst, formats.unpack_dst, FACE_R_DIM, FACE_R_DIM, params->num_faces, params->num_faces);
    _llk_unpack_configure_stoch_rnd_<STOCHASTIC_RND>();

    // Initialize tilize unpacker
    _llk_unpack_tilize_init_(
        formats.unpack_src,
        formats.unpack_dst,
        BLOCK_CT_DIM,
        FACE_R_DIM,
        params->NARROW_TILE // narrow_tile disabled for now
    );

    std::uint32_t read_offset = 0;

    const std::uint32_t block_ct_dim = is_blackhole ? 0 : BLOCK_CT_DIM;
    const std::uint32_t num_faces    = is_blackhole ? 4 : params->num_faces;

    // Main tilize loop - handle different tile configurations
    for (std::uint32_t row = 0; row < BLOCK_RT_DIM; ++row)
    {
        std::uint32_t tile_row_addr = L1_ADDRESS(params->buffer_A[read_offset]);
        for (std::uint32_t col = 0; col < BLOCK_CT_DIM; ++col)
        {
            _llk_unpack_tilize_(
                tile_row_addr,
                col,
                formats.unpack_src,
                formats.unpack_dst,
                block_ct_dim,
                FACE_R_DIM,
                num_faces,
                false // narrow_tile disabled for now
            );
        }
        read_offset += BLOCK_CT_DIM;
    }
}

#endif

const bool TILIZE = true;

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "params.h"

using namespace ckernel;
const bool is_int_fpu_en = false;

void run_kernel(const volatile struct RuntimeParams *params)
{
    // Copy srca to dest with tilize flag
#ifdef ARCH_BLACKHOLE
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, TILIZE, is_int_fpu_en>(params->num_faces, formats.math);
#else
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, is_int_fpu_en>(params->num_faces, formats.math);
#endif

    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    for (int i = 0; i < params->TILE_CNT; ++i)
    {
        LLK_ASSERT(
            (i < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()), "Block tile index exceeds maximum destination tiles");
#ifdef ARCH_BLACKHOLE
        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            i, formats.math, formats.math, params->num_faces);
#else
        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            i, formats.math, formats.math);
#endif
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
    const bool UNTILIZE             = false;
    const std::uint32_t DATUM_COUNT = 16 * 16 * params->num_faces;

#ifdef ARCH_BLACKHOLE
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, UNTILIZE, TILIZE>(formats.pack_src, formats.pack_dst, DATUM_COUNT, FACE_R_DIM, TILE_C_DIM, params->num_faces);
    _llk_pack_init_<UNTILIZE, false, TILIZE>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, params->num_faces);
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
#else
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, UNTILIZE>(formats.pack_src, formats.pack_dst, DATUM_COUNT, FACE_R_DIM, params->num_faces);
    _llk_pack_init_<UNTILIZE, false>(formats.pack_dst, FACE_R_DIM, params->num_faces);
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en, UNTILIZE>();
#endif

    _llk_packer_wait_for_math_done_();
    for (int i = 0; i < params->TILE_CNT; ++i)
    {
        LLK_ASSERT(
            (i < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()), "Block tile index exceeds maximum destination tiles");
        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, UNTILIZE>(i, L1_ADDRESS(params->buffer_Res[i]));
    }
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif
