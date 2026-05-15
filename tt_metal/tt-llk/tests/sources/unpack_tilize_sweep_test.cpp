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
#include "llk_lib_unpack_wrappers.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, params.num_faces, params.num_faces);
    _llk_unpack_configure_stoch_rnd_<STOCHASTIC_RND>();

    // Initialize tilize unpacker
    _llk_unpack_tilize_init_wrapper_(formats.unpack_A_src, formats.unpack_A_dst, params.BLOCK_CT_DIM, FACE_R_DIM, params.NARROW_TILE);

    std::uint32_t read_offset = 0;

    const std::uint32_t block_ct_dim = _llk_unpack_tilize_block_ct_dim_wrapper_(params.BLOCK_CT_DIM);

    const std::uint32_t num_faces = _llk_unpack_tilize_num_faces_wrapper_(params.num_faces);

    // Main tilize loop - handle different tile configurations
    for (std::uint32_t row = 0; row < params.BLOCK_RT_DIM; ++row)
    {
        std::uint32_t tile_row_addr = L1_ADDRESS(params.buffer_A[read_offset]);
        for (std::uint32_t col = 0; col < params.BLOCK_CT_DIM; ++col)
        {
            _llk_unpack_tilize_wrapper_(
                tile_row_addr,
                col,
                formats.unpack_A_src,
                formats.unpack_A_dst,
                block_ct_dim,
                FACE_R_DIM,
                num_faces,
                false // narrow_tile disabled for now
            );
        }
        read_offset += params.BLOCK_CT_DIM;
    }
}

#endif

const bool TILIZE = true;

#ifdef LLK_TRISC_MATH

#include "llk_lib_math_wrappers.h"
#include "params.h"

using namespace ckernel;
const bool is_int_fpu_en = false;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    // Copy srca to dest with tilize flag
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, TILIZE, is_int_fpu_en>(
        params.num_faces, formats.math);

    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
    {
        LLK_ASSERT(
            (i < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()), "Block tile index exceeds maximum destination tiles");
        _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            i, formats.math, formats.math, params.num_faces);
    }
    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const bool UNTILIZE             = false;
    const std::uint32_t DATUM_COUNT = 16 * 16 * params.num_faces;

    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, UNTILIZE, TILIZE>(
        formats.pack_src, formats.pack_dst, DATUM_COUNT, FACE_R_DIM, TILE_C_DIM, params.num_faces);
    _llk_pack_init_wrapper_<UNTILIZE, false /* zero_output */, TILIZE>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, params.num_faces);
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    _llk_packer_wait_for_math_done_();
    for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
    {
        LLK_ASSERT(
            (i < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()), "Block tile index exceeds maximum destination tiles");
        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, UNTILIZE>(i, L1_ADDRESS(params.buffer_Res[i]));
    }
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif
