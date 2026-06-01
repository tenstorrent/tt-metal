// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/* Eltwise unary datacopy with llk_dprint::dprint_tensix_dest_reg called on MATH
   between datacopy and dest_section_done. Quasar is unsupported. */

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"

#if defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE)
#include "dprint_tensix.h"
#endif

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_lib_unpack_wrappers.h"
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, params.num_faces, params.num_faces);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0, 0, FACE_R_DIM, params.num_faces, formats.unpack_A_src, formats.unpack_A_dst);

    const std::uint32_t num_tiles = params.NUM_BLOCKS * params.NUM_TILES_IN_BLOCK;
    for (std::uint32_t i = 0; i < num_tiles; ++i)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(params.buffer_A[i]), formats.unpack_A_src, formats.unpack_A_dst);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#ifdef FORMAT_INT32
const bool is_int_fpu_en = true;
#else
const bool is_int_fpu_en = false;
#endif

#include "llk_lib_math_wrappers.h"
#include "params.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

    _llk_math_eltwise_unary_datacopy_init_wrapper_<
        DataCopyType::A2D,
        is_fp32_dest_acc_en,
        BroadcastType::NONE,
        is_int_fpu_en,
        llk_test_pack_mode_v<false, false>>(params.num_faces, formats.math);
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    for (int block_num = 0; block_num < params.NUM_BLOCKS; ++block_num)
    {
        _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
        for (std::uint32_t tile_num = 0; tile_num < params.NUM_TILES_IN_BLOCK; ++tile_num)
        {
            LLK_ASSERT(
                (params.DST_INDEX + tile_num < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                "tile_num exceeds max dest tiles");
            _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                params.DST_INDEX + tile_num, formats.math, formats.math, params.num_faces);
        }

#if defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE)
        // Dump DEST before releasing the dest section, so pack reads what we printed.
        llk_dprint::dprint_tensix_dest_reg(params.DST_INDEX);
#endif

        _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }
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

    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, llk_test_pack_mode_v<false, false>>(
        formats.pack_src, formats.pack_dst, 16 * 16 * 4 /* tile_size */, FACE_R_DIM, TILE_C_DIM, params.num_faces);
    _llk_pack_init_wrapper_<llk_test_pack_mode_v<false, false>, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, params.num_faces);
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    for (int block_num = 0; block_num < params.NUM_BLOCKS; ++block_num)
    {
        _llk_packer_wait_for_math_done_();
        for (std::uint32_t tile_num = 0; tile_num < params.NUM_TILES_IN_BLOCK; ++tile_num)
        {
            LLK_ASSERT(
                (params.DST_INDEX + tile_num < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                "tile_num exceeds max dest tiles");
            _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(
                params.DST_INDEX + tile_num, L1_ADDRESS(params.buffer_Res[block_num * params.NUM_TILES_IN_BLOCK + tile_num]));
        }
        _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }
}

#endif
