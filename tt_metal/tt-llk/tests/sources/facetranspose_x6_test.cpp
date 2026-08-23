// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// X6 face-transpose VEHICLE (lane FV, 2026-08-22): the transpose_dest
// kernel with the math-thread FPU face transpose spelled on the TYPED X6
// surface (sfpi::face_transpose_dst_32b_batch) instead of the hand LLK's
// MOP/replay path (_llk_math_transpose_dest_<false, true>).  Covers the
// 32-bit within-face combination (transpose_of_faces handled by the
// unpacker, UNPACK_TRANSPOSE_FACES=Yes) -- the topk_xl deep-phase /
// deepseek transpose-ingest instruction algebra on a real kernel.
// Hand arm/control = the untouched transpose_dest_test.cpp.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, params.num_faces, params.num_faces);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        params.UNPACK_TRANSPOSE_FACES,
        0 /* within_face_16x16_transpose */,
        ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, params.num_faces),
        formats.unpack_A_src,
        formats.unpack_A_dst);

    // Feed SrcA per tile; ONE bank grant per block: the typed X6 arm holds
    // the SrcA/SrcB banks for the whole block and releases once
    // (face_transpose_release_banks), unlike the hand LLK's per-tile
    // CLR_AB/dummy-valid pairing.
    for (int block = 0; block < params.NUM_BLOCKS; ++block)
    {
        for (std::uint32_t tile = 0; tile < params.NUM_TILES_IN_BLOCK; ++tile)
        {
            _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                L1_ADDRESS(params.buffer_A[block * params.NUM_TILES_IN_BLOCK + tile]), formats.unpack_A_src, formats.unpack_A_dst);
        }
        _llk_unpack_set_srcb_dummy_valid_();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_lib_math_wrappers.h"

using namespace ckernel;

// One tile's four faces on the typed surface (Dst rows 0..63 relative to
// the tile base programmed via set_dst_write_addr).  OuterCfg=false: the
// block-level caller owns the cfg block.
inline void x6_tile_faces()
{
    sfpi::face_transpose_dst_32b_batch<4, 0, /*OuterCfg=*/false>();
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    static_assert(is_fp32_dest_acc_en, "X6 vehicle is the 32-bit Dst combination only");

    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    for (int block = 0; block < params.NUM_BLOCKS; ++block)
    {
        _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();

        _llk_math_eltwise_unary_datacopy_init_wrapper_<
            DataCopyType::A2D,
            is_fp32_dest_acc_en,
            BroadcastType::NONE,
            false /* is_int_fpu_en */,
            PackMode::Default>(params.num_faces, formats.math);
        for (std::uint32_t tile = 0; tile < params.NUM_TILES_IN_BLOCK; ++tile)
        {
            LLK_ASSERT(
                (tile < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                "Block tile index exceeds maximum destination tiles");
            _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                tile, formats.math, formats.math);
        }

        // Typed X6 phase: one cfg block for the whole Dst bank; per tile,
        // program the tile base (the LLK's own addressing seam) and
        // transpose its four faces in place.
        sfpi::face_transpose_cfg_enter();
        for (std::uint32_t tile = 0; tile < params.NUM_TILES_IN_BLOCK; ++tile)
        {
            math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(tile);
            math::reset_counters(p_setrwc::SET_ABD_F);
            x6_tile_faces();
        }
        sfpi::face_transpose_cfg_leave();
        // Restore the tile-base offset and return the banks (the hand
        // LLK's per-tile SETRWC CLR_AB, once per block here).
        math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
        sfpi::face_transpose_release_banks();

        _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
        formats.pack_src, formats.pack_dst, 16 * 16 * 4 /* tile_size */, FACE_R_DIM, TILE_C_DIM, params.num_faces);
    _llk_pack_init_wrapper_<PackMode::Default, false>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, params.num_faces);
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    for (int block = 0; block < params.NUM_BLOCKS; ++block)
    {
        _llk_packer_wait_for_math_done_();
        for (std::uint32_t tile = 0; tile < params.NUM_TILES_IN_BLOCK; ++tile)
        {
            _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(
                tile, L1_ADDRESS(params.buffer_Res[block * params.NUM_TILES_IN_BLOCK + tile]));
        }
        _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }
}

#endif
