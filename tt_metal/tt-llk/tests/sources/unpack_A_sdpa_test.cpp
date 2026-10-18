
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// =============================================================================
// GOLDEN DERIVATION (read this before checking results on a Blackhole card)
// =============================================================================
// This exercises the SDPA-specific unpack-A path promoted in PR #53295:
//     experimental/llk_unpack_A_sdpa.h  (_llk_unpack_A_sdpa_init_ / _mop_config_)
//     experimental/llk_unpack_A_sdpa_api.h (llk_unpack_A_sdpa_init)
//
// The SDPA header is a *stripped-down clone* of the generic llk_unpack_A.h. It
// only defines the init + MOP config (and two dummy-DVALID helpers); it reuses
// the generic _llk_unpack_A_ as the execute call, exactly as sdpa.h does
// (sdpa_bcast_col_reuse_tiles_init programs it with llk_unpack_A_sdpa_init, then
// the tile loop runs the generic llk_unpack_A execute).
//
// What the SDPA MOP actually does (llk_unpack_A_sdpa.h:34-49), for the only
// configuration it supports (BType==NONE, acc_to_dest==false, reuse==NONE,
// unpack_to_dest==false):
//   unpack_srca = UNPACR(SrcA, Z-inc=1, OvrdThreadId=1, SetDvalid=1, ...)
//   num_faces==1: outerloop=1, innerloop=num_tiles,        one unpack_srca/iter
//   num_faces>1 : outerloop=1, innerloop=num_tiles*nf/2,   two unpack_srca/iter
// i.e. it streams exactly (num_tiles * num_faces) plain SrcA UNPACRs, each with
// Z-increment and Set-Dvalid, and nothing else. There is NO transpose, NO
// padding, NO broadcast, NO SrcB dummy-DVALID, NO within-face 16x16 transpose
// applied to the data (Haloize is set from within_face_16x16_transpose, which we
// leave 0). That is a straight face-by-face copy of SrcA into the source
// register — the "internal move (identical)" from the blaze note.
//
// Compared to the generic llk_unpack_A.h NONE / non-acc / non-transpose path
// (llk_unpack_A.h:229-236), which does `unpack_srca` start-op + a per-face
// `unpack_srcb_set_dvalid` inner op, the SDPA MOP simply drops the SrcB
// dummy-DVALID and folds everything into a single MOP. The SrcA data movement is
// bit-for-bit identical, so the observable result is a plain datacopy identity.
//
// GOLDEN = DataCopyGolden(src_A) — input tile copied to the output, converted to
// the output format (the math thread runs a normal A2D datacopy, the packer
// writes it to L1). We drive num_tiles==1 so one MOP run == one full tile, which
// matches how the execute (_llk_unpack_A_) runs the programmed MOP once per tile.
//
// Because there is no Blackhole card here, this file's bar is a clean BH compile
// plus a golden that mirrors the header's real math (identity datacopy).
// =============================================================================

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

// Use the tt-llk experimental header directly (llk_lib), NOT the metal-side
// llk_api wrapper (experimental/llk_unpack_A_sdpa_api.h). The api wrapper pulls
// in llk_unpack_common_api.h -> circular_buffer_interface.h ->
// tt-metalium/circular_buffer_constants.h, which is not on this LLK test's
// include path. This test only needs the underscore-prefixed
// _llk_unpack_A_sdpa_init_ from the llk_lib header, so include that directly.
// The SDPA MOP config in llk_unpack_A_sdpa.h leaves transpose_of_faces /
// unpack_src_format / unpack_dst_format unused for the only configuration it
// supports; suppress the resulting -Werror=unused-parameter for this promoted
// experimental header (the header cannot be edited here).
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "experimental/llk_unpack_A_sdpa.h"
#pragma GCC diagnostic pop
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t num_tiles_in_block = params.NUM_TILES_IN_BLOCK;
    const std::uint32_t num_blocks         = params.NUM_BLOCKS;

    // Only the datacopy-identity configuration the SDPA unpack path supports.
    constexpr BroadcastType bcast_type              = BroadcastType::NONE;
    constexpr bool acc_to_dest                      = false;
    constexpr EltwiseBinaryReuseDestType reuse_dest = EltwiseBinaryReuseDestType::NONE;
    constexpr bool sdpa_unpack_to_dest              = false;
    // SDPA init programs a single MOP that streams num_tiles*num_faces SrcA UNPACRs.
    // Drive one tile per MOP run so the execute below maps 1 run -> 1 tile.
    constexpr std::uint32_t sdpa_num_tiles = 1;

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src,
        formats.unpack_B_src,
        formats.unpack_A_dst,
        formats.unpack_B_dst,
        params.TEST_FACE_R_DIM,
        params.TEST_FACE_R_DIM,
        params.num_faces,
        params.num_faces);
    _llk_unpack_configure_stoch_rnd_<STOCHASTIC_RND>();

    // SDPA-specific unpack-A init. Note it takes scalar face_r_dim + num_faces
    // (NOT a TensorShape like the generic _llk_unpack_A_init_). We pass
    // transpose_of_faces=0 and within_face_16x16_transpose=0 (the identity path).
    _llk_unpack_A_sdpa_init_<sdpa_num_tiles, bcast_type, acc_to_dest, reuse_dest, sdpa_unpack_to_dest>(
        params.UNPACK_TRANSPOSE_FACES,
        params.UNPACK_TRANSPOSE_WITHIN_FACE,
        params.TEST_FACE_R_DIM,
        params.num_faces,
        formats.unpack_A_src,
        formats.unpack_A_dst);

    // The SDPA unpack path is a pure SrcA producer: _llk_unpack_A_sdpa_init_
    // programs a MOP that streams only SrcA UNPACRs (each with Set-Dvalid) and
    // deliberately DROPS the per-face SrcB dummy-DVALID that the generic
    // llk_unpack_A.h NONE MOP publishes (llk_unpack_A.h:231-234, inner op
    // unpack_srcb_set_dvalid). SrcB is never made valid on this thread — which is
    // the SDPA design. The math thread below is therefore driven to consume SrcA
    // only and to clear SrcA only (SETRWC CLR_A, not CLR_AB), so it never touches
    // the never-valid SrcB bank. See the long note in the LLK_TRISC_MATH block.
    for (std::uint32_t i = 0; i < num_tiles_in_block * num_blocks; ++i)
    {
        // Execute reuses the generic single-operand-A unpack: it runs the MOP
        // that the SDPA init programmed once, streaming one tile's SrcA UNPACRs.
        _llk_unpack_A_<bcast_type, acc_to_dest, reuse_dest, sdpa_unpack_to_dest>(L1_ADDRESS(params.buffer_A[i]), formats.unpack_A_src, formats.unpack_A_dst);
    }
    _llk_unpack_A_uninit_<bcast_type>();
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
    const std::uint32_t num_tiles_in_block = params.NUM_TILES_IN_BLOCK;
    const std::uint32_t num_blocks         = params.NUM_BLOCKS;

    // Test configuration constants
    constexpr DstSync sync_mode = DstSync::SyncHalf;

    // SDPA unpack keeps data in SrcA (BType::NONE, unpack_to_dest=false) -> A2D copy.
    constexpr DataCopyType copy_type = DataCopyType::A2D;
    _llk_math_eltwise_unary_datacopy_init_wrapper_<copy_type, is_fp32_dest_acc_en, BroadcastType::NONE, is_int_fpu_en, PackMode::Default>(
        params.num_faces, formats.math);

    // A2D datacopy MOP override: end-op clears ONLY SrcA (CLR_A, not the stock CLR_AB).
    // llk_unpack_A_sdpa is SrcA-only and never validates SrcB, so the stock CLR_AB would
    // clear a never-valid SrcB bank -- HW-tolerated but non-contractual (ttsim flags it).
    // Otherwise bit-for-bit the stock MOVA2D MOP; valid only for the datacopy-identity
    // path this test drives (bf16/fp32, MOVA2D branch -- not the ELWADD/MOVB2D branches).
    {
        constexpr std::uint32_t innerloop = 16 >> 3; // MOV_8_ROWS -> 2 inner iterations per face
        const std::uint32_t outerloop     = params.num_faces;
        ckernel::ckernel_template tmp(outerloop, innerloop, TT_OP_MOVA2D(ckernel::p_mov::DEST_NORM, 0, ADDR_MOD_2, ckernel::p_mova2d::MOV_8_ROWS, 0));
        tmp.set_end_op(TT_OP_SETRWC(ckernel::p_setrwc::CLR_A, 0, 0, 0, 0, ckernel::p_setrwc::SET_AB));
        tmp.program();
    }

    _llk_math_pack_sync_init_<sync_mode, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    for (std::uint32_t block = 0; block < num_blocks; ++block)
    {
        _llk_math_wait_for_dest_available_<sync_mode>();
        for (std::uint32_t tile_in_block = 0; tile_in_block < num_tiles_in_block; ++tile_in_block)
        {
            LLK_ASSERT(
                (tile_in_block < get_dest_max_tiles<sync_mode, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                "Block tile index exceeds maximum destination tiles");
            _llk_math_eltwise_unary_datacopy_<copy_type, sync_mode, is_fp32_dest_acc_en, BroadcastType::NONE, false>(tile_in_block, formats.math, formats.math);
        }
        _llk_math_dest_section_done_<sync_mode, is_fp32_dest_acc_en>();
    }
    _llk_math_eltwise_unary_datacopy_uninit_<BroadcastType::NONE, false>();
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
    const std::uint32_t num_tiles_in_block = params.NUM_TILES_IN_BLOCK;
    const std::uint32_t num_blocks         = params.NUM_BLOCKS;

    // Test configuration constants
    constexpr DstSync sync_mode = DstSync::SyncHalf;
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
        formats.pack_src, formats.pack_dst, params.TEST_FACE_R_DIM * params.TEST_FACE_C_DIM * 4, params.TEST_FACE_R_DIM, TILE_C_DIM, params.num_faces);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, params.TEST_FACE_R_DIM, TILE_C_DIM, params.num_faces);

    _llk_pack_dest_init_wrapper_<sync_mode, is_fp32_dest_acc_en, PackMode::Default>();

    for (std::uint32_t block = 0; block < num_blocks; ++block)
    {
        _llk_packer_wait_for_math_done_();
        for (std::uint32_t tile_in_block = 0; tile_in_block < num_tiles_in_block; ++tile_in_block)
        {
            LLK_ASSERT(
                (tile_in_block < get_dest_max_tiles<sync_mode, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                "Block tile index exceeds maximum destination tiles");
            _llk_pack_<sync_mode, is_fp32_dest_acc_en, ckernel::PackMode::Default>(
                tile_in_block, L1_ADDRESS(params.buffer_Res[(block * num_tiles_in_block) + tile_in_block]));
        }
        _llk_pack_dest_section_done_<sync_mode, is_fp32_dest_acc_en>();
    }
}
#endif
