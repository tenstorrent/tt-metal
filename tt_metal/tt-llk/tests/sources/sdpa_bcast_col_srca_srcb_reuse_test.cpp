// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Blackhole-only unit test for the experimental LLK
//   _llk_math_sdpa_bcast_col_srca_srcb_reuse_
// (llk_lib/experimental/llk_math_sdpa_bcast_col_srca_srcb_reuse.h, promoted by tt-metal #53295).
//
// WHAT THE OP DOES (derived from the header, this is where the golden comes from)
// --------------------------------------------------------------------------------
// This is a MATH-only, DEST-resident column-broadcast eltwise op. Unlike the plain
// sub_bcast_col path (which unpacks SrcA from L1 and reuses only SrcB), this variant
// reuses BOTH source registers *out of DEST*:
//
//   * SrcB is loaded from DEST via the preamble
//     `_llk_math_sdpa_bcast_col_srca_srcb_reuse_preamble_<...>(isrc)`:
//       TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset, isrc + dest_base)
//       TTI_MOVD2B(0, SRC_ZERO_OFFSET+0, ADDR_MOD_1, MOV_4_ROWS, 0)   // DEST rows 0..3 -> SrcB 0..3
//       TTI_MOVD2B(0, SRC_ZERO_OFFSET+4, ADDR_MOD_1, MOV_4_ROWS, 4)   // DEST rows 4..7 -> SrcB 4..7
//     i.e. SrcB is populated from the DEST tile at `isrc`.
//
//   * SrcA is loaded from DEST inside the MOP start-op (the `d2a_instr` replay):
//       TTI_MOVD2A(0, MATH_HALO_ROWS+0, ADDR_MOD_1, MOV_4_ROWS, 0)    // DEST rows 0..3 -> SrcA 0..3
//       TTI_MOVD2A(0, MATH_HALO_ROWS+4, ADDR_MOD_1, MOV_4_ROWS, 4)    // DEST rows 4..7 -> SrcA 4..7
//     i.e. SrcA is populated from the DEST tile at `dst_index` (the op's SETC16 target).
//
//   * The MOP body is TT_OP_ELW{ADD,SUB,MUL} with broadcast_type = p_elwise::SRCB_BCAST_COL
//     and CLR_NONE (SrcB is never cleared inside the mop — it is reused). The result is
//     accumulated back into DEST at `dst_index`. num_faces is asserted == 2 (a 16x32 tiny tile).
//
// On both Blackhole and Wormhole B0, MATH_HALO_ROWS == 0 and SRC_ZERO_OFFSET == 0
// (common/inc/ckernel_instr_params.h), so both MOVD2A and MOVD2B read from the *base* of
// their SETC16-selected DEST tile.
//
// NET RESULT for one 16x32 (2-face) tile, with ELWSUB:
//     DEST[dst_index]  =  A  -  bcast_col(B)
//   where A is the DEST tile at `dst_index`, B is the DEST tile at `isrc`, and
//   bcast_col replicates each face's first column across all 16 columns of that face
//   (BroadcastType::COL semantics — faces 0 and 1 each use their own column-0 values;
//   for a 2-face tile that is faces 0 and 1).
//
// HOW THIS TEST FEEDS THE OP
// --------------------------------------------------------------------------------
// The op reads its operands out of DEST, so we must first place them there. Following the
// established DEST-preload pattern (cf. custom_mm_uninit_restore_test.cpp), the MATH thread
// A2D-datacopies two known operand tiles into DEST — tile 0 (the SrcA operand, A) and tile 1
// (the SrcB operand, B) — within a single DEST section. We then run the reuse op with
// dst_index=0 (SrcA source + output) and isrc=1 (SrcB source), and pack DEST tile 0.
//
// The UNPACK thread streams the two operand tiles into SrcA for the datacopies, then issues
// the dummy SrcA/SrcB dvalid the op's preamble requires (matching the compute API's
// sdpa_bcast_col_srca_srcb_reuse_preamble UNPACK side, llk_unpack_A_sdpa.h).
//
// GOLDEN: computed in Python as EltwiseBinary(ELWSUB, A, bcast_col(B)) over the 2-face tile —
// the same column-broadcast SUB the sibling sub_bcast_col golden validates, just DEST-sourced.
// All 512 datums of the 16x32 tile are defined and validated at the output format tolerance.

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_defs.h"
#include "tensor_shape.h"

using namespace ckernel;

// Globals required by the test framework.
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr DstSync DST_SYNC = DstSync::SyncHalf;

// 16x32 tiny tile: 2 faces of 16x16. The op asserts num_faces == 2.
static constexpr std::uint32_t NUM_FACES = 2;

// DEST tile slots the two operands are datacopied into. dst_index selects SrcA (and the
// output); isrc selects the SrcB (column-broadcast) source.
static constexpr std::uint32_t DST_INDEX_A = 0; // SrcA source + result
static constexpr std::uint32_t ISRC_B      = 1; // SrcB source

#ifdef LLK_TRISC_UNPACK

#include "experimental/llk_unpack_A_sdpa.h"
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src,
        formats.unpack_B_src,
        formats.unpack_A_dst,
        formats.unpack_B_dst,
        FACE_R_DIM /* unpA_face_r_dim */,
        FACE_R_DIM /* unpB_face_r_dim */,
        NUM_FACES /* unpA_num_faces */,
        NUM_FACES /* unpB_num_faces */);

    // Stream operand A (tile 0) then operand B (tile 1) through SrcA so the MATH thread can
    // A2D-datacopy each into its DEST slot. 2-face (16x32) tiny-tile geometry.
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false /* unpack_to_dest */>(
        0 /* transpose_of_faces */,
        0 /* within_face_16x16_transpose */,
        ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, NUM_FACES),
        formats.unpack_A_src,
        formats.unpack_A_dst);

    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false /* unpack_to_dest */>(
        L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src, formats.unpack_A_dst);
    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false /* unpack_to_dest */>(
        L1_ADDRESS(params.buffer_A[1]), formats.unpack_A_src, formats.unpack_A_dst);

    // The reuse op's preamble/main move data DEST->SrcA/SrcB (MOVD2A/MOVD2B) and assume the
    // unpacker has posted a dummy dvalid for both SrcA and SrcB, so the MATH stall on
    // SRCA_VLD|SRCB_VLD releases. This mirrors the compute API's
    // sdpa_bcast_col_srca_srcb_reuse_preamble UNPACK side.
    _llk_unpack_A_sdpa_set_srca_srcb_dummy_valid_();
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_sdpa_bcast_col_srca_srcb_reuse.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    _llk_math_wait_for_dest_available_<DST_SYNC>();

    // Preload DEST: tile 0 <- operand A (SrcA source), tile 1 <- operand B (SrcB source).
    // A plain A2D datacopy places each operand at the standard 64-row DEST tile spacing.
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */>(
        NUM_FACES /* num_faces */, formats.math);
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, false /* unpack_to_dest */>(
        DST_INDEX_A, formats.math, formats.math);
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, false /* unpack_to_dest */>(
        ISRC_B, formats.math, formats.math);

    // Configure the reuse op: ADDR_MODs + MOP for a column-broadcast SUB over 2 faces.
    // num_tiles == 1 (single output tile), acc_to_dest == 0. Reprograms ADDR_MOD_0/1/2.
    _llk_math_sdpa_bcast_col_srca_srcb_reuse_init_<EltwiseBinaryType::ELWSUB, 1 /* num_tiles */, MATH_FIDELITY>(NUM_FACES /* num_faces */, 0 /* acc_to_dest */);

    // Preamble: load SrcB from DEST tile isrc (MOVD2B). clear_dest=false: DEST[dst_index]
    // already holds operand A, which the op consumes and overwrites.
    _llk_math_sdpa_bcast_col_srca_srcb_reuse_preamble_<DST_SYNC, is_fp32_dest_acc_en, false /* clear_dest */>(ISRC_B);

    // Main: load SrcA from DEST tile dst_index (MOVD2A in the MOP), compute
    // DEST[dst_index] = SrcA - bcast_col(SrcB), signalling the packer once (output_granularity=1).
    _llk_math_sdpa_bcast_col_srca_srcb_reuse_<
        EltwiseBinaryType::ELWSUB,
        1 /* num_tiles */,
        DST_SYNC,
        is_fp32_dest_acc_en,
        MATH_FIDELITY,
        false /* clear_dest */,
        false /* skip_signalling */,
        1 /* output_granularity */>(DST_INDEX_A);

    _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
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

    // 16x32 tiny tile: 2 faces, face_r_dim=16, total_col_dim=32.
    const std::uint32_t tile_size = FACE_R_DIM * FACE_C_DIM * NUM_FACES;

    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
        formats.pack_src, formats.pack_dst, tile_size, FACE_R_DIM, TILE_C_DIM, NUM_FACES, false /* partial_face */, false /* narrow_tile */);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(
        formats.pack_dst, FACE_R_DIM, TILE_C_DIM, NUM_FACES, false /* partial_face */, false /* narrow_tile */);
    _llk_pack_dest_init_wrapper_<DST_SYNC, is_fp32_dest_acc_en, PackMode::Default>(FACE_R_DIM, false /* narrow_tile */);

    // Pack the validated result out of DEST tile dst_index.
    _llk_packer_wait_for_math_done_();
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(DST_INDEX_A, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}

#endif
