// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Cross-op tile-geometry transition test (Phase 3b Direction B): a REGULAR
// `unpack_tilize` (the polluter) followed by a TINY `unpack_AB_matmul`
// (num_faces < 4 / partial-face operand).
//
// This closes the "matmul with num_faces < 4 after a tilize" gap left by
// `test_matmul_unpack_tilize` (which only ever runs num_faces=4). As with
// Direction A, a coupled pipeline cannot mix tile geometries, so the tiny matmul
// reads its own independent, pre-tilized operands from buffer_A[0] / buffer_B[0];
// run-0's regular tilize only exists to leave the unpacker in a regular-tilize
// state (its packed output goes to scratch and is discarded). Between the two,
// `_llk_unpack_tilize_uninit_` restores the SrcA baseline (gated by DO_RESTORE).
//
// run-1 is the proven tiny-matmul sequence (verbatim from unpack_matmul_test.cpp,
// retargeted at formats_array[1]); it re-runs `_llk_unpack_hw_configure_` for the
// tiny operands, so this primarily validates that a regular tilize -> uninit ->
// tiny matmul SEQUENCE produces correct results. NOTE: because run-1 self-configures,
// DO_RESTORE=false is expected to behave the same on a correct build; the toggle is
// kept to probe whether an un-uninit'd tilize can corrupt the subsequent matmul.

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// Scratch L1 address for the discarded run-0 (polluter) tilize output.
constexpr std::uint32_t buffer_polluter_scratch = 0xA0000;

// run-0 polluter is always a regular 32x32 tilize.
constexpr std::uint32_t POLLUTER_FACE_R_DIM = 16;
constexpr std::uint32_t POLLUTER_NUM_FACES  = 4;

#ifdef LLK_TRISC_UNPACK

#include "llk_lib_unpack_wrappers.h"
#include "llk_unpack_AB_matmul.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig(&formats_array)[2] = params.formats;
#endif

    // ---- Run 0: regular tilize "polluter" (output discarded to scratch) ----
    int run                              = 0;
    const std::uint32_t pol_block_ct_dim = _llk_unpack_tilize_block_ct_dim_wrapper_(1);
    const std::uint32_t pol_tilize_nf    = POLLUTER_NUM_FACES;

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats_array[run].unpack_A_src,
        formats_array[run].unpack_B_src,
        formats_array[run].unpack_A_dst,
        formats_array[run].unpack_B_dst,
        POLLUTER_FACE_R_DIM,
        POLLUTER_FACE_R_DIM,
        POLLUTER_NUM_FACES,
        POLLUTER_NUM_FACES);
    _llk_unpack_tilize_init_wrapper_(
        formats_array[run].unpack_A_src, formats_array[run].unpack_A_dst, 1 /* ct_dim */, POLLUTER_FACE_R_DIM, false /* narrow_tile */, pol_tilize_nf);
    // Read the regular operand B buffer (full 32x32 tile) as raw row-major input.
    _llk_unpack_tilize_wrapper_(
        L1_ADDRESS(params.buffer_B[0]),
        0 /* tile_index */,
        formats_array[run].unpack_A_src,
        formats_array[run].unpack_A_dst,
        pol_block_ct_dim,
        POLLUTER_FACE_R_DIM,
        pol_tilize_nf,
        false /* narrow_tile */);

    // Wait until the run-0 packer has drained before touching unpacker state.
    t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE);
    t6_semaphore_get<>(semaphore::PACK_DONE);

    if constexpr (DO_RESTORE)
    {
        // Restore the canonical SrcA baseline mutated by the regular tilize.
#ifdef ARCH_WORMHOLE
        _llk_unpack_tilize_uninit_wrapper_(formats_array[1].unpack_A_dst, POLLUTER_NUM_FACES, POLLUTER_FACE_R_DIM);
#else
        _llk_unpack_tilize_uninit_wrapper_(formats_array[1].unpack_A_dst, POLLUTER_NUM_FACES);
#endif
    }

    // ---- Run 1: TINY matmul on independent pre-tilized operands (verbatim) ----
    run = 1;
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats_array[run].unpack_A_src,
        formats_array[run].unpack_B_src,
        formats_array[run].unpack_A_dst,
        formats_array[run].unpack_B_dst,
        params.in1_tile_r_dim < FACE_R_DIM ? params.in1_tile_r_dim : FACE_R_DIM,
        params.in0_tile_r_dim < FACE_R_DIM ? params.in0_tile_r_dim : FACE_R_DIM,
        params.num_faces_B, // in1
        params.num_faces_A, // in0
        params.TILE_SIZE_UNPACK_B,
        params.TILE_SIZE_UNPACK_A);
    _llk_unpack_configure_stoch_rnd_<STOCHASTIC_RND>();
    _llk_unpack_AB_matmul_init_<>(
        params.UNPACK_TRANSPOSE_FACES,
        params.CT_DIM,
        params.RT_DIM,
        params.KT_DIM,
        params.in1_tile_r_dim < FACE_R_DIM ? params.in1_tile_r_dim : FACE_R_DIM,
        params.in0_tile_r_dim < FACE_R_DIM ? params.in0_tile_r_dim : FACE_R_DIM,
        params.num_faces_B,     // in1
        params.num_faces_A,     // in0
        params.PARTIAL_FACE_B,  // in1
        params.PARTIAL_FACE_A); // in0
    for (std::uint32_t j = 0; j < params.KT_DIM; j++)
    {
        _llk_unpack_AB_matmul_<>(
            L1_ADDRESS(params.buffer_A[0]),
            L1_ADDRESS(params.buffer_B[0]),
            j,
            j * params.CT_DIM,
            params.TILE_SIZE_UNPACK_B,
            params.TILE_SIZE_UNPACK_A,
            params.PARTIAL_FACE_B, // in1
            params.PARTIAL_FACE_A, // in0
            params.CT_DIM,
            params.RT_DIM,
            params.KT_DIM);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_lib_math_wrappers.h"
#include "llk_math_common.h"
#include "llk_math_matmul.h"
#include "params.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig(&formats_array)[2] = params.formats;
#endif
    const bool is_int_fpu_en = false;

    // ---- Run 0: datacopy consuming the regular tilize polluter ----
    int run                      = 0;
    static constexpr bool TILIZE = true;
    _llk_math_eltwise_unary_datacopy_init_wrapper_<
        DataCopyType::A2D,
        is_fp32_dest_acc_en,
        BroadcastType::NONE,
        is_int_fpu_en,
        llk_test_pack_mode_v<false, TILIZE>>(POLLUTER_NUM_FACES, formats_array[run].math);
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats_array[run].math, formats_array[run].math);
    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, false>(
        0 /* dst */, formats_array[run].math, formats_array[run].math);
    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    // ---- Run 1: tiny matmul ----
    run = 1;
    _llk_math_reconfig_data_format_srca_<is_fp32_dest_acc_en, false>(formats_array[run].math);
    _llk_math_reconfig_data_format_srcb_<is_fp32_dest_acc_en, false>(formats_array[run].math);
    _llk_math_matmul_init_<MATH_FIDELITY, THROTTLE_LEVEL>(
        params.in0_tile_r_dim,
        params.in0_tile_c_dim,
        params.in1_tile_r_dim,
        params.in1_tile_c_dim,
        params.PARTIAL_FACE_MATH,
        params.UNPACK_TRANSPOSE_FACES,
        params.CT_DIM,
        params.RT_DIM);
    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    for (std::uint32_t j = 0; j < params.KT_DIM; j++)
    {
        _llk_math_matmul_<MATH_FIDELITY, THROTTLE_LEVEL>(params.DST_INDEX, params.CT_DIM, params.RT_DIM);
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
    const FormatConfig(&formats_array)[2] = params.formats;
#endif
    static constexpr bool UNTILIZE = false;
    static constexpr bool TILIZE   = true;

    // ---- Run 0: pack the (discarded) regular tilize result to scratch ----
    int run = 0;
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, llk_unpack_tilize_sweep_pack_cfg_mode_v<UNTILIZE, TILIZE>>(
        formats_array[run].pack_src,
        formats_array[run].pack_dst,
        POLLUTER_FACE_R_DIM * TILE_C_DIM * POLLUTER_NUM_FACES /* tile_size */,
        POLLUTER_FACE_R_DIM,
        TILE_C_DIM,
        POLLUTER_NUM_FACES);
    _llk_pack_init_wrapper_<llk_unpack_tilize_sweep_pack_cfg_mode_v<UNTILIZE, TILIZE>, false /* zero_output */>(
        formats_array[run].pack_dst, POLLUTER_FACE_R_DIM, TILE_C_DIM, POLLUTER_NUM_FACES);
    _llk_pack_dest_init_wrapper_<DstSync::SyncHalf, is_fp32_dest_acc_en, llk_test_pack_mode_v<UNTILIZE, false>>();

    _llk_packer_wait_for_math_done_();
    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, pack_exec_mode_v<UNTILIZE>>(0 /* dst */, L1_ADDRESS(buffer_polluter_scratch));
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    // Signal the unpacker that run-0 has fully drained.
    t6_semaphore_post<>(semaphore::PACK_DONE);

    // ---- Run 1: pack the tiny matmul result to the output buffer (verbatim) ----
    run = 1;
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
        formats_array[run].pack_src,
        formats_array[run].pack_dst,
        params.TILE_SIZE_PACK,
        params.in0_tile_r_dim < FACE_R_DIM ? params.in0_tile_r_dim : FACE_R_DIM,
        TILE_C_DIM,
        params.num_faces,
        params.PARTIAL_FACE_PACK);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(
        formats_array[run].pack_dst, params.in0_tile_r_dim < FACE_R_DIM ? params.in0_tile_r_dim : FACE_R_DIM, TILE_C_DIM, params.num_faces);
    // NOTE: do NOT re-run `_llk_pack_dest_init_` here. In a two-run SyncHalf kernel the
    // dest-bank counter is initialised once (run-0); re-initialising it in run-1 resets
    // the counter to bank-0 while the math side has already toggled to bank-1, so the
    // packer would drain an unwritten bank (all-zeros result).
    _llk_packer_wait_for_math_done_();
    for (std::uint32_t i = 0; i < params.TILE_CNT; i++)
    {
        LLK_ASSERT((i < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()), "i exceeds max dest tiles");
        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(params.DST_INDEX + i, L1_ADDRESS(params.buffer_Res[i]));
    }
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif
