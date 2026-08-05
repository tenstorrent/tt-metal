// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Cross-op unpacker-state restore test for `_llk_unpack_tilize_uninit_`.
//
// Goal: prove that after a `unpack_tilize` op, `_llk_unpack_tilize_uninit_`
// restores the SrcA tile-descriptor (num_faces / Y-dim), the unpack config
// word-0 (tilize_mode etc.), and `Tile_x_dim_cntx0` back to the canonical
// operand baseline programmed by `configure_unpack_AB` — so that a *following*
// op that uses the SAME operand baseline (and therefore performs NO data-format
// reconfig) reads correct data.
//
// To isolate the uninit restore as the *only* state reset between the two ops,
// this test deliberately:
//   * uses the SAME data format for both runs (see python `same=True`), and
//   * does NOT call `_llk_unpack_reconfig_data_format_srca_impl_` before the
//     second op (a reconfig would re-establish the strides and mask a broken
//     uninit).
//
// Run 0: tilize operand A (parameterized num_faces) -> packed to a scratch L1
//         buffer (`buffer_A_tilized`).
// Run 1: plain `_llk_unpack_A_` datacopy of that tilized tile -> packed to the
//         result buffer. On a correct build this is an identity copy of the
//         tilized tile, so the result equals `TilizeGolden(src_A, num_faces)`.
//         If uninit fails to restore the operand baseline (e.g. leaves the
//         tile-descriptor Z-dim at a tilize-specific value, or leaves
//         tilize_mode set), this datacopy is corrupted and the test fails.
//
// The `num_faces ∈ {1, 2}` cases specifically exercise the tile-descriptor
// Z-dim restore (`_llk_unpack_tilize_uninit_` line restoring num_faces) that no
// existing tilize test covers — every other uninit call site uses num_faces=4.

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// Scratch L1 address that holds the tilized tile written by the run-0 packer
// and read back by the run-1 unpacker. Sits well above the stimuli/result
// buffers (which start at 0x21000), matching matmul_unpack_tilize_test.cpp.
constexpr std::uint32_t buffer_A_tilized = 0xA0000;

#ifdef LLK_TRISC_UNPACK

#include "llk_lib_unpack_wrappers.h"
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig(&formats_array)[2] = params.formats;
#endif
    const std::uint32_t num_faces = params.num_faces;
    // face_r_dim = 16 → normal tile (Phase 1); face_r_dim < 16 → tiny tile (Phase 2),
    // which exercises the `canonical_unpA_tile_x_dim_cntx(face_r_dim)` restore branch.
    const std::uint32_t face_r_dim = params.TEST_FACE_R_DIM;

    // ---- Run 0: tilize operand A into the scratch buffer ----
    int run = 0;
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats_array[run].unpack_A_src,
        formats_array[run].unpack_B_src,
        formats_array[run].unpack_A_dst,
        formats_array[run].unpack_B_dst,
        face_r_dim,
        face_r_dim,
        num_faces,
        num_faces);

    const std::uint32_t tilize_num_faces = _llk_unpack_tilize_num_faces_wrapper_(num_faces);
    const std::uint32_t block_ct_dim     = _llk_unpack_tilize_block_ct_dim_wrapper_(1);

    _llk_unpack_tilize_init_wrapper_(
        formats_array[run].unpack_A_src, formats_array[run].unpack_A_dst, 1 /* ct_dim */, face_r_dim, false /* narrow_tile */, tilize_num_faces);
    _llk_unpack_tilize_wrapper_(
        L1_ADDRESS(params.buffer_A[0]),
        0 /* tile_index */,
        formats_array[run].unpack_A_src,
        formats_array[run].unpack_A_dst,
        block_ct_dim,
        face_r_dim,
        tilize_num_faces,
        false /* narrow_tile */);

    // Wait until the packer has written the tilized tile to the scratch buffer.
    t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE);
    t6_semaphore_get<>(semaphore::PACK_DONE);

    // ---- Restore under test ----
    // Restore the SrcA operand baseline this op mutated. NOTE: intentionally NO
    // `_llk_unpack_reconfig_data_format_srca_impl_` here, so this uninit is the
    // sole reset of the unpacker state before the next op.
#ifdef ARCH_WORMHOLE
    // Wormhole threads face_r_dim so the restore matches a tiny-tile operand baseline.
    _llk_unpack_tilize_uninit_wrapper_(formats_array[run].unpack_A_dst, num_faces, face_r_dim);
#else
    _llk_unpack_tilize_uninit_wrapper_(formats_array[run].unpack_A_dst, num_faces);
#endif

    // ---- Run 1: plain datacopy of the tilized tile (same format, no reconfig) ----
    run = 1;
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0, 0, ckernel::make_tensor_shape_from_legacy(face_r_dim, num_faces), formats_array[run].unpack_A_src, formats_array[run].unpack_A_dst);
    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        L1_ADDRESS(buffer_A_tilized), formats_array[run].unpack_A_src, formats_array[run].unpack_A_dst);
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_lib_math_wrappers.h"
#include "params.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig(&formats_array)[2] = params.formats;
#endif
    const std::uint32_t num_faces   = params.num_faces;
    const bool is_int_fpu_en        = false;
    const std::uint32_t res_dst_idx = 0;

    // ---- Run 0: datacopy with tilize ----
    int run                      = 0;
    static constexpr bool TILIZE = true;
    _llk_math_eltwise_unary_datacopy_init_wrapper_<
        DataCopyType::A2D,
        is_fp32_dest_acc_en,
        BroadcastType::NONE,
        is_int_fpu_en,
        llk_test_pack_mode_v<false, TILIZE>>(num_faces, formats_array[run].math);
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats_array[run].math, formats_array[run].math);

    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        res_dst_idx, formats_array[run].math, formats_array[run].math, num_faces);
    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    // ---- Run 1: plain datacopy (no tilize) ----
    run                             = 1;
    static constexpr bool NO_TILIZE = false;
    _llk_math_eltwise_unary_datacopy_init_wrapper_<
        DataCopyType::A2D,
        is_fp32_dest_acc_en,
        BroadcastType::NONE,
        is_int_fpu_en,
        llk_test_pack_mode_v<false, NO_TILIZE>>(num_faces, formats_array[run].math);

    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        res_dst_idx, formats_array[run].math, formats_array[run].math, num_faces);
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
    const std::uint32_t num_faces   = params.num_faces;
    const std::uint32_t face_r_dim  = params.TEST_FACE_R_DIM;
    const std::uint32_t res_dst_idx = 0;
    // Tilized datum count for the (possibly tiny) tile being packed.
    const std::uint32_t tile_size  = face_r_dim * params.TEST_FACE_C_DIM * num_faces;
    static constexpr bool UNTILIZE = false;
    static constexpr bool TILIZE   = true;

    // ---- Run 0: pack the tilized tile to the scratch buffer ----
    int run = 0;
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, llk_unpack_tilize_sweep_pack_cfg_mode_v<UNTILIZE, TILIZE>>(
        formats_array[run].pack_src, formats_array[run].pack_dst, tile_size, face_r_dim, TILE_C_DIM, num_faces);
    _llk_pack_init_wrapper_<llk_unpack_tilize_sweep_pack_cfg_mode_v<UNTILIZE, TILIZE>, false /* zero_output */>(
        formats_array[run].pack_dst, face_r_dim, TILE_C_DIM, num_faces);
    _llk_pack_dest_init_wrapper_<DstSync::SyncHalf, is_fp32_dest_acc_en, llk_test_pack_mode_v<UNTILIZE, false>>();

    _llk_packer_wait_for_math_done_();
    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, pack_exec_mode_v<UNTILIZE>>(res_dst_idx, L1_ADDRESS(buffer_A_tilized));
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    // Signal the unpacker that the tilized tile is ready in L1.
    t6_semaphore_post<>(semaphore::PACK_DONE);

    // ---- Run 1: pack the datacopy result to the output buffer ----
    run = 1;
#ifdef ARCH_BLACKHOLE
    // Blackhole run-0 used PackMode::Tilize; re-establish a Default-mode packer
    // for the plain datacopy result. Wormhole's tilize pack cfg mode is already
    // Default, so it needs no reconfig here.
    _llk_pack_reconfig_data_format_wrapper_<is_fp32_dest_acc_en>(
        formats_array[run].pack_src, formats_array[run].pack_dst, tile_size, face_r_dim, TILE_C_DIM, num_faces);
    _llk_pack_init_wrapper_<ckernel::PackMode::Default, false /* zero_output */>(formats_array[run].pack_dst, face_r_dim, TILE_C_DIM, num_faces);
#endif

    _llk_packer_wait_for_math_done_();
    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(res_dst_idx, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif
