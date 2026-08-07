// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Cross-op tile-geometry transition test (state-leak design): a `unpack_tilize`
// at one geometry (the "polluter") followed by a REGULAR `unpack_AB_matmul` on
// independent operands.
//
// This covers the user-requested "tiny tile unpack tilize + regular matmul"
// transition. Because a coupled pipeline (tilize output feeding the matmul) cannot
// mix geometries, the matmul reads its own independent, pre-tilized 32x32 operands
// from `buffer_A[0]` / `buffer_B[0]`; the run-0 tilize only exists to leave the
// unpacker in a (possibly tiny `face_r_dim`) state. The run-1 transition is:
//   `_llk_unpack_tilize_uninit_` (restores the tilize-mutated SrcA baseline) +
//   `_llk_unpack_reconfig_data_format_src{a,b}_impl_<.. FACE_ROW_MAJOR>`
//   (re-establishes the canonical regular strides / Tile_x_dim / Z-dim).
//
// If that transition fails to fully reset the unpacker (e.g. leaves a tiny
// `Tile_x_dim_cntx0`, a tilize-specific Z-dim, or a stale Y-stride), the regular
// matmul reads its operands with the wrong layout and the result diverges from
// `tilize(MatmulGolden(A, B))`.
//
// NOTE: run-0 tilize reads `buffer_A[0]` (which holds the tilized matmul operand)
// as raw row-major input; its packed output goes to a scratch buffer and is
// discarded. Only the run-1 matmul result is validated.

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;
// Regular 32x32 (4-face) matmul-operand tile size used by the run-1 matmul.
std::uint32_t tile_size = ckernel::FACE_R_DIM * ckernel::FACE_C_DIM * 4 / 8;

// Scratch L1 address for the discarded run-0 (polluter) tilize output.
constexpr std::uint32_t buffer_polluter_scratch = 0xA0000;

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
    // Polluter (run-0) tilize geometry: face_r_dim=16 -> regular, <16 -> tiny.
    const std::uint32_t pol_face_r_dim   = params.TEST_FACE_R_DIM;
    const std::uint32_t pol_num_faces    = params.num_faces;
    constexpr std::uint32_t mm_num_faces = 4; // regular matmul operands

    // ---- Run 0: tilize "polluter" (output discarded) ----
    int run                              = 0;
    const std::uint32_t pol_block_ct_dim = _llk_unpack_tilize_block_ct_dim_wrapper_(1);
    const std::uint32_t pol_tilize_nf    = _llk_unpack_tilize_num_faces_wrapper_(pol_num_faces);

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats_array[run].unpack_A_src,
        formats_array[run].unpack_B_src,
        formats_array[run].unpack_A_dst,
        formats_array[run].unpack_B_dst,
        pol_face_r_dim,
        pol_face_r_dim,
        pol_num_faces,
        pol_num_faces);
    _llk_unpack_tilize_init_wrapper_(
        formats_array[run].unpack_A_src, formats_array[run].unpack_A_dst, 1 /* ct_dim */, pol_face_r_dim, false /* narrow_tile */, pol_tilize_nf);
    _llk_unpack_tilize_wrapper_(
        L1_ADDRESS(params.buffer_A[0]),
        0 /* tile_index */,
        formats_array[run].unpack_A_src,
        formats_array[run].unpack_A_dst,
        pol_block_ct_dim,
        pol_face_r_dim,
        pol_tilize_nf,
        false /* narrow_tile */);

    // Wait until the run-0 packer has drained before touching unpacker state.
    t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE);
    t6_semaphore_get<>(semaphore::PACK_DONE);

    // ---- Run 1: REGULAR matmul on independent pre-tilized operands ----
    run = 1;
    if constexpr (DO_RESTORE)
    {
        // Tear down the tilize-mutated SrcA state (PR C1) and then re-establish the regular
        // 32x32 operand baseline for BOTH operands before the matmul.
        //
        // `_llk_unpack_AB_matmul_init_` only reprograms x-end and the ZW counter (see its
        // uninit doc) -- it does NOT reset the SrcA Y/Z strides, Tile_x_dim, or the operand
        // tile descriptors, so without an explicit restore the polluter tilize state leaks
        // into the matmul.
        //
        // A *geometry change* (tiny polluter -> regular matmul) cannot be undone by
        // uninit + a stride-only reconfig alone: `_llk_unpack_reconfig_data_format_srca_impl_`
        // re-commits the SrcA Z-stride / Y-stride / Tile_x_dim_cntx0 and descriptor Z-dim, but
        // not the SrcA tile-descriptor X/Y-dim, which uninit left at the tiny geometry. The
        // idiomatic full reconfigure for a geometry change is `_llk_unpack_hw_configure_`
        // (configure_unpack_AB), which reprograms both descriptors to the regular baseline.
        // (For a geometry-matched polluter, face_r_dim==16, uninit alone already suffices.)
#ifdef ARCH_WORMHOLE
        _llk_unpack_tilize_uninit_wrapper_(formats_array[run].unpack_A_dst, pol_num_faces, pol_face_r_dim);
#else
        _llk_unpack_tilize_uninit_wrapper_(formats_array[run].unpack_A_dst, pol_num_faces);
#endif
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats_array[run].unpack_A_src,
            formats_array[run].unpack_B_src,
            formats_array[run].unpack_A_dst,
            formats_array[run].unpack_B_dst,
            FACE_R_DIM,
            FACE_R_DIM,
            mm_num_faces,
            mm_num_faces,
            tile_size,
            tile_size);
    }
    _llk_unpack_AB_matmul_init_<>();
    _llk_unpack_AB_matmul_<>(L1_ADDRESS(params.buffer_A[0]), L1_ADDRESS(params.buffer_B[0]), 0, 0, tile_size, tile_size);
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_lib_math_wrappers.h"
#include "llk_math_matmul.h"
#include "params.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig(&formats_array)[2] = params.formats;
#endif
    const bool is_int_fpu_en          = false;
    const std::uint32_t pol_num_faces = params.num_faces;
    const std::uint32_t res_dst_idx   = 0;

    // ---- Run 0: datacopy consuming the polluter tilize ----
    int run                      = 0;
    static constexpr bool TILIZE = true;
    _llk_math_eltwise_unary_datacopy_init_wrapper_<
        DataCopyType::A2D,
        is_fp32_dest_acc_en,
        BroadcastType::NONE,
        is_int_fpu_en,
        llk_test_pack_mode_v<false, TILIZE>>(pol_num_faces, formats_array[run].math);
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats_array[run].math, formats_array[run].math);
    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, false>(
        res_dst_idx, formats_array[run].math, formats_array[run].math);
    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    // ---- Run 1: regular matmul ----
    run = 1;
    _llk_math_reconfig_data_format_srca_<is_fp32_dest_acc_en, false>(formats_array[run].math);
    _llk_math_reconfig_data_format_srcb_<is_fp32_dest_acc_en, false>(formats_array[run].math);
    _llk_math_matmul_init_<MATH_FIDELITY>();
    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    _llk_math_matmul_<MATH_FIDELITY>(res_dst_idx);
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
    const std::uint32_t pol_face_r_dim = params.TEST_FACE_R_DIM;
    const std::uint32_t pol_num_faces  = params.num_faces;
    const std::uint32_t res_dst_idx    = 0;
    static constexpr bool UNTILIZE     = false;
    static constexpr bool TILIZE       = true;

    // ---- Run 0: pack the (discarded) polluter tilize result to scratch ----
    int run = 0;
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, llk_unpack_tilize_sweep_pack_cfg_mode_v<UNTILIZE, TILIZE>>(
        formats_array[run].pack_src,
        formats_array[run].pack_dst,
        pol_face_r_dim * params.TEST_FACE_C_DIM * pol_num_faces /* tile_size */,
        pol_face_r_dim,
        TILE_C_DIM,
        pol_num_faces);
    _llk_pack_init_wrapper_<llk_unpack_tilize_sweep_pack_cfg_mode_v<UNTILIZE, TILIZE>, false /* zero_output */>(
        formats_array[run].pack_dst, pol_face_r_dim, TILE_C_DIM, pol_num_faces);
    _llk_pack_dest_init_wrapper_<DstSync::SyncHalf, is_fp32_dest_acc_en, llk_test_pack_mode_v<UNTILIZE, false>>();

    _llk_packer_wait_for_math_done_();
    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, pack_exec_mode_v<UNTILIZE>>(res_dst_idx, L1_ADDRESS(buffer_polluter_scratch));
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    // Signal the unpacker that run-0 has fully drained.
    t6_semaphore_post<>(semaphore::PACK_DONE);

    // ---- Run 1: pack the matmul result to the output buffer ----
    // The matmul output is a regular 32x32 (4-face) tile. Run-0 configured the packer for the
    // (possibly tiny) polluter geometry (pol_num_faces / pol_face_r_dim), so a data-format-only
    // reconfig would pack the regular result with a tiny face layout. Re-run the full pack
    // hw_configure + init for the regular geometry (Default mode). NOTE: deliberately NO
    // `_llk_pack_dest_init_` here -- the SyncHalf dest-bank counter is initialised once in run-0.
    run                                   = 1;
    constexpr std::uint32_t mm_num_faces  = 4;
    const std::uint32_t mm_pack_tile_size = FACE_R_DIM * TILE_C_DIM * mm_num_faces;
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, ckernel::PackMode::Default>(
        formats_array[run].pack_src, formats_array[run].pack_dst, mm_pack_tile_size, FACE_R_DIM, TILE_C_DIM, mm_num_faces);
    _llk_pack_init_wrapper_<ckernel::PackMode::Default, false /* zero_output */>(formats_array[run].pack_dst, FACE_R_DIM, TILE_C_DIM, mm_num_faces);

    _llk_packer_wait_for_math_done_();
    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(res_dst_idx, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif
