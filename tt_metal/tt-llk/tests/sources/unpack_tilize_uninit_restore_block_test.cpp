// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Multi-tile-block variant of unpack_tilize_uninit_restore_test.cpp (Phase 6).
//
// Same cross-op restore design (tilize -> `_llk_unpack_tilize_uninit_` -> plain
// datacopy, NO data-format reconfig in between, so uninit is the SOLE state
// reset), but run 0 tilizes a BLOCK of `BLOCK_CT_DIM > 1` column tiles in one
// init/execute sweep instead of a single tile.
//
// Why this is a distinct case: `_llk_unpack_tilize_init_(ct_dim=BLOCK_CT_DIM)`
// programs the tilize `shift_amount` (config word-0) and per-column addressing
// from the block width, and the op leaves SrcA in that block-tilize state. The
// PR's `_llk_unpack_tilize_uninit_` must still restore the canonical single-tile
// operand baseline so the following plain datacopy of EACH tilized tile reads
// correctly. If the block tilize leaves residual state that uninit doesn't
// clear, the run-1 datacopy of the block diverges from the tilized golden.
//
// Run 0: tilize the `BLOCK_CT_DIM` column tiles of operand A -> packed to
//         `BLOCK_CT_DIM` scratch L1 slots.
// Run 1: plain `_llk_unpack_A_` datacopy of each scratch tilized tile -> result.
//         On a correct restore this is an identity copy, so the result equals
//         `TilizeGolden(src_A_block, num_faces)`.

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// Scratch L1 base for the tilized block; each tilized tile gets a generous
// fixed slot (>= the largest 4-face fp32 tile = 4 KiB) well above the
// stimuli/result buffers (which start at 0x21000).
constexpr std::uint32_t buffer_block_scratch = 0xA0000;
constexpr std::uint32_t scratch_tile_stride  = 0x4000;

constexpr std::uint32_t scratch_tile_addr(std::uint32_t i)
{
    return buffer_block_scratch + i * scratch_tile_stride;
}

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
    const std::uint32_t num_faces  = params.num_faces;
    const std::uint32_t face_r_dim = params.TEST_FACE_R_DIM;

    // ---- Run 0: tilize the BLOCK_CT_DIM column tiles of operand A into scratch ----
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

    const std::uint32_t tilize_num_faces = num_faces;
    const std::uint32_t block_ct_dim     = _llk_unpack_tilize_block_ct_dim_wrapper_(params.BLOCK_CT_DIM);

    _llk_unpack_tilize_init_wrapper_(
        formats_array[run].unpack_A_src, formats_array[run].unpack_A_dst, params.BLOCK_CT_DIM, face_r_dim, false /* narrow_tile */, tilize_num_faces);
    for (std::uint32_t col = 0; col < params.BLOCK_CT_DIM; ++col)
    {
        _llk_unpack_tilize_wrapper_(
            L1_ADDRESS(params.buffer_A[0]),
            col,
            formats_array[run].unpack_A_src,
            formats_array[run].unpack_A_dst,
            block_ct_dim,
            face_r_dim,
            tilize_num_faces,
            false /* narrow_tile */);
    }

    // Wait until the packer has written all BLOCK_CT_DIM tilized tiles to scratch.
    t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE);
    t6_semaphore_get<>(semaphore::PACK_DONE);

    // ---- Restore under test (NO reconfig — uninit is the sole reset) ----
#ifdef ARCH_WORMHOLE
    _llk_unpack_tilize_uninit_wrapper_(formats_array[run].unpack_A_dst, num_faces, face_r_dim);
#else
    _llk_unpack_tilize_uninit_wrapper_(formats_array[run].unpack_A_dst, num_faces);
#endif

    // ---- Run 1: plain datacopy of each tilized tile (same format, no reconfig) ----
    run = 1;
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0, 0, ckernel::make_tensor_shape_from_legacy(face_r_dim, num_faces), formats_array[run].unpack_A_src, formats_array[run].unpack_A_dst);
    for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(scratch_tile_addr(i)), formats_array[run].unpack_A_src, formats_array[run].unpack_A_dst);
    }
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
    const std::uint32_t num_faces = params.num_faces;
    const bool is_int_fpu_en      = false;

    // ---- Run 0: block datacopy with tilize ----
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
    for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
    {
        LLK_ASSERT(
            (i < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()), "Block tile index exceeds maximum destination tiles");
        _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            i, formats_array[run].math, formats_array[run].math, num_faces);
    }
    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    // ---- Run 1: plain block datacopy (no tilize) ----
    run                             = 1;
    static constexpr bool NO_TILIZE = false;
    _llk_math_eltwise_unary_datacopy_init_wrapper_<
        DataCopyType::A2D,
        is_fp32_dest_acc_en,
        BroadcastType::NONE,
        is_int_fpu_en,
        llk_test_pack_mode_v<false, NO_TILIZE>>(num_faces, formats_array[run].math);

    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
    {
        _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            i, formats_array[run].math, formats_array[run].math, num_faces);
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
    const std::uint32_t num_faces  = params.num_faces;
    const std::uint32_t face_r_dim = params.TEST_FACE_R_DIM;
    const std::uint32_t tile_size  = face_r_dim * params.TEST_FACE_C_DIM * num_faces;
    static constexpr bool UNTILIZE = false;
    static constexpr bool TILIZE   = true;

    // ---- Run 0: pack the BLOCK_CT_DIM tilized tiles to scratch ----
    int run = 0;
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, llk_unpack_tilize_sweep_pack_cfg_mode_v<UNTILIZE, TILIZE>>(
        formats_array[run].pack_src, formats_array[run].pack_dst, tile_size, face_r_dim, TILE_C_DIM, num_faces);
    _llk_pack_init_wrapper_<llk_unpack_tilize_sweep_pack_cfg_mode_v<UNTILIZE, TILIZE>, false /* zero_output */>(
        formats_array[run].pack_dst, face_r_dim, TILE_C_DIM, num_faces);
    _llk_pack_dest_init_wrapper_<DstSync::SyncHalf, is_fp32_dest_acc_en, llk_test_pack_mode_v<UNTILIZE, false>>();

    _llk_packer_wait_for_math_done_();
    for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
    {
        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, pack_exec_mode_v<UNTILIZE>>(i, L1_ADDRESS(scratch_tile_addr(i)));
    }
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    // Signal the unpacker that the tilized block is ready in L1.
    t6_semaphore_post<>(semaphore::PACK_DONE);

    // ---- Run 1: pack the datacopy results to the output buffers ----
    run = 1;
#ifdef ARCH_BLACKHOLE
    // Blackhole run-0 used PackMode::Tilize; re-establish a Default-mode packer
    // for the plain datacopy result (Wormhole tilize pack cfg mode is Default).
    _llk_pack_reconfig_data_format_wrapper_<is_fp32_dest_acc_en>(
        formats_array[run].pack_src, formats_array[run].pack_dst, tile_size, face_r_dim, TILE_C_DIM, num_faces);
    _llk_pack_init_wrapper_<ckernel::PackMode::Default, false /* zero_output */>(formats_array[run].pack_dst, face_r_dim, TILE_C_DIM, num_faces);
#endif

    _llk_packer_wait_for_math_done_();
    for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
    {
        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(i, L1_ADDRESS(params.buffer_Res[i]));
    }
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif
