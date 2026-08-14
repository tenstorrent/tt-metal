// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Cross-op packer-state restore test for custom_mm_block_uninit /
// compressed_custom_mm_block_uninit
// (api/compute/experimental/{custom_mm,compressed_custom_mm}.h, promoted by
// tt-metal #52727).
//
// Both uninits do nothing but conditionally undo two pieces of packer state:
//
//   dense_packing         -> cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>
//                            back to the default 64-row tile-to-tile stride
//   restore_tile_pack_mop -> _llk_pack_mop_config_<PackMode::Default>() , reinstalling
//                            the fixed 32x32 / 4-face tile-pack MOP
//
// Nothing in tt-llk called either uninit before this test, and the pack-MOP restore is
// the one codegen change in #52727 — so both were entirely unexercised.
//
// Shape of the test (the established tt-llk uninit-restore pattern, cf.
// unpack_tilize_uninit_restore_test.cpp and unpack_bcastA_B_uninit_restore_test.cpp):
//
//   run 0  Establish the Default pack baseline, apply the dense_packing W-stride if
//          selected, then swap in the block-contiguous packer MOP -- this is what a
//          caller's pack_block_contiguous_init (experimental/pack_block.h) does, and
//          the case the uninit's comment is about: that init "replaces the packer MOP
//          without owning it". Pack once through it.
//
//          run 0's output is deliberately NOT asserted. Its only job is to leave the
//          packer in the non-default state; with dense_packing the DEST stride the
//          packer assumes (32 rows) disagrees with where a plain datacopy put the
//          tiles (64 rows) on purpose, so its output is expected garbage.
//
//   uninit The function under test, replicated call-for-call from the compute API,
//          including the no-argument _llk_pack_mop_config_<Default>() the header
//          actually issues (all defaults: face_r_dim=16, tile_c_dim=32, num_faces=4,
//          num_tiles=1).
//
//   run 1  A plain per-tile _llk_pack_<PackMode::Default>, with NO packer re-init.
//          _llk_pack_ executes whatever MOP is installed (ckernel_template::run), so
//          this reads the restored state directly. On a correct restore it is an
//          identity copy of the DEST tiles the math thread datacopied in.
//
// Expectation, owned by the python side:
//   run 1 is correct  <=>  the uninit ran AND restore_tile_pack_mop was set,
// because run 0 always leaves the block-contiguous MOP installed and only the MOP
// restore can undo it. UNINIT_SKIP is a negative control that drops the uninit
// entirely, proving both restores are load-bearing rather than incidental.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"

// Globals required by the test framework.
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr ckernel::DstSync DST_SYNC = ckernel::DstSync::SyncHalf;

// Scratch L1 for run 0's (unasserted) output. Well above the stimuli/result buffers,
// which start at 0x21000 -- same convention as unpack_tilize_uninit_restore_test.cpp.
constexpr std::uint32_t buffer_run0_scratch = 0xA0000;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES);

    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, ckernel::DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);

    for (std::uint32_t tile = 0; tile < PACK_NUM_TILES; ++tile)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(params.buffer_A[tile]), formats.unpack_A_src, formats.unpack_A_dst);
    }
}

#endif

#ifdef LLK_TRISC_MATH

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

    // Plain datacopy: puts PACK_NUM_TILES known tiles at the standard 64-row DEST
    // spacing. The packer state -- not the math -- is what this test varies.
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
        TILE_NUM_FACES, formats.math);

    _llk_math_wait_for_dest_available_<DST_SYNC>();

    for (std::uint32_t tile = 0; tile < PACK_NUM_TILES; ++tile)
    {
        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            tile, formats.math, formats.math);
    }

    _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "experimental/llk_pack_block.h"
#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "params.h"

using namespace ckernel;

// The two W-stride values custom_mm_block_init / _uninit write. Spelled out here exactly
// as the compute API spells them so a change to either side shows up as a diff.
constexpr std::uint32_t DENSE_WSTRIDE   = (TILE_NUM_FACES / 2) * FACE_C_DIM * FACE_R_DIM * 2;
constexpr std::uint32_t DEFAULT_WSTRIDE = TILE_NUM_FACES * FACE_C_DIM * FACE_R_DIM * 2;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    // ---- run 0 setup: the Default baseline custom_mm_block_init establishes ----
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * TILE_NUM_FACES);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
    _llk_pack_dest_init_wrapper_<DST_SYNC, is_fp32_dest_acc_en, PackMode::Default>();

    if constexpr (UNINIT_DENSE_PACKING)
    {
        // custom_mm_block_init's dense_packing branch: 32 rows between tiles.
        cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(DENSE_WSTRIDE);
    }

    // What a caller's pack_block_contiguous_init does: replace the MOP, keep the
    // ADDR_MODs/strides from the init above.
    //
    // BLOCK_MOP_NUM_FACES is the geometry that MOP is programmed with. It matters which
    // value is used: the pack MOP bakes in tile geometry, so when the block MOP carries
    // the SAME geometry as the run-1 pack (4 faces), restoring it and not restoring it
    // are indistinguishable. Programming it with a different geometry (2 faces, i.e. a
    // 16x32 tiny tile) is the situation the uninit's own comment is about -- "installs
    // fixed 32x32 tile geometry, wrong for 1x32 follow-ons" -- and is what makes the
    // restore observable at all.
    _llk_pack_block_contiguous_mop_config_<false /* zero_output */>(FACE_R_DIM, BLOCK_MOP_NUM_FACES);

    _llk_packer_wait_for_math_done_();

    // run 0: pack through the block MOP. Output intentionally not asserted.
    _llk_pack_block_contiguous_<DST_SYNC, is_fp32_dest_acc_en>(0 /* tile_index */, L1_ADDRESS(buffer_run0_scratch), PACK_NUM_TILES);

    // ---- the function under test ----
    if constexpr (!UNINIT_SKIP)
    {
        if constexpr (UNINIT_DENSE_PACKING)
        {
            cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(DEFAULT_WSTRIDE);
        }
        if constexpr (UNINIT_RESTORE_MOP)
        {
            // Exactly the no-argument call the compute API issues.
            _llk_pack_mop_config_<PackMode::Default>();
        }
    }

    // ---- run 1: plain per-tile pack, deliberately with NO packer re-init ----
    for (std::uint32_t tile = 0; tile < PACK_NUM_TILES; ++tile)
    {
        _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile, L1_ADDRESS(params.buffer_Res[tile]));
    }

    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}

#endif
