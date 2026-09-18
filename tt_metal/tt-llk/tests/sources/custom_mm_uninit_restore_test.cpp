// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Packer W-stride restore test for custom_mm_block_uninit /
// compressed_custom_mm_block_uninit
// (api/compute/experimental/{custom_mm,compressed_custom_mm}.h, merged to main by
// tt-metal #52727).
//
// As merged, both uninits do exactly one thing:
//
//   dense_packing -> cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW> back to the
//                    default 64-row tile-to-tile stride
//
// An earlier revision of #52727 also restored the tile-pack MOP, first unconditionally
// and then behind a restore_tile_pack_mop flag. **Neither survived review** -- main's
// uninit has no MOP restore and no such flag, and the fused caller is expected to pair
// pack_block_contiguous_init with its own uninit instead. This test was written against
// that earlier revision and has been narrowed to what actually shipped; the MOP-restore
// coverage is gone because the behaviour is gone.
//
// Shape of the test (the established tt-llk uninit-restore pattern, cf.
// unpack_tilize_uninit_restore_test.cpp):
//
//   run 0  Establish the Default pack baseline, apply the dense_packing W-stride if
//          selected, then swap in the block-contiguous packer MOP -- what a caller's
//          pack_block_contiguous_init (experimental/pack_block.h) does. Pack once
//          through it; output deliberately NOT asserted, since with dense_packing the
//          stride the packer assumes (32 rows) disagrees with where the datacopy put the
//          tiles (64 rows) on purpose.
//
//   uninit The function under test, replicated statement-for-statement from the compute
//          API -- which is now just the conditional W-stride write.
//
//   run 1  A plain per-tile _llk_pack_<PackMode::Default>, with NO packer re-init, so it
//          packs through whatever state the uninit left.
//
// BLOCK_MOP_NUM_FACES selects which of the two packer states run 1 measures, and the
// python side builds this source at both values on purpose. Do not normalize it to one:
//
//   4 faces  the geometry run 1 needs, so the MOP is not a confound and run 1 is correct
//            exactly when the W-stride was restored. This is the value the W-stride tests
//            (the positive test and the skip-uninit control) require -- at a mismatched
//            geometry run 1 is wrong whatever the stride is, and the restore would be
//            unobservable.
//   2 faces  a 16x32 tiny tile, a geometry run 1 does NOT want, used by
//            test_custom_mm_uninit_leaves_the_caller_mop_installed. Run 1 can only come
//            back correct if something reinstalled the Default MOP -- which main's uninit
//            must not do -- so that test asserts run 1 is *wrong*. Deleting this case
//            deletes the only coverage that would catch a MOP restore being re-added.
//
// Expectation for the W-stride tests (4 faces), owned by the python side:
//   run 1 is correct  <=>  dense_packing was not set, OR the uninit ran and restored the
//                          stride.
// UNINIT_SKIP is the negative control that drops the uninit entirely, proving the stride
// restore is load-bearing rather than incidental.

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
    }

    // ---- run 1: plain per-tile pack, deliberately with NO packer re-init ----
    for (std::uint32_t tile = 0; tile < PACK_NUM_TILES; ++tile)
    {
        _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile, L1_ADDRESS(params.buffer_Res[tile]));
    }

    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}

#endif
