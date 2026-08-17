// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Coexistence test for the two experimental sort SFPU headers, i.e. the
// set_dst_write_addr_offset extraction in tt-metal #52713.
//
// Before that PR, ckernel_sfpu_topk_xl.h and ckernel_sfpu_deepseek_top32_rm.h each
// defined their own identical `set_dst_write_addr_offset`, so a math translation unit
// including both would fail with a redefinition error. tt-blaze papers over this with
// #ifndef guards in its copies; #52713 instead extracts the helper into
// sfpu/experimental/ckernel_sfpu_set_dst_write_addr_offset.h and has both headers
// include it.
//
// Nothing in the tree compiles both headers into the same TU, so the redefinition the
// PR fixes is unreachable by every other test -- topk_xl_test.cpp includes only its own
// header, and the top32_rm consumers only theirs. This test exists to make that
// combination reachable: it is primarily a COMPILE-time assertion. If the extraction
// regresses (either header redeclaring the helper locally), this file stops building.
//
// It also runs, so the shared helper is exercised rather than merely compiled: the math
// thread calls set_dst_write_addr_offset to rebase the Dst write pointer, restores it to
// 0, and then a plain datacopy must still land correctly. A helper that left the offset
// dirty would corrupt that copy.
//
// Scope note: the two families' inits are deliberately NOT both called. Measured on BH
// p100a, calling _top32_rm_init_() and _topk_xl_init_<K, fused>() in one kernel hangs the
// math thread -- both program overlapping ADDR_MOD slots, the MOP and the REPLAY buffer,
// so only one family can be live at a time. That is not a defect and no real kernel does
// it: the PR's claim, and this test's assertion, is that the two headers can coexist in
// one *translation unit*, not that both can be initialized simultaneously. Anyone
// intending to fuse the two families needs to re-init between them.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

// Globals required by the test framework.
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr ckernel::DstSync DST_SYNC = ckernel::DstSync::SyncHalf;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES);

    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, ckernel::DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);

    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src, formats.unpack_A_dst);
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

// THE POINT OF THIS FILE: both experimental sort SFPU headers in one TU. Each pulls in
// sfpu/experimental/ckernel_sfpu_set_dst_write_addr_offset.h for the shared helper, so
// this only compiles while that extraction holds.
#include "sfpu/experimental/ckernel_sfpu_deepseek_top32_rm.h"
#include "sfpu/experimental/ckernel_sfpu_topk_xl.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();

    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
        TILE_NUM_FACES, formats.math);

    // Invariant SFPU config only -- see the scope note above on why neither family's
    // init is called here.
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();

    _llk_math_wait_for_dest_available_<DST_SYNC>();

    // Exercise the shared helper, then put the Dst write offset back. The datacopy below
    // must be unaffected; a helper that leaked a non-zero offset would misplace it.
    _llk_math_eltwise_unary_sfpu_params_(
        []
        {
            ckernel::sfpu::set_dst_write_addr_offset(SORT_DST_WRITE_OFFSET);
            ckernel::sfpu::set_dst_write_addr_offset(0);
        },
        0 /* dst_index */,
        VectorMode::None);

    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        0 /* dst_index */, formats.math, formats.math);

    _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
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
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * TILE_NUM_FACES);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
    _llk_pack_dest_init_wrapper_<DST_SYNC, is_fp32_dest_acc_en, PackMode::Default>();

    _llk_packer_wait_for_math_done_();
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(0 /* tile_index */, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}

#endif
