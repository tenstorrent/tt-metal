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
// It also runs, which proves the combined translation unit executes rather than merely
// building: a coexistence regression that wedged the math thread would surface as a hang
// or a corrupted datacopy. The helper is called below so it is code-generated rather than
// only parsed.
//
// What the runtime half does NOT do is validate the offset the helper writes. The datacopy
// used to read DEST back is _llk_math_eltwise_unary_datacopy_, which itself calls
// math::set_dst_write_addr<Tile32x32, SrcRegs>(dst_index) (cmath_common.h) -- the same
// DEST_TARGET_REG_CFG_MATH_Offset_ADDR32 the helper writes -- before anything touches DEST.
// So whatever offset the helper left is overwritten, and the copy lands identically whether
// the helper is correct, writes garbage, or is deleted. Measuring the offset's effect needs
// a DEST consumer that does not reprogram that register first; the helper is covered in its
// real context by the topk_xl and deepseek_top32_rm kernels themselves.
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

    // Force the shared helper to be code-generated -- it is the symbol the extraction is
    // about -- and put the Dst write offset back. Per the note at the top of this file this
    // is a compile/codegen assertion, not a runtime one: the datacopy below reprograms the
    // offset register itself, so no golden check downstream can observe this value.
    // 2 Dst rows is the column-group flip topk_xl performs.
    static constexpr std::uint32_t SORT_DST_WRITE_OFFSET_ROWS = 2;
    _llk_math_eltwise_unary_sfpu_params_(
        []
        {
            ckernel::sfpu::set_dst_write_addr_offset(SORT_DST_WRITE_OFFSET_ROWS);
            ckernel::sfpu::set_dst_write_addr_offset(0 /*addr*/);
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
