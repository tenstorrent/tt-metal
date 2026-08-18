// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Driver for the upper-unclamped exp SFPU entry
// (tt_llk_blackhole/common/inc/sfpu/experimental/ckernel_sfpu_sdpa_exp_unclamped.h).
//
// That header only exposes leaf helpers that map a vFloat to a vFloat; there is
// no dst_reg loop and no _init_ entry. The loop lives in the metal llk_api tree
// (`ckernel::sfpu::calculate_sdpa_exp_unclamped`, one SFPU slot per iteration,
// 8 iterations per face, in
// hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_sdpa_exp_unclamped.h),
// alongside the other SDPA SFPU wrappers, and this test drives it through the
// standard VectorMode::RC dispatch.
//
// What is under test: `_ckernel_sfpu_exp_accurate_upper_unclamped_` is a copy of
// the accurate exp path with the *upper* input clamp removed. The clamped variant
// saturates xlog2 = val/ln2 + 127 at its upper bound, which is dead code for the
// SDPA use case where val <= 0 always.
//
// Scope of the sweep: every point the python side feeds is inside the domain where
// this kernel is bit-identical to `_sfpu_exp_21f_bf16_`, so what is verified is that
// removing the upper clamp changed nothing observable. INPUT_RANGES tops out at 4.0
// (xlog2 = 4*1.4427 + 127 = 132.8, well under the removed upper bound of 255), and it
// bottoms out at -400, which does engage the surviving *lower* clamp (it needs
// val <= -88.03) deeply enough for a missing clamp to be visible -- around
// val ~= -176 an unclamped xlog2 recombines to ~1.0 against a golden of 0.
//
// The removed clamp's own domain is not swept, on purpose: past the upper clamp point
// it is the unclamped variant that stops tracking exp(val) -- the float-to-int step in
// `_float_to_int32_for_exp_21f_` wraps there -- and exp() overflows bf16 above
// val ~= 88.7 regardless, so there is no reference to compare against. See the
// python docstring, which is the authority on the swept domain.
//
// NOTE: the LLK header has an inverted dependency -- it does `#include
// "ckernel_sfpu_exp.h"`, and there is no such file in tt-llk: the exp kernels
// live one layer up, in the metal llk_api tree
// (hw/ckernels/blackhole/metal/llk_api/llk_sfpu/). That unqualified spelling only
// resolves because the tt-llk test build now also puts llk_api/llk_sfpu on the
// include path (see setup_compilation_options in helpers/test_config.py).

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

    for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(params.buffer_A[tile]), formats.unpack_A_src, formats.unpack_A_dst);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h"

// ckernel_sfpu_sdpa_exp_unclamped.h reads a bare DST_ACCUM_MODE (it is written
// against the metal SFPU macro environment), so define it before including it.
#define DST_ACCUM_MODE is_fp32_dest_acc_en
#include "experimental/llk_sfpu/ckernel_sfpu_sdpa_exp_unclamped.h"
#undef DST_ACCUM_MODE

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

    _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();

    for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
    {
        _llk_math_wait_for_dest_available_<DST_SYNC>();

        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            0 /* dst_index */, formats.math, formats.math);

        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_sdpa_exp_unclamped, (SFPU_SCALE_EN), 0 /* dst_index */, VectorMode::RC, SFPU_UNARY_SCALAR);

        _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
    }
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

    for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
    {
        _llk_packer_wait_for_math_done_();
        _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(0 /* tile_index */, L1_ADDRESS(params.buffer_Res[tile]));
        _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
    }
}

#endif
