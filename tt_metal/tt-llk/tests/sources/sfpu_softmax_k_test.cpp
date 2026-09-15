// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Driver for the softmax_k SFPU entry
// (tt_llk_blackhole/common/inc/sfpu/experimental/ckernel_sfpu_softmax_k.h).
//
// `_softmax_k_<k>` is a single-shot (VectorMode::RC_custom) kernel over ONE
// 4-row band of face 0. It reads:
//   DEST addr 0  -> rows 0-3, even columns : x
//   DEST addr 2  -> rows 0-3, odd columns  : x
//   DEST addr 8  -> rows 8-11, even columns: max(x) per row, pre-broadcast
// and writes the finished softmax back over addr 0 / addr 2, i.e. rows 0-3.
// So it is a per-row softmax over the 16 columns of face 0, with the row maximum
// supplied by the caller in rows 8-11 rather than computed here.
//
// Padding: the sequence sets a condition code from |even-column value| before
// subtracting the max, and only calls SFPENCC after the exponential, so lanes
// whose EVEN column holds exactly 0.0 skip both the subtract and the exp and stay
// 0. Padding therefore has to be exact 0.0, and it predicates the even/odd column
// PAIR (the odd lane inherits its even partner's condition code). That is why odd
// k needs `_zero_paired_odd_tail_lane_`: for odd k the tail column k is odd and
// its even partner k-1 is a valid non-zero lane, so it would otherwise be
// exponentiated. Valid inputs must be non-zero.
//
// ckernel_sfpu_softmax_k.h self-includes its exp/reduce prerequisites and declares
// SFPCONFIG_TARGET_LREG11 / SFPCONFIG_MOD_SET_LREG11 itself, so this driver needs no
// prerequisite includes and no local #defines. Note the header still spells those
// includes unqualified ("ckernel_sfpu_exp.h"), and the exp/reduce kernels live one
// layer up in the metal tree (hw/ckernels/blackhole/metal/llk_api/llk_sfpu/), so the
// tt-llk test build has to put that directory on the include path -- see
// setup_compilation_options in helpers/test_config.py.
//
// Both even and odd k are swept. Odd k is what exercises
// `_zero_paired_odd_tail_lane_` and its 1u << (k - 1) tail mask.

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

#define DST_ACCUM_MODE is_fp32_dest_acc_en
#include "sfpu/experimental/ckernel_sfpu_softmax_k.h"
#undef DST_ACCUM_MODE

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    static_assert(SOFTMAX_K >= 2 && SOFTMAX_K <= 16, "softmax_k operates on the 16 columns of one face");

    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();

    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
        TILE_NUM_FACES, formats.math);

    _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
    ckernel::sfpu::_init_softmax_k_();

    for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
    {
        _llk_math_wait_for_dest_available_<DST_SYNC>();

        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            0 /* dst_index */, formats.math, formats.math);

        // RC_custom: the kernel does its own DEST addressing, so no per-face loop.
        SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, _softmax_k_, (SOFTMAX_K), 0 /* dst_index */, VectorMode::RC_custom);

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
