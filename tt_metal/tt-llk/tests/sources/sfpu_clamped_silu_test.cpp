// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Driver for the clamped-SiLU activation family (ckernel_sfpu_clamped_silu.h).
//
// _sfpu_sigmoid_ takes its reciprocal from sfpu_reciprocal_iter, which needs
// vConstFloatPrgm0 = 2.0f. The kernel neither programs it nor exposes an init;
// sigmoid_init<false>() is what does, on both dest_acc arms. Without it the
// reciprocal iterates against whatever the previous op left in Prgm0.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr ckernel::DstSync DST_SYNC = ckernel::DstSync::SyncHalf;

static constexpr int CLAMPED_SILU_ITERATIONS = 32;

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

using namespace ckernel;

// Math only stages the tile into Dest, the activation runs on Pack.
void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();

    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
        TILE_NUM_FACES, formats.math);

    for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
    {
        _llk_math_wait_for_dest_available_<DST_SYNC>();

        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            0 /* dst_index */, formats.math, formats.math);

        _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_pack_common.h"
#include "llk_sfpu/ckernel_sfpu_sigmoid.h" // has to be qualified with llk_sfpu/

#define TRISC_PACK
#include "experimental/llk_sfpu/ckernel_sfpu_clamped_silu.h"
#undef TRISC_PACK

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * TILE_NUM_FACES);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
    _llk_pack_dest_init_wrapper_<DST_SYNC, is_fp32_dest_acc_en, PackMode::Default>();

    // Programs vConstFloatPrgm0 = 2.0f for the reciprocal inside _sfpu_sigmoid_.
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
    ckernel::sfpu::sigmoid_init<false /* APPROXIMATION_MODE */>();

    for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
    {
        _llk_packer_wait_for_math_done_();

        _llk_math_eltwise_sfpu_start_(0 /* dst_index */);

#if defined(CLAMPED_SILU_OP_GATE)
        ckernel::sfpu::calculate_clamped_silu_gate<is_fp32_dest_acc_en, CLAMPED_SILU_ITERATIONS>(CLAMPED_SILU_SCALAR0, CLAMPED_SILU_SCALAR1);
#elif defined(CLAMPED_SILU_OP_UP)
        ckernel::sfpu::calculate_clamped_up<is_fp32_dest_acc_en, CLAMPED_SILU_ITERATIONS>(CLAMPED_SILU_SCALAR0);
#elif defined(CLAMPED_SILU_OP_CLAMP_ONLY)
        ckernel::sfpu::calculate_clamped<is_fp32_dest_acc_en, CLAMPED_SILU_ITERATIONS>(CLAMPED_SILU_SCALAR0);
#elif defined(CLAMPED_SILU_OP_SITU_GATE)
        ckernel::sfpu::calculate_situ_gate<is_fp32_dest_acc_en, CLAMPED_SILU_ITERATIONS>(CLAMPED_SILU_SCALAR0, CLAMPED_SILU_SCALAR1);
#elif defined(CLAMPED_SILU_OP_SCALED_TANH)
        ckernel::sfpu::calculate_scaled_tanh<is_fp32_dest_acc_en, CLAMPED_SILU_ITERATIONS>(CLAMPED_SILU_SCALAR0, CLAMPED_SILU_SCALAR1);
#else
#error "no CLAMPED_SILU_OP_* selected"
#endif

        _llk_math_eltwise_sfpu_done_();

        _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(0 /* tile_index */, L1_ADDRESS(params.buffer_Res[tile]));
        _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
    }
}

#endif
