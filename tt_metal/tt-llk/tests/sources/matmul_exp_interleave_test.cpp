// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// One matmul init, then per output tile: matmul into Dest, accurate exp on that Dest tile.
//
// The math thread records the matmul MOP body in the FPU half of the per-thread replay
// buffer at init and replays it for every tile; the accurate exp records its own body in
// the SFPU half. The two halves meet at ckernel::math::replay_buf_offset. A record that
// crosses it rewrites the matmul program, and every matmul after the first exp then
// replays a corrupted sequence. Tile 0 is computed before any exp ran; tiles 1.. are
// computed after one, so a crossing shows as tile 0 correct, the rest wrong.
//
// SFPU_SCALE_EN selects the scaled exp, exp(EXP_WITH_BASE_SCALE * x), whose loop body is
// the longest instantiation of the accurate bf16 kernel.

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "params.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_AB_matmul.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t TILE_SIZE_UNPACK_A = params.TILE_SIZE_UNPACK_A;
    const std::uint32_t TILE_SIZE_UNPACK_B = params.TILE_SIZE_UNPACK_B;
    const std::uint32_t TILE_CNT           = params.TILE_CNT;
    const std::uint32_t CT_DIM             = params.CT_DIM;
    const std::uint32_t RT_DIM             = params.RT_DIM;
    const std::uint32_t KT_DIM             = params.KT_DIM;
    const Operand& buffer_A                = params.buffer_A;
    const Operand& buffer_B                = params.buffer_B;
#endif

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src,
        formats.unpack_B_src,
        formats.unpack_A_dst,
        formats.unpack_B_dst,
        FACE_R_DIM,
        FACE_R_DIM,
        TILE_NUM_FACES,
        TILE_NUM_FACES,
        TILE_SIZE_UNPACK_A,
        TILE_SIZE_UNPACK_B);
    _llk_unpack_AB_matmul_init_<>(
        0 /*transpose*/,
        CT_DIM,
        RT_DIM,
        KT_DIM,
        FACE_R_DIM,
        FACE_R_DIM,
        TILE_NUM_FACES,
        TILE_NUM_FACES,
        false /*unpA_partial_face*/,
        false /*unpB_partial_face*/);

    // Output tile t = A[t] x B (one B tile, shared by every output tile).
    for (std::uint32_t tile = 0; tile < TILE_CNT; tile++)
    {
        LLK_ASSERT(is_valid_L1_address(L1_ADDRESS(buffer_A[tile])), "unpack A address is outside L1");
        _llk_unpack_AB_matmul_<>(
            L1_ADDRESS(buffer_A[tile]),
            L1_ADDRESS(buffer_B[0]),
            0 /*tile_index_a*/,
            0 /*tile_index_b*/,
            TILE_SIZE_UNPACK_A,
            TILE_SIZE_UNPACK_B,
            false /*unpA_partial_face*/,
            false /*unpB_partial_face*/,
            CT_DIM,
            RT_DIM,
            KT_DIM);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_matmul.h"
#include "sfpu_operations.h"

using namespace ckernel;
using namespace ckernel::sfpu;

// exp_init bakes the scale into the approximate clamp path's constants only; the accurate
// paths take it from the calculate call. Passed for parity with the harness's exp_with_base
// init (sfpu_operations.h).
constexpr std::uint32_t EXP_INIT_SCALE_FP32 = SFPU_SCALE_EN ? EXP_WITH_BASE_SCALE_FP32 : 0x3F800000u /*1.0f*/;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t TILE_CNT = params.TILE_CNT;
    const std::uint32_t CT_DIM   = params.CT_DIM;
    const std::uint32_t RT_DIM   = params.RT_DIM;
#endif

    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    // Both inits once, before the tile loop, the way a compute kernel calls
    // mm_init + exp_tile_init once and then alternates matmul_tiles / exp_tile.
    _llk_math_matmul_init_<MATH_FIDELITY, THROTTLE_LEVEL>(
        TILE_R_DIM, TILE_C_DIM, TILE_R_DIM, TILE_C_DIM, false /*partial_face*/, 0 /*transpose*/, CT_DIM, RT_DIM);
    llk_math_eltwise_unary_sfpu_init<SfpuType::exponential>(exp_init<APPROX_MODE, EXP_INIT_SCALE_FP32, CLAMP_NEGATIVE, is_fp32_dest_acc_en>);
    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();

    LLK_ASSERT((TILE_CNT <= get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()), "TILE_CNT exceeds max dest tiles");

    _llk_math_wait_for_dest_available_<dest_sync>();
    for (std::uint32_t tile = 0; tile < TILE_CNT; tile++)
    {
        _llk_math_matmul_<MATH_FIDELITY, THROTTLE_LEVEL>(tile, CT_DIM, RT_DIM);
        if constexpr (SFPU_SCALE_EN)
        {
            // exp_tile's accurate path with a scale, exp(EXP_WITH_BASE_SCALE * x). The bf16
            // scale travels as a template argument (the kernel lowers it to an SFPMULI
            // immediate), through the same test adapter the harness's exp_with_base uses.
            SFPU_UNARY_CALL(
                dest_sync,
                is_fp32_dest_acc_en,
                calculate_exponential_const_scale,
                (APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS, CLAMP_NEGATIVE, EXP_WITH_BASE_SCALE_BF16),
                tile,
                VectorMode::RC);
        }
        else
        {
            // exp_tile's accurate path: calculate_exponential<APPROX_MODE, fp32 dest, SCALE_EN=false, ITERATIONS, CLAMP_NEGATIVE>
            SFPU_UNARY_CALL(
                dest_sync,
                is_fp32_dest_acc_en,
                calculate_exponential,
                (APPROX_MODE, is_fp32_dest_acc_en, false /*SCALE_EN*/, ITERATIONS, CLAMP_NEGATIVE),
                tile,
                VectorMode::RC,
                p_sfpu::kCONST_1_FP16B /*exp_base_scale_factor, unused with SCALE_EN=false*/);
        }
    }
    _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();

    _llk_math_matmul_uninit_();
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

#ifndef SPEED_OF_LIGHT
    const std::uint32_t TILE_SIZE_PACK = params.TILE_SIZE_PACK;
    const std::uint32_t TILE_CNT       = params.TILE_CNT;
    const Operand& buffer_Res          = params.buffer_Res;
#endif

    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, TILE_SIZE_PACK);
    _llk_pack_init_wrapper_<PackMode::Default, false /*zero_output*/>(formats.pack_dst);
    _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();

    _llk_packer_wait_for_math_done_();
    for (std::uint32_t tile = 0; tile < TILE_CNT; tile++)
    {
        LLK_ASSERT(is_valid_L1_address(L1_ADDRESS(buffer_Res[tile])), "pack result address is outside L1");
        _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile, L1_ADDRESS(buffer_Res[tile]));
    }
    _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
}

#endif
