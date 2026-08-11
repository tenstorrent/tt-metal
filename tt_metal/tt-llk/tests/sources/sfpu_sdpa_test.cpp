// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Driver for Metal's llk_sfpu/ckernel_sfpu_sdpa.h.
 *
 * Those bodies have no LLK API of their own, so each consumer declares its own wrapper. This test
 * declares one too, dispatching each body at VectorMode::C the way a consumer does, with a minimal
 * init sufficient to exercise it.
 *
 * Every body runs ITERATIONS_HALF_FACE = 4 iterations at a dst_reg stride of 2, on faces 0 and 2
 * under VectorMode::C. That writes columns {0,2,4,6,8,10,12,14} of all 32 rows and leaves the rest
 * of the tile alone. Only column 0 carries meaning for the caller, since these operands are
 * row-reduce outputs broadcast down a column.
 *
 * OP_CORRECTION drives the same skeleton over five DEST tiles instead of one, in the order ttnn's
 * correction_block assigns dst_reg_0..dst_reg_4:
 *
 *   tile 0  in: prev_max    out: exp(scale * (prev_max   - cur_max))
 *   tile 1  in: worker_max  out: exp(scale * (worker_max - cur_max))
 *   tile 2  in: ignored     out: cur_max = max(prev_max, worker_max)
 *   tile 3  in: prev_sum    out: exp_worker*worker_sum + exp_prev*prev_sum
 *   tile 4  in: worker_sum  out: exp_worker * worker_sum
 */

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// Which body to drive. Mirrors SdpaOp in helpers/llk_params.py.
constexpr int OP_RECIP_LEGACY = 0; // calculate_recip_first_column<true>,  _reciprocal_compat_
constexpr int OP_RECIP_ITER   = 1; // calculate_recip_first_column<false>, sfpu_reciprocal_iter
constexpr int OP_EXP_ACCURATE = 2; // calculate_exponential_first_column<true,  EXP_SCALE_BF16>
constexpr int OP_EXP_POLY     = 3; // calculate_exponential_first_column<false, EXP_SCALE_BF16>
constexpr int OP_SOFTPLUS     = 4; // calculate_softplus_first_column
constexpr int OP_CORRECTION   = 5; // calculate_fused_max_sub_exp_add_tile

static_assert(SDPA_OP >= OP_RECIP_LEGACY && SDPA_OP <= OP_CORRECTION, "unhandled SDPA_OP");

constexpr bool SDPA_OP_IS_EXP = (SDPA_OP == OP_EXP_ACCURATE || SDPA_OP == OP_EXP_POLY);

// Derived from the op rather than passed in, so the tile count cannot disagree with the body.
// Only the correction body works on more than one tile.
constexpr std::uint32_t NUM_DST_TILES = (SDPA_OP == OP_CORRECTION) ? 5 : 1;

static_assert(
    NUM_DST_TILES <= ckernel::get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, ckernel::DstTileShape::Tile32x32>(),
    "this configuration needs more Dest tiles than the dest_sync / dest_acc pair can hold");

// The dispatch always targets the base tile. The correction body reaches its other four regions
// by fixed dst_reg offsets from there.
constexpr std::uint32_t SDPA_DST_INDEX = 0;

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
        0 /* transpose_of_faces */,
        0 /* within_face_16x16_transpose */,
        ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, TILE_NUM_FACES),
        formats.unpack_A_src,
        formats.unpack_A_dst);

    for (std::uint32_t tile = 0; tile < NUM_DST_TILES; ++tile)
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

// ckernel_sfpu_sdpa.h needs these.
static constexpr bool DST_ACCUM_MODE = is_fp32_dest_acc_en;
static constexpr bool APPROX         = APPROX_MODE;
#ifndef ALWI
#define ALWI inline __attribute__((always_inline))
#endif

#include "experimental/llk_sfpu/ckernel_sfpu_sdpa.h"
#include "llk_sfpu/ckernel_sfpu_exp.h"
#include "llk_sfpu/ckernel_sfpu_recip.h"

using namespace ckernel;

inline void sdpa_op_init()
{
    if constexpr (SDPA_OP == OP_RECIP_LEGACY || SDPA_OP == OP_RECIP_ITER)
    {
        sfpu::recip_init<APPROX_MODE, is_fp32_dest_acc_en, SDPA_OP == OP_RECIP_LEGACY /* legacy_compat */>();
    }
    else if constexpr (SDPA_OP_IS_EXP || SDPA_OP == OP_CORRECTION)
    {
        sfpu::exp_init<SDPA_OP != OP_EXP_POLY /* APPROXIMATION_MODE */, 0x3F800000 /* scale, unused here */, true /* CLAMP_NEGATIVE */, is_fp32_dest_acc_en>();
    }
    else
    {
        _llk_math_eltwise_unary_sfpu_init_once_();
        sfpu::softplus_init();
    }
}

inline void sdpa_op(const std::uint32_t dst_index)
{
    if constexpr (SDPA_OP == OP_RECIP_LEGACY)
    {
        _llk_math_eltwise_unary_sfpu_params_(sfpu::calculate_recip_first_column<true /* legacy_compat */>, dst_index, VectorMode::C);
    }
    else if constexpr (SDPA_OP == OP_RECIP_ITER)
    {
        _llk_math_eltwise_unary_sfpu_params_(sfpu::calculate_recip_first_column<false /* legacy_compat */>, dst_index, VectorMode::C);
    }
    else if constexpr (SDPA_OP == OP_EXP_ACCURATE)
    {
        _llk_math_eltwise_unary_sfpu_params_(
            sfpu::calculate_exponential_first_column<true /* SDPA_EXP_APPROX_MODE */, EXP_SCALE_BF16>, dst_index, VectorMode::C);
    }
    else if constexpr (SDPA_OP == OP_EXP_POLY)
    {
        _llk_math_eltwise_unary_sfpu_params_(
            sfpu::calculate_exponential_first_column<false /* SDPA_EXP_APPROX_MODE */, EXP_SCALE_BF16>, dst_index, VectorMode::C);
    }
    else if constexpr (SDPA_OP == OP_SOFTPLUS)
    {
        _llk_math_eltwise_unary_sfpu_params_(
            sfpu::calculate_softplus_first_column, dst_index, VectorMode::C, SOFTPLUS_BETA_BITS, SOFTPLUS_BETA_RECIPROCAL_BITS, SOFTPLUS_THRESHOLD_BITS);
    }
    else
    {
        _llk_math_eltwise_unary_sfpu_params_(sfpu::calculate_fused_max_sub_exp_add_tile, dst_index, VectorMode::C, static_cast<int>(EXP_SCALE_BF16));
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
        TILE_NUM_FACES, formats.math);
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();

    sdpa_op_init();

    _llk_math_wait_for_dest_available_<dest_sync>();

    for (std::uint32_t tile = 0; tile < NUM_DST_TILES; ++tile)
    {
        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, dest_sync, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            tile, formats.math, formats.math);
    }

    sdpa_op(SDPA_DST_INDEX);

    _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
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
    _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();

    _llk_packer_wait_for_math_done_();

    // Every tile is packed out, since the correction body modifies four of its five regions in place.
    for (std::uint32_t tile = 0; tile < NUM_DST_TILES; ++tile)
    {
        _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile, L1_ADDRESS(params.buffer_Res[tile]));
    }

    _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
}

#endif
