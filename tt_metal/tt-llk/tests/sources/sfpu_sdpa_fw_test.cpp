// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Driver for Metal's llk_sfpu/ckernel_sfpu_sdpa_fw.h.
 *
 * Like the bodies in the neighbouring ckernel_sfpu_sdpa.h, these have no LLK API of their own.
 * The consumer declares its own wrapper and dispatches the body at VectorMode::C. This test
 * declares one too, with a minimal init sufficient to exercise the bodies.
 *
 * The header holds two:
 *
 *   calculate_recip_first_column        sfpu_reciprocal_iter<2> on an fp32 dest, else <1> plus a
 *                                       round to bf16.
 *   calculate_exponential_first_column  _ckernel_sfpu_exp_accurate_ with SCALE_EN, scale as a
 *                                       uint16_t bf16 pattern.
 *
 * Both run ITERATIONS_HALF_FACE = 4 iterations at a dst_reg stride of 2, on faces 0 and 2
 * under VectorMode::C. That writes columns {0,2,4,6,8,10,12,14} of all 32 rows and leaves the rest
 * of the tile alone. Only column 0 carries meaning for the caller, since these operands are
 * row-reduce outputs broadcast down a column, but the other seven are computed and so have to be
 * modelled.
 */

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// Which body to drive. Mirrors SdpaFwOp in helpers/llk_params.py.
constexpr int OP_FW_RECIP = 0; // calculate_recip_first_column
constexpr int OP_FW_EXP   = 1; // calculate_exponential_first_column<EXP_SCALE_BF16>

static_assert(SDPA_FW_OP == OP_FW_RECIP || SDPA_FW_OP == OP_FW_EXP, "unhandled SDPA_FW_OP");

// Both work on one Dest tile in place.
constexpr std::uint32_t SDPA_FW_DST_INDEX = 0;

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

    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src, formats.unpack_A_dst);
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

// ckernel_sfpu_sdpa_fw.h needs DST_ACCUM_MODE.
static constexpr bool DST_ACCUM_MODE = is_fp32_dest_acc_en;

#include "experimental/llk_sfpu/ckernel_sfpu_sdpa_fw.h"

using namespace ckernel;

inline void sdpa_fw_op_init()
{
    if constexpr (SDPA_FW_OP == OP_FW_RECIP)
    {
        sfpu::recip_init<APPROX_MODE, is_fp32_dest_acc_en, false /* legacy_compat */>();
    }
    else
    {
        _llk_math_eltwise_unary_sfpu_init_once_();
    }
}

inline void sdpa_fw_op(const std::uint32_t dst_index)
{
    if constexpr (SDPA_FW_OP == OP_FW_RECIP)
    {
        _llk_math_eltwise_unary_sfpu_params_(sfpu::calculate_recip_first_column, dst_index, VectorMode::C);
    }
    else
    {
        _llk_math_eltwise_unary_sfpu_params_(sfpu::calculate_exponential_first_column<EXP_SCALE_BF16>, dst_index, VectorMode::C);
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

    sdpa_fw_op_init();

    _llk_math_wait_for_dest_available_<dest_sync>();

    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, dest_sync, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        SDPA_FW_DST_INDEX, formats.math, formats.math);

    sdpa_fw_op(SDPA_FW_DST_INDEX);

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

    _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(SDPA_FW_DST_INDEX, L1_ADDRESS(params.buffer_Res[0]));

    _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
}

#endif
