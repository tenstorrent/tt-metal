// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Dest-side SFPU broadcast forms that read more than one operand tile or write only part of a tile:
// the partial column broadcasts (sfpu_*_bcast_col with VectorMode::R, sfpu_*_bcast_col_band), the parked
// scalar (sfpu_bcast_scalar_load + sfpu_mul_bcast_scalar) and the fused normalisations
// (sfpu_*normalize_bcast_*). Each runs through the same Quasar llk_sfpu entry point and SFPU_BINARY_CALL
// form the Compute API uses.
//
// Buffer layout: the SFPU_BCAST_TILE_CNT tiles of buffer_A are unpacked to Dest[0 .. TILE_CNT - 1]
// (unpack_to_dest, fp32 Dest). Operand tiles T0..T3 (SFPU_BCAST_T0..3) per form:
//   ROWS_0_15, BAND        data = T0 (in place), column tile = T1
//   MUL_SCALAR             data = T0 and T2 (in place, the scalar parked once, the common SFPU init run between
//                          the two multiplies), scalar tile = T1
//   NORM_COL, NORM_SCALAR  data = T0 (in place), mean = T1, inv_std = T2
//   NORM_COL_TWO_TILES     data = T0 and T3 (in place), mean = T1, inv_std = T2
//   RESIDUAL_NORM_COL      data = T0 (in place), mean = T1, inv_std = T2, residual = T3
// PACK writes T0 to result tile 0 and, for the two-output forms, the second data tile to result tile 1.

#include <cstdint>

#include "ckernel.h"
#include "counters.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "perf.h"
#include "profiler.h"
#include "quasar_test_common.h"
#include "sfpu_stub.h"

// Values of SFPU_BCAST_FORM (helpers/test_variant_parameters.py SFPU_BCAST_FORM.FORMS, in order).
enum SfpuBcastForm : int
{
    FORM_ROWS_0_15          = 0,
    FORM_BAND               = 1,
    FORM_MUL_SCALAR         = 2,
    FORM_NORM_COL           = 3,
    FORM_NORM_COL_TWO_TILES = 4,
    FORM_RESIDUAL_NORM_COL  = 5,
    FORM_NORM_SCALAR        = 6,
};

// The second packed tile, when the form writes two data tiles.
constexpr bool sfpu_bcast_form_has_second_output(const int form)
{
    return form == FORM_MUL_SCALAR || form == FORM_NORM_COL_TWO_TILES;
}

#ifdef LLK_TRISC_UNPACK

#include "cfg_defines.h"
#include "ckernel_buf_desc.h"
#include "llk_math_common.h"
#include "llk_unpack_common.h"
#include "llk_unpack_unary_operand_to_dest.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t TEST_FACE_R_DIM = params.TEST_FACE_R_DIM;
    const std::uint32_t num_faces       = params.num_faces;
    const Operand& buffer_A             = params.buffer_A;
    const std::uint32_t TILE_CNT        = params.TILE_CNT;
#endif
    static_assert(unpack_to_dest, "the broadcast forms stage their fp32 operands with unpack_to_dest");
    static_assert(PERF_RUN_TYPE == PerfRunType::L1_TO_L1, "functional test only");

    {
        START_PERF_MEASURE("INIT")
        set_up_unpack_to_sfpu_to_pack_dest_dvalid_chain<dest_dvalid_client::UNPACK>();
        ckernel::trisc::program_bfd<ckernel::trisc::BFD_ID_UNP0>(
            ckernel::tensor_shape_from_num_faces(TEST_FACE_R_DIM, num_faces), L1_ADDRESS(buffer_A[0]), formats.unpack_A_src);
        _llk_unpack_configure_unary_<UNPACKER_ENGINE_SEL>(static_cast<DataFormat>(formats.unpack_A_dst));
        _llk_unpack_dest_init_();
        _llk_unpack_unary_operand_to_dest_init_(ckernel::trisc::BFD_ID_UNP0);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        _llk_unpack_unary_operand_to_dest_block_(0 /*l1_tile_idx*/, 0 /*dst_tile_idx*/, TILE_CNT);
        _llk_unpack_dest_dvalid_section_done_<dest_sync>();
        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_UNPACK

#ifdef LLK_TRISC_MATH

#include "cfg_defines.h"
#include "cmath_common.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_sfpu/ckernel_sfpu_bcast_scalar.h"
#include "llk_sfpu/ckernel_sfpu_binary_bcast.h"
#include "llk_sfpu/llk_math_eltwise_binary_sfpu_macros.h"
#include "params.h"

using namespace ckernel;
using namespace ckernel::math;
using namespace ckernel::sfpu;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    static_assert(is_fp32_dest_acc_en, "the broadcast forms are fp32 Dest only");
    static_assert(SFPU_BCAST_FORM >= FORM_ROWS_0_15 && SFPU_BCAST_FORM <= FORM_NORM_SCALAR, "unknown SFPU_BCAST_FORM");
    constexpr BinaryOp OP = SFPU_BINARY_OP;
    constexpr std::uint32_t T0 = SFPU_BCAST_T0;
    constexpr std::uint32_t T1 = SFPU_BCAST_T1;
    constexpr std::uint32_t T2 = SFPU_BCAST_T2;
    constexpr std::uint32_t T3 = SFPU_BCAST_T3;
    const DataFormat math_format = static_cast<DataFormat>(formats.math);

    {
        START_PERF_MEASURE("INIT")
        set_up_unpack_to_sfpu_to_pack_dest_dvalid_chain<dest_dvalid_client::SFPU>();
        _llk_math_srcAB_hw_configure_<is_fp32_dest_acc_en>(math_format, math_format);
        _llk_math_eltwise_sfpu_init_();
        if constexpr (SFPU_BCAST_FORM == FORM_MUL_SCALAR)
        {
            _sfpu_bcast_scalar_init_();
        }
        else if constexpr (SFPU_BCAST_FORM == FORM_NORM_SCALAR)
        {
            _sfpu_binary_bcast_init_<BroadcastType::ROW>(); // sfpu_normalize_bcast_scalar_init
        }
        else
        {
            _sfpu_binary_bcast_init_<BroadcastType::COL>(); // sfpu_bcast_col_init
        }
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (SFPU_BCAST_FORM == FORM_ROWS_0_15)
        {
            // sfpu_{add,sub,mul}_bcast_col(T0, T1, VectorMode::R)
            SFPU_BINARY_CALL(dest_sync, is_fp32_dest_acc_en, _calculate_sfpu_binary_bcast_col_rows_0_15_, (OP), T0, T1, T0, VectorMode::None);
        }
        else if constexpr (SFPU_BCAST_FORM == FORM_BAND)
        {
            // sfpu_{add,sub,mul}_bcast_col_band(T0, T1, data_band, bcast_band)
            SFPU_BINARY_CALL(
                dest_sync,
                is_fp32_dest_acc_en,
                _calculate_sfpu_binary_bcast_col_band_,
                (OP),
                T0,
                T1,
                T0,
                VectorMode::None,
                SFPU_BCAST_DATA_BAND,
                SFPU_BCAST_COL_BAND);
        }
        else if constexpr (SFPU_BCAST_FORM == FORM_MUL_SCALAR)
        {
            // sfpu_bcast_scalar_load(T1), then sfpu_mul_bcast_scalar<ITERATIONS>(T0 / T2, vector_mode): the
            // second tile reuses the scalar parked by the one load.
            SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(dest_sync, is_fp32_dest_acc_en, _sfpu_bcast_scalar_load_, T1, T1, T1, VectorMode::None);
            SFPU_BINARY_CALL(
                dest_sync,
                is_fp32_dest_acc_en,
                _calculate_mul_bcast_scalar_,
                (SFPU_BCAST_SCALAR_ITERATIONS),
                T0,
                T0,
                T0,
                VectorMode::None,
                SFPU_BCAST_VECTOR_MODE);
            // An SFPU init between the two multiplies, as another op's init would run in a compute kernel: the
            // common SFPU init must leave the parked LREG14 alone.
            _llk_math_eltwise_sfpu_init_();
            SFPU_BINARY_CALL(
                dest_sync,
                is_fp32_dest_acc_en,
                _calculate_mul_bcast_scalar_,
                (SFPU_BCAST_SCALAR_ITERATIONS),
                T2,
                T2,
                T2,
                VectorMode::None,
                SFPU_BCAST_VECTOR_MODE);
        }
        else if constexpr (SFPU_BCAST_FORM == FORM_NORM_COL)
        {
            // sfpu_normalize_bcast_col(T0, mean = T1, inv_std = T2)
            SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
                dest_sync, is_fp32_dest_acc_en, _calculate_sfpu_normalize_bcast_col_full_tile_, T0, T1, T0, VectorMode::None, T2);
        }
        else if constexpr (SFPU_BCAST_FORM == FORM_NORM_COL_TWO_TILES)
        {
            // sfpu_normalize_bcast_col_two_tiles(T0, T3, mean = T1, inv_std = T2)
            SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
                dest_sync, is_fp32_dest_acc_en, _calculate_sfpu_normalize_bcast_col_two_tiles_, T0, T3, T0, VectorMode::None, T1, T2);
        }
        else if constexpr (SFPU_BCAST_FORM == FORM_RESIDUAL_NORM_COL)
        {
            // sfpu_residual_normalize_bcast_col(T0, residual = T3, mean = T1, inv_std = T2)
            SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
                dest_sync, is_fp32_dest_acc_en, _calculate_sfpu_residual_normalize_bcast_col_full_tile_, T0, T3, T0, VectorMode::None, T1, T2);
        }
        else if constexpr (SFPU_BCAST_FORM == FORM_NORM_SCALAR)
        {
            // sfpu_normalize_bcast_scalar(T0, mean = T1, inv_std = T2)
            SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
                dest_sync, is_fp32_dest_acc_en, _calculate_sfpu_normalize_bcast_scalar_full_tile_, T0, T1, T0, VectorMode::None, T2);
        }
        _llk_math_set_dvalid_<p_cleardvalid::SFPU, dest_sync>();
        wait_sfpu_idle();
        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_MATH

#ifdef LLK_TRISC_PACK

#include "cfg_defines.h"
#include "ckernel_buf_desc.h"
#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t TEST_FACE_R_DIM = params.TEST_FACE_R_DIM;
    const std::uint32_t num_faces       = params.num_faces;
    const Operand& buffer_Res           = params.buffer_Res;
#endif
    constexpr std::uint32_t second_out = SFPU_BCAST_FORM == FORM_MUL_SCALAR ? SFPU_BCAST_T2 : SFPU_BCAST_T3;

    {
        START_PERF_MEASURE("INIT")
        set_up_unpack_to_sfpu_to_pack_dest_dvalid_chain<dest_dvalid_client::PACK>();
        ckernel::trisc::program_bfd<ckernel::trisc::BFD_ID_PACK0>(
            ckernel::tensor_shape_from_num_faces(TEST_FACE_R_DIM, num_faces), L1_ADDRESS(buffer_Res[0]), formats.pack_dst);
        _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ckernel::ReluConfig::none());
        _llk_pack_init_(ckernel::trisc::BFD_ID_PACK0, ckernel::DEFAULT_TENSOR_SHAPE, 1 /*num_tiles*/);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        _llk_pack_(SFPU_BCAST_T0, 0 /*start_l1_tile_idx*/, ckernel::DEFAULT_TENSOR_SHAPE);
        if constexpr (sfpu_bcast_form_has_second_output(SFPU_BCAST_FORM))
        {
            _llk_pack_(second_out, 1 /*start_l1_tile_idx*/, ckernel::DEFAULT_TENSOR_SHAPE);
        }
        _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
        PROFILER_SYNC();
    }
}
#endif // LLK_TRISC_PACK
