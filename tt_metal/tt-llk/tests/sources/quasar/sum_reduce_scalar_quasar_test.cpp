// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Copies every input tile into DEST and sums them into datum 0 with the mul_reduce_scalar reduce tail.
// DEST is handed over with semaphores because the tail mixes FPU and SFPU work in one section.

#include <cstdint>

#include "ckernel.h"
#include "counters.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "perf.h"
#include "profiler.h"
#include "quasar_test_common.h"
#include "sfpu_stub.h"
#include "tensor_shape.h"

#ifdef LLK_TRISC_UNPACK

#include "ckernel_buf_desc.h"
#include "experimental/llk_unpack_mul_reduce_scalar.h"
#include "llk_unpack_common.h"
#include "llk_unpack_unary_operand.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
    const Operand& buffer_A         = params.buffer_A;
#endif
    const ckernel::TensorShape tensor_shape = TENSOR_SHAPE_FROM_PARAMS(params);

    {
        START_PERF_MEASURE("INIT")
        ckernel::trisc::program_bfd<ckernel::trisc::BFD_ID_UNP0>(tensor_shape, L1_ADDRESS(buffer_A[0]), formats.unpack_A_src);
        constexpr std::uint8_t bfd_a = ckernel::trisc::BFD_ID_UNP0;
        if constexpr (is_fp32_dest_acc_en)
        {
            // A 32-bit DEST datacopy runs as ELWADD, so SrcB needs a format too.
            _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(
                static_cast<DataFormat>(formats.unpack_A_dst), static_cast<DataFormat>(formats.unpack_A_dst));
        }
        else
        {
            _llk_unpack_configure_unary_<p_unpacr::UNP_A>(static_cast<DataFormat>(formats.unpack_A_dst));
        }
        _llk_unpack_unary_operand_init_<p_unpacr::UNP_A, false /*transpose*/, is_fp32_dest_acc_en>(bfd_a, tensor_shape, 1 /*num_tiles*/);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
        {
            for (std::uint32_t tile = 0; tile < TILE_CNT; tile++)
            {
                _llk_unpack_unary_operand_<p_unpacr::UNP_A>(tile, tensor_shape);
            }
            _llk_unpack_mul_reduce_scalar_switch_to_reduce_();
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/ckernel_sfpu_fill.h"
#include "experimental/llk_math_mul_reduce_scalar.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_sfpu_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "params.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
#endif
    const ckernel::TensorShape tensor_shape = TENSOR_SHAPE_FROM_PARAMS(params);
    const DataFormat math_format            = static_cast<DataFormat>(formats.math);
    // The copy leaves the input format in SrcA, and SrcB is configured to the same format.
    const DataFormat src_format             = static_cast<DataFormat>(formats.unpack_A_dst);
    constexpr float scaler                  = 1.0f;

    {
        START_PERF_MEASURE("INIT")
        _llk_math_pack_sync_init_<dest_sync>();
        _llk_math_srcAB_hw_configure_<is_fp32_dest_acc_en>(math_format, math_format);
        _llk_math_eltwise_sfpu_init_();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
        {
            _llk_math_wait_for_dest_available_();
            // The reduce tail reprograms the MOP, so the copy init runs every pass.
            _configure_alu_cfg_<AluCfgType::DEFAULT, is_fp32_dest_acc_en>(math_format, math_format);
            _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en>(
                tensor_shape.face_r_dim * tensor_shape.total_num_faces() /*num_rows_per_matrix*/, 1 /*num_matrices*/);
            for (std::uint32_t tile = 0; tile < TILE_CNT; tile++)
            {
                _llk_math_eltwise_unary_datacopy_(tile);
            }

            _llk_math_mul_reduce_scalar_init_<MATH_FIDELITY>(src_format, src_format, tensor_shape);
            _llk_math_mul_reduce_scalar_move_dest_to_src_<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(0 /*tile_idx*/, tensor_shape);
            // Load the scaler into SrcB through DEST tile 0, then zero that tile for the sums.
            _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::_calculate_fill_<2 /*ITERATIONS*/>, 0 /*dst_index*/, VectorMode::RC_custom, scaler);
            _llk_math_mul_reduce_scalar_move_dest_to_src_<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(0 /*tile_idx*/, tensor_shape);
            _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::_calculate_fill_<2 /*ITERATIONS*/>, 0 /*dst_index*/, VectorMode::RC_custom, 0.0f);

            _llk_math_mul_reduce_column_<MATH_FIDELITY>(0 /*dst_index*/, src_format, src_format, tensor_shape);
            for (std::uint32_t tile = 1; tile < TILE_CNT; tile++)
            {
                _llk_math_mul_reduce_scalar_move_dest_to_src_<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(tile, tensor_shape);
                _llk_math_mul_reduce_column_<MATH_FIDELITY>(0 /*dst_index*/, src_format, src_format, tensor_shape);
            }
            _llk_math_mul_reduce_scalar_<MATH_FIDELITY, is_fp32_dest_acc_en>(src_format, src_format);
            _llk_math_mul_reduce_scalar_clear_dvalid_();

            _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_PACK

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
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const Operand& buffer_Res       = params.buffer_Res;
#endif
    const ckernel::TensorShape tensor_shape = TENSOR_SHAPE_FROM_PARAMS(params);

    {
        START_PERF_MEASURE("INIT")
        ckernel::trisc::program_bfd<ckernel::trisc::BFD_ID_PACK0>(tensor_shape, L1_ADDRESS(buffer_Res[0]), formats.pack_dst);
        constexpr std::uint8_t bfd_pack = ckernel::trisc::BFD_ID_PACK0;
        _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ckernel::ReluConfig::none());
        _llk_pack_init_(bfd_pack, tensor_shape, 1 /*num_tiles_per_pack*/);
        _llk_pack_dest_init_<p_pacr::PACK0, dest_sync>();
        _llk_pack_reduce_mask_config_<ReduceDim::REDUCE_SCALAR>(tensor_shape);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
        {
            _llk_packer_wait_for_math_done_();
            _llk_pack_(0 /*dest_tile_idx*/, 0 /*l1_tile_idx*/, tensor_shape);
            _llk_pack_dest_semaphore_section_done_<p_pacr::PACK0, dest_sync, is_fp32_dest_acc_en>();
        }
        _llk_pack_reduce_mask_clear_();
        PROFILER_SYNC();
    }
}

#endif
