// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Perf kernel of the Blackhole mul_reduce_scalar row (experimental/llk_math_mul_reduce_scalar.h and
// llk_unpack_mul_reduce_scalar.h, the fused multiply and reduce-to-scalar of the DeepSeek RMSNorm). Per iteration one
// row of TILE_CNT 32x32 tiles as the functional kernel runs it: the multiply phase (an AB unpack and an ELWMUL per
// tile), the unpack switch to the reduce phase (SrcA and SrcB dummy valids), the reduce tail (its init, 16 MOVD2A of
// tile 0, the SFPU fill of SrcB, MOVD2B, the SFPU clear of DEST[0], the GAPOOL column reduces with 16 MOVD2A per
// further tile, the scalar collapse, CLEARDVALID), and one masked pack of the reduced tile. The pack mask is configured
// once in INIT, as the functional kernel does. Data-valid cadence per row: four SrcA and four SrcB per tile in the
// multiply phase, then one SrcA and one SrcB for the reduce phase. The copy form (sum_reduce_scalar, a datacopy in
// place of the multiply) is not in this kernel: in the perf harness it hung the core in every run type while its
// functional test passes, and that hang is not understood yet.

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "counters.h"
#include "profiler.h"
#include "tensor_shape.h"

using namespace ckernel;

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr DstSync DST_SYNC       = DstSync::SyncHalf;
static constexpr std::uint32_t DST_INDEX = 0;

#ifdef LLK_TRISC_UNPACK

#include "experimental/llk_unpack_mul_reduce_scalar.h"
#include "llk_unpack_AB.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const ckernel::TensorShape tensor_shape = {
        static_cast<std::uint8_t>(FACE_R_DIM), static_cast<std::uint8_t>(FACE_C_DIM), static_cast<std::uint8_t>(params.num_faces_r_dim_A), static_cast<std::uint8_t>(params.num_faces_c_dim_A)};
    const std::uint32_t num_faces = tensor_shape.total_num_faces();
    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, tensor_shape.face_r_dim, tensor_shape.face_r_dim, num_faces, num_faces);
        _llk_unpack_AB_init_<BroadcastType::NONE>(tensor_shape, ckernel::Transpose::None);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t i = 0; i < params.TILE_CNT * num_faces; ++i)
                {
                    _perf_unpack_set_valid(ckernel::SrcA);
                    _perf_unpack_set_valid(ckernel::SrcB);
                }
                _perf_unpack_set_valid(ckernel::SrcA);
                _perf_unpack_set_valid(ckernel::SrcB);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
                {
                    _llk_unpack_AB_<BroadcastType::NONE>(L1_ADDRESS(params.buffer_A[i]), L1_ADDRESS(params.buffer_B[i]));
                }
                _llk_unpack_mul_reduce_scalar_switch_to_reduce_();
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_mul_reduce_scalar.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

namespace ckernel::sfpu
{
// The float fill of sfpu/ckernel_sfpu_fill.h (the header's integer fills do not parse under this SFPI, see
// sum_reduce_scalar_test.cpp).
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_fill_x_(const float value)
{
    sfpi::vFloat fill_val = value;
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::dst_reg[0] = fill_val;
        sfpi::dst_reg++;
    }
}
} // namespace ckernel::sfpu

static constexpr float REDUCE_SCALER = 1.0f;

inline void row_math(const std::uint32_t tile_cnt, const ckernel::TensorShape& tensor_shape, const std::uint32_t math_format)
{
    _llk_math_eltwise_binary_init_<EltwiseBinaryType::ELWMUL, BroadcastType::NONE, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(tensor_shape, 0);
    for (std::uint32_t i = 0; i < tile_cnt; ++i)
    {
        _llk_math_eltwise_binary_<EltwiseBinaryType::ELWMUL, BroadcastType::NONE, DST_SYNC, is_fp32_dest_acc_en, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(
            tensor_shape, i, true /* clear_fp32_dst_acc */);
    }
    _llk_math_mul_reduce_scalar_init_<is_fp32_dest_acc_en, MATH_FIDELITY, false>();
    _llk_math_mul_reduce_scalar_move_dest_to_src_<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(DST_INDEX);
    _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::_calculate_fill_x_<false, 2>, DST_INDEX, VectorMode::RC_custom, REDUCE_SCALER);
    _llk_math_mul_reduce_scalar_move_dest_to_src_<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(DST_INDEX);
    _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::_calculate_fill_x_<false, 2>, DST_INDEX, VectorMode::RC_custom, 0.0f);
    _llk_math_mul_reduce_column_<MATH_FIDELITY>(DST_INDEX, tensor_shape);
    for (std::uint32_t i = 1; i < tile_cnt; ++i)
    {
        _llk_math_mul_reduce_scalar_move_dest_to_src_<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(i);
        _llk_math_mul_reduce_column_<MATH_FIDELITY>(DST_INDEX, tensor_shape);
    }
    _llk_math_mul_reduce_scalar_<MATH_FIDELITY>();
    _llk_math_mul_reduce_scalar_clear_dvalid_();
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t tile_cnt    = params.TILE_CNT;
    const ckernel::TensorShape tensor_shape = {
        static_cast<std::uint8_t>(FACE_R_DIM), static_cast<std::uint8_t>(FACE_C_DIM), static_cast<std::uint8_t>(params.num_faces_r_dim_A), static_cast<std::uint8_t>(params.num_faces_c_dim_A)};
    const std::uint32_t num_faces = tensor_shape.total_num_faces();
    {
        START_PERF_MEASURE("INIT")
        _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        _llk_math_eltwise_unary_sfpu_init_once_();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t i = 0; i < tile_cnt * num_faces; ++i)
                {
                    _perf_math_clear_valid(ckernel::SrcA);
                    _perf_math_clear_valid(ckernel::SrcB);
                }
                _perf_math_clear_valid(ckernel::SrcA);
                _perf_math_clear_valid(ckernel::SrcB);
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                row_math(tile_cnt, tensor_shape, formats.math);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                _llk_math_wait_for_dest_available_<DST_SYNC>();
                row_math(tile_cnt, tensor_shape, formats.math);
                _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const ckernel::TensorShape tensor_shape = {
        static_cast<std::uint8_t>(FACE_R_DIM), static_cast<std::uint8_t>(FACE_C_DIM), static_cast<std::uint8_t>(params.num_faces_r_dim_A), static_cast<std::uint8_t>(params.num_faces_c_dim_A)};
    const std::uint32_t tile_size = tensor_shape.total_tensor_size();
    const std::uint32_t num_faces = tensor_shape.total_num_faces();
    const bool partial_face       = tensor_shape.face_r_dim < FACE_R_DIM;
    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_<is_fp32_dest_acc_en, PackMode::Default>(
            formats.pack_src, formats.pack_dst, tile_size, tensor_shape.face_r_dim, tensor_shape.total_col_dim(), num_faces, partial_face);
        _llk_pack_init_<PackMode::Default, false, false, true>(formats.pack_src, tensor_shape.face_r_dim, tensor_shape.total_col_dim(), num_faces, 1, false);
        _llk_pack_reduce_mask_config_<ReduceDim::REDUCE_SCALAR>();
        _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE || PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(DST_INDEX, L1_ADDRESS(params.buffer_Res[0]));
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                _llk_packer_wait_for_math_done_();
                _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(DST_INDEX, L1_ADDRESS(params.buffer_Res[0]));
                _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
    _llk_pack_reduce_mask_clear_();
}

#endif
