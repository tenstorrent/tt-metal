// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Shadow of the dest-reuse op's callers, per iteration: TILE_CNT scalar tiles copied into DEST, a LoFi ELWSUB on
// tile 0 and a MATH_FIDELITY ELWMUL on each further tile, every call with its own init, then the pack. TILE_CNT 2 is
// blaze softmax_lanes and tt-lang's attention residual, 3 blaze softmax_top_p, 4 the DeepSeek sampling top-p path.
// With RMSNORM_SHADOW_SFPU the math thread also runs the callers' SFPU steps: the exponential of the difference
// (columns, as exp_tile with VectorMode::C) and the reciprocal of each multiply's scalar.

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr ckernel::DstSync DST_SYNC = ckernel::DstSync::SyncHalf;

#ifdef LLK_TRISC_UNPACK

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "experimental/llk_unpack_A_rmsnorm.h"
#pragma GCC diagnostic pop
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
        {
            _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                0, 0, ckernel::DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);
            for (std::uint32_t tile = 0; tile < TILE_CNT; ++tile)
            {
                _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                    L1_ADDRESS(params.buffer_B[0]), formats.unpack_A_src, formats.unpack_A_dst);
            }
            for (std::uint32_t tile = 0; tile < TILE_CNT; ++tile)
            {
                _llk_unpack_A_rmsnorm_init_<1, BroadcastType::SCALAR, true, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                    0, 0, FACE_R_DIM, 4, 0, 0, tile > 0 && RMSNORM_WHOLE_TILE);
                _llk_unpack_A_<BroadcastType::SCALAR, true, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                    L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src, formats.unpack_A_dst);
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "experimental/llk_math_rmsnorm_bcast_scalar_dest_reuse.h"
#pragma GCC diagnostic pop
#include "llk_lib_math_wrappers.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "sfpu_operations.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
    {
        START_PERF_MEASURE("INIT")
        _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
        {
            _llk_math_wait_for_dest_available_<DST_SYNC>();
            _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, PackMode::Default>(
                TILE_NUM_FACES, formats.math);
            for (std::uint32_t tile = 0; tile < TILE_CNT; ++tile)
            {
                _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                    tile, formats.math, formats.math);
            }
            _llk_math_rmsnorm_bcast_scalar_dest_reuse_init_<EltwiseBinaryType::ELWSUB, 1, MathFidelity::LoFi>(4, 0);
            _llk_math_rmsnorm_bcast_scalar_dest_reuse_<EltwiseBinaryType::ELWSUB, 1, DST_SYNC, is_fp32_dest_acc_en, MathFidelity::LoFi, false>(0, 0);
            if constexpr (RMSNORM_SHADOW_SFPU)
            {
                llk_math_eltwise_unary_sfpu_init<SfpuType::exponential>(ckernel::sfpu::exp_init<false, 0x3F800000, false, is_fp32_dest_acc_en>);
                SFPU_UNARY_CALL(
                    DST_SYNC, is_fp32_dest_acc_en, calculate_exponential, (false, is_fp32_dest_acc_en, false, 8, false), 0, VectorMode::C, p_sfpu::kCONST_1_FP16B);
            }
            for (std::uint32_t tile = 1; tile < TILE_CNT; ++tile)
            {
                if constexpr (RMSNORM_SHADOW_SFPU)
                {
                    llk_math_eltwise_unary_sfpu_init<SfpuType::reciprocal>(ckernel::sfpu::recip_init<false, is_fp32_dest_acc_en>);
                    SFPU_UNARY_CALL(DST_SYNC, is_fp32_dest_acc_en, calculate_reciprocal, (false, is_fp32_dest_acc_en, 1), tile, VectorMode::RC_custom);
                }
                _llk_math_rmsnorm_bcast_scalar_dest_reuse_init_<EltwiseBinaryType::ELWMUL, 1, MATH_FIDELITY>(4, 0, RMSNORM_WHOLE_TILE);
                _llk_math_rmsnorm_bcast_scalar_dest_reuse_<EltwiseBinaryType::ELWMUL, 1, DST_SYNC, is_fp32_dest_acc_en, MATH_FIDELITY, false>(tile, tile);
            }
            _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
        }
        PROFILER_SYNC();
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
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * TILE_NUM_FACES);
        _llk_pack_init_wrapper_<PackMode::Default, false>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
        _llk_pack_dest_init_wrapper_<DST_SYNC, is_fp32_dest_acc_en, PackMode::Default>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
        {
            _llk_packer_wait_for_math_done_();
            for (std::uint32_t tile = 0; tile < TILE_CNT; ++tile)
            {
                _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile, L1_ADDRESS(params.buffer_Res[tile]));
            }
            _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
        }
        PROFILER_SYNC();
    }
}

#endif
