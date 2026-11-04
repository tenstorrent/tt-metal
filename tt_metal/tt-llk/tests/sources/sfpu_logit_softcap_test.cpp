// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Driver for calculate_logit_softcap (llk_sfpu/ckernel_sfpu_logit_softcap.h).
//
// LOGIT_SOFTCAP_POLYNOMIAL selects tanh_init<false, false>() for the polynomial
// variant or tanh_init<false, true>() for the accurate variant, independently of
// destination precision. Both variants use shared vConstFloatPrgm0/1/2 constants.

#include <cstdint>

#include "ckernel.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr int LOGIT_SOFTCAP_ITERATIONS = 32;
#ifndef LOGIT_SOFTCAP_POLYNOMIAL
#define LOGIT_SOFTCAP_POLYNOMIAL false
#endif
// The polynomial regression streams several tiles. Quiesce MATH while PACK
// uses the shared SFPU/DEST addressing state, then permit the next copy.
static constexpr ckernel::DstSync DST_SYNC = LOGIT_SOFTCAP_POLYNOMIAL ? ckernel::DstSync::SyncFull : ckernel::DstSync::SyncHalf;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
#endif
    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES);

        _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, ckernel::DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            if constexpr (!unpack_to_dest)
            {
                _perf_unpack_loop_set_valid<true /* src A */, is_fp32_dest_acc_en>(LOOP_FACTOR * params.TILE_CNT * TILE_NUM_FACES);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
                {
                    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                        L1_ADDRESS(params.buffer_A[tile]), formats.unpack_A_src, formats.unpack_A_dst);
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
#endif
    {
        START_PERF_MEASURE("INIT")
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();

        _llk_math_eltwise_unary_datacopy_init_wrapper_<
            DataCopyType::A2D,
            is_fp32_dest_acc_en,
            BroadcastType::NONE,
            false /* is_int_fpu_en */,
            PackMode::Default>(ckernel::DEFAULT_TENSOR_SHAPE, formats.math);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            if constexpr (unpack_to_dest)
            {
                for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
                {
                    for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
                    {
                        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                            0 /* dst_index */, formats.math, formats.math);
                    }
                }
            }
            else
            {
                _perf_math_loop_clear_valid<true /* src A */, true /* src B */>(LOOP_FACTOR * params.TILE_CNT * TILE_NUM_FACES);
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            if constexpr (!unpack_to_dest)
            {
                for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
                {
                    for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
                    {
                        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                            0 /* dst_index */, formats.math, formats.math);
                    }
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
                {
                    _llk_math_wait_for_dest_available_<DST_SYNC>();

                    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                        0 /* dst_index */, formats.math, formats.math);

                    _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_pack_common.h"

// This will collide with another header if not qualified with llk_sfpu.
#include "llk_sfpu/ckernel_sfpu_tanh.h"

#define TRISC_PACK
#include "experimental/llk_sfpu/ckernel_sfpu_logit_softcap.h"
#undef TRISC_PACK

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
#endif
    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * TILE_NUM_FACES);
        _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
        _llk_pack_dest_init_wrapper_<DST_SYNC, is_fp32_dest_acc_en, PackMode::Default>();

        _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
        ckernel::sfpu::tanh_init<false /* APPROXIMATION_MODE */, !LOGIT_SOFTCAP_POLYNOMIAL>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
        }
        else
        {
            constexpr bool sync_with_math = PERF_RUN_TYPE == PerfRunType::L1_TO_L1;
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
                {
                    if constexpr (sync_with_math)
                    {
                        _llk_packer_wait_for_math_done_();
                    }

                    _llk_math_eltwise_sfpu_start_(0 /* dst_index */);
                    if constexpr (LOGIT_SOFTCAP_POLYNOMIAL)
                    {
                        ckernel::sfpu::calculate_logit_softcap<is_fp32_dest_acc_en, LOGIT_SOFTCAP_ITERATIONS>(SFPU_UNARY_SCALAR);
                    }
                    else
                    {
                        ckernel::sfpu::calculate_logit_softcap<LOGIT_SOFTCAP_ITERATIONS>(SFPU_UNARY_SCALAR, 0 /* unread */);
                    }
                    _llk_math_eltwise_sfpu_done_();

                    if constexpr (PERF_RUN_TYPE != PerfRunType::MATH_ISOLATE)
                    {
                        _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(0 /* tile_index */, L1_ADDRESS(params.buffer_Res[tile]));
                        if constexpr (sync_with_math)
                        {
                            _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
                        }
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif
