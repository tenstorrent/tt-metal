// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Perf twin of sources/sdpa_custom_mm_reuse_dest_srcb_test.cpp. L1_TO_L1 repeats the whole pass (P preload included)
// per iteration; the isolates preload P once in INIT and loop the reuse matmul. The pack thread stands in for the SFPU.

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"

#include "counters.h"
#include "profiler.h"

// Globals required by the test framework.
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr ckernel::DstSync DST_SYNC = ckernel::DstSync::SyncHalf;

#ifndef DST_FIRST
#define DST_FIRST false
#endif
// DST_FIRST: O at tile 0 and P above it (the placement of compute_sdpa_chunk); otherwise P at tile 0 and O above it.
constexpr std::uint32_t P_TILES    = (KT_DIM + 1) / 2;                                      // datacopy tiles holding the KT 16-row P chunks
constexpr std::uint32_t O_TILES    = (NT_DIM + 3) / 4;                                      // 32x32 tiles covered by the NT 16-row O tiles
constexpr std::uint32_t SRC_TILE   = DST_FIRST ? (O_TILES > 2 ? O_TILES : 2) : 0;           // P tile base for the datacopy preload
constexpr std::uint32_t SRC_INDEX  = 64 * SRC_TILE;                                         // P DEST_TARGET offset (SrcB source)
constexpr std::uint32_t DST_TILE   = DST_FIRST ? 0 : (P_TILES > 2 ? P_TILES : 2);           // O tile base
constexpr std::uint32_t DST_INDEX  = 64 * DST_TILE;                                         // O DEST_TARGET offset (64-datum units)
constexpr std::uint32_t PACK_TILES = (DST_TILE + NT_DIM <= 8) ? NT_DIM : (8 - DST_TILE);   // full-tile packs that stay inside the half
static_assert(SRC_TILE + P_TILES <= 8, "P tiles must stay inside the DEST half");
constexpr std::uint32_t SEM_MAX    = (2 * KT_DIM < 15) ? 2 * KT_DIM : 15;                // UNPACK_MATH_DONE depth: one call ahead where it fits
constexpr bool PRELOAD_IN_LOOP     = (PERF_RUN_TYPE == PerfRunType::L1_TO_L1) || (PERF_RUN_TYPE == PerfRunType::L1_CONGESTION);

#ifdef LLK_TRISC_UNPACK

#include "experimental/llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb.h"
#include "experimental/llk_unpack_A_sdpa.h"
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

inline void preload_p(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false>(
        0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, ckernel::DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);
    for (std::uint32_t k = 0; k < P_TILES; ++k)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false>(
            L1_ADDRESS(params.buffer_B[k]), formats.unpack_A_src, formats.unpack_A_dst);
    }
}

inline void reuse_unpack(RUNTIME_PARAMETERS params)
{
    _llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb_init_(NT_DIM, FACE_R_DIM, 4 /* unpA_num_faces */);
    _llk_unpack_A_sdpa_set_srcb_dummy_valid_();
    _llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb_(
        L1_ADDRESS(params.buffer_A[0]), 0 /* tile_index_a */, params.TILE_SIZE_UNPACK_A, KT_DIM, NT_DIM, 1 /* in1_k_stride */);
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src,
            formats.unpack_B_src,
            formats.unpack_A_dst,
            formats.unpack_B_dst,
            FACE_R_DIM,
            FACE_R_DIM,
            4 /* unpA_num_faces */,
            4 /* unpB_num_faces */,
            params.TILE_SIZE_UNPACK_A,
            params.TILE_SIZE_UNPACK_B);
        if constexpr (!PRELOAD_IN_LOOP && PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE)
        {
            preload_p(params);
        }
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
                _perf_unpack_set_valid(ckernel::SrcB);
                for (std::uint32_t i = 0; i < KT_DIM * NT_DIM; ++i)
                {
                    _perf_unpack_set_valid(ckernel::SrcA);
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                if constexpr (PRELOAD_IN_LOOP)
                {
                    preload_p(params);
                }
                reuse_unpack(params);
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "experimental/llk_math_sdpa_custom_mm_reuse_dest_srcb.h"
#pragma GCC diagnostic pop
#include "llk_lib_math_wrappers.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"

inline void preload_p_math(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
        TILE_NUM_FACES, formats.math);
    for (std::uint32_t k = 0; k < P_TILES; ++k)
    {
        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, false>(
            SRC_TILE + k, formats.math, formats.math);
    }
}

inline void reuse_math()
{
    _llk_math_sdpa_custom_mm_reuse_dest_srcb_init_<MATH_FIDELITY>(
        TILE_R_DIM, TILE_C_DIM, TILE_R_DIM, TILE_C_DIM, false /* partial_face */, 0 /* transpose */, KT_DIM);
    _llk_math_sdpa_custom_mm_reuse_dest_srcb_<1 /* output_granularity */, 1 /* input_granularity */>(
        SRC_INDEX, DST_INDEX, false /* transpose */, KT_DIM, NT_DIM, false /* signal_output */);
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    {
        START_PERF_MEASURE("INIT")
        _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        if constexpr (!PRELOAD_IN_LOOP && PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE)
        {
            _llk_math_wait_for_dest_available_<DST_SYNC>();
            preload_p_math(params);
        }
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t i = 0; i < KT_DIM * NT_DIM; ++i)
                {
                    _perf_math_clear_valid(ckernel::SrcA);
                }
                _perf_math_clear_valid(ckernel::SrcB);
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                reuse_math();
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                _llk_math_wait_for_dest_available_<DST_SYNC>();
                preload_p_math(params);
                reuse_math();
                _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"

using namespace ckernel;

// SFPU-producer stand-in: KT tokens per math call, each posted once the semaphore is below its max.
inline void post_tokens()
{
    for (std::uint32_t i = 0; i < KT_DIM; ++i)
    {
        t6_semaphore_wait_on_max<p_stall::STALL_SYNC>(semaphore::UNPACK_MATH_DONE);
        t6_semaphore_post<>(semaphore::UNPACK_MATH_DONE);
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, params.TILE_SIZE_PACK);
        _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst);
        _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en>();
        t6_semaphore_init(semaphore::UNPACK_MATH_DONE, 0, SEM_MAX);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                post_tokens();
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t i = 0; i < PACK_TILES; ++i)
                {
                    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(DST_TILE + i, L1_ADDRESS(params.buffer_Res[i]));
                }
            }
        }
        else
        {
            post_tokens(); // call 0
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                if (loop + 1 < LOOP_FACTOR)
                {
                    post_tokens(); // call loop + 1, one ahead of the pack
                }
                _llk_packer_wait_for_math_done_();
                for (std::uint32_t i = 0; i < PACK_TILES; ++i)
                {
                    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(DST_TILE + i, L1_ADDRESS(params.buffer_Res[i]));
                }
                _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
}

#endif
