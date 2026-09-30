// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Perf kernel of the Blackhole sdpa_custom_mm LLK (the SDPA Q K^T matmul with its FPU -> SFPU semaphore posts), the perf
// twin of sources/sdpa_custom_mm_test.cpp. Per loop iteration: one unpack call, one math call (ZEROACC of the ct output
// tiles, the kt-deep MVMUL walk, one FPU_SFPU post per SIGNAL_GRANULARITY tiles on the last k step) and the pack thread
// standing in for the SFPU consumer (one FPU_SFPU get per post), as the functional kernel does. TILE_COUNT is kt x ct
// K (in1) tiles per call. dvalid mocks: the custom_mm cadence (rt_dim 1). No mask re-entry.
// SIGNAL_GRANULARITY, READ_TRANSPOSED and MM_TRANSPOSE come from SDPA_CUSTOM_MM_FLAGS.

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

#ifndef SIGNAL_GRANULARITY
#define SIGNAL_GRANULARITY 1
#endif
#ifndef READ_TRANSPOSED
#define READ_TRANSPOSED false
#endif
#ifndef MM_TRANSPOSE
#define MM_TRANSPOSE false
#endif

// The pack thread takes its FPU_SFPU tokens after the matmul, so the posts of one call must fit the 4-bit Tensix
// semaphore; sixteen posts wedge the core.
static_assert(
    CT_DIM / SIGNAL_GRANULARITY <= ckernel::semaphore::SEMAPHORE_MAX_VALUE,
    "CT_DIM / SIGNAL_GRANULARITY FPU->SFPU posts per call must fit the 4-bit Tensix semaphore (at most 15)");

#ifdef LLK_TRISC_UNPACK

#include "experimental/llk_unpack_AB_sdpa_custom_mm.h"
#include "llk_unpack_common.h"

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
            params.in1_face_r_dim,
            params.in0_face_r_dim,
            params.num_faces_A,
            params.num_faces_B,
            params.TILE_SIZE_UNPACK_A,
            params.TILE_SIZE_UNPACK_B);
        _llk_unpack_AB_custom_mm_init_<MM_TRANSPOSE>(params.in0_face_r_dim, formats.unpack_A_dst, CT_DIM);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            _perf_unpack_matmul_mock(LOOP_FACTOR, 1 /* rt */, KT_DIM, CT_DIM);
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                _llk_unpack_AB_sdpa_custom_mm_<READ_TRANSPOSED, false /* configure_mask_extent */>(
                    L1_ADDRESS(params.buffer_A[0]), // in1 (SrcA, rhs)
                    L1_ADDRESS(params.buffer_B[0]), // in0 (SrcB, lhs)
                    0,                              // base_address_mask
                    0,                              // tile_index_a
                    0,                              // tile_index_b
                    params.TILE_SIZE_UNPACK_A,
                    params.TILE_SIZE_UNPACK_B,
                    KT_DIM,
                    CT_DIM,
                    false, // mask_chunk
                    params.in0_face_r_dim);
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "experimental/llk_math_sdpa_custom_mm.h"
#pragma GCC diagnostic pop
#include "llk_math_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    {
        START_PERF_MEASURE("INIT")
        _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        _llk_math_sdpa_custom_mm_init_<MM_TRANSPOSE>(params.in0_face_r_dim, CT_DIM);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            _perf_math_matmul_mock(LOOP_FACTOR, 1 /* rt */, KT_DIM, CT_DIM);
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                _llk_math_sdpa_custom_mm_<SIGNAL_GRANULARITY>(params.in0_face_r_dim, 0 /* dst_index */, KT_DIM, CT_DIM, false /* mask_chunk */);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                _llk_math_sdpa_custom_mm_<SIGNAL_GRANULARITY>(params.in0_face_r_dim, 0 /* dst_index */, KT_DIM, CT_DIM, false /* mask_chunk */);
                _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
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

// The SFPU consumer stand-in: one FPU_SFPU get per post (CT_DIM / SIGNAL_GRANULARITY posts per call). Waiting until the
// semaphore is non-zero before each get keeps the pack thread from running ahead of the math in MATH_ISOLATE.
inline void consume_fpu_sfpu_posts()
{
    for (std::uint32_t signal = 0; signal < CT_DIM / SIGNAL_GRANULARITY; ++signal)
    {
        t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::FPU_SFPU);
        t6_semaphore_get<p_stall::PACK>(semaphore::FPU_SFPU);
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
        _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
            formats.pack_src, formats.pack_dst, params.TILE_SIZE_PACK, params.in0_face_r_dim, TILE_C_DIM, params.num_faces, true);
        _llk_pack_init_<PackMode::Default, false /*zero_output*/, false /*skip_addrmod_config*/, true /*skip_packer_strides*/>(
            formats.pack_src, params.in0_face_r_dim, TILE_C_DIM, params.num_faces, 1 /*num_tiles*/, false /*skip_bh_tilize_workaround*/);
        // sdpa_custom_mm_block_init_pack_short(): the dense 16-row output tile.
        cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Zstride_RMW>(FACE_C_DIM * 8 * 2);
        cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>((TILE_NUM_FACES / 2) * FACE_C_DIM * 8 * 2);
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
                consume_fpu_sfpu_posts();
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t i = 0; i < CT_DIM; ++i)
                {
                    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(i, L1_ADDRESS(params.buffer_Res[i]));
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                _llk_packer_wait_for_math_done_();
                consume_fpu_sfpu_posts();
                for (std::uint32_t i = 0; i < CT_DIM; ++i)
                {
                    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(i, L1_ADDRESS(params.buffer_Res[i]));
                }
                _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
    // sdpa_custom_mm_block_uninit(): restore the default tile strides.
    cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Zstride_RMW>(FACE_C_DIM * FACE_R_DIM * 2);
    cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(TILE_NUM_FACES * FACE_C_DIM * FACE_R_DIM * 2);
}

#endif
