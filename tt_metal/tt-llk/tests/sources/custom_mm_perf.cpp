// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Perf twin of sources/custom_mm_test.cpp: one custom_mm call per loop iteration, TILE_COUNT = kt x ct weight tiles.
// SPLIT_ACC, FINALIZE, DENSE_PACKING and CLEAR_SRC come from perf_custom_mm.py (CUSTOM_MM_PERF_FLAGS).

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"

#include "counters.h"
#include "profiler.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static_assert(!FINALIZE || SPLIT_ACC, "finalize needs split_acc");

#ifdef LLK_TRISC_UNPACK

#include "experimental/llk_unpack_AB_custom_mm.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    {
        START_PERF_MEASURE("INIT")
        // in1 (full tiles, buffer_B) -> SrcA, in0 ([M, 32], buffer_A) -> SrcB: the functional kernel's operand swap.
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_B_src,
            formats.unpack_A_src,
            formats.unpack_B_dst,
            formats.unpack_A_dst,
            params.in1_face_r_dim,
            params.in0_face_r_dim,
            params.num_faces_B,
            params.num_faces_A,
            params.TILE_SIZE_UNPACK_B,
            params.TILE_SIZE_UNPACK_A);
        _llk_unpack_AB_custom_mm_init_<false /* transpose */>(params.in0_face_r_dim, formats.unpack_B_dst, CT_DIM);
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
                _llk_unpack_AB_custom_mm_<false /* read_transposed */, CLEAR_SRC>(
                    L1_ADDRESS(params.buffer_B[0]),
                    L1_ADDRESS(params.buffer_A[0]),
                    0 /* tile_index_a */,
                    0 /* tile_index_b */,
                    params.TILE_SIZE_UNPACK_B,
                    params.TILE_SIZE_UNPACK_A,
                    KT_DIM,
                    CT_DIM);
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_custom_mm.h"
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
        _llk_math_custom_mm_init_<false /* transpose */, SPLIT_ACC, DENSE_PACKING>(params.in0_face_r_dim, CT_DIM);
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
                _llk_math_custom_mm_<FINALIZE>(params.in0_face_r_dim, 0 /* dst_index */, KT_DIM, CT_DIM);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                _llk_math_custom_mm_<FINALIZE>(params.in0_face_r_dim, 0 /* dst_index */, KT_DIM, CT_DIM);
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
        if constexpr (DENSE_PACKING)
        {
            cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>((TILE_NUM_FACES / 2) * FACE_C_DIM * FACE_R_DIM * (is_fp32_dest_acc_en ? 4 : 2));
        }
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
                for (std::uint32_t i = 0; i < CT_DIM; i++)
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
                for (std::uint32_t i = 0; i < CT_DIM; i++)
                {
                    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(i, L1_ADDRESS(params.buffer_Res[i]));
                }
                _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
    if constexpr (DENSE_PACKING)
    {
        cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(TILE_NUM_FACES * FACE_C_DIM * FACE_R_DIM * (is_fp32_dest_acc_en ? 4 : 2));
    }
}

#endif
