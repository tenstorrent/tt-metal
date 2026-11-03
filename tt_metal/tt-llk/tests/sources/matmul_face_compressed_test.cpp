// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// PerfConfig reserves operand buffers but intentionally skips host stimulus writes. The face metadata controls
// both source publication and math clears, so copy the exact compile-time stream into reserved L1 before profiling.
inline void initialize_perf_compressed_meta(const std::uint32_t meta_base)
{
#ifdef LLK_PROFILER
    volatile std::uint32_t* meta_ptr = reinterpret_cast<volatile std::uint32_t*>(meta_base);
    for (std::uint32_t word = 0; word < PERF_COMPRESSED_META_WORD_COUNT; ++word)
    {
        meta_ptr[word] = PERF_COMPRESSED_META_WORDS[word];
    }
#else
    (void)meta_base;
#endif
}

// Mirror the concrete MOP's source schedule for isolate modes. Every aligned four-face-row block publishes SrcB
// immediately before its first nonzero weight face, publishes/clears one SrcA per nonzero face, and clears SrcB at
// the block's final face. CT_DIM==1 then publishes/clears one extra pair for split-accumulation finalization.
template <bool post>
inline void face_compressed_isolate_dvalids(const std::uint32_t loop_factor, const std::uint32_t meta_base)
{
    constexpr std::uint32_t metas_per_word  = 5;
    constexpr std::uint32_t meta_stride     = 6;
    constexpr std::uint32_t metas_per_block = 2 * CT_DIM;
    const std::uint32_t* meta_ptr           = reinterpret_cast<const std::uint32_t*>(meta_base);

    for (std::uint32_t loop = 0; loop < loop_factor; ++loop)
    {
        std::uint32_t meta_index = 0;
        for (std::uint32_t block = 0; block < KT_DIM / 2; ++block)
        {
            bool posted_b = false;
            for (std::uint32_t m = 0; m < metas_per_block; ++m, ++meta_index)
            {
                const std::uint32_t meta      = (meta_ptr[meta_index / metas_per_word] >> (meta_stride * (meta_index % metas_per_word))) & 0x3fu;
                const std::uint32_t face_bits = (meta >> 2) & 0xfu;
                for (std::uint32_t face = 0; face < 4; ++face)
                {
                    const bool nonzero = (face_bits & (1u << face)) != 0;
                    const bool last    = m + 1 == metas_per_block && face == 3;
                    if constexpr (post)
                    {
                        if (nonzero)
                        {
                            if (!posted_b)
                            {
                                _perf_unpack_loop_set_valid<false, true>(1);
                                posted_b = true;
                            }
                            _perf_unpack_loop_set_valid<true, false>(1);
                        }
                    }
                    else if (nonzero)
                    {
                        if (last)
                        {
                            _perf_math_loop_clear_valid<true, true>(1);
                        }
                        else
                        {
                            _perf_math_loop_clear_valid<true, false>(1);
                        }
                    }
                    else if (last)
                    {
                        _perf_math_loop_clear_valid<false, true>(1);
                    }
                }
            }
        }
        if constexpr (CT_DIM == 1)
        {
            if constexpr (post)
            {
                _perf_unpack_loop_set_valid<false, true>(1);
                _perf_unpack_loop_set_valid<true, false>(1);
            }
            else
            {
                _perf_math_loop_clear_valid<true, true>(1);
            }
        }
    }
}

#ifdef LLK_TRISC_UNPACK

#include "experimental/llk_unpack_AB_face_compressed_mm.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
#endif

    initialize_perf_compressed_meta(params.buffer_C[0]);

    {
        START_PERF_MEASURE("INIT")
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

        _llk_unpack_AB_face_compressed_mm_init_<false>(params.in0_face_r_dim);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            face_compressed_isolate_dvalids<true>(LOOP_FACTOR, params.buffer_C[0]);
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                _llk_unpack_AB_face_compressed_mm_<CT_DIM, true, true>(L1_ADDRESS(params.buffer_A[0]), params.buffer_C[0], KT_DIM);
            }
        }
        PROFILER_SYNC();
    }

    _llk_unpack_AB_face_compressed_mm_uninit_(params.num_faces_B);
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_face_compressed_mm.h"
#include "llk_math_common.h"

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
        _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

        _llk_math_face_compressed_mm_init_<CT_DIM>(params.in0_face_r_dim);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            face_compressed_isolate_dvalids<false>(LOOP_FACTOR, params.buffer_C[0]);
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                }
                _llk_math_face_compressed_mm_<CT_DIM, true>(params.buffer_C[0], params.in0_face_r_dim, 0, KT_DIM);
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "params.h"

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
        _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
            formats.pack_src, formats.pack_dst, params.TILE_SIZE_PACK, params.in0_face_r_dim, TILE_C_DIM, params.num_faces, true);

        _llk_pack_init_<PackMode::Default, false /*zero_output*/, false /*skip_addrmod_config*/, true /*skip_packer_strides*/, true /*mutex_ADC*/>(
            formats.pack_src, params.in0_face_r_dim, TILE_C_DIM, params.num_faces, 1 /*num_tiles*/, false /*skip_bh_tilize_workaround*/);
        cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>((TILE_NUM_FACES / 2) * FACE_C_DIM * FACE_R_DIM * 2);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_packer_wait_for_math_done_();
                }
                for (std::uint32_t i = 0; i < CT_DIM; i++)
                {
                    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default, true /* mutex_ADC */>(i, L1_ADDRESS(params.buffer_Res[i]));
                }
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }

    cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(TILE_NUM_FACES * FACE_C_DIM * FACE_R_DIM * 2);
}

#endif
