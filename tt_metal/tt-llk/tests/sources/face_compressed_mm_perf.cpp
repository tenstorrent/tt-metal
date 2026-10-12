// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The unit is one weight tile.

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"

#include "counters.h"
#include "profiler.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static const std::uint32_t META_L1[META_WORDS] = META;
constexpr std::uint32_t BLOCKS                = KT_DIM / 2;

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
    const auto& in1_face_r_dim = params.in1_face_r_dim;
    const auto& in0_face_r_dim = params.in0_face_r_dim;
    const auto& num_faces_B = params.num_faces_B;
    const auto& num_faces_A = params.num_faces_A;
    const auto& TILE_SIZE_UNPACK_B = params.TILE_SIZE_UNPACK_B;
    const auto& TILE_SIZE_UNPACK_A = params.TILE_SIZE_UNPACK_A;
    const auto& buffer_A = params.buffer_A;
#endif
    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_B_src,
            formats.unpack_A_src,
            formats.unpack_B_dst,
            formats.unpack_A_dst,
            in1_face_r_dim,
            in0_face_r_dim,
            num_faces_B,
            num_faces_A,
            TILE_SIZE_UNPACK_B,
            TILE_SIZE_UNPACK_A);
        _llk_unpack_AB_face_compressed_mm_init_<false /* transpose */>(in0_face_r_dim);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            constexpr std::uint32_t nz[BLOCKS] = META_NZ_BLOCKS;
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t b = 0; b < BLOCKS; ++b)
                {
                    _perf_unpack_set_valid(ckernel::SrcB);
                    for (std::uint32_t i = 0; i < nz[b]; ++i)
                    {
                        _perf_unpack_set_valid(ckernel::SrcA);
                    }
                }
                if constexpr (CT_DIM == 1)
                {
                    _perf_unpack_set_valid(ckernel::SrcB);
                    _perf_unpack_set_valid(ckernel::SrcA);
                }
            }
        }
        else
        {
            const std::uint32_t meta = reinterpret_cast<std::uint32_t>(META_L1);
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                _llk_unpack_AB_face_compressed_mm_<CT_DIM, true /* finalize */>(L1_ADDRESS(buffer_A[0]), meta, KT_DIM);
            }
        }
        PROFILER_SYNC();
    }
    _llk_unpack_AB_face_compressed_mm_uninit_(num_faces_B);
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
    const auto& in0_face_r_dim = params.in0_face_r_dim;
#endif
    {
        START_PERF_MEASURE("INIT")
        _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        _llk_math_face_compressed_mm_init_<CT_DIM>(in0_face_r_dim);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            constexpr std::uint32_t nz[BLOCKS] = META_NZ_BLOCKS;
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t b = 0; b < BLOCKS; ++b)
                {
                    for (std::uint32_t i = 0; i < nz[b]; ++i)
                    {
                        _perf_math_clear_valid(ckernel::SrcA);
                    }
                    _perf_math_clear_valid(ckernel::SrcB);
                }
                if constexpr (CT_DIM == 1)
                {
                    _perf_math_clear_valid(ckernel::SrcA);
                    _perf_math_clear_valid(ckernel::SrcB);
                }
            }
        }
        else
        {
            const std::uint32_t meta = reinterpret_cast<std::uint32_t>(META_L1);
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                if constexpr (PERF_RUN_TYPE != PerfRunType::MATH_ISOLATE)
                {
                    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                }
                _llk_math_face_compressed_mm_<CT_DIM, true /* finalize */>(meta, in0_face_r_dim, 0 /*dst_index*/, KT_DIM);
                if constexpr (PERF_RUN_TYPE != PerfRunType::MATH_ISOLATE)
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

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const auto& TILE_SIZE_PACK = params.TILE_SIZE_PACK;
    const auto& in0_face_r_dim = params.in0_face_r_dim;
    const auto& num_faces = params.num_faces;
    const auto& buffer_Res = params.buffer_Res;
#endif
    {
        START_PERF_MEASURE("INIT")
        _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
            formats.pack_src, formats.pack_dst, TILE_SIZE_PACK, in0_face_r_dim, TILE_C_DIM, num_faces, true /*partial_face*/);
        _llk_pack_init_<PackMode::Default, false /*zero_output*/, false /*skip_addrmod_config*/, true /*skip_packer_strides*/, true /*mutex_ADC*/>(
            formats.pack_src, in0_face_r_dim, TILE_C_DIM, num_faces, 1 /*num_tiles*/, false /*skip_bh_tilize_workaround*/);
        cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>((TILE_NUM_FACES / 2) * FACE_C_DIM * FACE_R_DIM * 2);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t i = 0; i < CT_DIM; ++i)
                {
                    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default, true /*mutex_ADC*/>(i, L1_ADDRESS(buffer_Res[i]));
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                _llk_packer_wait_for_math_done_();
                for (std::uint32_t i = 0; i < CT_DIM; ++i)
                {
                    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default, true /*mutex_ADC*/>(i, L1_ADDRESS(buffer_Res[i]));
                }
                _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
    cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(TILE_NUM_FACES * FACE_C_DIM * FACE_R_DIM * 2);
}

#endif
