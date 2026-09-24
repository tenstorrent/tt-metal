// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// matmul_test.cpp with the K loop owned by the LLK: _llk_unpack_AB_matmul_kloop_ / _llk_math_matmul_kloop_ (Wormhole
// alternates the 2x2 MVMUL order between K tiles). Golden path only.

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "counters.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr std::uint32_t MAX_TILES_DEST = ckernel::get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, ckernel::DstTileShape::Tile32x32>();

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_AB_matmul.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR        = params.LOOP_FACTOR;
    const std::uint32_t TILE_SIZE_UNPACK_A = params.TILE_SIZE_UNPACK_A;
    const std::uint32_t TILE_SIZE_UNPACK_B = params.TILE_SIZE_UNPACK_B;
    const std::uint32_t num_faces_A        = params.num_faces_A;
    const std::uint32_t num_faces_B        = params.num_faces_B;

    const std::uint32_t CT_DIM        = params.CT_DIM;
    const std::uint32_t RT_DIM        = params.RT_DIM;
    const std::uint32_t KT_DIM        = params.KT_DIM;
    const bool UNPACK_TRANSPOSE_FACES = params.UNPACK_TRANSPOSE_FACES;
    const Operand& buffer_A           = params.buffer_A;
    const Operand& buffer_B           = params.buffer_B;
#endif

    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src,
            formats.unpack_B_src,
            formats.unpack_A_dst,
            formats.unpack_B_dst,
            FACE_R_DIM,
            FACE_R_DIM,
            num_faces_A,
            num_faces_B,
            TILE_SIZE_UNPACK_A,
            TILE_SIZE_UNPACK_B);
        _llk_unpack_AB_matmul_init_<>(UNPACK_TRANSPOSE_FACES, CT_DIM, RT_DIM, KT_DIM, FACE_R_DIM, FACE_R_DIM, num_faces_A, num_faces_B, false, false);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            return;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            _perf_unpack_matmul_mock(LOOP_FACTOR, RT_DIM, KT_DIM, CT_DIM);
            return;
        }
        else
        {
            LLK_ASSERT(LOOP_FACTOR == 1, "matmul_kloop_test: only the golden (LOOP_FACTOR == 1) path is wired");
            LLK_ASSERT(is_valid_L1_address(L1_ADDRESS(buffer_A[(RT_DIM - 1) * KT_DIM + KT_DIM - 1])), "unpack A real-buffer top address is outside L1");
            LLK_ASSERT(is_valid_L1_address(L1_ADDRESS(buffer_B[(KT_DIM - 1) * CT_DIM + CT_DIM - 1])), "unpack B real-buffer top address is outside L1");
            // in0 is [RT x KT] tiles row-major (K stride 1, row stride KT); in1 is [KT x CT] tiles row-major (K stride CT).
            _llk_unpack_AB_matmul_kloop_<>(
                L1_ADDRESS(buffer_A[0]),
                L1_ADDRESS(buffer_B[0]),
                /* tile_index_a */ 0,
                /* tile_index_b */ 0,
                TILE_SIZE_UNPACK_A,
                TILE_SIZE_UNPACK_B,
                /* partial face */ false,
                /* partial face */ false,
                CT_DIM,
                RT_DIM,
                KT_DIM,
                /* in1_kt_stride */ CT_DIM);
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_matmul.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR   = params.LOOP_FACTOR;
    const std::uint32_t CT_DIM        = params.CT_DIM;
    const std::uint32_t RT_DIM        = params.RT_DIM;
    const std::uint32_t KT_DIM        = params.KT_DIM;
    const bool UNPACK_TRANSPOSE_FACES = params.UNPACK_TRANSPOSE_FACES;
#endif

    {
        START_PERF_MEASURE("INIT")
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        static_assert(THROTTLE_LEVEL == 0, "matmul_kloop_test: the K-loop entry point has no throttled variant");
        _llk_math_matmul_kloop_init_<MATH_FIDELITY>(
            /* tile A */ TILE_R_DIM,
            /* tile A */ TILE_C_DIM,
            /* tile B */ TILE_R_DIM,
            /* tile B */ TILE_C_DIM,
            /* partial face */ false,
            UNPACK_TRANSPOSE_FACES,
            CT_DIM,
            RT_DIM);
        _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            return;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            _perf_math_matmul_mock(LOOP_FACTOR, RT_DIM, KT_DIM, CT_DIM);
            return;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                LLK_ASSERT(
                    (get_dest_max_matmul_tiles(0, CT_DIM, RT_DIM) < get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                    "Block tile index exceeds maximum destination tiles for matmul");

                _llk_math_matmul_kloop_<MATH_FIDELITY>(/* dest_index */ 0, CT_DIM, RT_DIM, KT_DIM);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                LLK_ASSERT(
                    (get_dest_max_matmul_tiles(0, CT_DIM, RT_DIM) < get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                    "Block tile index exceeds maximum destination tiles for matmul");

                _llk_math_wait_for_dest_available_<dest_sync>();
                _llk_math_matmul_kloop_<MATH_FIDELITY>(/* dest_index */ 0, CT_DIM, RT_DIM, KT_DIM);
                _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
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
    const std::uint32_t LOOP_FACTOR    = params.LOOP_FACTOR;
    const std::uint32_t TILE_SIZE_PACK = params.TILE_SIZE_PACK;
    const std::uint32_t CT_DIM         = params.CT_DIM;
    const std::uint32_t RT_DIM         = params.RT_DIM;
    const Operand& buffer_Res          = params.buffer_Res;
#endif

    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, TILE_SIZE_PACK);
        _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst);
        _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE || PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            return;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t tile = 0; tile < CT_DIM * RT_DIM; tile++)
                {
                    const std::uint32_t tile_index = tile % MAX_TILES_DEST;
                    LLK_ASSERT(
                        (tile_index < get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                        "Block tile index exceeds maximum destination tiles for matmul");
                    _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile_index, PERF_ADDRESS(PERF_OUTPUT, tile_index));
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                _llk_packer_wait_for_math_done_();
                // Pack dest occupancy (RT×CT).
                for (std::uint32_t i = 0; i < CT_DIM * RT_DIM; i++)
                {
                    LLK_ASSERT((i < get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()), "i exceeds max dest tiles");
                    // Golden packs unique result tiles. Perf K=32 dest-fill Float32
                    // places buffer_Res past L1; use the PERF_ADDRESS ring like matmul_perf.
                    const std::uint32_t addr = LOOP_FACTOR > 1 ? PERF_ADDRESS(PERF_OUTPUT, i) : L1_ADDRESS(buffer_Res[i]);
                    if (LOOP_FACTOR == 1)
                    {
                        LLK_ASSERT(is_valid_L1_address(L1_ADDRESS(buffer_Res[i])), "pack result real-buffer address is outside L1");
                    }
                    _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(i, addr);
                }
                _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
}

#endif
