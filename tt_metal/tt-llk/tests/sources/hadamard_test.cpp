// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/* H128 (1x128) Hadamard transform LLK test (Blackhole only).

   operandA (h16) is in srcB, operandB (the 1x128 input) is in srcA. buffer_A therefore holds
   the persistent H_16 tile and buffer_B the input tiles, one per transform.
   The output is in the first face of each tile. */

#include <cstdint>

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

// The op reads exactly one face per operand; see the file header.
constexpr std::uint32_t HADAMARD_NUM_FACES = 1;

// Each tile posts SrcA+SrcB, then SrcA; isolation mocks must preserve that order.

#ifdef LLK_TRISC_UNPACK

#include "experimental/llk_unpack_hadamard.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT           = params.TILE_CNT;
    const std::uint32_t TILE_SIZE_UNPACK_A = params.TILE_SIZE_UNPACK_A;
    const std::uint32_t TILE_SIZE_UNPACK_B = params.TILE_SIZE_UNPACK_B;
    const Operand& buffer_A                = params.buffer_A;
    const Operand& buffer_B                = params.buffer_B;
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
            HADAMARD_NUM_FACES,
            HADAMARD_NUM_FACES);
        _llk_unpack_hadamard_h128_init_(L1_ADDRESS(buffer_A[HADAMARD_H16_TILE_INDEX]));
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t tile = 0; tile < LOOP_FACTOR * TILE_CNT; ++tile)
            {
                _perf_unpack_stream_set_valid<true /* set_a */, true /* set_b */, PERF_RUN_TYPE>(1);
                _perf_unpack_stream_set_valid<true /* set_a */, false /* set_b */, PERF_RUN_TYPE>(1);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t tile = 0; tile < TILE_CNT; ++tile)
                {
                    _llk_unpack_hadamard_h128_(
                        L1_ADDRESS(buffer_A[0]), L1_ADDRESS(buffer_B[0]), HADAMARD_H16_TILE_INDEX, tile, TILE_SIZE_UNPACK_A, TILE_SIZE_UNPACK_B);
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_hadamard.h"
#include "llk_math_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
#endif
    {
        START_PERF_MEASURE("INIT")
        _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        _llk_math_hadamard_h128_init_<MATH_FIDELITY, HADAMARD_NORMALIZE>();
        _llk_math_wait_for_dest_available_<dest_sync>();
        TTI_ZEROACC(p_zeroacc::CLR_ALL, is_fp32_dest_acc_en, 0, ADDR_MOD_7, 0);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            for (std::uint32_t tile = 0; tile < LOOP_FACTOR * TILE_CNT; ++tile)
            {
                _perf_math_stream_clear_valid<true /* clear_a */, false /* clear_b */, PERF_RUN_TYPE>(1);
                _perf_math_stream_clear_valid<true /* clear_a */, true /* clear_b */, PERF_RUN_TYPE>(1);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                // PACK's section release clears the bank before MATH reacquires it.
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    if (loop != 0)
                    {
                        _llk_math_wait_for_dest_available_<dest_sync>();
                    }
                }

                for (std::uint32_t tile = 0; tile < TILE_CNT; ++tile)
                {
                    LLK_ASSERT(
                        (tile < get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                        "Hadamard tile index exceeds maximum destination tiles");
                    _llk_math_hadamard_h128_<MATH_FIDELITY, HADAMARD_NORMALIZE>(tile);
                }

                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
    _llk_math_hadamard_h128_uninit_();
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

#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT       = params.TILE_CNT;
    const std::uint32_t TILE_SIZE_PACK = params.TILE_SIZE_PACK;
    const std::uint32_t num_faces      = params.num_faces;
    const Operand& buffer_Res          = params.buffer_Res;
#endif
    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_<is_fp32_dest_acc_en, ckernel::PackMode::Default>(
            formats.pack_src, formats.pack_dst, TILE_SIZE_PACK, FACE_R_DIM, ckernel::TILE_C_DIM, num_faces);
        _llk_pack_init_<ckernel::PackMode::Default, false /* zero_output */>(
            formats.pack_src, FACE_R_DIM, ckernel::TILE_C_DIM, num_faces, 1 /* num_tiles */, false /* skip_bh_tilize_workaround */);
        _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
        }
        else
        {
            constexpr bool sync_with_math = PERF_RUN_TYPE == PerfRunType::L1_TO_L1;
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                if constexpr (sync_with_math)
                {
                    _llk_packer_wait_for_math_done_();
                }
                for (std::uint32_t tile = 0; tile < TILE_CNT; ++tile)
                {
                    _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile, L1_ADDRESS(buffer_Res[tile]));
                }
                if constexpr (sync_with_math)
                {
                    _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif
