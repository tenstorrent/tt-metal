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

// Per tile the unpack posts SrcA+SrcB (phase 1) then SrcA (phase 2). Math
// clears SrcA after MM1 and SrcA+SrcB after MM2. The two banks are depth 2,
// so the isolate mock has to follow that order rather than posting every SrcA first.

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
        _llk_unpack_hadamard_h128_init_(L1_ADDRESS(params.buffer_A[HADAMARD_H16_TILE_INDEX]));
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t tile = 0; tile < LOOP_FACTOR * params.TILE_CNT; ++tile)
            {
                _perf_unpack_loop_set_valid<true, true>(1);
                _perf_unpack_loop_set_valid<true, false>(1);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
                {
                    _llk_unpack_hadamard_h128_(
                        L1_ADDRESS(params.buffer_A[0]),
                        L1_ADDRESS(params.buffer_B[0]),
                        HADAMARD_H16_TILE_INDEX,
                        tile,
                        params.TILE_SIZE_UNPACK_A,
                        params.TILE_SIZE_UNPACK_B);
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
#endif
    {
        START_PERF_MEASURE("INIT")
        _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        _llk_math_hadamard_h128_init_<MATH_FIDELITY, HADAMARD_NORMALIZE>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t tile = 0; tile < LOOP_FACTOR * params.TILE_CNT; ++tile)
            {
                _perf_math_loop_clear_valid<true, false>(1);
                _perf_math_loop_clear_valid<true, true>(1);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_math_wait_for_dest_available_<dest_sync>();
                }

                for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
                {
                    LLK_ASSERT(
                        (tile < get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                        "Hadamard tile index exceeds maximum destination tiles");
                    // CLR_ALL would also wipe the half the packer is draining on
                    // the next LOOP_FACTOR iteration. ZEROACC picks the bank from
                    // the destination offset, so the face index stays bank-relative.
                    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(tile);
                    constexpr std::uint32_t tiles_per_bank = is_fp32_dest_acc_en ? 4 : 8;
                    const std::uint32_t local_tile         = tile & (tiles_per_bank - 1);
                    for (std::uint32_t face = 0; face < 4; ++face)
                    {
                        TT_ZEROACC(p_zeroacc::CLR_16, is_fp32_dest_acc_en, 0, ADDR_MOD_7, get_dest_index_in_faces(local_tile, face));
                    }
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
#endif
    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_<is_fp32_dest_acc_en, ckernel::PackMode::Default>(
            formats.pack_src, formats.pack_dst, params.TILE_SIZE_PACK, FACE_R_DIM, ckernel::TILE_C_DIM, params.num_faces);
        _llk_pack_init_<ckernel::PackMode::Default, false /* zero_output */>(
            formats.pack_src, FACE_R_DIM, ckernel::TILE_C_DIM, params.num_faces, 1 /* num_tiles */, false /* skip_bh_tilize_workaround */);
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
                for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
                {
                    _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile, L1_ADDRESS(params.buffer_Res[tile]));
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
