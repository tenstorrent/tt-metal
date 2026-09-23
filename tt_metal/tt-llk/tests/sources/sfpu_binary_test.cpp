// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "counters.h"
#include "llk_defs.h"
#include "perf.h"
#include "profiler.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t num_faces   = params.num_faces;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
    const Operand& buffer_A         = params.buffer_A;
#endif
    // Isolate mocks must match _llk_unpack_A_ dvalids (not halved by even/odd SFPU pairing).
    // Per tile: NONE = num_faces SrcA plus a SrcB zerosrc dvalid (WA #1230) every face,
    // including dest_acc=No; ROW = num_faces SrcB; COL = dummy SrcA + 2 SrcB;
    // SCALAR = dummy SrcA + 1 SrcB.
    const std::uint32_t tile_iters = LOOP_FACTOR * TILE_CNT;

    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, num_faces, num_faces);
        _llk_unpack_A_init_<BROADCAST_TYPE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            0 /* transpose_of_faces */,
            0 /* within_face_16x16_transpose */,
            ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, num_faces),
            formats.unpack_A_src,
            formats.unpack_A_dst);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            // When unpack_to_dest is set, dest is assumed ready and math does not wait on src dvalids.
            if constexpr (!unpack_to_dest)
            {
                if constexpr (BROADCAST_TYPE == BroadcastType::NONE)
                {
                    // Real NONE unpack posts SrcA plus a SrcB zerosrc dvalid (WA #1230)
                    // every face, including dest_acc=No. MATH_ISOLATE must match that.
                    _perf_unpack_loop_set_valid</* src A */ true, /* src B */ true>(/* iterations */ tile_iters * num_faces);
                }
                else if constexpr (BROADCAST_TYPE == BroadcastType::ROW)
                {
                    _perf_unpack_loop_set_valid</* src A */ false, /* src B */ true>(/* iterations */ tile_iters * num_faces);
                }
                else if constexpr (BROADCAST_TYPE == BroadcastType::COL)
                {
                    // Interleave per tile: posting all A+B then all extra B fills the 2-deep src banks
                    // while math is still waiting for the second SrcB of tile 0.
                    for (std::uint32_t i = 0; i < tile_iters; ++i)
                    {
                        _perf_unpack_loop_set_valid</* src A */ true, /* src B */ true>(/* iterations */ 1);
                        _perf_unpack_loop_set_valid</* src A */ false, /* src B */ true>(/* iterations */ 1);
                    }
                }
                else
                {
                    _perf_unpack_loop_set_valid</* src A */ true, /* src B */ true>(/* iterations */ tile_iters);
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t i = 0; i < TILE_CNT; i++)
                {
                    _llk_unpack_A_<BROADCAST_TYPE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                        L1_ADDRESS(buffer_A[i]), formats.unpack_A_src, formats.unpack_A_dst);
                }
            }
        }
        PROFILER_SYNC();
    }
    _llk_unpack_A_uninit_<BROADCAST_TYPE>();
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_defs.h"
#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_binary_sfpu.h"
#include "params.h"
#include "sfpu_operations.h"

using namespace ckernel::sfpu;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR        = params.LOOP_FACTOR;
    const std::uint32_t num_faces          = params.num_faces;
    const int NUM_BLOCKS                   = params.NUM_BLOCKS;
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;
    const std::uint32_t TILE_CNT           = params.TILE_CNT;
#endif
    const bool is_int_fpu_en         = false;
    constexpr DataCopyType copy_type = (BROADCAST_TYPE == BroadcastType::NONE || unpack_to_dest) ? DataCopyType::A2D : DataCopyType::B2D;
    const std::uint32_t tile_iters   = LOOP_FACTOR * TILE_CNT;

    {
        START_PERF_MEASURE("INIT")
        _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        _llk_math_eltwise_unary_datacopy_init_wrapper_<copy_type, is_fp32_dest_acc_en, BROADCAST_TYPE, is_int_fpu_en, PackMode::Default>(
            num_faces, formats.math);
        test_utils::call_binary_sfpu_operation_init<APPROX_MODE, is_fp32_dest_acc_en, SFPU_BINARY_OPERATION, ITERATIONS, formats.math>();
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
                    for (int block = 0; block < NUM_BLOCKS; ++block)
                    {
                        for (std::uint32_t tile = 0; tile < NUM_TILES_IN_BLOCK; ++tile)
                        {
                            _llk_math_eltwise_unary_datacopy_<copy_type, DstSync::SyncHalf, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                                tile, formats.math, formats.math);
                        }
                    }
                }
            }
            else if constexpr (BROADCAST_TYPE == BroadcastType::NONE)
            {
                _perf_math_loop_clear_valid</* src A */ true, /* src B */ true>(/* iterations */ tile_iters * num_faces);
            }
            else if constexpr (BROADCAST_TYPE == BroadcastType::ROW)
            {
                _perf_math_loop_clear_valid</* src A */ false, /* src B */ true>(/* iterations */ tile_iters * num_faces);
            }
            else if constexpr (BROADCAST_TYPE == BroadcastType::COL)
            {
                for (std::uint32_t i = 0; i < tile_iters; ++i)
                {
                    _perf_math_loop_clear_valid</* src A */ true, /* src B */ true>(/* iterations */ 1);
                    _perf_math_loop_clear_valid</* src A */ false, /* src B */ true>(/* iterations */ 1);
                }
            }
            else
            {
                _perf_math_loop_clear_valid</* src A */ true, /* src B */ true>(/* iterations */ tile_iters);
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (int block = 0; block < NUM_BLOCKS; ++block)
                {
                    if constexpr (!unpack_to_dest)
                    {
                        for (std::uint32_t tile = 0; tile < NUM_TILES_IN_BLOCK; ++tile)
                        {
                            _llk_math_eltwise_unary_datacopy_<copy_type, DstSync::SyncHalf, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                                tile, formats.math, formats.math);
                        }
                    }

                    for (std::uint32_t tile = 0; tile < NUM_TILES_IN_BLOCK; tile += 2)
                    {
                        test_utils::
                            call_binary_sfpu_operation<DstSync::SyncHalf, is_fp32_dest_acc_en, APPROX_MODE, SFPU_BINARY_OPERATION, ITERATIONS, formats.math>(
                                tile, tile + 1, tile);
                    }
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (int block = 0; block < NUM_BLOCKS; ++block)
                {
                    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                    for (std::uint32_t tile = 0; tile < NUM_TILES_IN_BLOCK; ++tile)
                    {
                        _llk_math_eltwise_unary_datacopy_<copy_type, DstSync::SyncHalf, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                            tile, formats.math, formats.math);
                    }

                    for (std::uint32_t tile = 0; tile < NUM_TILES_IN_BLOCK; tile += 2)
                    {
                        test_utils::
                            call_binary_sfpu_operation<DstSync::SyncHalf, is_fp32_dest_acc_en, APPROX_MODE, SFPU_BINARY_OPERATION, ITERATIONS, formats.math>(
                                tile, tile + 1, tile);
                    }
                    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
    _llk_math_eltwise_unary_datacopy_uninit_<BROADCAST_TYPE, unpack_to_dest>();
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
    const std::uint32_t LOOP_FACTOR        = params.LOOP_FACTOR;
    const int NUM_BLOCKS                   = params.NUM_BLOCKS;
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;
    const Operand& buffer_Res              = params.buffer_Res;
#endif
    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, 16 * 16 /* tile_size */);
        _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst);
        _llk_pack_dest_init_wrapper_<DstSync::SyncHalf, is_fp32_dest_acc_en, PackMode::Default>();
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
                for (int block = 0; block < NUM_BLOCKS; ++block)
                {
                    for (std::uint32_t tile = 0; tile < NUM_TILES_IN_BLOCK; ++tile)
                    {
                        const std::uint32_t result_tile = block * NUM_TILES_IN_BLOCK + tile;
                        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile, L1_ADDRESS(buffer_Res[result_tile]));
                    }
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (int block = 0; block < NUM_BLOCKS; ++block)
                {
                    _llk_packer_wait_for_math_done_();
                    for (std::uint32_t tile = 0; tile < NUM_TILES_IN_BLOCK; ++tile)
                    {
                        const std::uint32_t result_tile = block * NUM_TILES_IN_BLOCK + tile;
                        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile, L1_ADDRESS(buffer_Res[result_tile]));
                    }
                    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif
