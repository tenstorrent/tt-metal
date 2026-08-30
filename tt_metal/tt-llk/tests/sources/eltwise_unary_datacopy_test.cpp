
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
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

#include "llk_lib_unpack_wrappers.h"
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

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
    const Operand& buffer_A                = params.buffer_A;
#endif
    const std::uint32_t num_tiles = NUM_BLOCKS * NUM_TILES_IN_BLOCK;
    // unpack_A posts one DVALID per face. Tilize posts one per tile on Blackhole
    // (per face on Wormhole); see _llk_unpack_tilize_num_dvalids_wrapper_.
    const std::uint32_t src_handshake_iters = LOOP_FACTOR * (tilize_en ? _llk_unpack_tilize_num_dvalids_wrapper_(num_tiles, num_faces) : num_tiles * num_faces);

    {
        START_PERF_MEASURE("INIT")
        if constexpr (!tilize_en)
        {
            _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
                formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, num_faces, num_faces);
            _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                0 /* transpose_of_faces */,
                0 /* within_face_16x16_transpose */,
                ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, num_faces),
                formats.unpack_A_src,
                formats.unpack_A_dst);
        }
        else
        {
            _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
                formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, num_faces, num_faces);
            _llk_unpack_tilize_init_wrapper_(formats.unpack_A_src, formats.unpack_A_dst, BLOCK_CT_DIM, FACE_R_DIM, false /* narrow_tile */);
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
            // Tilize A2D still waits on SrcA (and SrcB when dest is FP32). Leaving
            // unpack idle here deadlocks MATH_ISOLATE.
            if constexpr (!unpack_to_dest)
            {
                _perf_unpack_loop_set_valid</* src A */ true, /* src B */ tilize_en || is_fp32_dest_acc_en>(src_handshake_iters);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                if constexpr (!tilize_en)
                {
                    for (std::uint32_t i = 0; i < num_tiles; ++i)
                    {
                        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                            L1_ADDRESS(buffer_A[i]), formats.unpack_A_src, formats.unpack_A_dst);
                    }
                }
                else
                {
                    for (std::uint32_t i = 0; i < BLOCK_RT_DIM; i++)
                    {
                        const std::uint32_t read_offset = i * BLOCK_CT_DIM;
                        for (std::uint32_t j = 0; j < BLOCK_CT_DIM; j++)
                        {
                            _llk_unpack_tilize_wrapper_(
                                L1_ADDRESS(buffer_A[read_offset]),
                                j,
                                formats.unpack_A_src,
                                formats.unpack_A_dst,
                                0 /* block_ct_dim */,
                                FACE_R_DIM,
                                4 /* num_faces */,
                                false /* narrow_tile */);
                        }
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#ifdef FORMAT_INT32
const bool is_int_fpu_en = true;
#else
const bool is_int_fpu_en = false;
#endif

#include "llk_lib_math_wrappers.h"
#include "llk_lib_unpack_wrappers.h"
#include "params.h"

using namespace ckernel;

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
    const int DST_INDEX                    = params.DST_INDEX;
#endif
    const std::uint32_t num_tiles           = NUM_BLOCKS * NUM_TILES_IN_BLOCK;
    const std::uint32_t src_handshake_iters = LOOP_FACTOR * (tilize_en ? _llk_unpack_tilize_num_dvalids_wrapper_(num_tiles, num_faces) : num_tiles * num_faces);

    {
        START_PERF_MEASURE("INIT")
        _llk_math_eltwise_unary_datacopy_init_wrapper_<
            DataCopyType::A2D,
            is_fp32_dest_acc_en,
            BroadcastType::NONE,
            is_int_fpu_en,
            llk_test_pack_mode_v<false, tilize_en>>(num_faces, formats.math);
        _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
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
                    for (int block_num = 0; block_num < NUM_BLOCKS; ++block_num)
                    {
                        for (std::uint32_t tile_num = 0; tile_num < NUM_TILES_IN_BLOCK; ++tile_num)
                        {
                            _llk_math_eltwise_unary_datacopy_wrapper_<
                                DataCopyType::A2D,
                                DstSync::SyncHalf,
                                is_fp32_dest_acc_en,
                                BroadcastType::NONE,
                                unpack_to_dest>(DST_INDEX + tile_num, formats.math, formats.math, num_faces);
                        }
                    }
                }
            }
            else
            {
                _perf_math_loop_clear_valid</* src A */ true, /* src B */ true>(src_handshake_iters);
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            // Unpack-to-dest fills dest in unpack; math has no source consumer
            // and must not wait on unpack dvalids.
            if constexpr (!unpack_to_dest)
            {
                for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
                {
                    for (int block_num = 0; block_num < NUM_BLOCKS; ++block_num)
                    {
                        for (std::uint32_t tile_num = 0; tile_num < NUM_TILES_IN_BLOCK; ++tile_num)
                        {
                            LLK_ASSERT(
                                (DST_INDEX + tile_num < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                                "tile_num exceeds max dest tiles");
                            _llk_math_eltwise_unary_datacopy_wrapper_<
                                DataCopyType::A2D,
                                DstSync::SyncHalf,
                                is_fp32_dest_acc_en,
                                BroadcastType::NONE,
                                unpack_to_dest>(DST_INDEX + tile_num, formats.math, formats.math, num_faces);
                        }
                    }
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (int block_num = 0; block_num < NUM_BLOCKS; ++block_num)
                {
                    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                    for (std::uint32_t tile_num = 0; tile_num < NUM_TILES_IN_BLOCK; ++tile_num)
                    {
                        LLK_ASSERT(
                            (DST_INDEX + tile_num < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                            "tile_num exceeds max dest tiles");
                        _llk_math_eltwise_unary_datacopy_wrapper_<
                            DataCopyType::A2D,
                            DstSync::SyncHalf,
                            is_fp32_dest_acc_en,
                            BroadcastType::NONE,
                            unpack_to_dest>(DST_INDEX + tile_num, formats.math, formats.math, num_faces);
                    }
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
    const std::uint32_t LOOP_FACTOR        = params.LOOP_FACTOR;
    const std::uint32_t num_faces          = params.num_faces;
    const int NUM_BLOCKS                   = params.NUM_BLOCKS;
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;
    const int DST_INDEX                    = params.DST_INDEX;
    const Operand& buffer_Res              = params.buffer_Res;
#endif
    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, llk_test_pack_mode_v<false, tilize_en>>(
            formats.pack_src, formats.pack_dst, 16 * 16 * 4 /* tile_size */, FACE_R_DIM, TILE_C_DIM, num_faces);
        _llk_pack_init_wrapper_<llk_test_pack_mode_v<false, tilize_en>, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, num_faces);
        _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
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
                for (int block_num = 0; block_num < NUM_BLOCKS; ++block_num)
                {
                    for (std::uint32_t tile_num = 0; tile_num < NUM_TILES_IN_BLOCK; ++tile_num)
                    {
                        LLK_ASSERT(
                            (DST_INDEX + tile_num < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                            "tile_num exceeds max dest tiles");
                        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(
                            DST_INDEX + tile_num, L1_ADDRESS(buffer_Res[block_num * NUM_TILES_IN_BLOCK + tile_num]));
                    }
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (int block_num = 0; block_num < NUM_BLOCKS; ++block_num)
                {
                    _llk_packer_wait_for_math_done_();
                    for (std::uint32_t tile_num = 0; tile_num < NUM_TILES_IN_BLOCK; ++tile_num)
                    {
                        LLK_ASSERT(
                            (DST_INDEX + tile_num < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                            "tile_num exceeds max dest tiles");
                        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(
                            DST_INDEX + tile_num, L1_ADDRESS(buffer_Res[block_num * NUM_TILES_IN_BLOCK + tile_num]));
                    }
                    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
}
#endif
