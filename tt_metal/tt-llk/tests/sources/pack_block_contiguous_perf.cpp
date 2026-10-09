// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Perf kernel of the Blackhole block pack: per iteration one block, packed with one _llk_pack_block_contiguous_ call
// (PACK_BLOCK_CONTIGUOUS) or one standard _llk_pack_ per tile. PACK_ISOLATE runs the pack thread alone, no math hand-off.

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const int num_tiles_in_block    = params.NUM_TILES_IN_BLOCK;
    const int num_blocks            = params.NUM_BLOCKS;
    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src,
            formats.unpack_B_src,
            formats.unpack_A_dst,
            formats.unpack_B_dst,
            params.TEST_FACE_R_DIM,
            params.TEST_FACE_R_DIM,
            params.num_faces,
            params.num_faces);
        _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false>(
            0, 0, ckernel::make_tensor_shape_from_legacy(params.TEST_FACE_R_DIM, params.num_faces), formats.unpack_A_src, formats.unpack_A_dst);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (int i = 0; i < num_tiles_in_block * num_blocks; ++i)
                {
                    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false>(
                        L1_ADDRESS(params.buffer_A[i]), formats.unpack_A_src, formats.unpack_A_dst);
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_lib_math_wrappers.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const int num_tiles_in_block    = params.NUM_TILES_IN_BLOCK;
    const int num_blocks            = params.NUM_BLOCKS;
    {
        START_PERF_MEASURE("INIT")
        _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, PackMode::Default>(
            params.num_faces, formats.math);
        _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (int block = 0; block < num_blocks; block++)
                {
                    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                    for (int tile = 0; tile < num_tiles_in_block; tile++)
                    {
                        _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, false>(
                            tile, formats.math, formats.math, params.num_faces);
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

#include "experimental/llk_pack_block.h"
#include "llk_lib_pack_wrappers.h"
#include "llk_pack.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const int num_tiles_in_block    = params.NUM_TILES_IN_BLOCK;
    const int num_blocks            = params.NUM_BLOCKS;
    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
            formats.pack_src, formats.pack_dst, 16 * 16 * 4, params.TEST_FACE_R_DIM, params.in0_tile_c_dim, params.num_faces);
        _llk_pack_init_with_src_wrapper_<PackMode::Default, false /* zero_output */>(
            formats.pack_src,
            formats.pack_dst,
            params.TEST_FACE_R_DIM,
            params.in0_tile_c_dim,
            params.num_faces,
            false /* partial_face */,
            false /* narrow_tile */,
            1 /* num_tiles */);
        _llk_pack_dest_init_wrapper_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        reconfigure_packer_l1_acc(params.L1_ACC);
        if constexpr (PACK_BLOCK_CONTIGUOUS)
        {
            _llk_pack_block_contiguous_mop_config_<>(params.TEST_FACE_R_DIM, params.num_faces);
        }
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (int block = 0; block < num_blocks; block++)
                {
                    if constexpr (PACK_BLOCK_CONTIGUOUS)
                    {
                        _llk_pack_block_contiguous_<DstSync::SyncHalf, is_fp32_dest_acc_en>(
                            0, L1_ADDRESS(params.buffer_Res[block * num_tiles_in_block]), num_tiles_in_block);
                    }
                    else
                    {
                        for (int tile = 0; tile < num_tiles_in_block; ++tile)
                        {
                            _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(
                                tile, L1_ADDRESS(params.buffer_Res[block * num_tiles_in_block + tile]));
                        }
                    }
                }
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (int block = 0; block < num_blocks; block++)
                {
                    _llk_packer_wait_for_math_done_();
                    if constexpr (PACK_BLOCK_CONTIGUOUS)
                    {
                        _llk_pack_block_contiguous_<DstSync::SyncHalf, is_fp32_dest_acc_en>(
                            0, L1_ADDRESS(params.buffer_Res[block * num_tiles_in_block]), num_tiles_in_block);
                    }
                    else
                    {
                        for (int tile = 0; tile < num_tiles_in_block; ++tile)
                        {
                            _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(
                                tile, L1_ADDRESS(params.buffer_Res[block * num_tiles_in_block + tile]));
                        }
                    }
                    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif
