// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>

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

static_assert(PERF_RUN_TYPE != PerfRunType::MATH_ISOLATE, "Math isolation not supported for unpack_tilize");

#ifdef LLK_TRISC_UNPACK

#include "llk_lib_unpack_wrappers.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR  = params.LOOP_FACTOR;
    const std::uint32_t num_faces    = params.num_faces;
    const std::uint32_t BLOCK_CT_DIM = params.BLOCK_CT_DIM;
    const std::uint32_t BLOCK_RT_DIM = params.BLOCK_RT_DIM;
    const Operand& buffer_A          = params.buffer_A;
#endif

    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, num_faces, num_faces);
        _llk_unpack_tilize_init_wrapper_(formats.unpack_A_src, formats.unpack_A_dst, BLOCK_CT_DIM, FACE_R_DIM, false /* narrow_tile */, num_faces);
        PROFILER_SYNC();
    }

    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE)
        {
            const std::uint32_t block_ct_dim = _llk_unpack_tilize_block_ct_dim_wrapper_(BLOCK_CT_DIM);
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                std::uint32_t read_offset = 0;
                for (std::uint32_t i = 0; i < BLOCK_RT_DIM; ++i)
                {
                    const std::uint32_t tile_row_addr = LOOP_FACTOR > 1 ? PERF_ADDRESS(PERF_INPUT_A, read_offset) : L1_ADDRESS(buffer_A[read_offset]);
                    for (std::uint32_t j = 0; j < BLOCK_CT_DIM; ++j)
                    {
                        _llk_unpack_tilize_wrapper_(
                            tile_row_addr, j, formats.unpack_A_src, formats.unpack_A_dst, block_ct_dim, FACE_R_DIM, num_faces, false /* narrow_tile */);
                    }
                    read_offset += BLOCK_CT_DIM;
                }
            }
        }
        PROFILER_SYNC();
    }
    _llk_unpack_tilize_uninit_wrapper_(formats.unpack_A_dst, num_faces);
}

#endif

#ifdef LLK_TRISC_MATH

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
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;
    const std::uint32_t NUM_BLOCKS         = params.NUM_BLOCKS;
#endif
    const bool is_int_fpu_en = false;

    {
        START_PERF_MEASURE("INIT")
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        // copy srca to dest
        const bool TILIZE = true;
        _llk_math_eltwise_unary_datacopy_init_wrapper_<
            DataCopyType::A2D,
            is_fp32_dest_acc_en,
            BroadcastType::NONE,
            is_int_fpu_en,
            llk_test_pack_mode_v<false, TILIZE>>(num_faces, formats.math);
        _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
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
                    for (std::uint32_t block = 0; block < NUM_BLOCKS; ++block)
                    {
                        for (std::uint32_t tile = 0; tile < NUM_TILES_IN_BLOCK; ++tile)
                        {
                            _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                                tile, formats.math, formats.math);
                        }
                    }
                }
            }
            else
            {
                const std::uint32_t num_tiles   = NUM_BLOCKS * NUM_TILES_IN_BLOCK;
                const std::uint32_t num_dvalids = _llk_unpack_tilize_num_dvalids_wrapper_(num_tiles, num_faces);
                _perf_math_loop_clear_valid<true, true>(LOOP_FACTOR * num_dvalids);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block = 0; block < NUM_BLOCKS; ++block)
                {
                    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                    for (std::uint32_t tile = 0; tile < NUM_TILES_IN_BLOCK; ++tile)
                    {
                        LLK_ASSERT(
                            (tile < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                            "Block tile index exceeds maximum destination tiles");
                        _llk_math_eltwise_unary_datacopy_wrapper_<
                            DataCopyType::A2D,
                            DstSync::SyncHalf,
                            is_fp32_dest_acc_en,
                            BroadcastType::NONE,
                            unpack_to_dest>(tile, formats.math, formats.math, num_faces);
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
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;
    const std::uint32_t NUM_BLOCKS         = params.NUM_BLOCKS;
    const Operand& buffer_Res              = params.buffer_Res;
#endif
    static constexpr bool UNTILIZE = false;

    {
        START_PERF_MEASURE("INIT")
        static constexpr bool TILIZE = true;
        const bool is_8bit_format    = _llk_pack_skip_bh_tilize_workaround_wrapper_(formats.unpack_A_src);
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, llk_test_pack_mode_v<UNTILIZE, false>>(
            formats.pack_src, formats.pack_dst, 16 * 16 * 4 /* tile_size */, FACE_R_DIM, TILE_C_DIM, num_faces);
        _llk_pack_init_with_src_wrapper_<llk_unpack_tilize_sweep_pack_cfg_mode_v<UNTILIZE, TILIZE>, false /* zero_output */>(
            formats.pack_src,
            formats.pack_dst,
            FACE_R_DIM,
            TILE_C_DIM,
            num_faces,
            false /* partial_face */,
            false /* narrow_tile */,
            1 /* num_tiles */,
            is_8bit_format /* skip_bh_tilize_workaround */);
        _llk_pack_dest_init_wrapper_<DstSync::SyncHalf, is_fp32_dest_acc_en, llk_test_pack_mode_v<UNTILIZE, false>>();
        PROFILER_SYNC();
    }

    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block = 0; block < NUM_BLOCKS; ++block)
                {
                    for (std::uint32_t tile = 0; tile < NUM_TILES_IN_BLOCK; ++tile)
                    {
                        const std::uint32_t res_tile_idx = block * NUM_TILES_IN_BLOCK + tile;
                        LLK_ASSERT(
                            (tile < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                            "Block tile index exceeds maximum destination tiles");
                        const std::uint32_t output_addr = LOOP_FACTOR > 1 ? PERF_ADDRESS(PERF_OUTPUT, res_tile_idx) : L1_ADDRESS(buffer_Res[res_tile_idx]);
                        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, pack_exec_mode_v<UNTILIZE>>(tile, output_addr);
                    }
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block = 0; block < NUM_BLOCKS; ++block)
                {
                    _llk_packer_wait_for_math_done_();
                    for (std::uint32_t tile = 0; tile < NUM_TILES_IN_BLOCK; ++tile)
                    {
                        const std::uint32_t res_tile_idx = block * NUM_TILES_IN_BLOCK + tile;
                        LLK_ASSERT(
                            (tile < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                            "Block tile index exceeds maximum destination tiles");
                        const std::uint32_t output_addr = LOOP_FACTOR > 1 ? PERF_ADDRESS(PERF_OUTPUT, res_tile_idx) : L1_ADDRESS(buffer_Res[res_tile_idx]);
                        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, pack_exec_mode_v<UNTILIZE>>(tile, output_addr);
                    }
                    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif
