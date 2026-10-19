// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Scratch perf kernel for the tilize all-configuration sweep (not for merge). unpack_tilize_perf.cpp with the face
// geometry, the DEST sync, the tiles per DEST section and the block form as macros:
//   TA_FACE_R face rows, TA_NF faces, TA_FULL 1 = DstSync::SyncFull, TA_SECT tiles per DEST section (0 = as many as fit),
//   UNPACK_BLOCK 1 = one _llk_unpack_tilize_block_ per row and one _llk_pack_block_ per section (#60672 only).

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "build.h"
#include "ckernel.h"
#include "counters.h"
#include "cunpack_common.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

#ifndef UNPACK_BLOCK
#define UNPACK_BLOCK 0
#endif
#ifndef TA_FACE_R
#define TA_FACE_R 16
#endif
#ifndef TA_NF
#define TA_NF 4
#endif
#ifndef TA_FULL
#define TA_FULL 0
#endif
#ifndef TA_SECT
#define TA_SECT 0
#endif

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static_assert(PERF_RUN_TYPE != PerfRunType::MATH_ISOLATE, "Math isolation not supported for unpack_tilize");

static constexpr ckernel::DstSync TA_SYNC     = TA_FULL ? ckernel::DstSync::SyncFull : ckernel::DstSync::SyncHalf;
static constexpr std::uint32_t MAX_TILES_DEST = (is_fp32_dest_acc_en ? 4 : 8) * (TA_FULL ? 2 : 1);
static constexpr std::uint32_t SECT_TILES     = (TA_SECT == 0 || TA_SECT > MAX_TILES_DEST) ? MAX_TILES_DEST : TA_SECT;

#ifdef LLK_TRISC_UNPACK

#include "llk_lib_unpack_wrappers.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR  = params.LOOP_FACTOR;
    const std::uint32_t BLOCK_CT_DIM = params.BLOCK_CT_DIM;
    const std::uint32_t BLOCK_RT_DIM = params.BLOCK_RT_DIM;
#endif
    constexpr std::uint32_t src = 0x65000;
    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, TA_FACE_R, TA_FACE_R, TA_NF, TA_NF);
#if UNPACK_BLOCK
        _llk_unpack_tilize_init_wrapper_(formats.unpack_A_src, formats.unpack_A_dst, BLOCK_CT_DIM, TA_FACE_R, false, TA_NF, true);
#else
        _llk_unpack_tilize_init_wrapper_(formats.unpack_A_src, formats.unpack_A_dst, BLOCK_CT_DIM, TA_FACE_R, false, TA_NF);
#endif
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t i = 0; i < BLOCK_RT_DIM; i++)
                {
                    const std::uint32_t tile_row_addr = L1_ADDRESS(src + (i % 8) * 0x1000);
#if UNPACK_BLOCK
                    _llk_unpack_tilize_block_wrapper_(
                        tile_row_addr, 0, BLOCK_CT_DIM, formats.unpack_A_src, formats.unpack_A_dst, 0, TA_FACE_R, TA_NF, false);
#else
                    for (std::uint32_t j = 0; j < BLOCK_CT_DIM; j++)
                    {
                        _llk_unpack_tilize_wrapper_(tile_row_addr, j, formats.unpack_A_src, formats.unpack_A_dst, 0, TA_FACE_R, TA_NF, false);
                    }
#endif
                }
            }
        }
        PROFILER_SYNC();
    }
    _llk_unpack_tilize_uninit_wrapper_(formats.unpack_A_dst);
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_lib_math_wrappers.h"
#include "llk_lib_unpack_wrappers.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
#endif
    const bool is_int_fpu_en = false;
    {
        START_PERF_MEASURE("INIT")
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, is_int_fpu_en, llk_test_pack_mode_v<false, true>>(
            TA_NF, formats.math);
        _llk_math_pack_sync_init_<TA_SYNC, is_fp32_dest_acc_en>();
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
            const std::uint32_t NUM_DVALIDS = _llk_unpack_tilize_num_dvalids_wrapper_(TILE_CNT, TA_NF);
            if constexpr (!unpack_to_dest)
            {
                _perf_math_loop_clear_valid<true, true>(LOOP_FACTOR * NUM_DVALIDS);
                return;
            }
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t i = 0; i < TILE_CNT; i++)
                {
                    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, TA_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                        i % MAX_TILES_DEST, formats.math, formats.math);
                }
            }
            return;
        }
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
        {
            std::uint32_t remaining_tiles = TILE_CNT;
            while (remaining_tiles > 0)
            {
                _llk_math_wait_for_dest_available_<TA_SYNC>();
                const std::uint32_t num_tiles = std::min(remaining_tiles, SECT_TILES);
                for (std::uint32_t i = 0; i < num_tiles; ++i)
                {
                    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, TA_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                        i, formats.math, formats.math);
                }
                _llk_math_dest_section_done_<TA_SYNC, is_fp32_dest_acc_en>();
                remaining_tiles -= num_tiles;
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"

#if UNPACK_BLOCK
inline void pack_tilize_block(const std::uint32_t num_tiles, const std::uint32_t unpack_src_format)
{
    if (_llk_pack_skip_bh_tilize_workaround_wrapper_(unpack_src_format))
    {
        _llk_pack_block_<TA_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(0, PERF_ADDRESS(PERF_OUTPUT, 0), num_tiles);
    }
    else
    {
        _llk_pack_block_<TA_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Tilize>(0, PERF_ADDRESS(PERF_OUTPUT, 0), num_tiles);
    }
}
#endif

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
#endif
    static constexpr bool UNTILIZE = false;
    {
        START_PERF_MEASURE("INIT")
        const bool skip_bh_tilize_workaround = _llk_pack_skip_bh_tilize_workaround_wrapper_(formats.unpack_A_src);
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, llk_unpack_tilize_sweep_pack_cfg_mode_v<UNTILIZE, false>>(
            formats.pack_src, formats.pack_dst, 16 * 16 * 4, TA_FACE_R, TILE_C_DIM, TA_NF, false, false, 0);
        _llk_pack_init_with_src_wrapper_<llk_unpack_tilize_sweep_pack_cfg_mode_v<UNTILIZE, true>, false>(
            formats.pack_src, formats.pack_dst, TA_FACE_R, TILE_C_DIM, TA_NF, false, false, 1, skip_bh_tilize_workaround);
        _llk_pack_dest_init_wrapper_<TA_SYNC, is_fp32_dest_acc_en, llk_test_pack_mode_v<UNTILIZE, false>>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            return;
        }
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t remaining_tiles = TILE_CNT; remaining_tiles > 0;)
                {
                    const std::uint32_t num_tiles = std::min(remaining_tiles, SECT_TILES);
#if UNPACK_BLOCK
                    if (num_tiles > 1)
                    {
                        pack_tilize_block(num_tiles, formats.unpack_A_src);
                        remaining_tiles -= num_tiles;
                        continue;
                    }
#endif
                    for (std::uint32_t i = 0; i < num_tiles; ++i)
                    {
                        _llk_pack_<TA_SYNC, is_fp32_dest_acc_en, pack_exec_mode_v<UNTILIZE>>(i, PERF_ADDRESS(PERF_OUTPUT, i));
                    }
                    remaining_tiles -= num_tiles;
                }
            }
            PROFILER_SYNC();
            return;
        }
        for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
        {
            std::uint32_t remaining_tiles = TILE_CNT;
            while (remaining_tiles > 0)
            {
                const std::uint32_t num_tiles = std::min(remaining_tiles, SECT_TILES);
                _llk_packer_wait_for_math_done_();
#if UNPACK_BLOCK
                if (num_tiles > 1)
                {
                    pack_tilize_block(num_tiles, formats.unpack_A_src);
                    _llk_pack_dest_section_done_<TA_SYNC, is_fp32_dest_acc_en>();
                    remaining_tiles -= num_tiles;
                    continue;
                }
#endif
                for (std::uint32_t i = 0; i < num_tiles; ++i)
                {
                    _llk_pack_<TA_SYNC, is_fp32_dest_acc_en, pack_exec_mode_v<UNTILIZE>>(i, PERF_ADDRESS(PERF_OUTPUT, i));
                }
                _llk_pack_dest_section_done_<TA_SYNC, is_fp32_dest_acc_en>();
                remaining_tiles -= num_tiles;
            }
        }
        PROFILER_SYNC();
    }
}

#endif
