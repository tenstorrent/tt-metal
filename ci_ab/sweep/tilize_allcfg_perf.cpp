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
// 1 where the tree's tilize_block unpacks a row to a BFP output one call per tile
#ifndef TA_BFP_PER_TILE
#if __has_include("llk_tilize_bfp_per_tile.h")
#define TA_BFP_PER_TILE 1
#else
#define TA_BFP_PER_TILE 0
#endif
#endif
#ifndef TA_BFP2_PER_TILE
#if __has_include("llk_tilize_bfp2_per_tile.h")
#define TA_BFP2_PER_TILE 1
#else
#define TA_BFP2_PER_TILE 0
#endif
#endif
// 1 where the tree's llk_unpack_tilize_block unpacks 8-bit and unpack-to-DEST rows one call per tile (the X3 fix)
#ifndef TA_BLOCK_U2D_PER_TILE
#if __has_include("llk_tilize_block_per_tile_32b.h")
#define TA_BLOCK_U2D_PER_TILE 1
#else
#define TA_BLOCK_U2D_PER_TILE 0
#endif
#endif

// 1 where the tree packs a BFP output block with one closed tile per MOP run
#ifndef TA_BFP_CLOSED
#if __has_include("llk_tilize_bfp_closed.h")
#define TA_BFP_CLOSED 1
#else
#define TA_BFP_CLOSED 0
#endif
#endif

// 1 where the tree's tilize_block takes the unpack to DEST tiles of a row's DEST section under one handshake
#ifndef TA_DEST_RUNS
#if __has_include("llk_tilize_dest_runs.h")
#define TA_DEST_RUNS 1
#else
#define TA_DEST_RUNS 0
#endif
#endif

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static_assert(PERF_RUN_TYPE != PerfRunType::MATH_ISOLATE, "Math isolation not supported for unpack_tilize");

static constexpr ckernel::DstSync TA_SYNC     = TA_FULL ? ckernel::DstSync::SyncFull : ckernel::DstSync::SyncHalf;
static constexpr std::uint32_t MAX_TILES_DEST = (is_fp32_dest_acc_en ? 4 : 8) * (TA_FULL ? 2 : 1);
static constexpr std::uint32_t SECT_TILES     = (TA_SECT == 0 || TA_SECT > MAX_TILES_DEST) ? MAX_TILES_DEST : TA_SECT;
// As tilize_block with TA_DEST_RUNS: per row, one DEST section and one handshake per run of up to SECT_TILES four-face tiles
static constexpr bool DEST_RUNS = UNPACK_BLOCK && TA_DEST_RUNS && unpack_to_dest && is_fp32_dest_acc_en && TA_NF == 4;

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
#if UNPACK_BLOCK && TA_DEST_RUNS
                    if constexpr (DEST_RUNS)
                    {
                        _llk_unpack_tilize_dest_block_(
                            tile_row_addr, 0, BLOCK_CT_DIM, formats.unpack_A_src, formats.unpack_A_dst, TA_FACE_R, TA_NF, false, SECT_TILES);
                        continue;
                    }
#endif
#if UNPACK_BLOCK
                    const bool bfp2_per_tile = (TA_BFP_PER_TILE || (TA_BFP2_PER_TILE && BLOCK_CT_DIM == 2)) && IS_BFP_FORMAT(formats.pack_dst);
                    if (!bfp2_per_tile && (!TA_BLOCK_U2D_PER_TILE || !(unpack_to_dest || IS_8BIT_FORMAT(formats.unpack_A_src))))
                    {
                        _llk_unpack_tilize_block_wrapper_(
                            tile_row_addr, 0, BLOCK_CT_DIM, formats.unpack_A_src, formats.unpack_A_dst, 0, TA_FACE_R, TA_NF, false);
                    }
                    else
                    {
                        for (std::uint32_t j = 0; j < BLOCK_CT_DIM; j++)
                        {
                            _llk_unpack_tilize_wrapper_(tile_row_addr, j, formats.unpack_A_src, formats.unpack_A_dst, 0, TA_FACE_R, TA_NF, false);
                        }
                    }
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

#if TA_DEST_RUNS
// As llk_math_eltwise_unary_datacopy_block: a run of two or more tiles takes one handshake
inline void math_dest_run(const std::uint32_t n, const std::uint32_t fmt)
{
    if (n > 1)
    {
        _llk_math_eltwise_unary_datacopy_block_<DataCopyType::A2D, TA_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(0, n, fmt, fmt, TA_NF);
        return;
    }
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, TA_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(0, fmt, fmt);
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
#if !defined(SPEED_OF_LIGHT) && TA_DEST_RUNS
    [[maybe_unused]] const std::uint32_t BLOCK_CT_DIM = params.BLOCK_CT_DIM;
    [[maybe_unused]] const std::uint32_t BLOCK_RT_DIM = params.BLOCK_RT_DIM;
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
#if TA_DEST_RUNS
            if constexpr (DEST_RUNS)
            {
                for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
                {
                    for (std::uint32_t row = 0; row < BLOCK_RT_DIM; row++)
                    {
                        for (std::uint32_t t = 0; t < BLOCK_CT_DIM; t += SECT_TILES)
                        {
                            const std::uint32_t n = std::min(BLOCK_CT_DIM - t, SECT_TILES);
                            math_dest_run(n, formats.math);
                        }
                    }
                }
                return;
            }
#endif
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
#if TA_DEST_RUNS
        if constexpr (DEST_RUNS)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t row = 0; row < BLOCK_RT_DIM; row++)
                {
                    for (std::uint32_t t = 0; t < BLOCK_CT_DIM; t += SECT_TILES)
                    {
                        const std::uint32_t n = std::min(BLOCK_CT_DIM - t, SECT_TILES);
                        _llk_math_wait_for_dest_available_<TA_SYNC>();
                        math_dest_run(n, formats.math);
                        _llk_math_dest_section_done_<TA_SYNC, is_fp32_dest_acc_en>();
                    }
                }
            }
            PROFILER_SYNC();
            return;
        }
#endif
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
// As llk_pack_tilize_block (#60672): one block pack run only for full 32x32 tiles with a non-BFP output, else per tile.
inline bool pack_block_ok(const std::uint32_t pack_dst_format, const std::uint32_t unpack_src_format)
{
    return TA_NF == 4 && TA_FACE_R == 16 &&
           (!IS_BFP_FORMAT(pack_dst_format) || (TA_BFP_CLOSED && !_llk_pack_skip_bh_tilize_workaround_wrapper_(unpack_src_format)));
}

inline void pack_tilize_block(const std::uint32_t num_tiles, const std::uint32_t unpack_src_format, const std::uint32_t pack_dst_format)
{
#if TA_BFP_CLOSED
    if (IS_BFP_FORMAT(pack_dst_format))
    {
        _llk_pack_tilize_block_closed_<TA_SYNC, is_fp32_dest_acc_en>(0, PERF_ADDRESS(PERF_OUTPUT, 0), num_tiles, 4096);
        return;
    }
#endif
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
#if !defined(SPEED_OF_LIGHT) && TA_DEST_RUNS
    [[maybe_unused]] const std::uint32_t BLOCK_CT_DIM = params.BLOCK_CT_DIM;
    [[maybe_unused]] const std::uint32_t BLOCK_RT_DIM = params.BLOCK_RT_DIM;
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
#if TA_DEST_RUNS && UNPACK_BLOCK
        if constexpr (DEST_RUNS)
        {
            constexpr bool wait_math = !(PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION);
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t row = 0; row < BLOCK_RT_DIM; row++)
                {
                    for (std::uint32_t t = 0; t < BLOCK_CT_DIM; t += SECT_TILES)
                    {
                        const std::uint32_t n = std::min(BLOCK_CT_DIM - t, SECT_TILES);
                        if constexpr (wait_math)
                        {
                            _llk_packer_wait_for_math_done_();
                        }
                        if (n > 1 && pack_block_ok(formats.pack_dst, formats.unpack_A_src))
                        {
                            pack_tilize_block(n, formats.unpack_A_src, formats.pack_dst);
                        }
                        else
                        {
                            for (std::uint32_t i = 0; i < n; ++i)
                            {
                                _llk_pack_<TA_SYNC, is_fp32_dest_acc_en, pack_exec_mode_v<UNTILIZE>>(i, PERF_ADDRESS(PERF_OUTPUT, i));
                            }
                        }
                        if constexpr (wait_math)
                        {
                            _llk_pack_dest_section_done_<TA_SYNC, is_fp32_dest_acc_en>();
                        }
                    }
                }
            }
            PROFILER_SYNC();
            return;
        }
#endif
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t remaining_tiles = TILE_CNT; remaining_tiles > 0;)
                {
                    const std::uint32_t num_tiles = std::min(remaining_tiles, SECT_TILES);
#if UNPACK_BLOCK
                    if (num_tiles > 1 && pack_block_ok(formats.pack_dst, formats.unpack_A_src))
                    {
                        pack_tilize_block(num_tiles, formats.unpack_A_src, formats.pack_dst);
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
                if (num_tiles > 1 && pack_block_ok(formats.pack_dst, formats.unpack_A_src))
                {
                    pack_tilize_block(num_tiles, formats.unpack_A_src, formats.pack_dst);
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
