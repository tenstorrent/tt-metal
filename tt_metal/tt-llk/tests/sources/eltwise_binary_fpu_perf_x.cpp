// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// llk_analysis stage 2 (Blackhole eltwise binary): measurement copy of tests/sources/eltwise_binary_fpu_perf.cpp.
// Lives outside the worktree (shadow tree only). Differences from the original, all selected by compile-time
// constants that the driver perf_eltwise_binary_x.py emits:
//   BROADCAST_TYPE   (ckernel::BroadcastType)             NONE / ROW / COL / SCALAR on the standard path
//   REUSE_DEST_TYPE  (ckernel::EltwiseBinaryReuseDestType) NONE, or DEST_TO_SRCA / DEST_TO_SRCB: the dest-reuse path
//                    (unpack: _llk_unpack_A_<NONE, acc_to_dest=true, reuse>, math: _llk_math_eltwise_binary_<.., reuse>,
//                    the template arguments tt-metal's binary_reuse_dest_init uses on Blackhole)
//   PER_TILE_INIT    (bool) re-run the unpack and math init before every tile (the ttnn per-chunk binary_tiles_init pattern)
//   BLOCK_TILES      (uint32) tiles per DEST section (0 = the original MAX_TILES_DEST: 8, or 4 with fp32 DEST)
// The MATH_ISOLATE / UNPACK_ISOLATE mocks publish and clear the data-valids in the pattern of the selected path
// (NONE and reuse: A and B once per tile (SrcDvalid::PerTile); ROW: A and B per face; COL: B, A, A per face row;
// SCALAR: B once, then A per face). Stage 4 copy: the fix branch's per-tile hand-off on the NONE and dest-reuse paths;
// round 2: HANDOFF selects the binary_ng switch (per-face for add and sub, per-tile for ELWMUL).
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
#include "tensor_shape.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

using namespace ckernel;

static constexpr std::uint32_t MAX_TILES_DEST = (BLOCK_TILES > 0) ? BLOCK_TILES : (is_fp32_dest_acc_en ? 4u : 8u);
static constexpr bool REUSE                   = (REUSE_DEST_TYPE != EltwiseBinaryReuseDestType::NONE);
// Final round (r3): HANDOFF 0 is the compute API default, the per-face hand-off everywhere (every kernel that does
// not opt in); HANDOFF 1 is binary_ng's opt-in (per-tile for ELWMUL); HANDOFF 2 a kernel that opts in (per-tile on
// the standard and dest-reuse paths). The broadcast forms keep the per-face program.
// Round 3 (#58723): HANDOFF 3 is a kernel that opts its broadcast multiplies in (per-tile on the standard path with a broadcast).
static constexpr bool PER_TILE =
    ((REUSE || (BROADCAST_TYPE == BroadcastType::NONE)) && ((HANDOFF == 2) || (HANDOFF == 1 && ELTWISE_BINARY_OP == EltwiseBinaryType::ELWMUL && !REUSE))) ||
    (HANDOFF == 3 && ELTWISE_BINARY_OP == EltwiseBinaryType::ELWMUL && !REUSE);
static constexpr bool MOCK_AB_PER_TILE = PER_TILE;
static constexpr bool MOCK_AB_PER_FACE = !PER_TILE && (REUSE || (BROADCAST_TYPE == BroadcastType::NONE) || (BROADCAST_TYPE == BroadcastType::ROW));
static constexpr SrcDvalid SRC_DVALID  = PER_TILE ? SrcDvalid::PerTile : SrcDvalid::PerFace;

// data-valid publications of one tile, as the real unpack MOP of the selected path issues them
inline void mock_unpack_tile()
{
    if constexpr (MOCK_AB_PER_TILE)
    {
        _perf_unpack_loop_set_valid<true, true>(1);
    }
    else if constexpr (MOCK_AB_PER_FACE)
    {
        _perf_unpack_loop_set_valid<true, true>(TILE_NUM_FACES);
    }
    else if constexpr (BROADCAST_TYPE == BroadcastType::COL)
    {
        for (std::uint32_t r = 0; r < 2; r++)
        {
            _perf_unpack_set_valid(ckernel::SrcB);
            _perf_unpack_set_valid(ckernel::SrcA);
            _perf_unpack_set_valid(ckernel::SrcA);
        }
    }
    else // SCALAR
    {
        _perf_unpack_set_valid(ckernel::SrcB);
        for (std::uint32_t f = 0; f < TILE_NUM_FACES; f++)
        {
            _perf_unpack_set_valid(ckernel::SrcA);
        }
    }
}

// data-valid clears of one tile, as the real math MOP of the selected path issues them
inline void mock_math_tile()
{
    if constexpr (MOCK_AB_PER_TILE)
    {
        _perf_math_loop_clear_valid<true, true>(1);
    }
    else if constexpr (MOCK_AB_PER_FACE)
    {
        _perf_math_loop_clear_valid<true, true>(TILE_NUM_FACES);
    }
    else if constexpr (BROADCAST_TYPE == BroadcastType::COL)
    {
        for (std::uint32_t r = 0; r < 2; r++)
        {
            _perf_math_clear_valid(ckernel::SrcA);
            _perf_math_clear_valid(ckernel::SrcA);
            _perf_math_clear_valid(ckernel::SrcB);
        }
    }
    else // SCALAR
    {
        for (std::uint32_t f = 0; f < TILE_NUM_FACES; f++)
        {
            _perf_math_clear_valid(ckernel::SrcA);
        }
        _perf_math_clear_valid(ckernel::SrcB);
    }
}

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_AB.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
#endif

    auto unpack_init = [&]()
    {
        if constexpr (REUSE)
        {
            _llk_unpack_A_init_<BroadcastType::NONE, true /* acc_to_dest */, REUSE_DEST_TYPE, false, SRC_DVALID>(
                0, 0, DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);
        }
        else
        {
            _llk_unpack_AB_init_<BROADCAST_TYPE, SRC_DVALID>(DEFAULT_TENSOR_SHAPE);
        }
    };
    auto unpack_tile = [&](std::uint32_t tile)
    {
        if constexpr (REUSE)
        {
            _llk_unpack_A_<BroadcastType::NONE, true /* acc_to_dest */, REUSE_DEST_TYPE, false>(
                PERF_ADDRESS(PERF_INPUT_A, tile), formats.unpack_A_src, formats.unpack_A_dst);
        }
        else
        {
            _llk_unpack_AB_<BROADCAST_TYPE>(PERF_ADDRESS(PERF_INPUT_A, tile), PERF_ADDRESS(PERF_INPUT_B, tile));
        }
    };

    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src,
            formats.unpack_B_src,
            formats.unpack_A_dst,
            formats.unpack_B_dst,
            FACE_R_DIM,
            FACE_R_DIM,
            /* num_faces */ 4,
            /* num_faces */ 4);
        unpack_init();
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
            for (std::uint32_t i = 0; i < LOOP_FACTOR * TILE_CNT; i++)
            {
                mock_unpack_tile();
            }
            return;
        }
        else
        {
            if constexpr (UNPACK_BLOCK && !REUSE && !PER_TILE_INIT && BROADCAST_TYPE == BroadcastType::NONE)
            {
                // round 3: one _llk_unpack_AB_block_ call per DEST section (the add_block / sub_block / mul_block form)
                constexpr std::uint32_t TILE_STRIDE_16B = 4096 / 16;
                for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
                {
                    for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
                    {
                        const std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);
                        _llk_unpack_AB_block_<BroadcastType::NONE>(
                            PERF_ADDRESS(PERF_INPUT_A, block_start), PERF_ADDRESS(PERF_INPUT_B, block_start), block_tiles, TILE_STRIDE_16B, TILE_STRIDE_16B);
                    }
                }
            }
            else
            {
                for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
                {
                    for (std::uint32_t tile = 0; tile < TILE_CNT; tile++)
                    {
                        if constexpr (PER_TILE_INIT)
                        {
                            unpack_init();
                        }
                        unpack_tile(tile);
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
#endif

    auto math_init = [&]()
    {
        _llk_math_eltwise_binary_init_<ELTWISE_BINARY_OP, BROADCAST_TYPE, MATH_FIDELITY, REUSE_DEST_TYPE, SRC_DVALID>(DEFAULT_TENSOR_SHAPE, 0 /* acc_to_dest */);
    };
    auto math_tile = [&](std::uint32_t block_tile)
    {
        _llk_math_eltwise_binary_<ELTWISE_BINARY_OP, BROADCAST_TYPE, DstSync::SyncHalf, is_fp32_dest_acc_en, MATH_FIDELITY, REUSE_DEST_TYPE, SRC_DVALID>(
            DEFAULT_TENSOR_SHAPE, block_tile, false);
    };

    {
        START_PERF_MEASURE("INIT")
        _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        math_init();
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
            for (std::uint32_t i = 0; i < LOOP_FACTOR * TILE_CNT; i++)
            {
                mock_math_tile();
            }
            return;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);

                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; block_tile++)
                    {
                        if constexpr (PER_TILE_INIT)
                        {
                            math_init();
                        }
                        math_tile(block_tile);
                    }
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);

                    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; block_tile++)
                    {
                        if constexpr (PER_TILE_INIT)
                        {
                            math_init();
                        }
                        math_tile(block_tile);
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
#include "llk_pack.h"
#include "llk_pack_common.h"

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
        _llk_pack_hw_configure_<is_fp32_dest_acc_en, ckernel::PackMode::Default>(formats.pack_src, formats.pack_dst, TILE_WIDTH * TILE_HEIGHT);
        _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst);
        _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            return;
        }
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);

                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; block_tile++)
                    {
                        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en>(block_tile, PERF_ADDRESS(PERF_OUTPUT, block_start + block_tile));
                    }
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);

                    _llk_packer_wait_for_math_done_();
                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; block_tile++)
                    {
                        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en>(block_tile, PERF_ADDRESS(PERF_OUTPUT, block_start + block_tile));
                    }
                    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif
