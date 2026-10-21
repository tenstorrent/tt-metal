// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// llk_analysis transpose family shadow perf kernel (Blackhole), not part of the tree. Three production forms:
//   TRANSPOSE_MODE 0: transpose_dest alone (welford/layernorm/dit/sdpa_bw/topk_large_indices pattern): per tile the unpack
//                     thread publishes the dummy SrcB+SrcA valids, the math thread runs _llk_math_transpose_dest_, the pack
//                     thread packs; all five run types except L1_CONGESTION.
//   TRANSPOSE_MODE 1: transpose_tile of a 32-bit tensor (api/compute/transpose.h, unpack-to-dest branch): unpack to dest
//                     with the face order, dummy valids, datacopy and transpose_dest<false, true> per tile, one init.
//   TRANSPOSE_MODE 2: transpose_tile of a 16-bit tensor: the unpacker transpose (faces and within face, the init with
//                     acc_to_dest = true as transpose_init programs it) and the A2D datacopy.
// SECTION_TILES is the number of tiles per DEST section in L1_TO_L1 (the ttnn transpose kernel uses 1).
// TRANSPOSE_REINIT 1 repeats the op inits before every tile, as the groupnorm welford statistics loop calls transpose_init.

#include <algorithm>
#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifndef TRANSPOSE_REINIT
#define TRANSPOSE_REINIT 0
#endif

static constexpr std::uint32_t MAX_TILES_DEST = is_fp32_dest_acc_en ? 4 : 8;
static constexpr std::uint32_t NUM_FACES      = 4;
static_assert(SECTION_TILES >= 1 && SECTION_TILES <= MAX_TILES_DEST, "SECTION_TILES must fit one DEST half");
static_assert(TRANSPOSE_MODE != 1 || PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE, "unpack to dest runs L1_TO_L1");
static_assert(PERF_RUN_TYPE != PerfRunType::L1_CONGESTION, "no L1 congestion form");

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

inline void transpose_unpack_init(const std::uint32_t src_format, const std::uint32_t dst_format)
{
    if constexpr (TRANSPOSE_MODE == 1)
    {
        _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, true>(
            1, 0, ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, NUM_FACES), src_format, dst_format);
    }
    else if constexpr (TRANSPOSE_MODE == 2)
    {
        _llk_unpack_A_init_<BroadcastType::NONE, true, EltwiseBinaryReuseDestType::NONE, false>(
            1, 1, ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, NUM_FACES), src_format, dst_format);
    }
}

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
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, NUM_FACES, NUM_FACES);
        transpose_unpack_init(formats.unpack_A_src, formats.unpack_A_dst);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            _perf_unpack_loop_set_valid<true, true>(LOOP_FACTOR * TILE_CNT * (TRANSPOSE_MODE == 2 ? NUM_FACES : 1));
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t tile = 0; tile < TILE_CNT; tile++)
                {
                    if constexpr (TRANSPOSE_REINIT)
                    {
                        transpose_unpack_init(formats.unpack_A_src, formats.unpack_A_dst);
                    }
                    if constexpr (TRANSPOSE_MODE == 0)
                    {
                        _llk_unpack_set_srcb_dummy_valid_();
                    }
                    else if constexpr (TRANSPOSE_MODE == 1)
                    {
                        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, true>(
                            PERF_ADDRESS(PERF_INPUT_A, tile), formats.unpack_A_src, formats.unpack_A_dst);
                        _llk_unpack_set_srcb_dummy_valid_();
                    }
                    else
                    {
                        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false>(
                            PERF_ADDRESS(PERF_INPUT_A, tile), formats.unpack_A_src, formats.unpack_A_dst);
                    }
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

inline void transpose_math_init(const std::uint32_t math_format)
{
    if constexpr (TRANSPOSE_MODE == 0)
    {
        _llk_math_transpose_dest_init_<MATH_TRANSPOSE_FACES, is_fp32_dest_acc_en>();
    }
    else
    {
        _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, PackMode::Default>(
            NUM_FACES, math_format);
        if constexpr (TRANSPOSE_MODE == 1)
        {
            _llk_math_transpose_dest_init_<false, true>();
        }
    }
}

inline void transpose_math_tile(const std::uint32_t dst_index, const std::uint32_t math_format)
{
    if constexpr (TRANSPOSE_REINIT)
    {
        transpose_math_init(math_format);
    }
    if constexpr (TRANSPOSE_MODE == 0)
    {
        _llk_math_transpose_dest_wrapper_<is_fp32_dest_acc_en, MATH_TRANSPOSE_FACES, is_fp32_dest_acc_en>(dst_index);
    }
    else if constexpr (TRANSPOSE_MODE == 1)
    {
        _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, true>(
            dst_index, math_format, math_format, NUM_FACES);
        _llk_math_transpose_dest_wrapper_<is_fp32_dest_acc_en, false, true>(dst_index);
    }
    else
    {
        _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, false>(
            dst_index, math_format, math_format, NUM_FACES);
    }
}

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
        _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        transpose_math_init(formats.math);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            _perf_math_loop_clear_valid<true, true>(LOOP_FACTOR * TILE_CNT * (TRANSPOSE_MODE == 2 ? NUM_FACES : 1));
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t tile = 0; tile < TILE_CNT; tile++)
                {
                    transpose_math_tile(tile % MAX_TILES_DEST, formats.math);
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t section = 0; section < TILE_CNT; section += SECTION_TILES)
                {
                    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                    for (std::uint32_t tile = 0; tile < SECTION_TILES; tile++)
                    {
                        transpose_math_tile(tile, formats.math);
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
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t tile = 0; tile < TILE_CNT; tile++)
                {
                    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en>(tile % MAX_TILES_DEST, PERF_ADDRESS(PERF_OUTPUT, tile));
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t section = 0; section < TILE_CNT; section += SECTION_TILES)
                {
                    _llk_packer_wait_for_math_done_();
                    for (std::uint32_t tile = 0; tile < SECTION_TILES; tile++)
                    {
                        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en>(tile, PERF_ADDRESS(PERF_OUTPUT, section + tile));
                    }
                    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif
