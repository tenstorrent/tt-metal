// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Perf test of the reduce with a tilized operand A (tilizeA_B_reduce_init, unpack_tilizeA_B_block and the column reduce:
// the pool2d configuration). Operand A is a row-major block of TILE_CNT tiles of in0_tile_r_dim rows (2 faces wide),
// unpacked by NUM_BLOCKS calls of the tilizeA_B block unpack per loop against one-row scaler faces; the pack is the
// default pack with the reduce masks.

#include <algorithm>
#include <cstdint>

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

static constexpr std::uint32_t MAX_TILES_DEST = is_fp32_dest_acc_en ? 4 : 8;
static constexpr bool NEGINF_SRCA              = (POOL_TYPE == PoolType::MAX);
static constexpr bool ZERO_SRCA_REDUCE         = (POOL_TYPE != PoolType::MAX);
static constexpr std::uint32_t UNPB_FACE_R_DIM = 1; // the scaler CB of the pool: one row per face

inline TensorShape reduce_shape(const std::uint32_t num_faces)
{
    // as reduce_tile_math (api/compute/reduce.h): faces laid out row-wise first, a DEST face is always 16 rows
    return {
        MAX_FACE_R_DIM,
        MAX_FACE_C_DIM,
        (num_faces <= MAX_NUM_FACES_C_DIM) ? static_cast<std::uint8_t>(1) : MAX_NUM_FACES_R_DIM,
        (num_faces <= MAX_NUM_FACES_C_DIM) ? static_cast<std::uint8_t>(num_faces) : MAX_NUM_FACES_C_DIM};
}

#ifdef LLK_TRISC_UNPACK

#include "llk_lib_unpack_wrappers.h"
#include "llk_unpack_common.h"
#include "llk_unpack_tilize.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT    = params.TILE_CNT; // the tiles of one block row
    const std::uint32_t num_faces   = params.num_faces;
    const std::uint32_t face_r_dim  = params.in0_tile_r_dim / ((num_faces == 4) ? 2 : 1);
    const int NUM_BLOCKS            = params.NUM_BLOCKS;
#endif
    const std::uint32_t num_blocks      = static_cast<std::uint32_t>(NUM_BLOCKS);
    const std::uint32_t tiles_per_block = TILE_CNT / num_blocks;
    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, face_r_dim, UNPB_FACE_R_DIM, num_faces, num_faces);
        _llk_unpack_tilizeA_B_block_init_<NEGINF_SRCA, true /* reload_srcB */, false /* zero_srcA */, ZERO_SRCA_REDUCE>(
            formats.unpack_A_src, formats.unpack_A_dst, TILE_CNT, num_faces, UNPB_FACE_R_DIM, face_r_dim);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            // tilizeA_B publishes one SrcB and one SrcA bank per face
            _perf_unpack_loop_set_valid<true, true>(LOOP_FACTOR * TILE_CNT * num_faces);
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t block = 0; block < num_blocks; block++)
                {
                    _llk_unpack_tilizeA_B_block_<NEGINF_SRCA, true, false, ZERO_SRCA_REDUCE>(
                        formats.unpack_A_src,
                        face_r_dim,
                        PERF_ADDRESS(PERF_INPUT_A, 0),
                        PERF_ADDRESS(PERF_INPUT_B, 0),
                        block * tiles_per_block,
                        tiles_per_block,
                        num_faces);
                }
            }
        }
        PROFILER_SYNC();
    }
    _llk_unpack_tilizeA_B_block_uninit_(formats.unpack_A_dst, tensor_shape_from_num_faces(face_r_dim, num_faces));
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_lib_math_wrappers.h"
#include "llk_math_common.h"
#include "llk_math_reduce.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
    const std::uint32_t num_faces   = params.num_faces;
#endif
    constexpr bool IS_INT_FPU = false;
    const TensorShape shape   = reduce_shape(num_faces);
    {
        START_PERF_MEASURE("INIT")
        _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        _llk_math_reduce_init_<POOL_TYPE, REDUCE_DIM, is_fp32_dest_acc_en, MATH_FIDELITY>(shape);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            _perf_math_loop_clear_valid<true, true>(LOOP_FACTOR * TILE_CNT * num_faces);
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t tile = 0; tile < TILE_CNT; tile++)
                {
                    _llk_math_reduce_<POOL_TYPE, REDUCE_DIM, is_fp32_dest_acc_en, MATH_FIDELITY, IS_INT_FPU>(tile % MAX_TILES_DEST, shape);
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    const std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);
                    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; block_tile++)
                    {
                        _llk_math_reduce_<POOL_TYPE, REDUCE_DIM, is_fp32_dest_acc_en, MATH_FIDELITY, IS_INT_FPU>(block_tile, shape);
                    }
                    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
    _llk_math_reduce_uninit_();
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
        _llk_pack_reduce_mask_config_<REDUCE_DIM>();
        _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            _llk_pack_reduce_mask_clear_();
            return;
        }
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
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
                for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    const std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);
                    _llk_packer_wait_for_math_done_();
                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; block_tile++)
                    {
                        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en>(block_tile, PERF_ADDRESS(PERF_OUTPUT, block_start + block_tile));
                    }
                    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
                }
            }
        }
        _llk_pack_reduce_mask_clear_();
        PROFILER_SYNC();
    }
}

#endif
