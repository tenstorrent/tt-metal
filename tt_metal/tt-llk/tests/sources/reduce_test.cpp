// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"
#include "tensor_shape.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr bool IS_SWAPPED_REDUCE_ROW = (REDUCE_DIM == ckernel::ReduceDim::REDUCE_ROW) && (POOL_TYPE != ckernel::PoolType::MAX);

// Swapped 4-face REDUCE_ROW unpacks a whole tile with one UNPACR per operand; every other path posts one per face.
// Must match full_tile in _llk_unpack_AB_reduce_mop_config_: the isolate-mode dvalid mocks are sized from it.
inline std::uint32_t reduce_src_dvalids_per_tile(const ckernel::TensorShape& tensor_shape)
{
    const std::uint32_t num_faces = tensor_shape.total_num_faces();
    return (IS_SWAPPED_REDUCE_ROW && num_faces == ckernel::MAX_NUM_FACES) ? 1 : num_faces;
}

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_AB_reduce.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR        = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT           = params.TILE_CNT;
    const std::uint32_t TILE_SIZE_UNPACK_A = params.TILE_SIZE_UNPACK_A;
    const std::uint32_t TILE_SIZE_UNPACK_B = params.TILE_SIZE_UNPACK_B;
    const int in0_face_r_dim               = params.in0_face_r_dim;
    const int in0_face_c_dim               = params.in0_face_c_dim;
    const int num_faces_r_dim_A            = params.num_faces_r_dim_A;
    const int num_faces_c_dim_A            = params.num_faces_c_dim_A;
    const Operand& buffer_A                = params.buffer_A;
    const Operand& buffer_B                = params.buffer_B;
#endif
    TEST_TENSOR_SHAPE(tensor_shape, in0_face_r_dim, in0_face_c_dim, num_faces_r_dim_A, num_faces_c_dim_A);

    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src,
            formats.unpack_B_src,
            formats.unpack_A_dst,
            formats.unpack_B_dst,
            tensor_shape.face_r_dim,
            tensor_shape.face_r_dim,
            tensor_shape.total_num_faces(),
            tensor_shape.total_num_faces(),
            TILE_SIZE_UNPACK_A,
            TILE_SIZE_UNPACK_B);
        _llk_unpack_AB_reduce_init_<POOL_TYPE, REDUCE_DIM>(tensor_shape);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            _perf_unpack_loop_set_valid<true /*set_a*/, true /*set_b*/>(LOOP_FACTOR * TILE_CNT * reduce_src_dvalids_per_tile(tensor_shape));
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t tile = 0; tile < TILE_CNT; ++tile)
                {
                    // One resident scaler tile, as in production.
                    _llk_unpack_AB_reduce_<POOL_TYPE, REDUCE_DIM>(L1_ADDRESS(buffer_A[tile]), L1_ADDRESS(buffer_B[0]));
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_lib_math_wrappers.h"
#include "llk_math_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR        = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT           = params.TILE_CNT;
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;
    const bool IS_REDUCE_TO_ONE            = params.IS_REDUCE_TO_ONE;
    const int in0_face_r_dim               = params.in0_face_r_dim;
    const int in0_face_c_dim               = params.in0_face_c_dim;
    const int num_faces_r_dim_A            = params.num_faces_r_dim_A;
    const int num_faces_c_dim_A            = params.num_faces_c_dim_A;
#endif
    TEST_TENSOR_SHAPE(tensor_shape, in0_face_r_dim, in0_face_c_dim, num_faces_r_dim_A, num_faces_c_dim_A);
    constexpr bool is_int_fpu_en  = false;
    constexpr bool sync_with_pack = PERF_RUN_TYPE != PerfRunType::MATH_ISOLATE;

    {
        START_PERF_MEASURE("INIT")
        _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        _llk_math_reduce_init_<POOL_TYPE, REDUCE_DIM, is_fp32_dest_acc_en, MATH_FIDELITY>(tensor_shape);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            _perf_math_loop_clear_valid<true /*clear_a*/, true /*clear_b*/>(LOOP_FACTOR * TILE_CNT * reduce_src_dvalids_per_tile(tensor_shape));
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                // Reduce-to-one folds every tile into DEST index 0 inside one section, the production REDUCE_ROW shape.
                std::uint32_t remaining = TILE_CNT;
                while (remaining != 0)
                {
                    const std::uint32_t block_tiles = IS_REDUCE_TO_ONE ? remaining : std::min(remaining, NUM_TILES_IN_BLOCK);
                    if constexpr (sync_with_pack)
                    {
                        _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                    }
                    for (std::uint32_t tile = 0; tile < block_tiles; ++tile)
                    {
                        _llk_math_reduce_<POOL_TYPE, REDUCE_DIM, is_fp32_dest_acc_en, MATH_FIDELITY, is_int_fpu_en>(
                            IS_REDUCE_TO_ONE ? 0 : tile /*dst_index*/, tensor_shape);
                    }
                    if constexpr (sync_with_pack)
                    {
                        _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
                    }
                    remaining -= block_tiles;
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

template <bool sync_with_math>
inline void pack_tiles(const std::uint32_t output_tile_cnt, const std::uint32_t tiles_in_block, const Operand& buffer_Res)
{
    std::uint32_t remaining = output_tile_cnt;
    while (remaining != 0)
    {
        const std::uint32_t block_tiles = std::min(remaining, tiles_in_block);
        const std::uint32_t block_start = output_tile_cnt - remaining;
        if constexpr (sync_with_math)
        {
            _llk_packer_wait_for_math_done_();
        }
        for (std::uint32_t tile = 0; tile < block_tiles; ++tile)
        {
            _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile, L1_ADDRESS(buffer_Res[block_start + tile]));
        }
        if constexpr (sync_with_math)
        {
            _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        }
        remaining -= block_tiles;
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR        = params.LOOP_FACTOR;
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;
    const int OUTPUT_TILE_CNT              = params.OUTPUT_TILE_CNT;
    const int in0_face_r_dim               = params.in0_face_r_dim;
    const int in0_face_c_dim               = params.in0_face_c_dim;
    const int num_faces_r_dim_A            = params.num_faces_r_dim_A;
    const int num_faces_c_dim_A            = params.num_faces_c_dim_A;
    const Operand& buffer_Res              = params.buffer_Res;
#endif
    TEST_TENSOR_SHAPE(tensor_shape, in0_face_r_dim, in0_face_c_dim, num_faces_r_dim_A, num_faces_c_dim_A);

    const std::uint32_t tile_size = tensor_shape.total_tensor_size();
    const std::uint32_t num_faces = tensor_shape.total_num_faces();
    const bool partial_face       = tensor_shape.face_r_dim < FACE_R_DIM;
    const bool narrow_tile        = tensor_shape.num_faces_c_dim == 1;
    constexpr bool sync_with_math = PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE && PERF_RUN_TYPE != PerfRunType::L1_CONGESTION;

    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
            formats.pack_src, formats.pack_dst, tile_size, tensor_shape.face_r_dim, tensor_shape.total_col_dim(), num_faces, partial_face, narrow_tile);
        _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(
            formats.pack_dst, tensor_shape.face_r_dim, tensor_shape.total_col_dim(), num_faces, partial_face, narrow_tile);
        _llk_pack_reduce_mask_config_<REDUCE_DIM>(tensor_shape);
        _llk_pack_dest_init_wrapper_<DstSync::SyncHalf, is_fp32_dest_acc_en, PackMode::Default>(tensor_shape.face_r_dim, narrow_tile);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                pack_tiles<sync_with_math>(static_cast<std::uint32_t>(OUTPUT_TILE_CNT), NUM_TILES_IN_BLOCK, buffer_Res);
            }
        }
        PROFILER_SYNC();
    }
    _llk_pack_reduce_mask_clear_();
}

#endif
