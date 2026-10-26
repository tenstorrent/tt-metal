// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Fused tilize plus column reduce for compute_pool_2d on Wormhole and Blackhole.

#include <cstdint>

#include "ckernel.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

using namespace ckernel;

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_lib_unpack_wrappers.h"
#include "llk_unpack_common.h"
#include "llk_unpack_tilize.h"

constexpr bool NEGINF_SRCA       = POOL_TYPE == PoolType::MAX;
constexpr bool ZERO_SRCA         = false;
constexpr bool ZERO_SRCA_REDUCE  = POOL_TYPE == PoolType::AVG;
constexpr std::uint32_t RELOAD_B = true;

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
    const std::uint32_t TEST_FACE_R_DIM    = params.TEST_FACE_R_DIM;
    const int num_faces_r_dim_A            = params.num_faces_r_dim_A;
    const int num_faces_c_dim_A            = params.num_faces_c_dim_A;
    const Operand& buffer_A                = params.buffer_A;
    const Operand& buffer_B                = params.buffer_B;
#endif
    TEST_TENSOR_SHAPE(tensor_shape, TEST_FACE_R_DIM, FACE_C_DIM, num_faces_r_dim_A, num_faces_c_dim_A);
    const std::uint32_t num_faces = tensor_shape.total_num_faces();

    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src,
            formats.unpack_B_src,
            formats.unpack_A_dst,
            formats.unpack_B_dst,
            tensor_shape.face_r_dim,
            tensor_shape.face_r_dim,
            num_faces,
            num_faces,
            TILE_SIZE_UNPACK_A,
            TILE_SIZE_UNPACK_B);
        _llk_unpack_tilizeA_B_init_wrapper_<NEGINF_SRCA, RELOAD_B, ZERO_SRCA, ZERO_SRCA_REDUCE>(
            formats.unpack_A_src, formats.unpack_A_dst, TILE_CNT, num_faces, tensor_shape.face_r_dim);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            // Both architectures set one SrcA and one SrcB dvalid per face.
            _perf_unpack_loop_set_valid<true /*set_a*/, true /*set_b*/>(LOOP_FACTOR * TILE_CNT * num_faces);
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t tile = 0; tile < TILE_CNT; ++tile)
                {
                    _llk_unpack_tilizeA_B_wrapper_<NEGINF_SRCA, RELOAD_B, ZERO_SRCA, ZERO_SRCA_REDUCE>(
                        formats.unpack_A_src, tensor_shape.face_r_dim, L1_ADDRESS(buffer_A[0]), L1_ADDRESS(buffer_B[0]), tile, TILE_CNT, num_faces);
                }
            }
        }
        PROFILER_SYNC();
    }
    _llk_unpack_tilizeA_B_uninit_wrapper_(formats.unpack_A_dst, num_faces, tensor_shape.face_r_dim);
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_reduce.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR     = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT        = params.TILE_CNT;
    const std::uint32_t TEST_FACE_R_DIM = params.TEST_FACE_R_DIM;
    const int num_faces_r_dim_A         = params.num_faces_r_dim_A;
    const int num_faces_c_dim_A         = params.num_faces_c_dim_A;
#endif
    TEST_TENSOR_SHAPE(tensor_shape, TEST_FACE_R_DIM, FACE_C_DIM, num_faces_r_dim_A, num_faces_c_dim_A);
    const std::uint32_t num_faces = tensor_shape.total_num_faces();

    {
        START_PERF_MEASURE("INIT")
        _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
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
            _perf_math_loop_clear_valid<true /*clear_a*/, true /*clear_b*/>(LOOP_FACTOR * TILE_CNT * num_faces);
        }
        else
        {
            constexpr bool sync_with_pack = PERF_RUN_TYPE != PerfRunType::MATH_ISOLATE;
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                if constexpr (sync_with_pack)
                {
                    _llk_math_wait_for_dest_available_<dest_sync>();
                }
                for (std::uint32_t tile = 0; tile < TILE_CNT; ++tile)
                {
                    LLK_ASSERT((tile < get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()), "tile exceeds max dest tiles");
                    _llk_math_reduce_<POOL_TYPE, REDUCE_DIM, is_fp32_dest_acc_en, MATH_FIDELITY, false /* is_int_fpu_en */>(tile, tensor_shape);
                }
                if constexpr (sync_with_pack)
                {
                    _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
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
    const std::uint32_t LOOP_FACTOR     = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT        = params.TILE_CNT;
    const std::uint32_t TEST_FACE_R_DIM = params.TEST_FACE_R_DIM;
    const int num_faces_r_dim_A         = params.num_faces_r_dim_A;
    const int num_faces_c_dim_A         = params.num_faces_c_dim_A;
    const Operand& buffer_Res           = params.buffer_Res;
#endif
    TEST_TENSOR_SHAPE(tensor_shape, TEST_FACE_R_DIM, FACE_C_DIM, num_faces_r_dim_A, num_faces_c_dim_A);

    const std::uint32_t tile_size = tensor_shape.total_tensor_size();
    const std::uint32_t num_faces = tensor_shape.total_num_faces();
    const bool partial_face       = tensor_shape.face_r_dim < FACE_R_DIM;
    constexpr bool narrow_tile    = false;

    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
            formats.pack_src, formats.pack_dst, tile_size, tensor_shape.face_r_dim, tensor_shape.total_col_dim(), num_faces, partial_face, narrow_tile);
        _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(
            formats.pack_dst, tensor_shape.face_r_dim, tensor_shape.total_col_dim(), num_faces, partial_face, narrow_tile);
        _llk_pack_reduce_mask_config_<REDUCE_DIM>(tensor_shape);
        _llk_pack_dest_init_wrapper_<dest_sync, is_fp32_dest_acc_en, PackMode::Default>(tensor_shape.face_r_dim, narrow_tile);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
        }
        else
        {
            constexpr bool sync_with_math = PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE && PERF_RUN_TYPE != PerfRunType::L1_CONGESTION;
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                if constexpr (sync_with_math)
                {
                    _llk_packer_wait_for_math_done_();
                }
                for (std::uint32_t tile = 0; tile < TILE_CNT; ++tile)
                {
                    LLK_ASSERT((tile < get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()), "tile exceeds max dest tiles");
                    _llk_pack_<dest_sync, is_fp32_dest_acc_en, PackMode::Default>(tile, L1_ADDRESS(buffer_Res[tile]));
                }
                if constexpr (sync_with_math)
                {
                    _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
    _llk_pack_reduce_mask_clear_();
}

#endif
