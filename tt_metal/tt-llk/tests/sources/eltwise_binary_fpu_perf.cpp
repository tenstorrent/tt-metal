// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "counters.h"
#include "llk_assert.h"
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

#ifndef SPEED_OF_LIGHT
// SPEED_OF_LIGHT injects these names as constexpr; otherwise copy them from RuntimeParams.
#define ELTWISE_BINARY_PERF_BIND_RUNTIME(params)                      \
    const std::uint32_t LOOP_FACTOR     = (params).LOOP_FACTOR;       \
    const std::uint32_t TILE_CNT        = (params).TILE_CNT;          \
    const std::uint32_t TEST_FACE_R_DIM = (params).TEST_FACE_R_DIM;   \
    const std::uint32_t TEST_FACE_C_DIM = (params).TEST_FACE_C_DIM;   \
    const int num_faces_r_dim_A         = (params).num_faces_r_dim_A; \
    const int num_faces_c_dim_A         = (params).num_faces_c_dim_A; \
    const bool UNPACK_TRANSPOSE_FACES   = (params).UNPACK_TRANSPOSE_FACES
#else
#define ELTWISE_BINARY_PERF_BIND_RUNTIME(params) ((void)0)
#endif

// Isolate mocks must follow _llk_unpack_AB_mop_config_ / math CLR_A vs CLR_B.
// NONE/ROW unpack A and B 1:1 per face. COL unpacks SrcB once per face row
// (outerloop) and SrcA per face. SCALAR unpacks SrcB once per tile then SrcA
// per face. Combined <true,true> * num_faces deadlocks COL/SCALAR isolates.
template <BroadcastType BType>
inline void _perf_eltwise_binary_unpack_set_valid(std::uint32_t tiles, const TensorShape& tensor_shape, bool transpose_of_faces)
{
    const std::uint32_t num_faces = tensor_shape.total_num_faces();

    if constexpr (BType == BroadcastType::SCALAR)
    {
        LLK_ASSERT(!transpose_of_faces, "SrcA transpose is not supported with scalar broadcast");
        LLK_ASSERT(num_faces >= 1, "SCALAR broadcast requires at least one SrcA face");
        for (std::uint32_t tile = 0; tile < tiles; tile++)
        {
            _perf_unpack_loop_set_valid<false /*set_a*/, true /*set_b*/>(1);
            _perf_unpack_loop_set_valid<true /*set_a*/, false /*set_b*/>(num_faces);
        }
    }
    else if constexpr (BType == BroadcastType::COL)
    {
        LLK_ASSERT(tensor_shape.num_faces_c_dim >= tensor_shape.num_faces_r_dim, "COL broadcast is not supported when num_faces_c_dim < num_faces_r_dim");
        if (transpose_of_faces)
        {
            LLK_ASSERT(
                tensor_shape.num_faces_r_dim == tensor_shape.num_faces_c_dim, "num_faces_r_dim must equal num_faces_c_dim when transpose_of_faces is true");
        }
        const std::uint32_t outerloop = transpose_of_faces ? tensor_shape.num_faces_c_dim : tensor_shape.num_faces_r_dim;
        const std::uint32_t innerloop = transpose_of_faces ? tensor_shape.num_faces_r_dim : tensor_shape.num_faces_c_dim;
        LLK_ASSERT(outerloop >= 1 && innerloop >= 1, "COL broadcast MOP loops must be non-zero");
        for (std::uint32_t tile = 0; tile < tiles; tile++)
        {
            for (std::uint32_t row = 0; row < outerloop; row++)
            {
                _perf_unpack_loop_set_valid<false /*set_a*/, true /*set_b*/>(1);
                _perf_unpack_loop_set_valid<true /*set_a*/, false /*set_b*/>(innerloop);
            }
        }
    }
    else
    {
        _perf_unpack_loop_set_valid<true, true>(tiles * num_faces);
    }
}

template <BroadcastType BType>
inline void _perf_eltwise_binary_math_clear_valid(std::uint32_t tiles, const TensorShape& tensor_shape, bool transpose_of_faces)
{
    const std::uint32_t num_faces = tensor_shape.total_num_faces();

    if constexpr (BType == BroadcastType::SCALAR)
    {
        LLK_ASSERT(!transpose_of_faces, "SrcA transpose is not supported with scalar broadcast");
        for (std::uint32_t tile = 0; tile < tiles; tile++)
        {
            _perf_math_loop_clear_valid<true /*clear_a*/, false /*clear_b*/>(num_faces);
            _perf_math_loop_clear_valid<false /*clear_a*/, true /*clear_b*/>(1);
        }
    }
    else if constexpr (BType == BroadcastType::COL)
    {
        LLK_ASSERT(tensor_shape.num_faces_c_dim >= tensor_shape.num_faces_r_dim, "COL broadcast is not supported when num_faces_c_dim < num_faces_r_dim");
        if (transpose_of_faces)
        {
            LLK_ASSERT(
                tensor_shape.num_faces_r_dim == tensor_shape.num_faces_c_dim, "num_faces_r_dim must equal num_faces_c_dim when transpose_of_faces is true");
        }
        const std::uint32_t outerloop = transpose_of_faces ? tensor_shape.num_faces_c_dim : tensor_shape.num_faces_r_dim;
        const std::uint32_t innerloop = transpose_of_faces ? tensor_shape.num_faces_r_dim : tensor_shape.num_faces_c_dim;
        LLK_ASSERT(outerloop >= 1 && innerloop >= 1, "COL broadcast MOP loops must be non-zero");
        for (std::uint32_t tile = 0; tile < tiles; tile++)
        {
            for (std::uint32_t row = 0; row < outerloop; row++)
            {
                _perf_math_loop_clear_valid<true /*clear_a*/, false /*clear_b*/>(innerloop);
                _perf_math_loop_clear_valid<false /*clear_a*/, true /*clear_b*/>(1);
            }
        }
    }
    else
    {
        _perf_math_loop_clear_valid<true, true>(tiles * num_faces);
    }
}

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_AB.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

    ELTWISE_BINARY_PERF_BIND_RUNTIME(params);
    const TensorShape tensor_shape {
        static_cast<std::uint8_t>(TEST_FACE_R_DIM),
        static_cast<std::uint8_t>(TEST_FACE_C_DIM),
        static_cast<std::uint8_t>(num_faces_r_dim_A),
        static_cast<std::uint8_t>(num_faces_c_dim_A)};
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
            num_faces);
        _llk_unpack_AB_init_<BROADCAST_TYPE>(tensor_shape, UNPACK_TRANSPOSE_FACES);
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
            _perf_eltwise_binary_unpack_set_valid<BROADCAST_TYPE>(LOOP_FACTOR * TILE_CNT, tensor_shape, UNPACK_TRANSPOSE_FACES);
            return;
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t tile = 0; tile < TILE_CNT; tile++)
                {
                    _llk_unpack_AB_<BROADCAST_TYPE>(PERF_ADDRESS(PERF_INPUT_A, tile), PERF_ADDRESS(PERF_INPUT_B, tile));
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

    ELTWISE_BINARY_PERF_BIND_RUNTIME(params);
    const TensorShape tensor_shape {
        static_cast<std::uint8_t>(TEST_FACE_R_DIM),
        static_cast<std::uint8_t>(TEST_FACE_C_DIM),
        static_cast<std::uint8_t>(num_faces_r_dim_A),
        static_cast<std::uint8_t>(num_faces_c_dim_A)};

    {
        START_PERF_MEASURE("INIT")
        _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        _llk_math_eltwise_binary_init_<ELTWISE_BINARY_OP, BROADCAST_TYPE, MATH_FIDELITY>(tensor_shape, 0 /* acc_to_dest */);
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
            _perf_eltwise_binary_math_clear_valid<BROADCAST_TYPE>(LOOP_FACTOR * TILE_CNT, tensor_shape, UNPACK_TRANSPOSE_FACES);
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
                        _llk_math_eltwise_binary_<ELTWISE_BINARY_OP, BROADCAST_TYPE, DstSync::SyncHalf, is_fp32_dest_acc_en, MATH_FIDELITY>(
                            tensor_shape, block_tile, false);
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
                        _llk_math_eltwise_binary_<ELTWISE_BINARY_OP, BROADCAST_TYPE, DstSync::SyncHalf, is_fp32_dest_acc_en, MATH_FIDELITY>(
                            tensor_shape, block_tile, false);
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
