
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/* Shared functional/performance elementwise-binary kernel.
   Supports broadcast, transpose, destination reuse, and variable tile dimensions. */
#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"
#include "perf.h"
#include "profiler.h"
#include "tensor_shape.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

template <bool Produce, ckernel::BroadcastType BType>
inline void perf_binary_source_handshakes(
    const std::uint32_t loop_factor, const std::uint32_t num_tiles, const std::uint32_t num_faces_r, const std::uint32_t num_faces_c)
{
    const std::uint32_t num_faces = num_faces_r * num_faces_c;

    for (std::uint32_t loop = 0; loop < loop_factor; ++loop)
    {
        for (std::uint32_t tile = 0; tile < num_tiles; ++tile)
        {
            if constexpr (BType == ckernel::BroadcastType::COL)
            {
                // The COL MOP loads B once per face row, then consumes A once
                // per face column while retaining B until the row completes.
                for (std::uint32_t face_r = 0; face_r < num_faces_r; ++face_r)
                {
                    if constexpr (Produce)
                    {
                        _perf_unpack_loop_set_valid<false, true>(1);
                        _perf_unpack_loop_set_valid<true, false>(num_faces_c);
                    }
                    else
                    {
                        _perf_math_loop_clear_valid<true, false>(num_faces_c);
                        _perf_math_loop_clear_valid<false, true>(1);
                    }
                }
            }
            else if constexpr (BType == ckernel::BroadcastType::SCALAR)
            {
                // Scalar B remains valid while all A faces are consumed.
                if constexpr (Produce)
                {
                    _perf_unpack_loop_set_valid<false, true>(1);
                    _perf_unpack_loop_set_valid<true, false>(num_faces);
                }
                else
                {
                    _perf_math_loop_clear_valid<true, false>(num_faces);
                    _perf_math_loop_clear_valid<false, true>(1);
                }
            }
            else
            {
                // NONE and ROW load and clear both sources once per face.
                if constexpr (Produce)
                {
                    _perf_unpack_loop_set_valid<true, true>(num_faces);
                }
                else
                {
                    _perf_math_loop_clear_valid<true, true>(num_faces);
                }
            }
        }
    }
}

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_AB.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    // Cache volatile values to local variables first
    const std::uint8_t face_r_dim           = static_cast<std::uint8_t>(params.TEST_FACE_R_DIM);
    const std::uint8_t face_c_dim           = static_cast<std::uint8_t>(params.TEST_FACE_C_DIM);
    const std::uint8_t num_faces_r_dim      = static_cast<std::uint8_t>(params.num_faces_r_dim_A);
    const std::uint8_t num_faces_c_dim      = static_cast<std::uint8_t>(params.num_faces_c_dim_A);
    const ckernel::TensorShape tensor_shape = {face_r_dim, face_c_dim, num_faces_r_dim, num_faces_c_dim};
    const ckernel::Transpose transpose      = params.UNPACK_TRANSPOSE_FACES
                                                  ? (params.UNPACK_TRANSPOSE_WITHIN_FACE ? ckernel::Transpose::Both : ckernel::Transpose::InterFace)
                                                  : (params.UNPACK_TRANSPOSE_WITHIN_FACE ? ckernel::Transpose::IntraFace : ckernel::Transpose::None);
#ifdef EN_DEST_REUSE
    const std::uint32_t num_total_tiles = params.INPUT_NUM_TILES_IN_BLOCK * params.INPUT_NUM_BLOCKS;
#else
    const std::uint32_t num_total_tiles = params.NUM_TILES_IN_BLOCK * params.NUM_BLOCKS;
#endif
    const std::uint32_t loop_factor = params.LOOP_FACTOR;

    {
        ZONE_SCOPED("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src,
            formats.unpack_B_src,
            formats.unpack_A_dst,
            formats.unpack_B_dst,
            tensor_shape.face_r_dim,
            tensor_shape.face_r_dim,
            tensor_shape.total_num_faces(),
            tensor_shape.total_num_faces(),
            params.TILE_SIZE_UNPACK_A,
            params.TILE_SIZE_UNPACK_B);

        // Must follow HW configure, which overwrites the ALU stoch-rnd bits.
        _llk_unpack_configure_stoch_rnd_<StochRndType::None>();
        _llk_unpack_AB_init_<BROADCAST_TYPE>(tensor_shape, transpose);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            perf_binary_source_handshakes<true, BROADCAST_TYPE>(loop_factor, num_total_tiles, tensor_shape.num_faces_r_dim, tensor_shape.num_faces_c_dim);
        }
        else
        {
            for (std::uint32_t loop = 0; loop < loop_factor; ++loop)
            {
                for (std::uint32_t i = 0; i < num_total_tiles; ++i)
                {
                    _llk_unpack_AB_<BROADCAST_TYPE>(L1_ADDRESS(params.buffer_A[i]), L1_ADDRESS(params.buffer_B[i]));
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
#include "params.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    // Cache volatile values to local variables first
    const std::uint8_t face_r_dim      = static_cast<std::uint8_t>(params.TEST_FACE_R_DIM);
    const std::uint8_t face_c_dim      = static_cast<std::uint8_t>(params.TEST_FACE_C_DIM);
    const std::uint8_t num_faces_r_dim = static_cast<std::uint8_t>(params.num_faces_r_dim_A);
    const std::uint8_t num_faces_c_dim = static_cast<std::uint8_t>(params.num_faces_c_dim_A);
    const TensorShape tensor_shape     = {face_r_dim, face_c_dim, num_faces_r_dim, num_faces_c_dim};
    constexpr bool ACC_TO_DEST         = false;
    const std::uint32_t loop_factor    = params.LOOP_FACTOR;

    {
        ZONE_SCOPED("INIT")
        _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
#ifndef EN_DEST_REUSE
        constexpr auto REUSE_DEST_TYPE = ckernel::EltwiseBinaryReuseDestType::NONE;
        _llk_math_eltwise_binary_init_<ELTWISE_BINARY_OP, BROADCAST_TYPE, MATH_FIDELITY, REUSE_DEST_TYPE>(tensor_shape, ACC_TO_DEST);
#endif
        PROFILER_SYNC();
    }

    {
        ZONE_SCOPED("TILE_LOOP")
#ifdef EN_DEST_REUSE
        const std::uint32_t tiles_in_block          = params.OUTPUT_NUM_TILES_IN_BLOCK;
        const std::uint32_t num_tiles_accumulations = params.INPUT_NUM_TILES_IN_BLOCK / tiles_in_block;
        const std::uint32_t num_blocks              = params.INPUT_NUM_BLOCKS;
        const std::uint32_t num_input_tiles         = params.INPUT_NUM_TILES_IN_BLOCK * num_blocks;

        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            perf_binary_source_handshakes<false, BROADCAST_TYPE>(loop_factor, num_input_tiles, tensor_shape.num_faces_r_dim, tensor_shape.num_faces_c_dim);
        }
        else
        {
            // Seed each accumulation group without reuse, then fold the
            // remaining input tiles through the selected destination source.
            for (std::uint32_t loop = 0; loop < loop_factor; ++loop)
            {
                for (std::uint32_t block = 0; block < num_blocks; ++block)
                {
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                    {
                        _llk_math_wait_for_dest_available_<dest_sync>();
                    }

                    _llk_math_eltwise_binary_init_<ELTWISE_BINARY_OP, BROADCAST_TYPE, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(
                        tensor_shape, ACC_TO_DEST);
                    for (std::uint32_t tile = 0; tile < tiles_in_block; ++tile)
                    {
                        LLK_ASSERT(
                            (tile < get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                            "Block tile index exceeds maximum destination tiles");
                        _llk_math_eltwise_binary_<
                            ELTWISE_BINARY_OP,
                            BROADCAST_TYPE,
                            dest_sync,
                            is_fp32_dest_acc_en,
                            MATH_FIDELITY,
                            EltwiseBinaryReuseDestType::NONE>(tensor_shape, tile, false);
                    }

                    _llk_math_eltwise_binary_init_<ELTWISE_BINARY_OP, BROADCAST_TYPE, MATH_FIDELITY, REUSE_DEST_TYPE>(tensor_shape, ACC_TO_DEST);
                    for (std::uint32_t n = 1; n < num_tiles_accumulations; ++n)
                    {
                        for (std::uint32_t tile = 0; tile < tiles_in_block; ++tile)
                        {
                            LLK_ASSERT(
                                (tile < get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                                "Block tile index exceeds maximum destination tiles");
                            _llk_math_eltwise_binary_<ELTWISE_BINARY_OP, BROADCAST_TYPE, dest_sync, is_fp32_dest_acc_en, MATH_FIDELITY, REUSE_DEST_TYPE>(
                                tensor_shape, tile, false);
                        }
                    }

                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                    {
                        _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                    }
                }
            }
        }
#else
        const std::uint32_t tiles_in_block = params.NUM_TILES_IN_BLOCK;
        const std::uint32_t num_blocks     = params.NUM_BLOCKS;
        const std::uint32_t num_tiles      = tiles_in_block * num_blocks;
        constexpr auto REUSE_DEST_TYPE     = ckernel::EltwiseBinaryReuseDestType::NONE;

        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            perf_binary_source_handshakes<false, BROADCAST_TYPE>(loop_factor, num_tiles, tensor_shape.num_faces_r_dim, tensor_shape.num_faces_c_dim);
        }
        else
        {
            for (std::uint32_t loop = 0; loop < loop_factor; ++loop)
            {
                for (std::uint32_t block = 0; block < num_blocks; ++block)
                {
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                    {
                        _llk_math_wait_for_dest_available_<dest_sync>();
                    }
                    for (std::uint32_t tile = 0; tile < tiles_in_block; ++tile)
                    {
                        LLK_ASSERT(
                            (tile < get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                            "Block tile index exceeds maximum destination tiles");
                        _llk_math_eltwise_binary_<ELTWISE_BINARY_OP, BROADCAST_TYPE, dest_sync, is_fp32_dest_acc_en, MATH_FIDELITY, REUSE_DEST_TYPE>(
                            tensor_shape, tile, false);
                    }
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                    {
                        _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                    }
                }
            }
        }
#endif
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
    // Cache volatile values to local variables first
    const std::uint8_t face_r_dim           = static_cast<std::uint8_t>(params.TEST_FACE_R_DIM);
    const std::uint8_t face_c_dim           = static_cast<std::uint8_t>(params.TEST_FACE_C_DIM);
    const std::uint8_t num_faces_r_dim      = static_cast<std::uint8_t>(params.num_faces_r_dim_A);
    const std::uint8_t num_faces_c_dim      = static_cast<std::uint8_t>(params.num_faces_c_dim_A);
    const ckernel::TensorShape tensor_shape = {face_r_dim, face_c_dim, num_faces_r_dim, num_faces_c_dim};

    const std::uint32_t tile_size = tensor_shape.total_tensor_size();

    const std::uint32_t num_faces = tensor_shape.total_num_faces();
    const bool partial_face       = tensor_shape.face_r_dim < FACE_R_DIM;

    const bool narrow_tile          = (tensor_shape.num_faces_c_dim == 1);
    const std::uint32_t loop_factor = params.LOOP_FACTOR;

#ifdef EN_DEST_REUSE
    const std::uint32_t output_tiles_in_block = params.OUTPUT_NUM_TILES_IN_BLOCK;
    const std::uint32_t output_num_blocks     = params.OUTPUT_NUM_BLOCKS;
#else
    const std::uint32_t output_tiles_in_block = params.NUM_TILES_IN_BLOCK;
    const std::uint32_t output_num_blocks     = params.NUM_BLOCKS;
#endif

    {
        ZONE_SCOPED("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
            formats.pack_src, formats.pack_dst, tile_size, tensor_shape.face_r_dim, tensor_shape.total_col_dim(), num_faces, partial_face, narrow_tile);

        _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(
            formats.pack_dst, tensor_shape.face_r_dim, tensor_shape.total_col_dim(), num_faces, partial_face, narrow_tile);

        _llk_pack_dest_init_wrapper_<dest_sync, is_fp32_dest_acc_en, PackMode::Default>(tensor_shape.face_r_dim, narrow_tile);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
        }
        else
        {
            for (std::uint32_t loop = 0; loop < loop_factor; ++loop)
            {
                for (std::uint32_t block = 0; block < output_num_blocks; ++block)
                {
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                    {
                        _llk_packer_wait_for_math_done_();
                    }
                    for (std::uint32_t tile = 0; tile < output_tiles_in_block; ++tile)
                    {
                        const std::uint32_t res_tile_idx = block * output_tiles_in_block + tile;
                        LLK_ASSERT(
                            (tile < get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                            "Block tile index exceeds maximum destination tiles");
                        _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile, L1_ADDRESS(params.buffer_Res[res_tile_idx]));
                    }
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                    {
                        _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif
