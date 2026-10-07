
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/* Shared functional/performance elementwise-binary kernel.
   Supports broadcast, transpose, destination reuse, and variable tile dimensions. */
#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "counters.h"
#include "llk_defs.h"
#include "perf.h"
#include "profiler.h"
#include "tensor_shape.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#if defined(ARCH_BLACKHOLE)
// Each operand tile goes to math as one source bank (SrcDvalid::PerTile) unless the variant sets per_face_handoff, which the test
// does for a transposed SrcA; partial faces, and a column or row broadcast without 2 x 2 faces, fall back inside the LLK.
#define SRC_DVALID     (per_face_handoff ? ckernel::SrcDvalid::PerFace : ckernel::SrcDvalid::PerTile)
#define SRC_DVALID_ARG , SRC_DVALID
#define PER_TILE_MOCK(shape) \
    tile_handoff<BROADCAST_TYPE>(SRC_DVALID == ckernel::SrcDvalid::PerTile, (shape).face_r_dim, (shape).num_faces_r_dim, (shape).num_faces_c_dim)
#else
#define SRC_DVALID_ARG
#define PER_TILE_MOCK(shape) false
#endif

// Whether the LLK hands this tile shape over once per tile on both threads.
template <ckernel::BroadcastType BType>
inline bool tile_handoff(const bool per_tile, const std::uint32_t face_r_dim, const std::uint32_t num_faces_r, const std::uint32_t num_faces_c)
{
    const bool needs_2x2 = BType == ckernel::BroadcastType::COL || BType == ckernel::BroadcastType::ROW;
    return per_tile && face_r_dim == ckernel::FACE_R_DIM && (!needs_2x2 || (num_faces_r == 2 && num_faces_c == 2));
}

template <bool Produce, ckernel::BroadcastType BType>
inline void perf_binary_source_handshakes(
    const std::uint32_t loop_factor,
    const std::uint32_t num_tiles,
    const std::uint32_t num_faces_r,
    const std::uint32_t num_faces_c,
    const bool per_tile_handoff = false)
{
    const std::uint32_t num_faces = num_faces_r * num_faces_c;

    for (std::uint32_t loop = 0; loop < loop_factor; ++loop)
    {
        for (std::uint32_t tile = 0; tile < num_tiles; ++tile)
        {
            if (per_tile_handoff)
            {
                if constexpr (Produce)
                {
                    _perf_unpack_loop_set_valid<true /*set_a*/, true /*set_b*/>(1);
                }
                else
                {
                    _perf_math_loop_clear_valid<true /*clear_a*/, true /*clear_b*/>(1);
                }
            }
            else if constexpr (BType == ckernel::BroadcastType::COL)
            {
                // The COL MOP loads B once per face row, then consumes A once
                // per face column while retaining B until the row completes.
                for (std::uint32_t face_r = 0; face_r < num_faces_r; ++face_r)
                {
                    if constexpr (Produce)
                    {
                        _perf_unpack_loop_set_valid<false /*set_a*/, true /*set_b*/>(1);
                        _perf_unpack_loop_set_valid<true /*set_a*/, false /*set_b*/>(num_faces_c);
                    }
                    else
                    {
                        _perf_math_loop_clear_valid<true /*clear_a*/, false /*clear_b*/>(num_faces_c);
                        _perf_math_loop_clear_valid<false /*clear_a*/, true /*clear_b*/>(1);
                    }
                }
            }
            else if constexpr (BType == ckernel::BroadcastType::SCALAR)
            {
                // Scalar B remains valid while all A faces are consumed.
                if constexpr (Produce)
                {
                    _perf_unpack_loop_set_valid<false /*set_a*/, true /*set_b*/>(1);
                    _perf_unpack_loop_set_valid<true /*set_a*/, false /*set_b*/>(num_faces);
                }
                else
                {
                    _perf_math_loop_clear_valid<true /*clear_a*/, false /*clear_b*/>(num_faces);
                    _perf_math_loop_clear_valid<false /*clear_a*/, true /*clear_b*/>(1);
                }
            }
            else
            {
                // NONE and ROW load and clear both sources once per face.
                if constexpr (Produce)
                {
                    _perf_unpack_loop_set_valid<true /*set_a*/, true /*set_b*/>(num_faces);
                }
                else
                {
                    _perf_math_loop_clear_valid<true /*clear_a*/, true /*clear_b*/>(num_faces);
                }
            }
        }
    }
}

#if defined(ARCH_BLACKHOLE)
// Whether the dest-reuse unpack and math hand a folded tile over once per tile: full faces, or 8-row faces (no broadcast).
inline bool reuse_tile_handoff(const bool per_tile, const std::uint32_t face_r_dim, const std::uint32_t num_faces_r)
{
    return per_tile && (face_r_dim == ckernel::FACE_R_DIM || (face_r_dim == ckernel::MAX_FPU_ROWS && num_faces_r == 1));
}

// Each block seeds its output tiles with the two-operand op, then folds the rest with the dest-reuse op.
template <bool Produce>
inline void perf_dest_reuse_source_handshakes(
    const std::uint32_t loop_factor,
    const std::uint32_t num_tiles,
    const std::uint32_t num_blocks,
    const std::uint32_t seeds,
    const std::uint32_t num_faces,
    const bool seed_per_tile,
    const bool fold_per_tile)
{
    const std::uint32_t folds         = num_tiles / num_blocks - seeds;
    const std::uint32_t seed_handoffs = seed_per_tile ? seeds : seeds * num_faces;
    const std::uint32_t fold_handoffs = fold_per_tile ? folds : folds * num_faces;
    for (std::uint32_t i = 0; i < loop_factor * num_blocks; ++i)
    {
        if constexpr (Produce)
        {
            _perf_unpack_loop_set_valid<true /*set_a*/, true /*set_b*/>(seed_handoffs + fold_handoffs);
        }
        else
        {
            _perf_math_loop_clear_valid<true /*clear_a*/, true /*clear_b*/>(seed_handoffs + fold_handoffs);
        }
    }
}
#endif

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_AB.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#ifdef SPEED_OF_LIGHT
    constexpr ckernel::TensorShape tensor_shape = {
        static_cast<std::uint8_t>(TEST_FACE_R_DIM),
        static_cast<std::uint8_t>(TEST_FACE_C_DIM),
        static_cast<std::uint8_t>(num_faces_r_dim_A),
        static_cast<std::uint8_t>(num_faces_c_dim_A)};
    constexpr ckernel::Transpose transpose  = UNPACK_TRANSPOSE_FACES ? (UNPACK_TRANSPOSE_WITHIN_FACE ? ckernel::Transpose::Both : ckernel::Transpose::InterFace)
                                                                     : (UNPACK_TRANSPOSE_WITHIN_FACE ? ckernel::Transpose::IntraFace : ckernel::Transpose::None);
    constexpr std::uint32_t num_total_tiles = INPUT_NUM_TILES_IN_BLOCK * INPUT_NUM_BLOCKS;
#else
#ifdef RUNTIME_FORMATS
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t TEST_FACE_R_DIM          = params.TEST_FACE_R_DIM;
    const std::uint32_t TEST_FACE_C_DIM          = params.TEST_FACE_C_DIM;
    const int num_faces_r_dim_A                  = params.num_faces_r_dim_A;
    const int num_faces_c_dim_A                  = params.num_faces_c_dim_A;
    const bool UNPACK_TRANSPOSE_FACES            = params.UNPACK_TRANSPOSE_FACES;
    const bool UNPACK_TRANSPOSE_WITHIN_FACE      = params.UNPACK_TRANSPOSE_WITHIN_FACE;
    const std::uint32_t INPUT_NUM_TILES_IN_BLOCK = params.INPUT_NUM_TILES_IN_BLOCK;
    const int INPUT_NUM_BLOCKS                   = params.INPUT_NUM_BLOCKS;
    const std::uint32_t LOOP_FACTOR              = params.LOOP_FACTOR;
#if defined(ARCH_BLACKHOLE) && defined(EN_DEST_REUSE) && defined(DEST_REUSE_UNPACK_A)
    const std::uint32_t OUTPUT_NUM_TILES_IN_BLOCK = params.OUTPUT_NUM_TILES_IN_BLOCK;
#endif
    const std::uint32_t TILE_SIZE_UNPACK_A       = params.TILE_SIZE_UNPACK_A;
    const std::uint32_t TILE_SIZE_UNPACK_B       = params.TILE_SIZE_UNPACK_B;
    const Operand& buffer_A                      = params.buffer_A;
    const Operand& buffer_B                      = params.buffer_B;
    // Cache volatile values to local variables first
    const ckernel::TensorShape tensor_shape = {
        static_cast<std::uint8_t>(TEST_FACE_R_DIM),
        static_cast<std::uint8_t>(TEST_FACE_C_DIM),
        static_cast<std::uint8_t>(num_faces_r_dim_A),
        static_cast<std::uint8_t>(num_faces_c_dim_A)};
    const ckernel::Transpose transpose  = UNPACK_TRANSPOSE_FACES ? (UNPACK_TRANSPOSE_WITHIN_FACE ? ckernel::Transpose::Both : ckernel::Transpose::InterFace)
                                                                 : (UNPACK_TRANSPOSE_WITHIN_FACE ? ckernel::Transpose::IntraFace : ckernel::Transpose::None);
    const std::uint32_t num_total_tiles = INPUT_NUM_TILES_IN_BLOCK * INPUT_NUM_BLOCKS;
#endif

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

        // Must follow HW configure, which overwrites the ALU stoch-rnd bits.
        _llk_unpack_configure_stoch_rnd_<StochRndType::None>();
        _llk_unpack_AB_init_<BROADCAST_TYPE SRC_DVALID_ARG>(tensor_shape, transpose);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
#if defined(ARCH_BLACKHOLE) && defined(EN_DEST_REUSE) && defined(DEST_REUSE_UNPACK_A)
            perf_dest_reuse_source_handshakes<true>(
                LOOP_FACTOR,
                num_total_tiles,
                INPUT_NUM_BLOCKS,
                OUTPUT_NUM_TILES_IN_BLOCK,
                tensor_shape.total_num_faces(),
                PER_TILE_MOCK(tensor_shape),
                reuse_tile_handoff(SRC_DVALID == ckernel::SrcDvalid::PerTile, tensor_shape.face_r_dim, tensor_shape.num_faces_r_dim));
#else
            perf_binary_source_handshakes<true, BROADCAST_TYPE>(
                LOOP_FACTOR, num_total_tiles, tensor_shape.num_faces_r_dim, tensor_shape.num_faces_c_dim, PER_TILE_MOCK(tensor_shape));
#endif
        }
        else
        {
#if defined(ARCH_BLACKHOLE) && defined(EN_DEST_REUSE) && defined(DEST_REUSE_UNPACK_A)
            // The folds take the dest-reuse unpack the compute API pairs with the dest-reuse math: the L1 operand is B for
            // DEST_TO_SRCA and A for DEST_TO_SRCB
            static_assert(BROADCAST_TYPE == BroadcastType::NONE, "The dest-reuse unpack variant has no broadcast");
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t i = 0; i < num_total_tiles;)
                {
                    _llk_unpack_AB_init_<BROADCAST_TYPE SRC_DVALID_ARG>(tensor_shape, transpose);
                    for (std::uint32_t t = 0; t < OUTPUT_NUM_TILES_IN_BLOCK; ++t, ++i)
                    {
                        _llk_unpack_AB_<BROADCAST_TYPE>(L1_ADDRESS(buffer_A[i]), L1_ADDRESS(buffer_B[i]));
                    }
                    _llk_unpack_A_init_<BroadcastType::NONE, true /*acc_to_dest*/, REUSE_DEST_TYPE, false /*unpack_to_dest*/, SRC_DVALID>(
                        0, 0, tensor_shape, formats.unpack_A_src, formats.unpack_A_dst);
                    for (std::uint32_t t = OUTPUT_NUM_TILES_IN_BLOCK; t < INPUT_NUM_TILES_IN_BLOCK; ++t, ++i)
                    {
                        const std::uint32_t address =
                            REUSE_DEST_TYPE == EltwiseBinaryReuseDestType::DEST_TO_SRCA ? L1_ADDRESS(buffer_B[i]) : L1_ADDRESS(buffer_A[i]);
                        _llk_unpack_A_<BroadcastType::NONE, true /*acc_to_dest*/, REUSE_DEST_TYPE>(address, formats.unpack_A_src, formats.unpack_A_dst);
                    }
                }
            }
#else
#if defined(ARCH_BLACKHOLE)
            if constexpr (unpack_ab_block)
            {
                const std::uint32_t block_tiles = INPUT_NUM_TILES_IN_BLOCK;
                const std::uint32_t stride_a    = num_total_tiles > 1 ? L1_ADDRESS(buffer_A[1]) - L1_ADDRESS(buffer_A[0]) : 0;
                const std::uint32_t stride_b =
                    (unpack_ab_block == 2 || num_total_tiles == 1) ? 0 : L1_ADDRESS(buffer_B[1]) - L1_ADDRESS(buffer_B[0]);
                for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
                {
                    for (std::uint32_t block = 0; block < static_cast<std::uint32_t>(INPUT_NUM_BLOCKS); ++block)
                    {
                        _llk_unpack_AB_block_<BroadcastType::NONE>(
                            L1_ADDRESS(buffer_A[block * block_tiles]), L1_ADDRESS(buffer_B[block * block_tiles]), block_tiles, stride_a, stride_b);
                    }
                }
            }
            else
#endif
            {
                for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
                {
                    for (std::uint32_t i = 0; i < num_total_tiles; ++i)
                    {
                        _llk_unpack_AB_<BROADCAST_TYPE>(L1_ADDRESS(buffer_A[i]), L1_ADDRESS(buffer_B[i]));
                    }
                }
            }
#endif
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
#ifdef SPEED_OF_LIGHT
    constexpr TensorShape tensor_shape = {
        static_cast<std::uint8_t>(TEST_FACE_R_DIM),
        static_cast<std::uint8_t>(TEST_FACE_C_DIM),
        static_cast<std::uint8_t>(num_faces_r_dim_A),
        static_cast<std::uint8_t>(num_faces_c_dim_A)};
    constexpr std::uint32_t input_tiles_in_block   = INPUT_NUM_TILES_IN_BLOCK;
    constexpr std::uint32_t output_tiles_in_block  = OUTPUT_NUM_TILES_IN_BLOCK;
    constexpr std::uint32_t num_blocks             = INPUT_NUM_BLOCKS;
    constexpr std::uint32_t num_input_tiles        = input_tiles_in_block * num_blocks;
    constexpr std::uint32_t tiles_per_accumulation = input_tiles_in_block / output_tiles_in_block;
#else
#ifdef RUNTIME_FORMATS
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t TEST_FACE_R_DIM           = params.TEST_FACE_R_DIM;
    const std::uint32_t TEST_FACE_C_DIM           = params.TEST_FACE_C_DIM;
    const int num_faces_r_dim_A                   = params.num_faces_r_dim_A;
    const int num_faces_c_dim_A                   = params.num_faces_c_dim_A;
    const std::uint32_t LOOP_FACTOR               = params.LOOP_FACTOR;
    const std::uint32_t INPUT_NUM_TILES_IN_BLOCK  = params.INPUT_NUM_TILES_IN_BLOCK;
    const std::uint32_t OUTPUT_NUM_TILES_IN_BLOCK = params.OUTPUT_NUM_TILES_IN_BLOCK;
    const int INPUT_NUM_BLOCKS                    = params.INPUT_NUM_BLOCKS;
    // Cache volatile values to local variables first
    const TensorShape tensor_shape = {
        static_cast<std::uint8_t>(TEST_FACE_R_DIM),
        static_cast<std::uint8_t>(TEST_FACE_C_DIM),
        static_cast<std::uint8_t>(num_faces_r_dim_A),
        static_cast<std::uint8_t>(num_faces_c_dim_A)};
    const std::uint32_t input_tiles_in_block   = INPUT_NUM_TILES_IN_BLOCK;
    const std::uint32_t output_tiles_in_block  = OUTPUT_NUM_TILES_IN_BLOCK;
    const std::uint32_t num_blocks             = INPUT_NUM_BLOCKS;
    const std::uint32_t num_input_tiles        = input_tiles_in_block * num_blocks;
    const std::uint32_t tiles_per_accumulation = input_tiles_in_block / output_tiles_in_block;
#endif

    {
        START_PERF_MEASURE("INIT")
        _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
#ifndef EN_DEST_REUSE
        constexpr auto REUSE_DEST_TYPE = ckernel::EltwiseBinaryReuseDestType::NONE;
        _llk_math_eltwise_binary_init_<ELTWISE_BINARY_OP, BROADCAST_TYPE, MATH_FIDELITY, REUSE_DEST_TYPE SRC_DVALID_ARG>(tensor_shape, ACC_TO_DEST);
#endif
        PROFILER_SYNC();
    }

    {
        START_PERF_MEASURE("TILE_LOOP")
#ifdef EN_DEST_REUSE
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
#if defined(ARCH_BLACKHOLE) && defined(DEST_REUSE_UNPACK_A)
            perf_dest_reuse_source_handshakes<false>(
                LOOP_FACTOR,
                num_input_tiles,
                num_blocks,
                output_tiles_in_block,
                tensor_shape.total_num_faces(),
                PER_TILE_MOCK(tensor_shape),
                reuse_tile_handoff(SRC_DVALID == ckernel::SrcDvalid::PerTile, tensor_shape.face_r_dim, tensor_shape.num_faces_r_dim));
#else
            perf_binary_source_handshakes<false, BROADCAST_TYPE>(
                LOOP_FACTOR, num_input_tiles, tensor_shape.num_faces_r_dim, tensor_shape.num_faces_c_dim, PER_TILE_MOCK(tensor_shape));
#endif
        }
        else
        {
            // Seed each accumulation group without reuse, then fold the
            // remaining input tiles through the selected destination source.
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block = 0; block < num_blocks; ++block)
                {
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                    {
                        _llk_math_wait_for_dest_available_<dest_sync>();
                    }

                    _llk_math_eltwise_binary_init_<ELTWISE_BINARY_OP, BROADCAST_TYPE, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE SRC_DVALID_ARG>(
                        tensor_shape, ACC_TO_DEST);
                    for (std::uint32_t tile = 0; tile < output_tiles_in_block; ++tile)
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
                            EltwiseBinaryReuseDestType::NONE SRC_DVALID_ARG>(tensor_shape, tile, false /* clear_fp32_dst_acc */);
                    }

                    _llk_math_eltwise_binary_init_<ELTWISE_BINARY_OP, BROADCAST_TYPE, MATH_FIDELITY, REUSE_DEST_TYPE SRC_DVALID_ARG>(tensor_shape, ACC_TO_DEST);
                    for (std::uint32_t n = 1; n < tiles_per_accumulation; ++n)
                    {
                        for (std::uint32_t tile = 0; tile < output_tiles_in_block; ++tile)
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
                                REUSE_DEST_TYPE SRC_DVALID_ARG>(tensor_shape, tile, false /* clear_fp32_dst_acc */);
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
        LLK_ASSERT(output_tiles_in_block > 0, "Output block must contain at least one tile");
        LLK_ASSERT(input_tiles_in_block % output_tiles_in_block == 0, "Input tiles must divide evenly among accumulated output tiles");
        constexpr auto REUSE_DEST_TYPE = ckernel::EltwiseBinaryReuseDestType::NONE;

        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            perf_binary_source_handshakes<false, BROADCAST_TYPE>(
                LOOP_FACTOR, num_input_tiles, tensor_shape.num_faces_r_dim, tensor_shape.num_faces_c_dim, PER_TILE_MOCK(tensor_shape));
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block = 0; block < num_blocks; ++block)
                {
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                    {
                        _llk_math_wait_for_dest_available_<dest_sync>();
                    }
                    for (std::uint32_t tile = 0; tile < output_tiles_in_block; ++tile)
                    {
                        LLK_ASSERT(
                            (tile < get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                            "Block tile index exceeds maximum destination tiles");
                        for (std::uint32_t accumulation = 0; accumulation < tiles_per_accumulation; ++accumulation)
                        {
                            _llk_math_eltwise_binary_<
                                ELTWISE_BINARY_OP,
                                BROADCAST_TYPE,
                                dest_sync,
                                is_fp32_dest_acc_en,
                                MATH_FIDELITY,
                                REUSE_DEST_TYPE SRC_DVALID_ARG>(tensor_shape, tile, false /* clear_fp32_dst_acc */);
                        }
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
#ifdef SPEED_OF_LIGHT
    constexpr ckernel::TensorShape tensor_shape = {
        static_cast<std::uint8_t>(TEST_FACE_R_DIM),
        static_cast<std::uint8_t>(TEST_FACE_C_DIM),
        static_cast<std::uint8_t>(num_faces_r_dim_A),
        static_cast<std::uint8_t>(num_faces_c_dim_A)};
    constexpr std::uint32_t tile_size             = tensor_shape.total_tensor_size();
    constexpr std::uint32_t num_faces             = tensor_shape.total_num_faces();
    constexpr bool partial_face                   = tensor_shape.face_r_dim < FACE_R_DIM;
    constexpr bool narrow_tile                    = tensor_shape.num_faces_c_dim == 1;
    constexpr std::uint32_t output_tiles_in_block = OUTPUT_NUM_TILES_IN_BLOCK;
    constexpr std::uint32_t output_num_blocks     = OUTPUT_NUM_BLOCKS;
#else
#ifdef RUNTIME_FORMATS
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t TEST_FACE_R_DIM           = params.TEST_FACE_R_DIM;
    const std::uint32_t TEST_FACE_C_DIM           = params.TEST_FACE_C_DIM;
    const int num_faces_r_dim_A                   = params.num_faces_r_dim_A;
    const int num_faces_c_dim_A                   = params.num_faces_c_dim_A;
    const std::uint32_t LOOP_FACTOR               = params.LOOP_FACTOR;
    const std::uint32_t OUTPUT_NUM_TILES_IN_BLOCK = params.OUTPUT_NUM_TILES_IN_BLOCK;
    const int OUTPUT_NUM_BLOCKS                   = params.OUTPUT_NUM_BLOCKS;
    const Operand& buffer_Res                     = params.buffer_Res;
    // Cache volatile values to local variables first
    const ckernel::TensorShape tensor_shape = {
        static_cast<std::uint8_t>(TEST_FACE_R_DIM),
        static_cast<std::uint8_t>(TEST_FACE_C_DIM),
        static_cast<std::uint8_t>(num_faces_r_dim_A),
        static_cast<std::uint8_t>(num_faces_c_dim_A)};
    const std::uint32_t tile_size             = tensor_shape.total_tensor_size();
    const std::uint32_t num_faces             = tensor_shape.total_num_faces();
    const bool partial_face                   = tensor_shape.face_r_dim < FACE_R_DIM;
    const bool narrow_tile                    = tensor_shape.num_faces_c_dim == 1;
    const std::uint32_t output_tiles_in_block = OUTPUT_NUM_TILES_IN_BLOCK;
    const std::uint32_t output_num_blocks     = OUTPUT_NUM_BLOCKS;
#endif

    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
            formats.pack_src, formats.pack_dst, tile_size, tensor_shape.face_r_dim, tensor_shape.total_col_dim(), num_faces, partial_face, narrow_tile);

        _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(
            formats.pack_dst, tensor_shape.face_r_dim, tensor_shape.total_col_dim(), num_faces, partial_face, narrow_tile);

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
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
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
                        _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile, L1_ADDRESS(buffer_Res[res_tile_idx]));
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
