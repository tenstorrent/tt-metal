// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Binary dest-reuse add (experimental, Blackhole only).
//
// This expands the experimental Compute API
// (api/compute/experimental/eltwise_add_scalar.h) into its underlying _llk_*
// calls so the op runs inside the tt-llk harness. The op is the add analogue of
// the mul_scalar dest-reuse sibling:
//
//     deepseek_binary_dest_reuse_add_tiles<..., DEST_TO_SRCA>:
//         dest[idst] = dest[idst] + cb[in_tile_index]
//
//   i.e. the accumulator tile already resident in DEST is fed back as SrcA
//   (MOVD2A via the DEST_TO_SRCA reuse path) and the freshly-unpacked cb tile is
//   SrcB; the sum overwrites DEST[idst]. To match the header's seed-then-fold
//   usage shape, DEST is first seeded with a plain NONE-reuse ELWADD A_0 + B_0
//   on the first inner tile of each accumulation group; remaining inner tiles
//   fold in via DEST_TO_SRCA.
//
// NO HiFi INIT WORKAROUND (unlike eltwise_mul_scalar.h). The add header's
// deepseek_binary_dest_reuse_add_tiles_init (eltwise_add_scalar.h:27-34) uses
// the single shorthand
//   llk_math_eltwise_binary_init<ELWADD, NONE, MATH_FIDELITY, reuse>(icb0, icb0)
// for every fidelity — there is no reverted DEFAULT_TENSOR_SHAPE branch. ELWADD
// also only supports MathFidelity::LoFi (fidelity > LoFi is an ELWMUL-only
// hardware feature), so this test runs at LoFi and is NOT xfail.
//
// GOLDEN (mirrors the math above, derived from the header, not a guess):
//   For each output tile, seed = A_0 + B_0 (NONE-reuse ELWADD), then for each
//   remaining inner tile i: dest = dest + B_i (DEST_TO_SRCA). Every lane of
//   every output tile is defined (a full tile is packed), so the golden
//   validates all lanes at the format tolerance.

#include <cstdint>

#include "ckernel.h"
#include "counters.h"
#include "llk_defs.h"
#include "perf.h"
#include "profiler.h"
#include "tensor_shape.h"

using namespace ckernel;

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// This op only ever reuses DEST as SrcA (the deepseek default), exactly like the
// header's deepseek_binary_dest_reuse_add_tiles<..., DEST_TO_SRCA>.
static constexpr EltwiseBinaryReuseDestType REUSE_DEST_TYPE = EltwiseBinaryReuseDestType::DEST_TO_SRCA;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_AB.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint8_t face_r_dim           = static_cast<std::uint8_t>(params.TEST_FACE_R_DIM);
    const std::uint8_t face_c_dim           = static_cast<std::uint8_t>(params.TEST_FACE_C_DIM);
    const std::uint8_t num_faces_r_dim      = static_cast<std::uint8_t>(params.num_faces_r_dim_A);
    const std::uint8_t num_faces_c_dim      = static_cast<std::uint8_t>(params.num_faces_c_dim_A);
    const ckernel::TensorShape tensor_shape = {face_r_dim, face_c_dim, num_faces_r_dim, num_faces_c_dim};

    const std::uint32_t input_tiles_in_block  = params.INPUT_NUM_TILES_IN_BLOCK;
    const std::uint32_t output_tiles_in_block = params.OUTPUT_NUM_TILES_IN_BLOCK;
    const std::uint32_t num_blocks            = params.INPUT_NUM_BLOCKS;
    const std::uint32_t inner_dim             = input_tiles_in_block / output_tiles_in_block;
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
#endif

    {
        START_PERF_MEASURE("INIT")
        // compute_kernel_hw_startup
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src,
            formats.unpack_B_src,
            formats.unpack_A_dst,
            formats.unpack_B_dst,
            tensor_shape.face_r_dim,
            tensor_shape.face_r_dim,
            tensor_shape.total_num_faces(),
            tensor_shape.total_num_faces());

        _llk_unpack_configure_stoch_rnd_<StochRndType::None>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            const std::uint32_t num_faces = tensor_shape.total_num_faces();
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block = 0; block < num_blocks; ++block)
                {
                    for (std::uint32_t i = 0; i < inner_dim; ++i)
                    {
                        for (std::uint32_t tile = 0; tile < output_tiles_in_block; ++tile)
                        {
                            // Seed AB and DEST_TO_SRCA fold both publish SrcA+SrcB once per face
                            // (fold SrcA is a dummy; do not treat dest-reuse as SrcB-only).
                            _perf_unpack_loop_set_valid<true /*set_a*/, true /*set_b*/>(num_faces);
                        }
                    }
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block = 0; block < num_blocks; ++block)
                {
                    // One init per phase, and not until the previous phase's unpacks
                    // have run. Programming DEST_TO_SRCA in INIT overwrites the seed
                    // MOP before any seed tile is issued.
                    _llk_unpack_AB_init_<BroadcastType::NONE>(tensor_shape, ckernel::Transpose::None);
                    for (std::uint32_t tile = 0; tile < output_tiles_in_block; ++tile)
                    {
                        const std::uint32_t input_tile_idx = block * input_tiles_in_block + tile;
                        _llk_unpack_AB_<BroadcastType::NONE>(L1_ADDRESS(params.buffer_A[input_tile_idx]), L1_ADDRESS(params.buffer_B[input_tile_idx]));
                    }
                    if (inner_dim > 1)
                    {
                        _llk_unpack_A_init_<BroadcastType::NONE, true /* acc_to_dest */, REUSE_DEST_TYPE>(
                            0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, tensor_shape, formats.unpack_B_src, formats.unpack_B_dst);
                        for (std::uint32_t i = 1; i < inner_dim; ++i)
                        {
                            for (std::uint32_t tile = 0; tile < output_tiles_in_block; ++tile)
                            {
                                const std::uint32_t input_tile_idx = block * input_tiles_in_block + i * output_tiles_in_block + tile;
                                _llk_unpack_A_<BroadcastType::NONE, true /* acc_to_dest */, REUSE_DEST_TYPE>(
                                    L1_ADDRESS(params.buffer_B[input_tile_idx]), formats.unpack_B_src, formats.unpack_B_dst);
                            }
                        }
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
#include "params.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint8_t face_r_dim      = static_cast<std::uint8_t>(params.TEST_FACE_R_DIM);
    const std::uint8_t face_c_dim      = static_cast<std::uint8_t>(params.TEST_FACE_C_DIM);
    const std::uint8_t num_faces_r_dim = static_cast<std::uint8_t>(params.num_faces_r_dim_A);
    const std::uint8_t num_faces_c_dim = static_cast<std::uint8_t>(params.num_faces_c_dim_A);
    const TensorShape tensor_shape     = {face_r_dim, face_c_dim, num_faces_r_dim, num_faces_c_dim};
    constexpr bool ACC_TO_DEST         = false;

    const std::uint32_t input_tiles_in_block  = params.INPUT_NUM_TILES_IN_BLOCK;
    const std::uint32_t output_tiles_in_block = params.OUTPUT_NUM_TILES_IN_BLOCK;
    const std::uint32_t num_blocks            = params.INPUT_NUM_BLOCKS;
    const std::uint32_t inner_dim             = input_tiles_in_block / output_tiles_in_block;
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
#endif

    {
        START_PERF_MEASURE("INIT")
        // compute_kernel_hw_startup
        _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            const std::uint32_t num_faces = tensor_shape.total_num_faces();
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                _perf_math_loop_clear_valid<true /*clear_a*/, true /*clear_b*/>(num_blocks * inner_dim * output_tiles_in_block * num_faces);
            }
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

                    // Seed init: plain NONE-reuse ELWADD to establish DEST[tile] = A_0 + B_0.
                    _llk_math_eltwise_binary_init_<EltwiseBinaryType::ELWADD, BroadcastType::NONE, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(
                        tensor_shape, ACC_TO_DEST);
                    for (std::uint32_t tile = 0; tile < output_tiles_in_block; ++tile)
                    {
                        LLK_ASSERT(
                            (tile < get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                            "Seed tile index exceeds maximum destination tiles");
                        _llk_math_eltwise_binary_<
                            EltwiseBinaryType::ELWADD,
                            BroadcastType::NONE,
                            dest_sync,
                            is_fp32_dest_acc_en,
                            MATH_FIDELITY,
                            EltwiseBinaryReuseDestType::NONE>(tensor_shape, tile /* dst_index */, false /* clear_fp32_dst_acc */);
                    }

                    // deepseek_binary_dest_reuse_add_tiles_init (math half). Unlike the mul
                    // sibling, the add header (eltwise_add_scalar.h:27-34) uses a single
                    // shorthand init for every fidelity — there is no reverted HiFi
                    // DEFAULT_TENSOR_SHAPE branch. We mirror it with the general init on the
                    // real tensor_shape, which is the shorthand's expansion.
                    _llk_math_eltwise_binary_init_<EltwiseBinaryType::ELWADD, BroadcastType::NONE, MATH_FIDELITY, REUSE_DEST_TYPE>(tensor_shape, ACC_TO_DEST);

                    // Fold: dest[tile] = dest[tile] + B_i for each remaining inner tile.
                    for (std::uint32_t i = 1; i < inner_dim; ++i)
                    {
                        for (std::uint32_t tile = 0; tile < output_tiles_in_block; ++tile)
                        {
                            LLK_ASSERT(
                                (tile < get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                                "Fold tile index exceeds maximum destination tiles");
                            _llk_math_eltwise_binary_<
                                EltwiseBinaryType::ELWADD,
                                BroadcastType::NONE,
                                dest_sync,
                                is_fp32_dest_acc_en,
                                MATH_FIDELITY,
                                REUSE_DEST_TYPE>(tensor_shape, tile /* dst_index */, false /* clear_fp32_dst_acc */);
                        }
                    }

                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                    {
                        _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                    }
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
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint8_t face_r_dim           = static_cast<std::uint8_t>(params.TEST_FACE_R_DIM);
    const std::uint8_t face_c_dim           = static_cast<std::uint8_t>(params.TEST_FACE_C_DIM);
    const std::uint8_t num_faces_r_dim      = static_cast<std::uint8_t>(params.num_faces_r_dim_A);
    const std::uint8_t num_faces_c_dim      = static_cast<std::uint8_t>(params.num_faces_c_dim_A);
    const ckernel::TensorShape tensor_shape = {face_r_dim, face_c_dim, num_faces_r_dim, num_faces_c_dim};

    const std::uint32_t tile_size = tensor_shape.total_tensor_size();
    const std::uint32_t num_faces = tensor_shape.total_num_faces();
    const bool partial_face       = tensor_shape.face_r_dim < FACE_R_DIM;
    const bool narrow_tile        = (tensor_shape.num_faces_c_dim == 1);
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
#endif
    const std::uint32_t output_tiles_in_block = params.OUTPUT_NUM_TILES_IN_BLOCK;
    const std::uint32_t output_num_blocks     = params.OUTPUT_NUM_BLOCKS;

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
                        const std::uint32_t res_tile_idx = (block * output_tiles_in_block) + tile;
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
