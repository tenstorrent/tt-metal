
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The SDPA MOP streams SrcA without a SrcB dummy; A2D math is an identity copy.

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "counters.h"
#include "llk_defs.h"
#include "perf.h"
#include "profiler.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

// The metal API wrapper depends on headers outside the LLK test include path.
// This identity specialization leaves several experimental-header arguments unused.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "experimental/llk_unpack_A_sdpa.h"
#pragma GCC diagnostic pop
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR         = params.LOOP_FACTOR;
    const std::uint32_t NUM_TILES_IN_BLOCK  = params.NUM_TILES_IN_BLOCK;
    const std::uint32_t NUM_BLOCKS          = params.NUM_BLOCKS;
    const std::uint32_t TEST_FACE_R_DIM     = params.TEST_FACE_R_DIM;
    const std::uint32_t num_faces           = params.num_faces;
    const bool UNPACK_TRANSPOSE_FACES       = params.UNPACK_TRANSPOSE_FACES;
    const bool UNPACK_TRANSPOSE_WITHIN_FACE = params.UNPACK_TRANSPOSE_WITHIN_FACE;
    const Operand& buffer_A                 = params.buffer_A;
#endif

    // Only the datacopy-identity configuration the SDPA unpack path supports.
    constexpr BroadcastType bcast_type              = BroadcastType::NONE;
    constexpr bool acc_to_dest                      = false;
    constexpr EltwiseBinaryReuseDestType reuse_dest = EltwiseBinaryReuseDestType::NONE;
    constexpr bool sdpa_unpack_to_dest              = false;
    // SDPA init programs a single MOP that streams num_tiles*num_faces SrcA UNPACRs.
    // Drive one tile per MOP run so the execute below maps 1 run -> 1 tile.
    constexpr std::uint32_t sdpa_num_tiles = 1;

    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, TEST_FACE_R_DIM, TEST_FACE_R_DIM, num_faces, num_faces);
        _llk_unpack_configure_stoch_rnd_<STOCHASTIC_RND>();
        _llk_unpack_A_sdpa_init_<sdpa_num_tiles, bcast_type, acc_to_dest, reuse_dest, sdpa_unpack_to_dest>(
            UNPACK_TRANSPOSE_FACES, UNPACK_TRANSPOSE_WITHIN_FACE, TEST_FACE_R_DIM, num_faces, formats.unpack_A_src, formats.unpack_A_dst);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            const std::uint32_t src_handshake_iters = LOOP_FACTOR * NUM_TILES_IN_BLOCK * NUM_BLOCKS * num_faces;
            _perf_unpack_stream_set_valid<true /*set_a*/, false /*set_b*/, PERF_RUN_TYPE>(src_handshake_iters);
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t i = 0; i < NUM_TILES_IN_BLOCK * NUM_BLOCKS; ++i)
                {
                    _llk_unpack_A_<bcast_type, acc_to_dest, reuse_dest, sdpa_unpack_to_dest>(
                        L1_ADDRESS(buffer_A[i]), formats.unpack_A_src, formats.unpack_A_dst);
                }
            }
        }
        PROFILER_SYNC();
    }
    _llk_unpack_A_uninit_<bcast_type>();
}

#endif

#ifdef LLK_TRISC_MATH

#ifdef FORMAT_INT32
const bool is_int_fpu_en = true;
#else
const bool is_int_fpu_en = false;
#endif

#include "llk_lib_math_wrappers.h"
#include "params.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR        = params.LOOP_FACTOR;
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;
    const std::uint32_t NUM_BLOCKS         = params.NUM_BLOCKS;
    const std::uint32_t TEST_FACE_R_DIM    = params.TEST_FACE_R_DIM;
    const std::uint32_t num_faces          = params.num_faces;
#endif

    // Test configuration constants
    constexpr DstSync sync_mode = DstSync::SyncHalf;

    // SDPA unpack keeps data in SrcA (BType::NONE, unpack_to_dest=false) -> A2D copy.
    constexpr DataCopyType copy_type        = DataCopyType::A2D;
    const ckernel::TensorShape tensor_shape = ckernel::make_tensor_shape_from_legacy(TEST_FACE_R_DIM, num_faces);
    {
        START_PERF_MEASURE("INIT")
        _llk_math_eltwise_unary_datacopy_init_wrapper_<copy_type, is_fp32_dest_acc_en, BroadcastType::NONE, is_int_fpu_en, PackMode::Default>(
            tensor_shape, formats.math);

        // SrcB is never valid, so this identity path clears SrcA without touching SrcB.
        {
            constexpr std::uint32_t innerloop = 16 >> 3;
            const std::uint32_t outerloop     = num_faces;
            ckernel::ckernel_template tmp(outerloop, innerloop, TT_OP_MOVA2D(ckernel::p_mov::DEST_NORM, 0, ADDR_MOD_2, ckernel::p_mova2d::MOV_8_ROWS, 0));
            tmp.set_end_op(TT_OP_SETRWC(ckernel::p_setrwc::CLR_A, 0, 0, 0, 0, ckernel::p_setrwc::SET_AB));
            tmp.program();
        }

        _llk_math_pack_sync_init_<sync_mode, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            const std::uint32_t src_handshake_iters = LOOP_FACTOR * NUM_TILES_IN_BLOCK * NUM_BLOCKS * num_faces;
            _perf_math_stream_clear_valid<true /*clear_a*/, false /*clear_b*/, PERF_RUN_TYPE>(src_handshake_iters);
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block = 0; block < NUM_BLOCKS; ++block)
                {
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                    {
                        _llk_math_wait_for_dest_available_<sync_mode>();
                    }
                    for (std::uint32_t tile_in_block = 0; tile_in_block < NUM_TILES_IN_BLOCK; ++tile_in_block)
                    {
                        LLK_ASSERT(
                            (tile_in_block < get_dest_max_tiles<sync_mode, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                            "Block tile index exceeds maximum destination tiles");
                        _llk_math_eltwise_unary_datacopy_<copy_type, sync_mode, is_fp32_dest_acc_en, BroadcastType::NONE, false>(
                            tile_in_block, formats.math, formats.math, tensor_shape);
                    }
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                    {
                        _llk_math_dest_section_done_<sync_mode, is_fp32_dest_acc_en>();
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
    _llk_math_eltwise_unary_datacopy_uninit_<BroadcastType::NONE, false>();
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
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR        = params.LOOP_FACTOR;
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;
    const std::uint32_t NUM_BLOCKS         = params.NUM_BLOCKS;
    const std::uint32_t TEST_FACE_R_DIM    = params.TEST_FACE_R_DIM;
    const std::uint32_t TEST_FACE_C_DIM    = params.TEST_FACE_C_DIM;
    const std::uint32_t num_faces          = params.num_faces;
    const Operand& buffer_Res              = params.buffer_Res;
#endif

    // Test configuration constants
    constexpr DstSync sync_mode = DstSync::SyncHalf;
    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
            formats.pack_src, formats.pack_dst, TEST_FACE_R_DIM * TEST_FACE_C_DIM * 4, TEST_FACE_R_DIM, TILE_C_DIM, num_faces);
        _llk_pack_init_wrapper_<PackMode::Default, false /*zero_output*/>(formats.pack_dst, TEST_FACE_R_DIM, TILE_C_DIM, num_faces);
        _llk_pack_dest_init_wrapper_<sync_mode, is_fp32_dest_acc_en, PackMode::Default>();
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
                for (std::uint32_t block = 0; block < NUM_BLOCKS; ++block)
                {
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                    {
                        _llk_packer_wait_for_math_done_();
                    }
                    for (std::uint32_t tile_in_block = 0; tile_in_block < NUM_TILES_IN_BLOCK; ++tile_in_block)
                    {
                        LLK_ASSERT(
                            (tile_in_block < get_dest_max_tiles<sync_mode, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                            "Block tile index exceeds maximum destination tiles");
                        _llk_pack_<sync_mode, is_fp32_dest_acc_en, ckernel::PackMode::Default>(
                            tile_in_block, L1_ADDRESS(buffer_Res[(block * NUM_TILES_IN_BLOCK) + tile_in_block]));
                    }
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_PACK_SECTION_RELEASE)
                    {
                        _llk_pack_dest_section_done_<sync_mode, is_fp32_dest_acc_en>();
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
}
#endif
