// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// Source-register valids that one _llk_unpack_A_ call publishes and one
// _llk_math_eltwise_unary_datacopy_ call consumes, outside unpack-to-dest.
// The isolate modes replace one side of the handshake with these counts, pairing
// SrcA with SrcB face by face as the unpack MOPs do.
inline std::uint32_t srca_valids_per_tile(const std::uint32_t num_faces)
{
    if constexpr (BROADCAST_TYPE == ckernel::BroadcastType::NONE)
    {
        return num_faces;
    }
    else if constexpr (BROADCAST_TYPE == ckernel::BroadcastType::ROW)
    {
        return 0;
    }
    else
    {
        return 1; // COL and SCALAR publish one zeroed SrcA bank for ELWADD
    }
}

inline std::uint32_t srcb_valids_per_tile(const std::uint32_t num_faces)
{
    if constexpr (BROADCAST_TYPE == ckernel::BroadcastType::NONE)
    {
        return num_faces;
    }
    else if constexpr (BROADCAST_TYPE == ckernel::BroadcastType::ROW)
    {
#ifdef ARCH_WORMHOLE
        return 4; // Wormhole's ROW unpack and B2D MOPs always move four faces
#else
        return num_faces;
#endif
    }
    else if constexpr (BROADCAST_TYPE == ckernel::BroadcastType::COL)
    {
        return 2;
    }
    else
    {
        return 1;
    }
}

struct SrcValidsPerTile
{
    std::uint32_t paired;
    std::uint32_t a_only;
    std::uint32_t b_only;
};

inline SrcValidsPerTile src_valids_per_tile(const std::uint32_t num_faces)
{
    const std::uint32_t srca_valids = srca_valids_per_tile(num_faces);
    const std::uint32_t srcb_valids = srcb_valids_per_tile(num_faces);
    const std::uint32_t paired      = std::min(srca_valids, srcb_valids);
    return {paired, srca_valids - paired, srcb_valids - paired};
}

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR        = params.LOOP_FACTOR;
    const std::uint32_t num_faces          = params.num_faces;
    const int NUM_BLOCKS                   = params.NUM_BLOCKS;
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;
    const std::uint32_t TEST_FACE_R_DIM    = params.TEST_FACE_R_DIM;
    const std::uint32_t TEST_FACE_C_DIM    = params.TEST_FACE_C_DIM;
    const int num_faces_r_dim_A            = params.num_faces_r_dim_A;
    const int num_faces_c_dim_A            = params.num_faces_c_dim_A;
    const Operand& buffer_A                = params.buffer_A;
#endif
    const std::uint32_t num_tiles = NUM_BLOCKS * NUM_TILES_IN_BLOCK;

    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, TEST_FACE_R_DIM, TEST_FACE_R_DIM, num_faces, num_faces);
        _llk_unpack_configure_stoch_rnd_<StochRndType::None>();
        const ckernel::TensorShape tensor_shape = ckernel::make_tensor_shape(TEST_FACE_R_DIM, TEST_FACE_C_DIM, num_faces_r_dim_A, num_faces_c_dim_A);
        _llk_unpack_A_init_<BROADCAST_TYPE, false /* acc_to_dest */, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, tensor_shape, formats.unpack_A_src, formats.unpack_A_dst);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE && !unpack_to_dest)
        {
            const SrcValidsPerTile valids = src_valids_per_tile(num_faces);
            for (std::uint32_t tile = 0; tile < LOOP_FACTOR * num_tiles; ++tile)
            {
                _perf_unpack_loop_set_valid<true /* set_a */, true /* set_b */>(valids.paired);
                _perf_unpack_loop_set_valid<true /* set_a */, false /* set_b */>(valids.a_only);
                _perf_unpack_loop_set_valid<false /* set_a */, true /* set_b */>(valids.b_only);
            }
        }
        else
        {
            // Unpack-to-dest hands each tile to math through the UNPACK_TO_DEST semaphore and the
            // math->unpack mailbox, so every non-pack mode runs the real unpacker for it.
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t tile = 0; tile < num_tiles; ++tile)
                {
                    _llk_unpack_A_<BROADCAST_TYPE, false /* acc_to_dest */, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                        L1_ADDRESS(buffer_A[tile]), formats.unpack_A_src, formats.unpack_A_dst);
                }
            }
        }
        PROFILER_SYNC();
    }
    _llk_unpack_A_uninit_<BROADCAST_TYPE>();
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_lib_math_wrappers.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR        = params.LOOP_FACTOR;
    const std::uint32_t num_faces          = params.num_faces;
    const int NUM_BLOCKS                   = params.NUM_BLOCKS;
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;
#endif
    constexpr DstSync sync_mode = DstSync::SyncHalf;
    // Broadcasts arrive in SrcB and copy B2D; the NONE datacopy (and unpack-to-dest) copies A2D.
    constexpr DataCopyType copy_type = (BROADCAST_TYPE == BroadcastType::NONE || unpack_to_dest) ? DataCopyType::A2D : DataCopyType::B2D;

    {
        START_PERF_MEASURE("INIT")
        _llk_math_eltwise_unary_datacopy_init_wrapper_<copy_type, is_fp32_dest_acc_en, BROADCAST_TYPE, false /* is_int_fpu_en */, PackMode::Default>(
            num_faces, formats.math);
        _llk_math_pack_sync_init_<sync_mode, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr ((PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION) && !unpack_to_dest)
        {
            const SrcValidsPerTile valids = src_valids_per_tile(num_faces);
            for (std::uint32_t tile = 0; tile < LOOP_FACTOR * NUM_BLOCKS * NUM_TILES_IN_BLOCK; ++tile)
            {
                _perf_math_loop_clear_valid<true /* clear_a */, true /* clear_b */>(valids.paired);
                _perf_math_loop_clear_valid<true /* clear_a */, false /* clear_b */>(valids.a_only);
                _perf_math_loop_clear_valid<false /* clear_a */, true /* clear_b */>(valids.b_only);
            }
        }
        else
        {
            // Pack only drains Dest in L1_TO_L1; every other mode must not wait on it.
            constexpr bool dest_sync_en = (PERF_RUN_TYPE == PerfRunType::L1_TO_L1);
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (int block = 0; block < NUM_BLOCKS; ++block)
                {
                    if constexpr (dest_sync_en)
                    {
                        _llk_math_wait_for_dest_available_<sync_mode>();
                    }
                    for (std::uint32_t tile_in_block = 0; tile_in_block < NUM_TILES_IN_BLOCK; ++tile_in_block)
                    {
                        LLK_ASSERT(
                            (tile_in_block < get_dest_max_tiles<sync_mode, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                            "Block tile index exceeds maximum destination tiles");
                        _llk_math_eltwise_unary_datacopy_<copy_type, sync_mode, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                            tile_in_block, formats.math, formats.math);
                    }
                    if constexpr (dest_sync_en)
                    {
                        _llk_math_dest_section_done_<sync_mode, is_fp32_dest_acc_en>();
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
    _llk_math_eltwise_unary_datacopy_uninit_<BROADCAST_TYPE, unpack_to_dest>();
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
    const std::uint32_t LOOP_FACTOR        = params.LOOP_FACTOR;
    const std::uint32_t num_faces          = params.num_faces;
    const int NUM_BLOCKS                   = params.NUM_BLOCKS;
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;
    const std::uint32_t TEST_FACE_R_DIM    = params.TEST_FACE_R_DIM;
    const std::uint32_t TEST_FACE_C_DIM    = params.TEST_FACE_C_DIM;
    const Operand& buffer_Res              = params.buffer_Res;
#endif
    constexpr DstSync sync_mode = DstSync::SyncHalf;

    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
            formats.pack_src, formats.pack_dst, TEST_FACE_R_DIM * TEST_FACE_C_DIM * num_faces /* tile_size */, TEST_FACE_R_DIM, TILE_C_DIM, num_faces);
        _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, TEST_FACE_R_DIM, TILE_C_DIM, num_faces);
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
            // Math only fills Dest in L1_TO_L1; PACK_ISOLATE and L1_CONGESTION must not wait on it.
            constexpr bool dest_sync_en = (PERF_RUN_TYPE == PerfRunType::L1_TO_L1);
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (int block = 0; block < NUM_BLOCKS; ++block)
                {
                    if constexpr (dest_sync_en)
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
                    if constexpr (dest_sync_en)
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
