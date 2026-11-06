// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

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

// SrcA/SrcB valids one tile publishes outside unpack-to-dest; the isolate modes
// fake them for the absent unpack or math side.
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
        return 2; // COL unpacks SrcB face 0, then face 2 after srcb_set_z_2
    }
    else
    {
        return 1;
    }
}

struct SrcValidsPerTile
{
    std::uint32_t paired;
    std::uint32_t b_only;
};

inline SrcValidsPerTile src_valids_per_tile(const std::uint32_t num_faces)
{
    const std::uint32_t srca_valids = srca_valids_per_tile(num_faces);
    const std::uint32_t srcb_valids = srcb_valids_per_tile(num_faces);
    LLK_ASSERT(srca_valids <= srcb_valids, "Without acc_to_dest no broadcast type publishes more SrcA than SrcB valids");
    return {srca_valids, srcb_valids - srca_valids};
}

// Fakes the absent side's valids for `tiles` tiles: sets them in place of unpack, clears them in place of math.
// COL must alternate its paired and SrcB-only valid per tile; every other type posts one kind, issued in one call.
template <bool set_valids>
inline void fake_src_valids(const std::uint32_t tiles, const std::uint32_t num_faces)
{
    const SrcValidsPerTile valids      = src_valids_per_tile(num_faces);
    constexpr bool per_tile            = (BROADCAST_TYPE == ckernel::BroadcastType::COL);
    const std::uint32_t calls          = per_tile ? tiles : 1;
    const std::uint32_t tiles_per_call = per_tile ? 1 : tiles;
    for (std::uint32_t call = 0; call < calls; ++call)
    {
        if constexpr (set_valids)
        {
            _perf_unpack_loop_set_valid<true /*set_a*/, true /*set_b*/>(tiles_per_call * valids.paired /*iterations*/);
            _perf_unpack_loop_set_valid<false /*set_a*/, true /*set_b*/>(tiles_per_call * valids.b_only /*iterations*/);
        }
        else
        {
            _perf_math_loop_clear_valid<true /*clear_a*/, true /*clear_b*/>(tiles_per_call * valids.paired /*iterations*/);
            _perf_math_loop_clear_valid<false /*clear_a*/, true /*clear_b*/>(tiles_per_call * valids.b_only /*iterations*/);
        }
    }
}

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

// _llk_unpack_A_init_ and _llk_unpack_A_ must agree on these.
constexpr bool unpack_acc_to_dest                             = false;
constexpr EltwiseBinaryReuseDestType unpack_binary_reuse_dest = EltwiseBinaryReuseDestType::NONE;

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
        _llk_unpack_A_init_<BROADCAST_TYPE, unpack_acc_to_dest, unpack_binary_reuse_dest, unpack_to_dest>(
            0 /*transpose_of_faces*/, 0 /*within_face_16x16_transpose*/, tensor_shape, formats.unpack_A_src, formats.unpack_A_dst);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE && unpack_to_dest)
        {
            // Stand in for the unpacker's half of the unpack-to-dest handshake without unpacking:
            // take math's Dest index from the mailbox and release the tile through UNPACK_TO_DEST.
            for (std::uint32_t tile = 0; tile < LOOP_FACTOR * num_tiles; ++tile)
            {
                (void)mailbox_read(ThreadId::MathThreadId);
                wait_for_dest_available();
                t6_semaphore_post<p_stall::UNPACK0>(semaphore::UNPACK_TO_DEST);
#ifdef ARCH_BLACKHOLE
                if constexpr (BROADCAST_TYPE != BroadcastType::NONE)
                {
                    if (ckernel::unpacker::is_32bit_input(formats.unpack_A_src, formats.unpack_A_dst))
                    {
                        // Blackhole's _llk_unpack_A_ also hands math one SrcB bank per such tile; the datacopy waits for it.
                        TTI_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
                    }
                }
#endif
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            fake_src_valids<true /*set_valids*/>(LOOP_FACTOR * num_tiles /*tiles*/, num_faces);
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t tile = 0; tile < num_tiles; ++tile)
                {
                    _llk_unpack_A_<BROADCAST_TYPE, unpack_acc_to_dest, unpack_binary_reuse_dest, unpack_to_dest>(
                        L1_ADDRESS(buffer_A[tile]) /*address*/, formats.unpack_A_src, formats.unpack_A_dst);
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
    const std::uint32_t TEST_FACE_R_DIM    = params.TEST_FACE_R_DIM;
    const std::uint32_t TEST_FACE_C_DIM    = params.TEST_FACE_C_DIM;
    const int num_faces_r_dim_A            = params.num_faces_r_dim_A;
    const int num_faces_c_dim_A            = params.num_faces_c_dim_A;
#endif
    const ckernel::TensorShape tensor_shape = ckernel::make_tensor_shape(TEST_FACE_R_DIM, TEST_FACE_C_DIM, num_faces_r_dim_A, num_faces_c_dim_A);
    // Broadcasts arrive in SrcB and copy B2D; the NONE datacopy (and unpack-to-dest) copies A2D.
    constexpr DataCopyType copy_type = (BROADCAST_TYPE == BroadcastType::NONE || unpack_to_dest) ? DataCopyType::A2D : DataCopyType::B2D;

    {
        START_PERF_MEASURE("INIT")
        _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        _llk_math_eltwise_unary_datacopy_init_wrapper_<copy_type, is_fp32_dest_acc_en, BROADCAST_TYPE, false /*is_int_fpu_en*/, PackMode::Default>(
            tensor_shape, formats.math);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr ((PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION) && unpack_to_dest)
        {
            // Stand in for math's half of the unpack-to-dest handshake without copying: send the unpacker
            // each tile's Dest index through the mailbox and take the tile back through UNPACK_TO_DEST.
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (int block = 0; block < NUM_BLOCKS; ++block)
                {
                    for (std::uint32_t tile_in_block = 0; tile_in_block < NUM_TILES_IN_BLOCK; ++tile_in_block)
                    {
                        math::math_unpack_to_dest_math_ready();
                        math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::DestReg>(tile_in_block);
                        math::math_unpack_to_dest_tile_ready();
#ifdef ARCH_BLACKHOLE
                        if constexpr (BROADCAST_TYPE != BroadcastType::NONE)
                        {
                            if (math::is_32bit_input(formats.unpack_A_src, formats.unpack_A_dst))
                            {
                                // Blackhole's _llk_unpack_A_ hands math one SrcB bank per such tile; release it.
                                _perf_math_loop_clear_valid<false /*clear_a*/, true /*clear_b*/>(1 /*iterations*/);
                            }
                        }
#endif
                    }
                }
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            fake_src_valids<false /*set_valids*/>(LOOP_FACTOR * NUM_BLOCKS * NUM_TILES_IN_BLOCK /*tiles*/, num_faces);
        }
        else
        {
            // Only L1_TO_L1 pairs math with pack through the Dest section handshake.
            constexpr bool dest_sync_en = (PERF_RUN_TYPE == PerfRunType::L1_TO_L1);
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (int block = 0; block < NUM_BLOCKS; ++block)
                {
                    if constexpr (dest_sync_en)
                    {
                        _llk_math_wait_for_dest_available_<dest_sync>();
                    }
                    for (std::uint32_t tile_in_block = 0; tile_in_block < NUM_TILES_IN_BLOCK; ++tile_in_block)
                    {
                        LLK_ASSERT(
                            (tile_in_block < get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                            "Block tile index exceeds maximum destination tiles");
                        _llk_math_eltwise_unary_datacopy_wrapper_<copy_type, dest_sync, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                            tile_in_block, formats.math, formats.math, tensor_shape);
                    }
                    if constexpr (dest_sync_en)
                    {
                        _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
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
    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
            formats.pack_src, formats.pack_dst, TEST_FACE_R_DIM * TEST_FACE_C_DIM * num_faces /*tile_size*/, TEST_FACE_R_DIM, TILE_C_DIM, num_faces);
        _llk_pack_init_wrapper_<PackMode::Default, false /*zero_output*/>(formats.pack_dst, TEST_FACE_R_DIM, TILE_C_DIM, num_faces);
        _llk_pack_dest_init_wrapper_<dest_sync, is_fp32_dest_acc_en, PackMode::Default>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
        }
        else
        {
            // Only L1_TO_L1 pairs pack with math through the Dest section handshake.
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
                            (tile_in_block < get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                            "Block tile index exceeds maximum destination tiles");
                        _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(
                            tile_in_block, L1_ADDRESS(buffer_Res[(block * NUM_TILES_IN_BLOCK) + tile_in_block]) /*address*/);
                    }
                    if constexpr (dest_sync_en)
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
