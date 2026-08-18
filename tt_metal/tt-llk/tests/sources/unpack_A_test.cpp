
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// extern const struct RuntimeParams *__runtime_args_start;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t num_tiles_in_block = params.NUM_TILES_IN_BLOCK;
    const std::uint32_t num_blocks         = params.NUM_BLOCKS;

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src,
        formats.unpack_B_src,
        formats.unpack_A_dst,
        formats.unpack_B_dst,
        params.TEST_FACE_R_DIM,
        params.TEST_FACE_R_DIM,
        params.num_faces,
        params.num_faces);
    _llk_unpack_configure_stoch_rnd_<STOCHASTIC_RND>();
    const ckernel::TensorShape tensor_shape =
        ckernel::make_tensor_shape(params.TEST_FACE_R_DIM, params.TEST_FACE_C_DIM, params.num_faces_r_dim_A, params.num_faces_c_dim_A);
    _llk_unpack_A_init_<BROADCAST_TYPE, ACC_TO_DEST, REUSE_DEST_TYPE, unpack_to_dest>(
        params.UNPACK_TRANSPOSE_FACES, params.UNPACK_TRANSPOSE_WITHIN_FACE, tensor_shape, formats.unpack_A_src, formats.unpack_A_dst);

    // Accumulating broadcast: run every block twice. The first pass uses the plain broadcast staging and primes DEST
    // with bcast(src_A); the second uses the accumulating staging and adds the same broadcast on top, so the packed
    // result is 2 * bcast(src_A). Priming is what makes the check meaningful: the packer zeroes DEST between blocks,
    // so accumulating onto a fresh DEST would produce exactly what a plain copy produces.
    constexpr bool BCAST_ACC_TO_DEST = ACC_TO_DEST && (BROADCAST_TYPE != ckernel::BroadcastType::NONE) && !unpack_to_dest;

    if constexpr (BCAST_ACC_TO_DEST)
    {
        for (std::uint32_t block = 0; block < num_blocks; ++block)
        {
            _llk_unpack_A_init_<BROADCAST_TYPE, false /* acc_to_dest */, REUSE_DEST_TYPE, unpack_to_dest>(
                params.UNPACK_TRANSPOSE_FACES, params.UNPACK_TRANSPOSE_WITHIN_FACE, tensor_shape, formats.unpack_A_src, formats.unpack_A_dst);
            for (std::uint32_t tile_in_block = 0; tile_in_block < num_tiles_in_block; ++tile_in_block)
            {
                _llk_unpack_A_<BROADCAST_TYPE, false /* acc_to_dest */, REUSE_DEST_TYPE, unpack_to_dest>(
                    L1_ADDRESS(params.buffer_A[(block * num_tiles_in_block) + tile_in_block]), formats.unpack_A_src, formats.unpack_A_dst);
            }

            _llk_unpack_A_init_<BROADCAST_TYPE, BCAST_ACC_TO_DEST, REUSE_DEST_TYPE, unpack_to_dest>(
                params.UNPACK_TRANSPOSE_FACES, params.UNPACK_TRANSPOSE_WITHIN_FACE, tensor_shape, formats.unpack_A_src, formats.unpack_A_dst);
            for (std::uint32_t tile_in_block = 0; tile_in_block < num_tiles_in_block; ++tile_in_block)
            {
                _llk_unpack_A_<BROADCAST_TYPE, BCAST_ACC_TO_DEST, REUSE_DEST_TYPE, unpack_to_dest>(
                    L1_ADDRESS(params.buffer_A[(block * num_tiles_in_block) + tile_in_block]), formats.unpack_A_src, formats.unpack_A_dst);
            }
        }
    }
    else
    {
        for (std::uint32_t i = 0; i < num_tiles_in_block * num_blocks; ++i)
        {
            _llk_unpack_A_<BROADCAST_TYPE, ACC_TO_DEST, REUSE_DEST_TYPE, unpack_to_dest>(
                L1_ADDRESS(params.buffer_A[i]), formats.unpack_A_src, formats.unpack_A_dst);
        }
    }
    _llk_unpack_A_uninit_<BROADCAST_TYPE>();
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
    const std::uint32_t num_tiles_in_block = params.NUM_TILES_IN_BLOCK;
    const std::uint32_t num_blocks         = params.NUM_BLOCKS;

    // Test configuration constants
    constexpr DstSync sync_mode = DstSync::SyncHalf;

    // copy srca to dest
    // Use B2D for all broadcasts except NONE (data in srcB), A2D for NONE (data in srcA)
    constexpr DataCopyType copy_type = (BROADCAST_TYPE == BroadcastType::NONE || unpack_to_dest) ? DataCopyType::A2D : DataCopyType::B2D;

    // Accumulating broadcast: the plain broadcast copy primes DEST with bcast(src_A) and the accumulating copy adds
    // the same broadcast on top, so the section holds 2 * bcast(src_A). Both passes consume one unpacked operand per
    // tile, matching the two unpack passes on T0.
    constexpr bool BCAST_ACC_TO_DEST = ACC_TO_DEST && (BROADCAST_TYPE != BroadcastType::NONE) && !unpack_to_dest;

    _llk_math_eltwise_unary_datacopy_init_wrapper_<copy_type, is_fp32_dest_acc_en, BROADCAST_TYPE, is_int_fpu_en, PackMode::Default>(
        params.num_faces, formats.math);
    _llk_math_pack_sync_init_<sync_mode, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    for (std::uint32_t block = 0; block < num_blocks; ++block)
    {
        _llk_math_wait_for_dest_available_<sync_mode>();
        if constexpr (BCAST_ACC_TO_DEST)
        {
            _llk_math_eltwise_unary_datacopy_init_wrapper_<
                copy_type,
                is_fp32_dest_acc_en,
                BROADCAST_TYPE,
                is_int_fpu_en,
                PackMode::Default,
                false /* acc_to_dest */>(params.num_faces, formats.math);
        }
        for (std::uint32_t tile_in_block = 0; tile_in_block < num_tiles_in_block; ++tile_in_block)
        {
            LLK_ASSERT(
                (tile_in_block < get_dest_max_tiles<sync_mode, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                "Block tile index exceeds maximum destination tiles");
            _llk_math_eltwise_unary_datacopy_<copy_type, sync_mode, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                tile_in_block, formats.math, formats.math);
        }
        if constexpr (BCAST_ACC_TO_DEST)
        {
            _llk_math_eltwise_unary_datacopy_init_wrapper_<copy_type, is_fp32_dest_acc_en, BROADCAST_TYPE, is_int_fpu_en, PackMode::Default, BCAST_ACC_TO_DEST>(
                params.num_faces, formats.math);
            for (std::uint32_t tile_in_block = 0; tile_in_block < num_tiles_in_block; ++tile_in_block)
            {
                _llk_math_eltwise_unary_datacopy_<copy_type, sync_mode, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                    tile_in_block, formats.math, formats.math);
            }
        }
        _llk_math_dest_section_done_<sync_mode, is_fp32_dest_acc_en>();
    }
    _llk_math_eltwise_unary_datacopy_uninit_<BROADCAST_TYPE, unpack_to_dest>();
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
    const std::uint32_t num_tiles_in_block = params.NUM_TILES_IN_BLOCK;
    const std::uint32_t num_blocks         = params.NUM_BLOCKS;

    // Test configuration constants
    constexpr DstSync sync_mode = DstSync::SyncHalf;
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
        formats.pack_src, formats.pack_dst, params.TEST_FACE_R_DIM * params.TEST_FACE_C_DIM * 4, params.TEST_FACE_R_DIM, TILE_C_DIM, params.num_faces);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, params.TEST_FACE_R_DIM, TILE_C_DIM, params.num_faces);

    _llk_pack_dest_init_wrapper_<sync_mode, is_fp32_dest_acc_en, PackMode::Default>();

    for (std::uint32_t block = 0; block < num_blocks; ++block)
    {
        _llk_packer_wait_for_math_done_();
        for (std::uint32_t tile_in_block = 0; tile_in_block < num_tiles_in_block; ++tile_in_block)
        {
            LLK_ASSERT(
                (tile_in_block < get_dest_max_tiles<sync_mode, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                "Block tile index exceeds maximum destination tiles");
            _llk_pack_<sync_mode, is_fp32_dest_acc_en, ckernel::PackMode::Default>(
                tile_in_block, L1_ADDRESS(params.buffer_Res[(block * num_tiles_in_block) + tile_in_block]));
        }
        _llk_pack_dest_section_done_<sync_mode, is_fp32_dest_acc_en>();
    }
}
#endif
