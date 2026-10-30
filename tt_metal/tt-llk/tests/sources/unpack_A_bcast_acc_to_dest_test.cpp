// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Unary broadcast with acc_to_dest: a plain SEED_BROADCAST_TYPE pass of A seeds Dest, then a BROADCAST_TYPE pass of B
// runs with acc_to_dest = ACC_TO_DEST. The result is seed_bcast(A) + bcast(B) when accumulating, bcast(B) otherwise.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static_assert(BROADCAST_TYPE != ckernel::BroadcastType::NONE && SEED_BROADCAST_TYPE != ckernel::BroadcastType::NONE, "This test covers unary broadcasts only");
static_assert(!unpack_to_dest, "acc_to_dest is not supported when unpacking to dest");

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

template <ckernel::BroadcastType bcast_type, bool acc_to_dest>
inline void unpack_block(const RuntimeParams& params, const ckernel::TensorShape tensor_shape, const std::uint32_t block, const Operand& buffer)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_A_init_<bcast_type, acc_to_dest, EltwiseBinaryReuseDestType::NONE, false /*unpack_to_dest*/>(
        0 /*transpose_of_faces*/, 0 /*within_face_16x16_transpose*/, tensor_shape, formats.unpack_A_src, formats.unpack_A_dst);
    for (std::uint32_t tile = 0; tile < params.NUM_TILES_IN_BLOCK; ++tile)
    {
        _llk_unpack_A_<bcast_type, acc_to_dest, EltwiseBinaryReuseDestType::NONE, false /*unpack_to_dest*/>(
            L1_ADDRESS(buffer[(block * params.NUM_TILES_IN_BLOCK) + tile]), formats.unpack_A_src, formats.unpack_A_dst);
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src,
        formats.unpack_B_src,
        formats.unpack_A_dst,
        formats.unpack_B_dst,
        params.TEST_FACE_R_DIM,
        params.TEST_FACE_R_DIM,
        params.num_faces,
        params.num_faces);
    _llk_unpack_configure_stoch_rnd_<StochRndType::None>();
    const ckernel::TensorShape tensor_shape =
        ckernel::make_tensor_shape(params.TEST_FACE_R_DIM, params.TEST_FACE_C_DIM, params.num_faces_r_dim_A, params.num_faces_c_dim_A);

    for (std::uint32_t block = 0; block < static_cast<std::uint32_t>(params.NUM_BLOCKS); ++block)
    {
        unpack_block<SEED_BROADCAST_TYPE, false /*acc_to_dest*/>(params, tensor_shape, block, params.buffer_A);
        unpack_block<BROADCAST_TYPE, ACC_TO_DEST>(params, tensor_shape, block, params.buffer_B);
    }
    _llk_unpack_A_uninit_<BROADCAST_TYPE>();
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_lib_math_wrappers.h"

using namespace ckernel;

template <BroadcastType bcast_type, bool acc_to_dest>
inline void broadcast_block(const RuntimeParams& params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::B2D, is_fp32_dest_acc_en, bcast_type, false /*is_int_fpu_en*/, PackMode::Default, acc_to_dest>(
        params.num_faces, formats.math);
    for (std::uint32_t tile = 0; tile < params.NUM_TILES_IN_BLOCK; ++tile)
    {
        LLK_ASSERT(
            (tile < get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()), "Block tile index exceeds maximum destination tiles");
        _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::B2D, dest_sync, is_fp32_dest_acc_en, bcast_type, false /*unpack_to_dest*/, acc_to_dest>(
            tile, formats.math, formats.math, params.num_faces);
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    for (std::uint32_t block = 0; block < static_cast<std::uint32_t>(params.NUM_BLOCKS); ++block)
    {
        _llk_math_wait_for_dest_available_<dest_sync>();
        broadcast_block<SEED_BROADCAST_TYPE, false /*acc_to_dest*/>(params);
        broadcast_block<BROADCAST_TYPE, ACC_TO_DEST>(params);
        _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
    }
    _llk_math_eltwise_unary_datacopy_uninit_<BROADCAST_TYPE, false /*unpack_to_dest*/>();
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
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
        formats.pack_src,
        formats.pack_dst,
        params.TEST_FACE_R_DIM * params.TEST_FACE_C_DIM * params.num_faces /*tile_size*/,
        params.TEST_FACE_R_DIM,
        TILE_C_DIM,
        params.num_faces);
    _llk_pack_init_wrapper_<PackMode::Default, false /*zero_output*/>(formats.pack_dst, params.TEST_FACE_R_DIM, TILE_C_DIM, params.num_faces);
    _llk_pack_dest_init_wrapper_<dest_sync, is_fp32_dest_acc_en, PackMode::Default>();

    for (std::uint32_t block = 0; block < static_cast<std::uint32_t>(params.NUM_BLOCKS); ++block)
    {
        _llk_packer_wait_for_math_done_();
        for (std::uint32_t tile = 0; tile < params.NUM_TILES_IN_BLOCK; ++tile)
        {
            LLK_ASSERT(
                (tile < get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()), "Block tile index exceeds maximum destination tiles");
            _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(
                tile, L1_ADDRESS(params.buffer_Res[(block * params.NUM_TILES_IN_BLOCK) + tile]));
        }
        _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
    }
}

#endif
