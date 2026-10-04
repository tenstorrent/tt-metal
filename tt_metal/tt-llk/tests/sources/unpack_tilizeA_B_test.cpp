// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Functional test of the reduce with a tilized operand A (the pool2d configuration of tilizeA_B_reduce_init and
// unpack_tilizeA_B_block): operand A is a row-major block of TILE_CNT tiles with face_r_dim rows per face, unpacked by
// NUM_BLOCKS calls of the tilizeA_B block unpack against a one-row scaler operand B, reduced over its rows (REDUCE_COL)
// and packed with the reduce masks, one row per face (the reduce leaves its result in row 0 of each face).

#include <algorithm>
#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_defs.h"
#include "params.h"
#include "tensor_shape.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

using namespace ckernel;

static constexpr std::uint32_t MAX_TILES_DEST  = is_fp32_dest_acc_en ? 4 : 8;
static constexpr bool NEGINF_SRCA              = (POOL_TYPE == PoolType::MAX);
static constexpr bool ZERO_SRCA_REDUCE         = (POOL_TYPE != PoolType::MAX);
static constexpr std::uint32_t UNPB_FACE_R_DIM = 1;
static constexpr std::uint32_t PACK_FACE_R_DIM = 1;

// The math reduces 16-row faces laid out row-wise first, as reduce_tile_math does for the pool.
inline TensorShape reduce_shape(const std::uint32_t num_faces)
{
    return {
        MAX_FACE_R_DIM,
        MAX_FACE_C_DIM,
        (num_faces <= MAX_NUM_FACES_C_DIM) ? static_cast<std::uint8_t>(1) : MAX_NUM_FACES_R_DIM,
        (num_faces <= MAX_NUM_FACES_C_DIM) ? static_cast<std::uint8_t>(num_faces) : MAX_NUM_FACES_C_DIM};
}

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_common.h"
#include "llk_unpack_tilize.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t num_faces       = params.num_faces;
    const std::uint32_t face_r_dim      = params.TEST_FACE_R_DIM;
    const std::uint32_t num_blocks      = static_cast<std::uint32_t>(params.NUM_BLOCKS);
    const std::uint32_t tiles_per_block = params.TILE_CNT / num_blocks;

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, face_r_dim, UNPB_FACE_R_DIM, num_faces, num_faces);
    _llk_unpack_tilizeA_B_block_init_<NEGINF_SRCA, true /* reload_srcB */, false /* zero_srcA */, ZERO_SRCA_REDUCE>(
        formats.unpack_A_src, formats.unpack_A_dst, params.TILE_CNT, num_faces, UNPB_FACE_R_DIM, face_r_dim);
    for (std::uint32_t block = 0; block < num_blocks; block++)
    {
        _llk_unpack_tilizeA_B_block_<NEGINF_SRCA, true, false, ZERO_SRCA_REDUCE>(
            formats.unpack_A_src,
            face_r_dim,
            L1_ADDRESS(params.buffer_A[0]),
            L1_ADDRESS(params.buffer_B[0]),
            block * tiles_per_block,
            tiles_per_block,
            num_faces);
    }
    _llk_unpack_tilizeA_B_block_uninit_(formats.unpack_A_dst, tensor_shape_from_num_faces(face_r_dim, num_faces));
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
    const TensorShape shape = reduce_shape(params.num_faces);

    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_reduce_init_<POOL_TYPE, REDUCE_DIM, is_fp32_dest_acc_en, MATH_FIDELITY>(shape);
    for (std::uint32_t start = 0; start < params.TILE_CNT; start += MAX_TILES_DEST)
    {
        const std::uint32_t tiles = std::min(params.TILE_CNT - start, MAX_TILES_DEST);
        _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
        for (std::uint32_t tile = 0; tile < tiles; tile++)
        {
            _llk_math_reduce_<POOL_TYPE, REDUCE_DIM, is_fp32_dest_acc_en, MATH_FIDELITY, false>(tile, shape);
        }
        _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }
    _llk_math_reduce_uninit_();
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
    const std::uint32_t num_faces = params.num_faces;
    const std::uint32_t tile_size = PACK_FACE_R_DIM * FACE_C_DIM * num_faces;

    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
        formats.pack_src, formats.pack_dst, tile_size, PACK_FACE_R_DIM, TILE_C_DIM, num_faces, true /* partial_face */, false /* narrow_tile */);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, PACK_FACE_R_DIM, TILE_C_DIM, num_faces, true, false);
    _llk_pack_reduce_mask_config_<REDUCE_DIM>(PACK_FACE_R_DIM);
    _llk_pack_dest_init_wrapper_<DstSync::SyncHalf, is_fp32_dest_acc_en, PackMode::Default>(PACK_FACE_R_DIM, false);
    for (std::uint32_t start = 0; start < params.TILE_CNT; start += MAX_TILES_DEST)
    {
        const std::uint32_t tiles = std::min(params.TILE_CNT - start, MAX_TILES_DEST);
        _llk_packer_wait_for_math_done_();
        for (std::uint32_t tile = 0; tile < tiles; tile++)
        {
            _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, PackMode::Default>(tile, L1_ADDRESS(params.buffer_Res[start + tile]));
        }
        _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }
    _llk_pack_reduce_mask_clear_();
}

#endif
