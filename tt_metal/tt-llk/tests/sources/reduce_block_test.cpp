// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// reduce_test.cpp with the block calls: _llk_unpack_AB_reduce_block_ and _llk_math_reduce_block_ take a block of tiles per
// call. Reduce to one accumulates blocks of INPUT_NUM_TILES_IN_BLOCK tiles into DEST tile 0; otherwise each DEST section of
// NUM_TILES_IN_BLOCK tiles is one block, tile i into DEST tile i.
#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"
#include "tensor_shape.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_AB_reduce.h"
#include "llk_unpack_common.h"
#include "params.h"
#include "tensor_shape.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const ckernel::TensorShape tensor_shape = {
        static_cast<std::uint8_t>(params.in0_face_r_dim),
        static_cast<std::uint8_t>(params.in0_face_c_dim),
        static_cast<std::uint8_t>(params.num_faces_r_dim_A),
        static_cast<std::uint8_t>(params.num_faces_c_dim_A)};
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
    _llk_unpack_AB_reduce_init_<POOL_TYPE, REDUCE_DIM>(tensor_shape);

    const std::uint32_t tile_cnt = static_cast<std::uint32_t>(params.INPUT_TILE_CNT);
    const std::uint32_t stride   = tile_cnt > 1 ? L1_ADDRESS(params.buffer_A[1]) - L1_ADDRESS(params.buffer_A[0]) : 0;
    const std::uint32_t block    = params.IS_REDUCE_TO_ONE ? params.INPUT_NUM_TILES_IN_BLOCK : params.NUM_TILES_IN_BLOCK;
    for (std::uint32_t start = 0; start < tile_cnt; start += block)
    {
        const std::uint32_t n = std::min(block, tile_cnt - start);
        _llk_unpack_AB_reduce_block_<POOL_TYPE, REDUCE_DIM>(
            L1_ADDRESS(params.buffer_A[start]), L1_ADDRESS(params.buffer_B[0]), n, stride, formats.unpack_A_src, tensor_shape);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_lib_math_wrappers.h"
#include "llk_math_common.h"
#include "params.h"
#include "tensor_shape.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const ckernel::TensorShape tensor_shape = {
        static_cast<std::uint8_t>(params.in0_face_r_dim),
        static_cast<std::uint8_t>(params.in0_face_c_dim),
        static_cast<std::uint8_t>(params.num_faces_r_dim_A),
        static_cast<std::uint8_t>(params.num_faces_c_dim_A)};

    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_reduce_init_<POOL_TYPE, REDUCE_DIM, is_fp32_dest_acc_en, MATH_FIDELITY>(tensor_shape);

    const std::uint32_t tile_cnt = static_cast<std::uint32_t>(params.INPUT_TILE_CNT);
    if (params.IS_REDUCE_TO_ONE)
    {
        const std::uint32_t block = params.INPUT_NUM_TILES_IN_BLOCK;
        _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
        for (std::uint32_t start = 0; start < tile_cnt; start += block)
        {
            _llk_math_reduce_block_<POOL_TYPE, REDUCE_DIM, is_fp32_dest_acc_en, MATH_FIDELITY>(0, std::min(block, tile_cnt - start), 0, tensor_shape);
        }
        _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }
    else
    {
        for (std::uint32_t start = 0; start < tile_cnt; start += params.NUM_TILES_IN_BLOCK)
        {
            _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
            _llk_math_reduce_block_<POOL_TYPE, REDUCE_DIM, is_fp32_dest_acc_en, MATH_FIDELITY>(
                0, std::min(params.NUM_TILES_IN_BLOCK, tile_cnt - start), 1, tensor_shape);
            _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        }
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
    const ckernel::TensorShape tensor_shape = {
        static_cast<std::uint8_t>(params.in0_face_r_dim),
        static_cast<std::uint8_t>(params.in0_face_c_dim),
        static_cast<std::uint8_t>(params.num_faces_r_dim_A),
        static_cast<std::uint8_t>(params.num_faces_c_dim_A)};

    const std::uint32_t tile_size = tensor_shape.total_tensor_size();
    const std::uint32_t num_faces = tensor_shape.total_num_faces();
    const bool partial_face       = tensor_shape.face_r_dim < FACE_R_DIM;
    const bool narrow_tile        = tensor_shape.num_faces_c_dim == 1;

    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
        formats.pack_src, formats.pack_dst, tile_size, tensor_shape.face_r_dim, tensor_shape.total_col_dim(), num_faces, partial_face, narrow_tile);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(
        formats.pack_dst, tensor_shape.face_r_dim, tensor_shape.total_col_dim(), num_faces, partial_face, narrow_tile);
    _llk_pack_reduce_mask_config_<REDUCE_DIM>(tensor_shape.face_r_dim);
    _llk_pack_dest_init_wrapper_<DstSync::SyncHalf, is_fp32_dest_acc_en, PackMode::Default>(tensor_shape.face_r_dim, narrow_tile);

    int remaining_tiles = params.OUTPUT_TILE_CNT;
    while (remaining_tiles != 0)
    {
        int tiles_from_dest = std::min(remaining_tiles, static_cast<int>(params.NUM_TILES_IN_BLOCK));
        _llk_packer_wait_for_math_done_();
        for (int i = 0; i < tiles_from_dest; ++i)
        {
            _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(
                i, L1_ADDRESS(params.buffer_Res[params.OUTPUT_TILE_CNT - remaining_tiles + i]));
        }
        _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        remaining_tiles -= tiles_from_dest;
    }
    _llk_pack_reduce_mask_clear_();
}

#endif
