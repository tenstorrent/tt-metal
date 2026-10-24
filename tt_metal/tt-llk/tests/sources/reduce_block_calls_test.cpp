// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Reduce to one with a list of calls into DEST tile 0 on one init: REDUCE_CALLS holds, per call, n > 0 for a block call of
// n tiles or -n for n single-tile calls, in input order.
#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"
#include "tensor_shape.h"

constexpr bool SCALAR_PAIR = REDUCE_DIM == ckernel::ReduceDim::REDUCE_SCALAR && POOL_TYPE != ckernel::PoolType::MAX;

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
    _llk_unpack_AB_reduce_init_<POOL_TYPE, REDUCE_DIM, (MATH_FIDELITY != ckernel::MathFidelity::LoFi)>(tensor_shape);

    const std::uint32_t tile_cnt = static_cast<std::uint32_t>(params.INPUT_TILE_CNT);
    const std::uint32_t stride   = tile_cnt > 1 ? L1_ADDRESS(params.buffer_A[1]) - L1_ADDRESS(params.buffer_A[0]) : 0;
    std::uint32_t start          = 0;
    for (const int call : REDUCE_CALLS)
    {
        if (call > 0)
        {
            _llk_unpack_AB_reduce_block_<POOL_TYPE, REDUCE_DIM>(
                L1_ADDRESS(params.buffer_A[start]), L1_ADDRESS(params.buffer_B[0]), call, stride, formats.unpack_A_src, tensor_shape, SCALAR_PAIR);
            start += call;
        }
        else
        {
            for (int i = 0; i < -call; i++)
            {
                _llk_unpack_AB_reduce_<POOL_TYPE, REDUCE_DIM>(L1_ADDRESS(params.buffer_A[start++]), L1_ADDRESS(params.buffer_B[0]));
            }
        }
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
    _llk_math_reduce_init_<POOL_TYPE, REDUCE_DIM, is_fp32_dest_acc_en, MATH_FIDELITY, (MATH_FIDELITY != ckernel::MathFidelity::LoFi)>(tensor_shape);

    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    for (const int call : REDUCE_CALLS)
    {
        if (call > 0)
        {
            _llk_math_reduce_block_<POOL_TYPE, REDUCE_DIM, is_fp32_dest_acc_en, MATH_FIDELITY>(0, call, 0, tensor_shape, SCALAR_PAIR);
        }
        else
        {
            for (int i = 0; i < -call; i++)
            {
                _llk_math_reduce_<POOL_TYPE, REDUCE_DIM, is_fp32_dest_acc_en, MATH_FIDELITY>(0, tensor_shape);
            }
        }
    }
    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
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
    _llk_pack_init_wrapper_<PackMode::Default, false>(
        formats.pack_dst, tensor_shape.face_r_dim, tensor_shape.total_col_dim(), num_faces, partial_face, narrow_tile);
    _llk_pack_reduce_mask_config_<REDUCE_DIM>(tensor_shape.face_r_dim);
    _llk_pack_dest_init_wrapper_<DstSync::SyncHalf, is_fp32_dest_acc_en, PackMode::Default>(tensor_shape.face_r_dim, narrow_tile);

    _llk_packer_wait_for_math_done_();
    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(0, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_pack_reduce_mask_clear_();
}

#endif
