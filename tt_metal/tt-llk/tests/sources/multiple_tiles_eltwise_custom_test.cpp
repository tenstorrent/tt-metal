// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "experimental/llk_unpack_AB_sub_bcast_col_custom.h"
#include "llk_unpack_common.h"
#include "params.h"
#include "tensor_shape.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const ckernel::TensorShape tensor_shape = ckernel::tensor_shape_from_num_faces(params.TEST_FACE_R_DIM, params.num_faces);
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src,
        formats.unpack_B_src,
        formats.unpack_A_dst,
        formats.unpack_B_dst,
        params.TEST_FACE_R_DIM,
        params.TEST_FACE_R_DIM,
        params.num_faces,
        params.num_faces);
    for (int block = 0; block < params.NUM_BLOCKS; ++block)
    {
        _llk_unpack_AB_sub_bcast_col_init_custom_(tensor_shape);
        _llk_unpack_AB_sub_bcast_col_custom_(
            L1_ADDRESS(params.buffer_A[block * params.NUM_TILES_IN_BLOCK]), L1_ADDRESS(params.buffer_B[0]), params.NUM_TILES_IN_BLOCK);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_eltwise_binary_custom.h"
#include "llk_math_common.h"
#include "params.h"
#include "tensor_shape.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const ckernel::TensorShape tensor_shape = ckernel::tensor_shape_from_num_faces(params.TEST_FACE_R_DIM, params.num_faces);
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    // Call the custom LLK once per destination section. The templated MUL/SUB scaffold is
    // Blackhole-only; Wormhole has only the SUB-named wrapper.
    for (int block = 0; block < params.NUM_BLOCKS; ++block)
    {
        _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
        _llk_math_eltwise_binary_init_custom_<ELTWISE_BINARY_OP, BROADCAST_TYPE>(params.num_faces);
#ifdef ARCH_BLACKHOLE
        _llk_math_bcast_cols_reuse_custom_<ELTWISE_BINARY_OP>(params.NUM_TILES_IN_BLOCK, tensor_shape);
#else
        _llk_math_sub_bcast_cols_reuse_custom_(params.NUM_TILES_IN_BLOCK, tensor_shape);
#endif
        _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
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
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
        formats.pack_src,
        formats.pack_dst,
        params.TEST_FACE_R_DIM * FACE_C_DIM * params.num_faces /* tile_size */,
        params.TEST_FACE_R_DIM,
        TILE_C_DIM,
        params.num_faces);

    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, params.TEST_FACE_R_DIM, TILE_C_DIM, params.num_faces);

    _llk_pack_dest_init_wrapper_<DstSync::SyncHalf, is_fp32_dest_acc_en, PackMode::Default>(params.TEST_FACE_R_DIM);

    for (int block = 0; block < params.NUM_BLOCKS; ++block)
    {
        _llk_packer_wait_for_math_done_();
        for (std::uint32_t tile = 0; tile < params.NUM_TILES_IN_BLOCK; ++tile)
        {
            const std::uint32_t result_tile = block * params.NUM_TILES_IN_BLOCK + tile;
            _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile, L1_ADDRESS(params.buffer_Res[result_tile]));
        }
        _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }
}

#endif
