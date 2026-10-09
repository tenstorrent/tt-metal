// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Re-initialises the math/pack dest handshake between consecutive blocks of a datacopy with nothing
// else in between: the pack thread calls _llk_pack_dest_init_ straight after its previous block's last
// PACR, and the math thread calls _llk_math_pack_sync_init_ straight after its previous block's
// section-done. REINIT_DELAY RISC nops before the pack-side re-init move it relative to that PACR, so a
// sweep over REINIT_DELAY shows whether the re-init's config writes are ordered behind the pack by the
// LLK itself or only by luck.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_lib_unpack_wrappers.h"
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const FormatConfig& formats            = params.formats;
    const std::uint32_t num_faces          = params.num_faces;
    const int NUM_BLOCKS                   = params.NUM_BLOCKS;
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;
    const Operand& buffer_A                = params.buffer_A;
    const std::uint32_t num_tiles          = NUM_BLOCKS * NUM_TILES_IN_BLOCK;

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, num_faces, num_faces);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0 /* transpose_of_faces */,
        0 /* within_face_16x16_transpose */,
        ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, num_faces),
        formats.unpack_A_src,
        formats.unpack_A_dst);

    for (std::uint32_t i = 0; i < num_tiles; ++i)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(buffer_A[i]), formats.unpack_A_src, formats.unpack_A_dst);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_lib_math_wrappers.h"
#include "llk_lib_unpack_wrappers.h"
#include "params.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
    const FormatConfig& formats            = params.formats;
    const std::uint32_t num_faces          = params.num_faces;
    const int NUM_BLOCKS                   = params.NUM_BLOCKS;
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;
    const int DST_INDEX                    = params.DST_INDEX;

    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, PackMode::Default>(
        num_faces, formats.math);
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    for (int block_num = 0; block_num < NUM_BLOCKS; ++block_num)
    {
        // Re-init of the dest handshake for every block, directly behind the previous block's section-done.
        _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

        _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
        for (std::uint32_t tile_num = 0; tile_num < NUM_TILES_IN_BLOCK; ++tile_num)
        {
            _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                DST_INDEX + tile_num, formats.math, formats.math, num_faces);
        }
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
    const FormatConfig& formats            = params.formats;
    const std::uint32_t num_faces          = params.num_faces;
    const int NUM_BLOCKS                   = params.NUM_BLOCKS;
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;
    const int DST_INDEX                    = params.DST_INDEX;
    const Operand& buffer_Res              = params.buffer_Res;

    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
        formats.pack_src, formats.pack_dst, 16 * 16 * 4 /* tile_size */, FACE_R_DIM, TILE_C_DIM, num_faces);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, num_faces);

    for (int block_num = 0; block_num < NUM_BLOCKS; ++block_num)
    {
        // Move the re-init relative to the previous block's last PACR, which is still in flight here.
        _Pragma("GCC unroll 256") for (std::uint32_t i = 0; i < REINIT_DELAY; ++i)
        {
            asm volatile("nop");
        }
        // Re-init of the dest handshake for every block, directly behind the previous block's packs.
        _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

        _llk_packer_wait_for_math_done_();
        for (std::uint32_t tile_num = 0; tile_num < NUM_TILES_IN_BLOCK; ++tile_num)
        {
            _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(
                DST_INDEX + tile_num, L1_ADDRESS(buffer_Res[block_num * NUM_TILES_IN_BLOCK + tile_num]));
        }
        _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }
}

#endif
