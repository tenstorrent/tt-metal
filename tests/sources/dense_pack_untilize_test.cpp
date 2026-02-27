// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ckernel.h"
#include "llk_assert.h"
#include "llk_defs.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(const volatile struct RuntimeParams* params)
{
#ifdef RUNTIME_FORMATS
    const volatile FormatConfig& formats = params->formats;
#endif
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, params->num_faces, params->num_faces);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0, 0, FACE_R_DIM, params->num_faces, formats.unpack_A_src, formats.unpack_A_dst);

    for (std::uint32_t tile_index_within_block = 0; tile_index_within_block < BLOCK_CT_DIM; ++tile_index_within_block) // Loop over tiles in the block
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(params->buffer_A[tile_index_within_block]), formats.unpack_A_src, formats.unpack_A_dst);
    }
}
#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "params.h"

using namespace ckernel;

void run_kernel(const volatile struct RuntimeParams* params)
{
#ifdef RUNTIME_FORMATS
    const volatile FormatConfig& formats = params->formats;
#endif
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, false>(params->num_faces, formats.math);
    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_reconfig_remap_(true);

    _llk_math_wait_for_dest_available_<dest_sync>();
    for (std::uint32_t tile_index_within_block = 0; tile_index_within_block < BLOCK_CT_DIM; ++tile_index_within_block) // Loop over tiles in the block
    {
        LLK_ASSERT(
            (tile_index_within_block < get_dest_max_tiles<dest_sync, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
            "Block tile index exceeds maximum destination tiles");
        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, dest_sync, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            tile_index_within_block, formats.math, formats.math);
    }
    _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(const volatile struct RuntimeParams* params)
{
#ifdef RUNTIME_FORMATS
    const volatile FormatConfig& formats = params->formats;
#endif
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, false, false>(formats.pack_src, formats.pack_dst, 0 /* tile_size */);
    _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();
    // Ugly but we are converting one 4 face tile into two 2 face tiles side by side
    _llk_pack_untilize_init_<BLOCK_CT_DIM * 2, FULL_CT_DIM * 2, false, TILE_C_DIM, true>(
        formats.pack_src, formats.pack_dst, params->in0_tile_r_dim, params->num_faces / 2);

    _llk_packer_wait_for_math_done_();
    _llk_pack_untilize_<BLOCK_CT_DIM * 2, FULL_CT_DIM * 2, false, 0, true>(
        L1_ADDRESS(params->buffer_Res[0]), formats.pack_dst, params->in0_tile_r_dim, params->num_faces / 2, 0 /* tile_dst_rt_offset */);
    _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
}

#endif
