// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"

// Globals
uint32_t unp_cfg_context          = 0;
uint32_t pack_sync_tile_dst_ptr   = 0;
uint32_t math_sync_tile_dst_index = 0;

constexpr static std::uint32_t format_size_in_bytes(uint data_format)
{
    switch (static_cast<DataFormat>(data_format & 0xF))
    {
        case DataFormat::Int32:
        case DataFormat::Float32:
            return 4;

        case DataFormat::Float16:
        case DataFormat::Float16_b:
        case DataFormat::UInt16:
            return 2;

        default:
            return 1;
    }
}

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(const volatile struct RuntimeParams* params)
{
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_src, formats.unpack_src, formats.unpack_dst, formats.unpack_dst, FACE_R_DIM, FACE_R_DIM, params->num_faces, params->num_faces);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0, 0, FACE_R_DIM, params->num_faces, formats.unpack_src, formats.unpack_dst);

    const uint32_t num_blocks_per_col = FULL_CT_DIM / BLOCK_CT_DIM;

    for (uint32_t rt = 0; rt < FULL_RT_DIM; rt++) // Loop over all tiles vertically
    {
        for (uint32_t block_num = 0; block_num < num_blocks_per_col; ++block_num) // Loop over blocks in the column (dst reg)
        {
            for (uint32_t tile_index_within_block = 0; tile_index_within_block < BLOCK_CT_DIM; ++tile_index_within_block) // Loop over tiles in the block
            {
                uint32_t tile_index_in_memory = rt * FULL_CT_DIM + block_num * BLOCK_CT_DIM + tile_index_within_block;
                _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                    L1_ADDRESS(buffer_A[tile_index_in_memory]), formats.unpack_src, formats.unpack_dst);
            }
        }
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
    const bool is_int_fpu_en = false;

// copy srca to dest
#ifdef ARCH_BLACKHOLE
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, is_int_fpu_en>(params->num_faces, formats.math);
#else
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, is_int_fpu_en>(params->num_faces, formats.math);
#endif
    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_(formats.math, formats.math);
#ifdef ARCH_BLACKHOLE
    _llk_math_reconfig_remap_(true);
#endif

    const uint32_t num_blocks_per_col = FULL_CT_DIM / BLOCK_CT_DIM;

    for (uint32_t rt = 0; rt < FULL_RT_DIM; rt++) // Loop over all tiles vertically
    {
        for (uint32_t block_num = 0; block_num < num_blocks_per_col; ++block_num) // Loop over blocks in the column (dst reg)
        {
            _llk_math_wait_for_dest_available_<dest_sync>();
            for (uint32_t tile_index_within_block = 0; tile_index_within_block < BLOCK_CT_DIM; ++tile_index_within_block) // Loop over tiles in the block
            {
                _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, dest_sync, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                    tile_index_within_block, formats.math, formats.math);
            }
            _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
        }
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

constexpr uint32_t L1_ACCESS_ADDRESS_GRANULARITY = 16; // in bytes

void run_kernel(const volatile struct RuntimeParams* params)
{
    const bool UNTILIZE               = true;
    const uint32_t NUM_DATUMS_IN_TILE = FACE_R_DIM * FACE_C_DIM * params->num_faces;
    const uint32_t row_stride_16B     = (FULL_CT_DIM * NUM_DATUMS_IN_TILE * format_size_in_bytes(formats.pack_dst)) / L1_ACCESS_ADDRESS_GRANULARITY;
    const uint32_t block_stride_16B =
        (BLOCK_CT_DIM * ((params->num_faces > 2) ? params->num_faces / 2 : params->num_faces) * FACE_C_DIM * format_size_in_bytes(formats.pack_dst)) /
        L1_ACCESS_ADDRESS_GRANULARITY;
    const uint32_t base_addr_16B = L1_ADDRESS(buffer_Res[0]);

#ifdef ARCH_BLACKHOLE
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, UNTILIZE, false>(formats.pack_src, formats.pack_dst, NUM_DATUMS_IN_TILE /* tile_size */);
    _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();
    _llk_pack_untilize_init_<BLOCK_CT_DIM, FULL_CT_DIM>(formats.pack_src, formats.pack_dst, FACE_R_DIM, params->num_faces);
#else
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, UNTILIZE>(formats.pack_src, formats.pack_dst, NUM_DATUMS_IN_TILE /* tile_size */);
    _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en, UNTILIZE>();
    _llk_pack_untilize_init_<BLOCK_CT_DIM, FULL_CT_DIM>(formats.pack_dst, FACE_R_DIM, params->num_faces);
#endif
    const uint32_t num_blocks_per_col = FULL_CT_DIM / BLOCK_CT_DIM;

    for (uint32_t rt = 0; rt < FULL_RT_DIM; rt++) // Loop over all tiles vertically
    {
        for (uint32_t block_num = 0; block_num < num_blocks_per_col; ++block_num) // Loop over blocks in the column (dst reg)
        {
            uint32_t pack_addr_16B = base_addr_16B + rt * row_stride_16B + block_num * block_stride_16B;

            _llk_packer_wait_for_math_done_();
            _llk_pack_untilize_<BLOCK_CT_DIM, FULL_CT_DIM>(pack_addr_16B, formats.pack_dst, FACE_R_DIM, params->num_faces, 0 /* tile_dst_rt_offset */);
            _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
        }
    }
}

#endif
