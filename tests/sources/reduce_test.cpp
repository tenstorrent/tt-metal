// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_AB_reduce.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(const volatile struct RuntimeParams *params)
{
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_src,
        formats.unpack_src,
        formats.unpack_dst,
        formats.unpack_dst,
        params->TEST_FACE_R_DIM,
        params->TEST_FACE_R_DIM,
        params->num_faces,
        params->num_faces);
    _llk_unpack_AB_reduce_init_<POOL_TYPE, REDUCE_DIM, false /* enforce_fp32_accumulation */>(params->TEST_FACE_R_DIM, params->num_faces);
    for (int i = 0; i < params->INPUT_TILE_CNT; ++i)
    {
        _llk_unpack_AB_reduce_<POOL_TYPE, REDUCE_DIM>(L1_ADDRESS(params->buffer_A[i]), L1_ADDRESS(params->buffer_B[0]));
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_reduce.h"
#include "params.h"

void run_kernel(const volatile struct RuntimeParams *params)
{
    const bool is_int_fpu_en             = false;
    const bool enforce_fp32_accumulation = false;

    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_reduce_init_<POOL_TYPE, REDUCE_DIM, is_fp32_dest_acc_en, MATH_FIDELITY, enforce_fp32_accumulation>();

    if (params->IS_REDUCE_TO_ONE)
    {
        _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
        // Reduce all tiles in one go
        for (int i = 0; i < params->INPUT_TILE_CNT; ++i)
        {
            _llk_math_reduce_<POOL_TYPE, REDUCE_DIM, is_fp32_dest_acc_en, MATH_FIDELITY, is_int_fpu_en, enforce_fp32_accumulation>(0);
        }
        _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }
    else
    {
        int remaining_tiles = params->INPUT_TILE_CNT;
        while (remaining_tiles)
        {
            int tiles_to_dest = std::min(remaining_tiles, static_cast<int>(params->NUM_TILES_IN_BLOCK));
            _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
            for (int i = 0; i < tiles_to_dest; ++i)
            {
                _llk_math_reduce_<POOL_TYPE, REDUCE_DIM, is_fp32_dest_acc_en, MATH_FIDELITY, is_int_fpu_en, enforce_fp32_accumulation>(i);
            }
            _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            remaining_tiles -= tiles_to_dest;
        }
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(const volatile struct RuntimeParams *params)
{
    _llk_pack_init_<false, false>(formats.pack_dst);

#ifdef ARCH_BLACKHOLE
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, false /* untilize */, false /* tilize */>(formats.pack_src, formats.pack_dst, 16 * 16 * 4);
#else
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, false /* untilize */>(formats.pack_src, formats.pack_dst, 16 * 16 * 4);
#endif

    _llk_pack_reduce_mask_config_<false /* untilize */, REDUCE_DIM>();

#ifdef ARCH_BLACKHOLE
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
#else
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en, false /* untilize */>();
#endif
    int remaining_tiles = params->OUTPUT_TILE_CNT;
    while (remaining_tiles)
    {
        int tiles_from_dest = std::min(remaining_tiles, static_cast<int>(params->NUM_TILES_IN_BLOCK));
        _llk_packer_wait_for_math_done_();
        for (int i = 0; i < tiles_from_dest; ++i)
        {
            _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, false /* untilize */>(
                i, L1_ADDRESS(params->buffer_Res[params->OUTPUT_TILE_CNT - remaining_tiles + i]));
        }
        _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        remaining_tiles -= tiles_from_dest;
    }
    _llk_pack_reduce_mask_clear_();
}

#endif
