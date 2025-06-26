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

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_common.h"
#include "llk_unpack_tilize.h"
#include "params.h"

void run_kernel()
{
    _llk_unpack_tilize_hw_configure_<is_fp32_dest_acc_en, StochRndType::None>(UNPACK_A_IN, UNPACK_A_OUT, FACE_R_DIM, 0, 4);
    _llk_unpack_tilize_init_(UNPACK_A_IN, UNPACK_A_OUT, BLOCK_CT_DIM, FACE_R_DIM, false);

    uint32_t read_offset = 0;

    for (uint32_t i = 0; i < BLOCK_RT_DIM; i++)
    {
        for (uint32_t j = 0; j < BLOCK_CT_DIM; j++)
        {
            _llk_unpack_tilize_(L1_ADDRESS(buffer_A[read_offset]), j, UNPACK_A_IN, BLOCK_CT_DIM, FACE_R_DIM, 4, false);
        }
        read_offset += BLOCK_RT_DIM;
    }
}

#endif

const bool TILIZE = true;

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "params.h"

using namespace ckernel;

void run_kernel()
{
    const bool is_int_fpu_en = false;

// copy srca to dest
#ifdef ARCH_BLACKHOLE
    // set tilize flag to true
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, TILIZE, is_int_fpu_en>(0, 0, 4, MATH_FORMAT);
#else
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, is_int_fpu_en>(0, 0, 4, MATH_FORMAT);
#endif
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<false, false>(MATH_FORMAT, MATH_FORMAT);
    for (int i = 0; i < TILE_CNT; ++i)
    {
        _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, false>(i, MATH_FORMAT, MATH_FORMAT);
    }
    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel()
{
    const bool UNTILIIZE = false;

#ifdef ARCH_BLACKHOLE
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, UNTILIIZE, TILIZE>(PACK_IN, PACK_OUT, 16 * 16 * 4);
    _llk_pack_init_<UNTILIIZE, false, DstTileFaceLayout::RowMajor, false, TILIZE>(PACK_OUT);
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileFaceLayout::RowMajor>();
#else
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, UNTILIIZE>(PACK_IN, PACK_OUT, 16 * 16 * 4);
    _llk_pack_init_<UNTILIIZE, false, DstTileFaceLayout::RowMajor, false>(PACK_OUT);
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileFaceLayout::RowMajor, UNTILIIZE>();
#endif

    _llk_packer_wait_for_math_done_();
    for (int i = 0; i < TILE_CNT; ++i)
    {
        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, UNTILIIZE>(i, L1_ADDRESS(buffer_Res[i]));
    }
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif
