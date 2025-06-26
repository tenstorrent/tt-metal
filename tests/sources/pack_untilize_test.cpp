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

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel()
{
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(0, 0, FACE_R_DIM, 4, UNPACK_A_IN, UNPACK_A_OUT);
    _llk_unpack_A_hw_configure_<is_fp32_dest_acc_en, StochRndType::None>(UNPACK_A_IN, UNPACK_A_OUT, FACE_R_DIM, 0, 4);
    for (int i = 0; i < TILE_CNT; ++i)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(L1_ADDRESS(buffer_A[i]), 0, UNPACK_A_IN, UNPACK_A_OUT);
    }
}

#endif

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
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, is_int_fpu_en>(0, 0, 4, MATH_FORMAT);
#else
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, is_int_fpu_en>(0, 0, 4, MATH_FORMAT);
#endif
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<true, false>(MATH_FORMAT, MATH_FORMAT);
    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    for (int i = 0; i < TILE_CNT; ++i)
    {
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
    const std::uint32_t ct_dim = 1;
    const bool UNTILIZE        = true;

#ifdef ARCH_BLACKHOLE
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, UNTILIZE, false>(PACK_IN, PACK_OUT, 16 * 16 * 4);
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileFaceLayout::RowMajor>();
    _llk_pack_untilize_init_<BLOCK_CT_DIM>(PACK_IN, PACK_OUT, FACE_R_DIM, 4);
#else
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, UNTILIZE>(PACK_IN, PACK_OUT, 16 * 16 * 4);
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileFaceLayout::RowMajor, UNTILIZE>();
    _llk_pack_untilize_init_<BLOCK_CT_DIM>(PACK_OUT, FACE_R_DIM, 4);
#endif

    _llk_packer_wait_for_math_done_();

    _llk_pack_untilize_<BLOCK_CT_DIM>(L1_ADDRESS(buffer_Res[0]), PACK_OUT, FACE_R_DIM, 4, 0);
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif
