// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "ckernel.h"
#include "llk_defs.h"
#include "profiler.h"

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
    // Configure unpacker for Float16_b format
    _llk_unpack_A_hw_configure_<is_fp32_dest_acc_en, StochRndType::None>(formats.unpack_src, formats.unpack_dst, FACE_R_DIM, 0, 4);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0, 0, FACE_R_DIM, 4, formats.unpack_src, formats.unpack_dst);

    // Unpack tiles from L1 to source register A
    for (int i = 0; i < TILE_CNT; ++i)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(buffer_A[i]), formats.unpack_src, formats.unpack_dst);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "params.h"

using namespace ckernel;
using namespace ckernel::sfpu;

void run_kernel()
{
    // Initialize datacopy from srcA to dest
#ifdef ARCH_BLACKHOLE
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, false>(4, formats.math);
#else
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false>(4, formats.math);
#endif
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<false, false>(formats.math, formats.math);

    // Process each tile
    for (int i = 0; i < TILE_CNT; ++i)
    {
        // Wait for destination to be available
        _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();

        // Copy from srcA to dest
        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            i, formats.math, formats.math);
    }

    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif

#include "llk_math_eltwise_unary_sfpu.h"

#ifdef LLK_TRISC_PACK

#include "ckernel_sfpu_reduce_custom.h"
#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel()
{
    // Configure packer hardware
#ifdef ARCH_BLACKHOLE
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, false, false>(formats.pack_src, formats.pack_dst, 16 * 16 * 4);
#else
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, false>(formats.pack_src, formats.pack_dst, 16 * 16 * 4);
#endif

    _llk_pack_init_<false, false, DstTileFaceLayout::RowMajor, false>(formats.pack_dst);

    // Initialize destination for packing
#ifdef ARCH_BLACKHOLE
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileFaceLayout::RowMajor>();
#else
    _llk_pack_dest_init_<DstSync::SyncHalf, false, DstTileFaceLayout::RowMajor, false>();
#endif

    _llk_packer_wait_for_math_done_();

    // SFPU part
    // Initialize SFPU for reduce operation
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::reduce>();

    ckernel::sfpu::_init_reduce_max_col_subblock_4x2_<DataFormat::Float16_b>();

    _llk_math_eltwise_unary_sfpu_start_<DstSync::SyncHalf>(0);
    ckernel::sfpu::_reduce_max_col_subblock_4x2_prologue_();
    ckernel::sfpu::_calculate_reduce_max_col_subblock_4x2_<PoolType::MAX, REDUCE_COL, DataFormat::Float16_b>(BLOCK_RT_DIM);

    ckernel::sfpu::_reduce_max_col_subblock_4x2_epilogue_();

    _llk_math_eltwise_unary_sfpu_done_();

    // Wait for math to finish and pack tiles back to L1

    for (int i = 0; i < TILE_CNT; ++i)
    {
        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, false>(i, L1_ADDRESS(buffer_Res[i]));
    }
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif
