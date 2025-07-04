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

#include "llk_unpack_AB_matmul.h"
#include "params.h"

void run_kernel()
{
    std::uint32_t ct_dim = BLOCK_CT_DIM;
    std::uint32_t rt_dim = BLOCK_RT_DIM;
    std::uint32_t kt_dim = BLOCK_CT_DIM; // for square matrices, kt_dim == ct_dim

    std::uint32_t tile_size = 128;

    if constexpr (static_cast<DataFormat>(UNPACK_A_IN) == DataFormat::Bfp8_b)
    {
        tile_size = 68;
    }
    else if constexpr (static_cast<DataFormat>(UNPACK_A_IN) == DataFormat::Float32)
    {
        tile_size = 256;
    }

    _llk_unpack_AB_matmul_hw_configure_<is_fp32_dest_acc_en, StochRndType::None>(
        UNPACK_A_IN, UNPACK_B_IN, UNPACK_A_OUT, UNPACK_B_OUT, FACE_R_DIM, FACE_R_DIM, 0, 4, 4, tile_size, tile_size);
    _llk_unpack_AB_matmul_init_<>(0, ct_dim, rt_dim, kt_dim, FACE_R_DIM, FACE_R_DIM);
    for (uint32_t j = 0; j < kt_dim; j++)
    {
        _llk_unpack_AB_matmul_<>(
            L1_ADDRESS(buffer_A[0]),
            L1_ADDRESS(buffer_B[0]),
            j,
            j * ct_dim,
            tile_size,
            tile_size,
            FACE_R_DIM,
            FACE_R_DIM,
            false,
            false,
            ct_dim,
            rt_dim,
            kt_dim);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_matmul.h"
#include "params.h"

void run_kernel()
{
    std::uint32_t ct_dim = BLOCK_CT_DIM;
    std::uint32_t rt_dim = BLOCK_RT_DIM;
    std::uint32_t kt_dim = BLOCK_CT_DIM; // for square matrices, kt_dim == ct_dim

    _llk_math_matmul_init_<MATH_FIDELITY, DstTileFaceLayout::RowMajor>(TILE_R_DIM, TILE_C_DIM, TILE_R_DIM, TILE_C_DIM, false, 0, ct_dim, rt_dim, kt_dim);
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<false, false>(MATH_FORMAT, MATH_FORMAT);
    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    for (uint32_t j = 0; j < kt_dim; j++)
    {
        _llk_math_matmul_<MATH_FIDELITY, DstTileFaceLayout::RowMajor>(0, 0, ct_dim, rt_dim, kt_dim);
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
#ifdef ARCH_BLACKHOLE
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, false, false>(PACK_IN, PACK_OUT, 16 * 16 * 4);
    _llk_pack_init_<false, false, DstTileFaceLayout::RowMajor, false, false>(PACK_OUT);
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileFaceLayout::RowMajor>();
#else
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, false>(PACK_IN, PACK_OUT, 16 * 16 * 4);
    _llk_pack_init_<false, false, DstTileFaceLayout::RowMajor, false>(PACK_OUT);
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileFaceLayout::RowMajor, false>();
#endif

    for (int i = 0; i < TILE_CNT; i++)
    {
        _llk_packer_wait_for_math_done_();
        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, false>(i, L1_ADDRESS(buffer_Res[i]));
    }
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif
